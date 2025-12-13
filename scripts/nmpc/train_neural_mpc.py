"""Train neural MPC model to map discrete A* plans to spline parameters
This script loads the generated dataset and trains a neural network that learns
to predict B-spline control points from discrete graph plans, using ship dynamics information.
The spline parameters can then be used to generate smooth curved paths.
"""
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class PathDataset(Dataset):
    """Dataset for discrete plan -> spline control points mapping"""
    
    def __init__(self, dataset, max_discrete_nodes=50, num_spline_control_points=15):
        self.dataset = dataset
        self.max_discrete_nodes = max_discrete_nodes
        self.num_spline_control_points = num_spline_control_points
        
        # Preprocess dataset
        self.processed_samples = []
        for sample in dataset:
            processed = self._process_sample(sample)
            if processed is not None:
                self.processed_samples.append(processed)
        
        print(f"Processed {len(self.processed_samples)} samples from {len(dataset)} raw samples")
    
    def _process_sample(self, sample):
        """Process a single sample to fixed-size tensors"""
        discrete_plan = sample['discrete_plan']  # (N, 3)
        spline_control_points = sample.get('spline_control_points')  # (K, 2)
        start_pose = sample['start_pose']  # (3,)
        start_nu = sample['start_nu']  # (3,)
        goal = sample['goal']  # (2,)
        
        # If spline_control_points not available, skip (old dataset format)
        if spline_control_points is None:
            return None
        
        # Pad/truncate discrete plan
        N = len(discrete_plan)
        if N > self.max_discrete_nodes:
            # Take evenly spaced nodes
            indices = np.linspace(0, N-1, self.max_discrete_nodes, dtype=int)
            discrete_plan = discrete_plan[indices]
            N = self.max_discrete_nodes
        
        # Pad discrete plan with zeros
        discrete_padded = np.zeros((self.max_discrete_nodes, 3))
        discrete_padded[:N] = discrete_plan
        discrete_mask = np.zeros(self.max_discrete_nodes, dtype=bool)
        discrete_mask[:N] = True
        
        # Process spline control points
        K = len(spline_control_points)
        if K > self.num_spline_control_points:
            # Take evenly spaced control points
            indices = np.linspace(0, K-1, self.num_spline_control_points, dtype=int)
            spline_control_points = spline_control_points[indices]
            K = self.num_spline_control_points
        elif K < self.num_spline_control_points:
            # Pad with last point
            last_point = spline_control_points[-1] if K > 0 else np.array([0.0, 0.0])
            padding = np.tile(last_point, (self.num_spline_control_points - K, 1))
            spline_control_points = np.vstack([spline_control_points, padding])
            K = self.num_spline_control_points
        
        # Normalize positions relative to start
        discrete_normalized = discrete_padded.copy()
        discrete_normalized[:, :2] -= start_pose[:2]
        
        spline_normalized = spline_control_points.copy()
        spline_normalized[:, :2] -= start_pose[:2]
        
        # Normalize goal relative to start
        goal_normalized = goal - start_pose[:2]
        
        return {
            'discrete_plan': discrete_normalized.astype(np.float32),
            'discrete_mask': discrete_mask,
            'spline_control_points': spline_normalized.astype(np.float32),
            'start_pose': start_pose.astype(np.float32),
            'start_nu': start_nu.astype(np.float32),
            'goal': goal_normalized.astype(np.float32),
            'num_discrete': N,
            'num_spline_control': K
        }
    
    def __len__(self):
        return len(self.processed_samples)
    
    def __getitem__(self, idx):
        return self.processed_samples[idx]


class NeuralMPC(nn.Module):
    """Neural network for mapping discrete plans to spline control points"""
    
    def __init__(self, 
                 discrete_dim=3,  # (x, y, psi) per node
                 spline_dim=2,  # (x, y) per control point
                 hidden_dim=256,
                 num_layers=3,
                 max_discrete_nodes=50,
                 num_spline_control_points=15):
        super(NeuralMPC, self).__init__()
        
        self.max_discrete_nodes = max_discrete_nodes
        self.num_spline_control_points = num_spline_control_points
        
        # Encoder: Process discrete plan
        self.discrete_encoder = nn.LSTM(
            input_size=discrete_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Context encoder: Encode start state and goal
        self.context_encoder = nn.Sequential(
            nn.Linear(3 + 3 + 2, hidden_dim),  # start_pose (3) + start_nu (3) + goal (2)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Decoder: Generate spline control points
        # Use LSTM decoder
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim * 2 + hidden_dim,  # discrete features + context
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output projection to spline control points
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, spline_dim)  # Output (x, y) for each control point
        )
    
    def forward(self, discrete_plan, discrete_mask, start_pose, start_nu, goal):
        """
        Args:
            discrete_plan: (B, max_nodes, 3) - normalized discrete waypoints
            discrete_mask: (B, max_nodes) - mask for valid nodes
            start_pose: (B, 3) - [x, y, psi]
            start_nu: (B, 3) - [u, v, r]
            goal: (B, 2) - normalized goal position
        
        Returns:
            spline_control_points: (B, num_control_points, 2) - predicted spline control points
        """
        batch_size = discrete_plan.shape[0]
        
        # Encode discrete plan
        lengths = discrete_mask.sum(dim=1).cpu().numpy()
        packed = nn.utils.rnn.pack_padded_sequence(
            discrete_plan, lengths, batch_first=True, enforce_sorted=False
        )
        discrete_encoded, (h_n, c_n) = self.discrete_encoder(packed)
        discrete_encoded, _ = nn.utils.rnn.pad_packed_sequence(
            discrete_encoded, batch_first=True, total_length=self.max_discrete_nodes
        )
        
        # Get final discrete encoding (use last valid node)
        discrete_features = []
        for i in range(batch_size):
            valid_len = int(lengths[i])
            if valid_len > 0:
                discrete_features.append(discrete_encoded[i, valid_len-1])
            else:
                discrete_features.append(discrete_encoded[i, 0])
        discrete_features = torch.stack(discrete_features)  # (B, hidden_dim * 2)
        
        # Encode context
        context_input = torch.cat([start_pose, start_nu, goal], dim=1)
        context_features = self.context_encoder(context_input)  # (B, hidden_dim)
        
        # Combine features for decoder
        decoder_input = torch.cat([discrete_features, context_features], dim=1)  # (B, hidden_dim * 2 + hidden_dim)
        decoder_input = decoder_input.unsqueeze(1).expand(-1, self.num_spline_control_points, -1)
        
        # Decode spline control points
        decoder_output, _ = self.decoder_lstm(decoder_input)
        
        # Project to output (spline control points)
        spline_control_points = self.output_proj(decoder_output)  # (B, num_control_points, 2)
        
        return spline_control_points


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc='Training'):
        # Move to device
        discrete_plan = batch['discrete_plan'].to(device)
        discrete_mask = batch['discrete_mask'].to(device)
        spline_control_points = batch['spline_control_points'].to(device)
        start_pose = batch['start_pose'].to(device)
        start_nu = batch['start_nu'].to(device)
        goal = batch['goal'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        pred_control_points = model(discrete_plan, discrete_mask, start_pose, start_nu, goal)
        
        # Compute loss on spline control points
        loss = criterion(pred_control_points, spline_control_points)
        loss = loss.mean()  # Average over all control points
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            discrete_plan = batch['discrete_plan'].to(device)
            discrete_mask = batch['discrete_mask'].to(device)
            spline_control_points = batch['spline_control_points'].to(device)
            start_pose = batch['start_pose'].to(device)
            start_nu = batch['start_nu'].to(device)
            goal = batch['goal'].to(device)
            
            pred_control_points = model(discrete_plan, discrete_mask, start_pose, start_nu, goal)
            
            loss = criterion(pred_control_points, spline_control_points)
            loss = loss.mean()
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train neural MPC model')
    parser.add_argument('--dataset', type=str, default='neural_mpc_dataset.pkl',
                       help='Path to dataset pickle file')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='neural_mpc_models',
                       help='Output directory for saved models')
    parser.add_argument('--max_discrete', type=int, default=50,
                       help='Maximum number of discrete nodes')
    parser.add_argument('--num_spline_control', type=int, default=15,
                       help='Number of spline control points')
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    with open(args.dataset, 'rb') as f:
        data = pickle.load(f)
    
    dataset_list = data['dataset']
    config = data['config']
    
    print(f"Loaded {len(dataset_list)} samples")
    print(f"Config: {config}")
    
    # Create dataset
    path_dataset = PathDataset(
        dataset_list,
        max_discrete_nodes=args.max_discrete,
        num_spline_control_points=args.num_spline_control
    )
    
    # Split train/val
    train_size = int(0.8 * len(path_dataset))
    val_size = len(path_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        path_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    model = NeuralMPC(
        max_discrete_nodes=args.max_discrete,
        num_spline_control_points=args.num_spline_control
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }, output_dir / 'best_model.pth')
            print(f"Saved best model (val_loss: {val_loss:.6f})")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Curves')
    plt.savefig(output_dir / 'training_curves.png')
    plt.close()
    
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.6f}")
    print(f"Models saved to {output_dir}")


if __name__ == '__main__':
    main()

