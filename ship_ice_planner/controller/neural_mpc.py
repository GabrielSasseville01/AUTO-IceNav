"""Neural Model Predictive Controller for ship path planning
Uses a trained neural network to predict spline parameters from discrete A* plans,
then generates smooth spline curves. Supports gradient-based optimization using costmap at runtime.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from pathlib import Path
from scipy.interpolate import BSpline, splrep, splev

from ship_ice_planner.cost_map import CostMap
from ship_ice_planner.geometry.utils import Rxy
from ship_ice_planner.utils.utils import compute_path_length

try:
    from ship_ice_planner.controller.critic import HybridCritic, create_critic_for_costmap
    CRITIC_AVAILABLE = True
except ImportError:
    CRITIC_AVAILABLE = False


class NeuralMPC(nn.Module):
    """Neural network for mapping discrete plans to spline control points"""
    
    def __init__(self, 
                 discrete_dim=3,
                 spline_dim=2,
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
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim * 2 + hidden_dim,
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
        
        # Get final discrete encoding
        discrete_features = []
        for i in range(batch_size):
            valid_len = int(lengths[i])
            if valid_len > 0:
                discrete_features.append(discrete_encoded[i, valid_len-1])
            else:
                discrete_features.append(discrete_encoded[i, 0])
        discrete_features = torch.stack(discrete_features)
        
        # Encode context
        context_input = torch.cat([start_pose, start_nu, goal], dim=1)
        context_features = self.context_encoder(context_input)
        
        # Combine features for decoder
        decoder_input = torch.cat([discrete_features, context_features], dim=1)
        decoder_input = decoder_input.unsqueeze(1).expand(-1, self.num_spline_control_points, -1)
        
        # Decode spline control points
        decoder_output, _ = self.decoder_lstm(decoder_input)
        
        # Project to output (spline control points)
        spline_control_points = self.output_proj(decoder_output)
        
        return spline_control_points


def generate_spline_from_control_points(control_points, num_points=500, degree=3, start_pose=None):
    """
    Generate a smooth spline path from control points.
    
    Args:
        control_points: (N, 2) control points [x, y]
        num_points: Number of points to generate along the spline
        degree: Degree of the B-spline (3 = cubic)
        start_pose: Optional (3,) starting pose to ensure path starts correctly
    
    Returns:
        path: (num_points, 3) path [x, y, psi]
    """
    if len(control_points) < 2:
        # Not enough points, return straight line
        if start_pose is not None:
            path = np.zeros((num_points, 3))
            path[:, 0] = start_pose[0]
            path[:, 1] = np.linspace(start_pose[1], start_pose[1] + 100, num_points)
            path[:, 2] = start_pose[2]
            return path
        else:
            return np.zeros((num_points, 3))
    
    # Create parameter values
    n = len(control_points)
    if n < degree + 1:
        degree = max(1, n - 1)
    
    # Fit B-spline
    try:
        # Use uniform knot vector
        knots = np.linspace(0, 1, n + degree + 1)
        
        # Create B-spline
        t = np.linspace(0, 1, num_points)
        
        # Fit spline for x and y separately
        tck_x, u_x = splrep(np.linspace(0, 1, n), control_points[:, 0], k=degree, s=0)
        tck_y, u_y = splrep(np.linspace(0, 1, n), control_points[:, 1], k=degree, s=0)
        
        # Evaluate spline
        x_vals = splev(t, tck_x)
        y_vals = splev(t, tck_y)
        
        # Compute heading from path
        dx = np.diff(x_vals)
        dy = np.diff(y_vals)
        psi = np.arctan2(dy, dx)
        # Append last heading
        psi = np.append(psi, psi[-1])
        
        path = np.column_stack([x_vals, y_vals, psi])
        
        # Ensure path starts at start_pose if provided
        if start_pose is not None:
            path[0, :2] = start_pose[:2]
            path[0, 2] = start_pose[2]
        
        return path
    except Exception as e:
        # Fallback: linear interpolation
        t = np.linspace(0, 1, num_points)
        x_vals = np.interp(t, np.linspace(0, 1, n), control_points[:, 0])
        y_vals = np.interp(t, np.linspace(0, 1, n), control_points[:, 1])
        dx = np.diff(x_vals)
        dy = np.diff(y_vals)
        psi = np.arctan2(dy, dx)
        psi = np.append(psi, psi[-1])
        path = np.column_stack([x_vals, y_vals, psi])
        
        if start_pose is not None:
            path[0, :2] = start_pose[:2]
            path[0, 2] = start_pose[2]
        
        return path


class NeuralMPCController:
    """Neural Model Predictive Controller for ship path planning using splines"""
    
    def __init__(self,
                 model_path: str,
                 horizon: int = 200,
                 max_discrete_nodes: int = 50,
                 num_spline_control_points: int = 15,
                 device: str = 'cpu',
                 use_gradient_optimization: bool = True,
                 optimization_steps: int = 10,
                 learning_rate: float = 0.01,
                 critic_path: Optional[str] = None,
                 costmap: Optional[CostMap] = None):
        """
        Initialize Neural MPC Controller
        
        Args:
            model_path: Path to trained model checkpoint
            horizon: Planning horizon (number of steps)
            max_discrete_nodes: Maximum number of discrete nodes
            num_spline_control_points: Number of spline control points
            device: Device to run model on ('cpu' or 'cuda')
            use_gradient_optimization: Whether to use gradient-based optimization at runtime
            optimization_steps: Number of gradient steps for optimization
            learning_rate: Learning rate for gradient optimization
        """
        self.horizon = horizon
        self.max_discrete_nodes = max_discrete_nodes
        self.num_spline_control_points = num_spline_control_points
        self.device = torch.device(device)
        self.use_gradient_optimization = use_gradient_optimization
        self.optimization_steps = optimization_steps
        self.learning_rate = learning_rate
        
        # Load model
        print(f"Loading Neural MPC model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model
        self.model = NeuralMPC(
            max_discrete_nodes=max_discrete_nodes,
            num_spline_control_points=num_spline_control_points
        ).to(self.device)
        
        # Load weights (strict=False to handle minor differences)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)
        
        self.model.eval()
        print("Model loaded successfully")
        
        # Load critic if provided
        self.critic = None
        if critic_path and CRITIC_AVAILABLE and costmap is not None:
            try:
                checkpoint = torch.load(critic_path, map_location=self.device)
                from ship_ice_planner.controller.critic import PathCritic, DifferentiableCostFunction
                
                path_critic = PathCritic().to(self.device)
                if 'model_state_dict' in checkpoint:
                    path_critic.load_state_dict(checkpoint['model_state_dict'])
                else:
                    path_critic.load_state_dict(checkpoint)
                
                cost_function = DifferentiableCostFunction(costmap, collision_weight=1.0).to(self.device)
                self.critic = HybridCritic(path_critic, cost_function, 
                                          learned_weight=0.7, cost_weight=0.3).to(self.device)
                self.critic.eval()
                print(f"Loaded critic from {critic_path}")
            except Exception as e:
                print(f"Warning: Could not load critic from {critic_path}: {e}")
                self.critic = None
        
        # For gradient optimization
        if self.use_gradient_optimization:
            # Create a copy of the model for optimization (with gradients enabled)
            self.optim_model = NeuralMPC(
                max_discrete_nodes=max_discrete_nodes,
                num_spline_control_points=num_spline_control_points
            ).to(self.device)
            self.optim_model.load_state_dict(self.model.state_dict())
            self.optim_model.train()  # Enable gradients for optimization
    
    def predict_path(self,
                     discrete_plan: np.ndarray,
                     start_pose: np.ndarray,
                     start_nu: np.ndarray,
                     goal: np.ndarray,
                     num_path_points: int = 500) -> np.ndarray:
        """
        Predict continuous path from discrete plan using spline generation
        
        Args:
            discrete_plan: (N, 3) discrete waypoints [x, y, psi]
            start_pose: (3,) start pose [x, y, psi]
            start_nu: (3,) start velocities [u, v, r]
            goal: (2,) goal position [x, y]
            num_path_points: Number of points to generate along the spline
        
        Returns:
            continuous_path: (M, 3) predicted continuous path [x, y, psi]
        """
        # Normalize inputs
        discrete_normalized = discrete_plan.copy()
        discrete_normalized[:, :2] -= start_pose[:2]
        
        goal_normalized = goal - start_pose[:2]
        
        # Pad/truncate discrete plan
        N = len(discrete_plan)
        if N > self.max_discrete_nodes:
            indices = np.linspace(0, N-1, self.max_discrete_nodes, dtype=int)
            discrete_normalized = discrete_normalized[indices]
            N = self.max_discrete_nodes
        
        discrete_padded = np.zeros((self.max_discrete_nodes, 3), dtype=np.float32)
        discrete_padded[:N] = discrete_normalized
        discrete_mask = np.zeros(self.max_discrete_nodes, dtype=bool)
        discrete_mask[:N] = True
        
        # Convert to tensors
        discrete_tensor = torch.from_numpy(discrete_padded).unsqueeze(0).to(self.device)
        discrete_mask_tensor = torch.from_numpy(discrete_mask).unsqueeze(0).to(self.device)
        start_pose_tensor = torch.from_numpy(start_pose.astype(np.float32)).unsqueeze(0).to(self.device)
        start_nu_tensor = torch.from_numpy(start_nu.astype(np.float32)).unsqueeze(0).to(self.device)
        goal_tensor = torch.from_numpy(goal_normalized.astype(np.float32)).unsqueeze(0).to(self.device)
        
        # Predict spline control points
        with torch.no_grad():
            pred_control_points_normalized = self.model(
                discrete_tensor, discrete_mask_tensor,
                start_pose_tensor, start_nu_tensor, goal_tensor
            )
        
        # Denormalize control points
        pred_control_points = pred_control_points_normalized[0].cpu().numpy()
        pred_control_points[:, :2] += start_pose[:2]
        
        # Generate smooth spline path from control points
        path = generate_spline_from_control_points(
            pred_control_points,
            num_points=num_path_points,
            degree=3,
            start_pose=start_pose
        )
        
        return path
    
    def optimize_path_with_costmap(self,
                                    discrete_plan: np.ndarray,
                                    start_pose: np.ndarray,
                                    start_nu: np.ndarray,
                                    goal: np.ndarray,
                                    costmap: CostMap,
                                    critic=None) -> np.ndarray:
        """
        Optimize path using gradient-based optimization with costmap
        For now, we optimize the generated spline path directly.
        
        Args:
            discrete_plan: (N, 3) discrete waypoints
            start_pose: (3,) start pose
            start_nu: (3,) start velocities
            goal: (2,) goal position
            costmap: CostMap object for collision costs
        
        Returns:
            optimized_path: (M, 3) optimized continuous path
        """
        if not self.use_gradient_optimization:
            return self.predict_path(discrete_plan, start_pose, start_nu, goal)
        
        # Get initial prediction (spline path)
        initial_path = self.predict_path(discrete_plan, start_pose, start_nu, goal, num_path_points=500)
        
        # Normalize inputs
        discrete_normalized = discrete_plan.copy()
        discrete_normalized[:, :2] -= start_pose[:2]
        goal_normalized = goal - start_pose[:2]
        
        # Pad/truncate discrete plan
        N = len(discrete_plan)
        if N > self.max_discrete_nodes:
            indices = np.linspace(0, N-1, self.max_discrete_nodes, dtype=int)
            discrete_normalized = discrete_normalized[indices]
            N = self.max_discrete_nodes
        
        discrete_padded = np.zeros((self.max_discrete_nodes, 3), dtype=np.float32)
        discrete_padded[:N] = discrete_normalized
        discrete_mask = np.zeros(self.max_discrete_nodes, dtype=bool)
        discrete_mask[:N] = True
        
        # Convert to tensors (with gradients)
        discrete_tensor = torch.from_numpy(discrete_padded).unsqueeze(0).to(self.device)
        discrete_mask_tensor = torch.from_numpy(discrete_mask).unsqueeze(0).to(self.device)
        start_pose_tensor = torch.from_numpy(start_pose.astype(np.float32)).unsqueeze(0).to(self.device)
        start_nu_tensor = torch.from_numpy(start_nu.astype(np.float32)).unsqueeze(0).to(self.device)
        goal_tensor = torch.from_numpy(goal_normalized.astype(np.float32)).unsqueeze(0).to(self.device)
        
        # Create learnable path (start from initial spline prediction)
        # We optimize the path points directly (could optimize control points in future)
        path_normalized = initial_path.copy()
        path_normalized[:, :2] -= start_pose[:2]
        path_tensor = torch.from_numpy(path_normalized.astype(np.float32)).unsqueeze(0).to(self.device)
        path_tensor.requires_grad = True
        
        # Optimizer for path
        optimizer = torch.optim.Adam([path_tensor], lr=self.learning_rate)
        
        # Convert costmap to tensor for differentiable sampling
        costmap_tensor = torch.from_numpy(costmap.cost_map.astype(np.float32)).to(self.device)
        scale = costmap.scale
        
        # Gradient optimization loop
        for step in range(self.optimization_steps):
            optimizer.zero_grad()
            
            # Compute costmap collision cost (differentiable)
            # Denormalize path for costmap lookup
            path_denorm = path_tensor[0].clone()
            path_denorm[:, :2] += torch.from_numpy(start_pose[:2].astype(np.float32)).to(self.device)
            
            # Sample costmap using bilinear interpolation (differentiable)
            total_cost = torch.tensor(0.0, device=self.device)
            for i in range(len(path_denorm)):
                x, y = path_denorm[i, 0], path_denorm[i, 1]
                
                # Convert to costmap coordinates (continuous)
                x_scale = x * scale
                y_scale = y * scale
                
                # Bilinear interpolation for differentiable sampling
                x0 = torch.floor(x_scale).long()
                y0 = torch.floor(y_scale).long()
                x1 = x0 + 1
                y1 = y0 + 1
                
                # Get interpolation weights
                wx = x_scale - x0.float()
                wy = y_scale - y0.float()
                
                # Clamp indices to valid range
                h, w = costmap_tensor.shape
                x0 = torch.clamp(x0, 0, w - 1)
                y0 = torch.clamp(y0, 0, h - 1)
                x1 = torch.clamp(x1, 0, w - 1)
                y1 = torch.clamp(y1, 0, h - 1)
                
                # Bilinear interpolation
                cost_00 = costmap_tensor[y0, x0]
                cost_01 = costmap_tensor[y1, x0]
                cost_10 = costmap_tensor[y0, x1]
                cost_11 = costmap_tensor[y1, x1]
                
                cost_interp = (cost_00 * (1 - wx) * (1 - wy) +
                              cost_10 * wx * (1 - wy) +
                              cost_01 * (1 - wx) * wy +
                              cost_11 * wx * wy)
                
                total_cost = total_cost + cost_interp
            
            # Add smoothness penalty (encourage smooth paths)
            smoothness = torch.tensor(0.0, device=self.device)
            if len(path_tensor[0]) > 1:
                diff = path_tensor[0, 1:, :2] - path_tensor[0, :-1, :2]
                smoothness = torch.mean(torch.sum(diff ** 2, dim=1))
            
            # Use critic if provided, otherwise use costmap cost
            if critic is not None:
                # Build context [start_pose (3), start_nu (3), goal_normalized (2)] = 8
                context = torch.cat([
                    start_pose_tensor[0, :3],  # start_pose (x, y, psi)
                    start_nu_tensor[0],  # start_nu (u, v, r)
                    goal_tensor[0]  # goal relative to start (x, y)
                ]).unsqueeze(0)
                
                # Critic evaluates path (lower value = higher cost, so we minimize negative value)
                critic_value = critic(path_tensor, context, goal_tensor)
                total_loss = -critic_value.mean() + 0.1 * smoothness  # Minimize negative value
            else:
                # Total loss without critic
                total_loss = total_cost + 0.1 * smoothness
            
            # Backward pass
            total_loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_([path_tensor], max_norm=1.0)
            
            # Update
            optimizer.step()
            
            # Project back to valid range if needed
            with torch.no_grad():
                # Keep path reasonable
                path_tensor.clamp_(-1000, 1000)
        
        # Get optimized path
        optimized_path_normalized = path_tensor[0].detach().cpu().numpy()
        optimized_path = optimized_path_normalized.copy()
        optimized_path[:, :2] += start_pose[:2]
        
        return optimized_path
    
    def plan(self,
             discrete_plan: np.ndarray,
             start_pose: np.ndarray,
             start_nu: np.ndarray,
             goal: np.ndarray,
             costmap: Optional[CostMap] = None) -> np.ndarray:
        """
        Main planning function
        
        Args:
            discrete_plan: (N, 3) discrete waypoints from A*
            start_pose: (3,) start pose [x, y, psi]
            start_nu: (3,) start velocities [u, v, r]
            goal: (2,) goal position [x, y]
            costmap: Optional costmap for gradient optimization
        
        Returns:
            continuous_path: (M, 3) continuous path
        """
        if costmap is not None and self.use_gradient_optimization:
            return self.optimize_path_with_costmap(
                discrete_plan, start_pose, start_nu, goal, costmap, critic=self.critic
            )
        else:
            return self.predict_path(discrete_plan, start_pose, start_nu, goal)

