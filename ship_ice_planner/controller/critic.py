"""Differentiable Critic Network for Path Evaluation
Learns to evaluate paths based on costmap, collisions, path length, etc.
Can be backpropagated through for gradient-based optimization.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from ship_ice_planner.cost_map import CostMap
from ship_ice_planner.geometry.utils import Rxy


class DifferentiableCostFunction(nn.Module):
    """
    Differentiable cost function for path evaluation.
    Computes costs that can be backpropagated through.
    """
    def __init__(self, costmap: CostMap, collision_weight: float = 1.0, 
                 smoothness_weight: float = 0.1, goal_weight: float = 0.5):
        super(DifferentiableCostFunction, self).__init__()
        self.costmap = costmap
        self.collision_weight = collision_weight
        self.smoothness_weight = smoothness_weight
        self.goal_weight = goal_weight
        
        # Convert costmap to tensor for differentiable access
        self.register_buffer('costmap_tensor', torch.from_numpy(costmap.cost_map).float())
        
        # Costmap parameters for bilinear interpolation
        self.scale = costmap.scale
        self.costmap_shape = costmap.shape  # (height, width) in grid units
        
    def bilinear_interpolate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Bilinear interpolation of costmap at (x, y) positions.
        x, y are in meters, converted to grid coordinates.
        
        Args:
            x: (N,) tensor of x positions in meters
            y: (N,) tensor of y positions in meters
        
        Returns:
            costs: (N,) tensor of interpolated costs
        """
        # Convert to grid coordinates
        x_grid = x * self.scale
        y_grid = y * self.scale
        
        # Clamp to valid range
        x_grid = torch.clamp(x_grid, 0, self.costmap_shape[1] - 1.001)
        y_grid = torch.clamp(y_grid, 0, self.costmap_shape[0] - 1.001)
        
        # Get integer coordinates
        x0 = x_grid.floor().long()
        y0 = y_grid.floor().long()
        x1 = (x0 + 1).clamp(max=self.costmap_shape[1] - 1)
        y1 = (y0 + 1).clamp(max=self.costmap_shape[0] - 1)
        
        # Get fractional parts
        wx = x_grid - x0.float()
        wy = y_grid - y0.float()
        
        # Get four corner values
        c00 = self.costmap_tensor[y0, x0]
        c01 = self.costmap_tensor[y1, x0]
        c10 = self.costmap_tensor[y0, x1]
        c11 = self.costmap_tensor[y1, x1]
        
        # Bilinear interpolation
        c0 = c00 * (1 - wy) + c01 * wy
        c1 = c10 * (1 - wy) + c11 * wy
        cost = c0 * (1 - wx) + c1 * wx
        
        return cost
    
    def compute_path_cost(self, path: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
        Compute differentiable cost for a path.
        
        Args:
            path: (N, 3) tensor of path points [x, y, psi] in meters
            goal: (2,) tensor of goal position [x, y] in meters
        
        Returns:
            total_cost: scalar tensor
        """
        if path.shape[0] == 0:
            return torch.tensor(0.0, device=path.device)
        
        # Extract positions
        positions = path[:, :2]  # (N, 2)
        
        # 1. Collision cost (from costmap)
        collision_costs = self.bilinear_interpolate(positions[:, 0], positions[:, 1])
        collision_cost = torch.sum(collision_costs) * self.collision_weight
        
        # 2. Smoothness cost (penalize sharp turns and acceleration)
        if path.shape[0] > 1:
            # Heading changes
            psi = path[:, 2]
            heading_diffs = torch.diff(psi)
            # Wrap to [-pi, pi]
            heading_diffs = torch.atan2(torch.sin(heading_diffs), torch.cos(heading_diffs))
            smoothness_cost = torch.sum(heading_diffs ** 2) * self.smoothness_weight
            
            # Position changes (penalize large jumps)
            pos_diffs = torch.diff(positions, dim=0)
            distances = torch.norm(pos_diffs, dim=1)
            # Penalize large variations in step size
            if len(distances) > 1:
                distance_var = torch.var(distances)
                smoothness_cost += distance_var * self.smoothness_weight
        else:
            smoothness_cost = torch.tensor(0.0, device=path.device)
        
        # 3. Goal-reaching cost (distance from path end to goal)
        final_pos = positions[-1]
        goal_distance = torch.norm(final_pos - goal)
        goal_cost = goal_distance * self.goal_weight
        
        # 4. Path length cost (encourage shorter paths)
        if path.shape[0] > 1:
            path_length = torch.sum(torch.norm(torch.diff(positions, dim=0), dim=1))
        else:
            path_length = torch.tensor(0.0, device=path.device)
        
        total_cost = collision_cost + smoothness_cost + goal_cost + path_length * 0.01
        
        return total_cost


class PathCritic(nn.Module):
    """
    Neural network critic that learns to evaluate paths.
    Takes path and context (start, goal, obstacles) and outputs a value estimate.
    """
    def __init__(self, path_dim=3, context_dim=8, hidden_dim=256, num_layers=3):
        super(PathCritic, self).__init__()
        
        self.path_dim = path_dim
        self.context_dim = context_dim  # start_pose (3) + start_nu (3) + goal (2) = 8
        
        # Path encoder (LSTM)
        self.path_encoder = nn.LSTM(path_dim, hidden_dim, num_layers, batch_first=True)
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Fusion and value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Scalar value
        )
    
    def forward(self, path: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Evaluate a path given context.
        
        Args:
            path: (batch, N, 3) tensor of path points [x, y, psi]
            context: (batch, context_dim) tensor of context [start_pose, start_nu, goal]
        
        Returns:
            values: (batch,) tensor of path values (lower is better)
        """
        # Encode path
        path_out, (h_n, c_n) = self.path_encoder(path)
        path_feat = h_n[-1]  # Last layer hidden state (batch, hidden_dim)
        
        # Encode context
        context_feat = self.context_encoder(context)
        
        # Fuse features
        combined = torch.cat([path_feat, context_feat], dim=1)
        
        # Output value (negative cost)
        value = self.value_head(combined).squeeze(-1)
        
        return value


class HybridCritic(nn.Module):
    """
    Hybrid critic that combines learned features with differentiable cost function.
    """
    def __init__(self, path_critic: PathCritic, cost_function: DifferentiableCostFunction,
                 learned_weight: float = 0.5, cost_weight: float = 0.5):
        super(HybridCritic, self).__init__()
        self.path_critic = path_critic
        self.cost_function = cost_function
        self.learned_weight = learned_weight
        self.cost_weight = cost_weight
    
    def forward(self, path: torch.Tensor, context: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
        Evaluate path using both learned critic and cost function.
        
        Args:
            path: (batch, N, 3) or (N, 3) path points
            context: (batch, context_dim) or (context_dim,) context
            goal: (batch, 2) or (2,) goal position
        
        Returns:
            total_value: combined value estimate
        """
        # Ensure batch dimension
        if path.dim() == 2:
            path = path.unsqueeze(0)
        if context.dim() == 1:
            context = context.unsqueeze(0)
        if goal.dim() == 1:
            goal = goal.unsqueeze(0)
        
        # Learned value (higher is better)
        learned_value = self.path_critic(path, context)
        
        # Cost-based value (negative cost, so higher is better)
        batch_size = path.shape[0]
        cost_values = []
        for i in range(batch_size):
            cost = self.cost_function.compute_path_cost(path[i], goal[i])
            cost_values.append(-cost)  # Negate cost to get value
        
        cost_value = torch.stack(cost_values)
        
        # Combine
        total_value = (self.learned_weight * learned_value + 
                      self.cost_weight * cost_value)
        
        return total_value


def create_critic_for_costmap(costmap: CostMap, 
                              collision_weight: float = 1.0,
                              device: str = 'cpu') -> HybridCritic:
    """
    Create a hybrid critic for a given costmap.
    """
    path_critic = PathCritic().to(device)
    cost_function = DifferentiableCostFunction(
        costmap, 
        collision_weight=collision_weight
    ).to(device)
    
    critic = HybridCritic(path_critic, cost_function).to(device)
    return critic

