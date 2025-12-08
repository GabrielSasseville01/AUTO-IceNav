import numpy as np
import torch
from typing import Optional, List, Tuple
from tqdm import tqdm

from ship_ice_planner.controller.NRC_supply import NrcSupply
from ship_ice_planner.geometry.utils import Rxy


class ModelBasedDiffusionController(NrcSupply):
    """
    Model-Based Diffusion Controller following the reverse SDE approach from model-based-diffusion.
    Converted from JAX to PyTorch for ship ice navigation.
    """
    
    def __init__(self, 
                 horizon: int = 50,
                 num_samples: int = 2048,
                 num_diffusion_steps: int = 100,
                 temperature: float = 0.1,
                 beta0: float = 1e-4,
                 betaT: float = 1e-2,
                 seed: int = 0):
        """
        :param horizon: Prediction horizon (Hsample)
        :param num_samples: Number of samples per diffusion step (Nsample)
        :param num_diffusion_steps: Number of diffusion steps (Ndiffuse)
        :param temperature: Temperature for softmax weighting (temp_sample)
        :param beta0: Initial beta for diffusion schedule
        :param betaT: Final beta for diffusion schedule
        :param seed: Random seed
        """
        super().__init__()
        
        # Diffusion parameters
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_diffusion_steps = num_diffusion_steps
        self.temperature = temperature
        self.beta0 = beta0
        self.betaT = betaT
        
        # Control dimensions
        self.nu_dim = 3  # [surge, sway, yaw_rate]
        
        # Forward motion constraint: enforce constant forward speed
        self.enforce_forward_motion = True
        self.target_surge_speed = None  # Will be set based on dynamics
        
        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Compute diffusion schedule
        self._compute_diffusion_schedule()
        
        # Warm start for control sequence
        self.u_seq_mean = np.zeros((self.horizon, self.nu_dim))
        
        # Execution tracking
        self.execution_count = 0
        self.replan_interval = 10  # Replan every 10 steps (less frequent for speed)
        
        # Override limits for full scale simulation (approximate)
        self.input_lims = [5.0, 1.0, 10.0]  # [m/s, m/s, deg/s]
        
    def _compute_diffusion_schedule(self):
        """Compute diffusion schedule parameters (betas, alphas, alpha_bar, sigmas)"""
        betas = np.linspace(self.beta0, self.betaT, self.num_diffusion_steps)
        alphas = 1.0 - betas
        alphas_bar = np.cumprod(alphas)  # Cumulative product
        
        # Sigma for noise schedule
        sigmas = np.sqrt(1 - alphas_bar)
        
        # Conditional sigmas (for reverse diffusion)
        alphas_bar_rolled = np.roll(alphas_bar, 1)
        alphas_bar_rolled[0] = 1.0
        Sigmas_cond = ((1 - alphas) * (1 - np.sqrt(alphas_bar_rolled)) / (1 - alphas_bar))
        sigmas_cond = np.sqrt(Sigmas_cond)
        sigmas_cond[0] = 0.0
        
        # Store as attributes
        self.betas = betas
        self.alphas = alphas
        self.alphas_bar = alphas_bar
        self.sigmas = sigmas
        self.sigmas_cond = sigmas_cond
        
        print(f"Initial sigma = {sigmas[-1]:.2e}")
    
    def rollout_trajectory(self, 
                          eta_start: np.ndarray, 
                          nu_start: np.ndarray,
                          u_seq: np.ndarray,
                          dt: float,
                          costmap=None,
                          goal_y: Optional[float] = None) -> Tuple[np.ndarray, float, List[np.ndarray]]:
        """
        Rollout a single trajectory given initial state and control sequence.
        Returns: (trajectory, total_reward, states_list)
        - trajectory: (horizon+1, 3) array of [x, y, psi]
        - total_reward: scalar reward (negative of total cost)
        - states_list: list of intermediate states for collision checking
        """
        # Ensure inputs are numpy arrays (handle tuples/lists)
        eta = np.array(eta_start, dtype=float).copy()
        nu = np.array(nu_start, dtype=float).copy()
        trajectory = np.zeros((self.horizon + 1, 3))
        trajectory[0] = eta
        states_list = [eta.copy()]
        
        total_cost = 0.0
        
        for t in range(self.horizon):
            u_control = u_seq[t].copy()
            
            # ENFORCE FORWARD MOTION: ALWAYS apply strong forward surge control
            if self.target_surge_speed is not None and self.enforce_forward_motion:
                # ALWAYS enforce minimum forward control - override whatever diffusion gave
                # From dynamics model, we need strong positive u_control[0] to get forward motion
                current_surge = nu[0]
                if current_surge < self.target_surge_speed:
                    # Need to accelerate forward - use strong control
                    surge_error = self.target_surge_speed - current_surge
                    u_control[0] = max(1500.0, surge_error * 5000.0)  # Very strong forward control, minimum 1500
                else:
                    # Maintain speed - still need positive control
                    u_control[0] = max(1500.0, u_control[0])  # Minimum 1500 to maintain forward motion
            else:
                # Even if not enforcing, ensure we don't go negative
                u_control[0] = max(1500.0, u_control[0])
            
            # Dynamics step
            nu_deg = [nu[0], nu[1], np.rad2deg(nu[2])]
            nu_next_deg = self.dynamics(nu_deg[0], nu_deg[1], nu_deg[2], u_control)
            nu_next = [nu_next_deg[0], nu_next_deg[1], np.deg2rad(nu_next_deg[2])]
            
            # Integrate position
            u_g, v_g = Rxy(eta[2]) @ [nu_next[0], nu_next[1]]
            eta[0] += dt * u_g
            eta[1] += dt * v_g
            eta[2] = (eta[2] + dt * nu_next[2]) % (2 * np.pi)
            
            trajectory[t + 1] = eta
            states_list.append(eta.copy())
            nu = nu_next
            
            # Compute cost at this timestep
            # Costmap collision cost (ENABLED - diffusion should avoid obstacles!)
            coll_cost = 0.0
            if costmap is not None:
                c_x = int(eta[0] * costmap.scale)
                c_y = int(eta[1] * costmap.scale)
                
                if 0 <= c_x < costmap.shape[1] and 0 <= c_y < costmap.shape[0]:
                    map_val = costmap.cost_map[c_y, c_x]
                    if map_val > 0:
                        coll_cost = map_val * 10.0  # Scale up collision cost for stronger penalty
                else:
                    coll_cost = 1000.0  # Out of bounds - strong penalty
            
            # Control effort (reduced weight)
            ctrl_cost = np.sum((u_control / 1000.0) ** 2) * 0.001  # Much smaller weight
            
            total_cost += coll_cost + ctrl_cost
        
        # Goal-reaching reward (MINIMAL - discrete path already goes to goal!)
        # Only use goal_y for minimal forward progress encouragement, NOT terminal position
        goal_reward = 0.0
        if goal_y is not None:
            # REMOVED: Terminal reward pushing toward final goal (discrete path handles this)
            # REMOVED: Strong progress reward (waypoint anchors handle path following)
            
            # Minimal forward progress reward (just to avoid staying still)
            y_progress = trajectory[-1, 1] - eta_start[1]
            if y_progress < 0.1:  # If moved less than 0.1m forward
                goal_reward = -1000.0  # Small penalty for not moving (reduced from -50000)
            
            # Minimal forward speed reward (just to encourage motion)
            forward_speed_reward = 0.0
            for i in range(1, len(trajectory)):
                dy = trajectory[i, 1] - trajectory[i-1, 1]
                if dy < 0:
                    forward_speed_reward -= 500.0  # Small penalty for backward motion (reduced)
                # No strong reward for forward motion - let waypoints guide it
        
        # Reward is negative of cost + minimal goal rewards
        # Waypoint anchors and path tracking are the PRIMARY constraints
        total_reward = -total_cost + goal_reward
        
        return trajectory, total_reward, states_list
    
    def objective_function(self,
                          u_seq_batch: np.ndarray,
                          eta_start: np.ndarray,
                          nu_start: np.ndarray,
                          local_path: Optional[np.ndarray],
                          dt: float,
                          costmap=None,
                          goal_y: Optional[float] = None) -> np.ndarray:
        """
        Evaluate rewards for a batch of control sequences.
        Returns rewards (higher is better), not costs.
        
        :param u_seq_batch: (num_samples, horizon, 3) control sequences
        :param eta_start: (3,) initial pose [x, y, psi]
        :param nu_start: (3,) initial velocities [u, v, r]
        :param local_path: (horizon, 3) reference path or None
        :param dt: timestep
        :param costmap: CostMap object
        :return: (num_samples,) array of rewards
        """
        rewards = np.zeros(self.num_samples)
        
        for i in range(self.num_samples):
            trajectory, base_reward, states_list = self.rollout_trajectory(
                eta_start, nu_start, u_seq_batch[i], dt, costmap, goal_y=goal_y
            )
            
            # Add path tracking reward (following the continuous/local path)
            # This guides the trajectory along the reference path
            path_reward = 0.0
            if local_path is not None:
                for t in range(min(len(trajectory) - 1, len(local_path))):
                    pos_err = np.sum((trajectory[t+1, :2] - local_path[t, :2]) ** 2)
                    ang_err = trajectory[t+1, 2] - local_path[t, 2]
                    ang_err = (ang_err + np.pi) % (2 * np.pi) - np.pi
                    path_reward -= (pos_err + 50.0 * ang_err ** 2) * 1.0  # Increased weight - important constraint
            
            # Add waypoint anchoring reward (PRIMARY constraint - must pass near discrete plan nodes)
            # This is the main constraint - follow the discrete A* plan waypoints
            waypoint_reward = 0.0
            if hasattr(self, 'waypoint_anchors') and self.waypoint_anchors:
                for anchor_t, anchor_wp in self.waypoint_anchors.items():
                    # anchor_t is the time step where we want to be near anchor_wp
                    if anchor_t < len(trajectory):
                        traj_pos = trajectory[anchor_t, :2]
                        anchor_pos = anchor_wp[:2]
                        dist_to_anchor = np.linalg.norm(traj_pos - anchor_pos)
                        # STRONG reward for being close to anchor (within 100m for more flexibility)
                        if dist_to_anchor < 100.0:
                            waypoint_reward += 5000.0 / (1.0 + dist_to_anchor / 10.0)  # Increased reward
                        else:
                            # STRONG penalty for being far from anchor
                            waypoint_reward -= dist_to_anchor * 200.0  # Much stronger penalty
            
            # Combined reward: waypoint anchors are PRIMARY, path tracking is SECONDARY
            # Goal rewards are MINIMAL (just to prevent staying still)
            rewards[i] = base_reward + path_reward + waypoint_reward
        
        return rewards
    
    def reverse_diffusion_step(self,
                              i: int,
                              Ybar_i: np.ndarray,
                              eta_start: np.ndarray,
                              nu_start: np.ndarray,
                              local_path: Optional[np.ndarray],
                              dt: float,
                              costmap=None,
                              goal_y: Optional[float] = None) -> Tuple[np.ndarray, float]:
        """
        Single reverse diffusion step following the SDE formulation.
        
        :param i: Current diffusion step index
        :param Ybar_i: Current mean control sequence (horizon, nu_dim)
        :param eta_start: Initial pose
        :param nu_start: Initial velocities
        :param local_path: Reference path
        :param dt: Timestep
        :param costmap: CostMap object
        :return: (Ybar_im1, mean_reward) - updated mean and mean reward
        """
        # Convert Ybar_i to Yi (noisy version at step i)
        Yi = Ybar_i * np.sqrt(self.alphas_bar[i])
        
        # Sample from q_i: Y0s = eps * sigma_i + Ybar_i
        eps_u = np.random.normal(0, 1, (self.num_samples, self.horizon, self.nu_dim))
        Y0s = eps_u * self.sigmas[i] + Ybar_i
        
        # ENFORCE CONSTANT FORWARD SPEED: maintain surge control to keep forward motion
        # Only optimize sway and yaw, keep surge at level that maintains forward speed
        if self.target_surge_speed is not None and self.enforce_forward_motion:
            # ALWAYS enforce minimum forward surge control
            if len(Ybar_i) > 0:
                mean_surge = max(500.0, Ybar_i[:, 0].mean())  # Ensure minimum surge
                # Maintain surge at level that keeps forward motion - allow some variation
                Y0s[:, :, 0] = np.maximum(300.0, mean_surge + eps_u[:, :, 0] * self.sigmas[i] * 0.2)
            else:
                # Default forward speed control - STRONG forward control
                Y0s[:, :, 0] = 1000.0  # Strong base surge control for forward motion
        
        Y0s = np.clip(Y0s, -2000, 2000)  # Clip to reasonable control limits
        
        # Evaluate trajectories and compute rewards
        rewards = self.objective_function(Y0s, eta_start, nu_start, local_path, dt, costmap, goal_y=goal_y)
        
        # Normalize rewards
        rew_mean = rewards.mean()
        rew_std = rewards.std()
        rew_std = rew_std if rew_std > 1e-4 else 1.0
        
        # Compute log probabilities (logp0)
        logp0 = (rewards - rew_mean) / rew_std / self.temperature
        
        # Softmax weights
        weights = np.exp(logp0 - np.max(logp0))  # Numerically stable
        weights = weights / weights.sum()
        
        # Update mean: Ybar = weighted average of Y0s
        Ybar = np.einsum('n,nij->ij', weights, Y0s)
        
        # ENFORCE CONSTANT FORWARD SPEED: after updating mean, fix surge control
        if self.target_surge_speed is not None and self.enforce_forward_motion:
            # Maintain surge at level that keeps forward motion (don't let it drop too low)
            if len(Ybar_i) > 0:
                original_surge_mean = max(500.0, Ybar_i[:, 0].mean())
                Ybar[:, 0] = np.maximum(500.0, Ybar[:, 0])  # ENFORCE minimum 500 surge control
            else:
                Ybar[:, 0] = 1000.0  # Default strong forward control
        
        # Compute score function
        score = 1 / (1.0 - self.alphas_bar[i]) * (-Yi + np.sqrt(self.alphas_bar[i]) * Ybar)
        
        # Reverse step: Y_{i-1}
        Yim1 = 1 / np.sqrt(self.alphas[i]) * (Yi + (1.0 - self.alphas_bar[i]) * score)
        
        # Convert back to Ybar_{i-1}
        Ybar_im1 = Yim1 / np.sqrt(self.alphas_bar[i - 1])
        
        # ENFORCE CONSTANT FORWARD SPEED: maintain surge in final result
        if self.target_surge_speed is not None and self.enforce_forward_motion:
            # ALWAYS enforce minimum forward surge
            Ybar_im1[:, 0] = np.maximum(500.0, Ybar_im1[:, 0])  # Never let surge go below 500
        
        return Ybar_im1, rew_mean
    
    def DPcontrol(self,
                 pose: np.ndarray,
                 setpoint: np.ndarray,
                 dt: float,
                 nu: Optional[np.ndarray] = None,
                 local_path: Optional[np.ndarray] = None,
                 costmap=None,
                 goal_y: Optional[float] = None):
        """
        Main control function using model-based diffusion.
        
        :param pose: [x, y, psi] current pose
        :param setpoint: [x_ref, y_ref, psi_ref] target pose
        :param dt: timestep
        :param nu: [u, v, r] current velocities
        :param local_path: (N, 3) reference path
        :param costmap: CostMap object
        :return: Control input [u_control_0, u_control_1, u_control_2]
        """
        if nu is None:
            nu = np.array([0.0, 0.0, 0.0])
        
        # Set target surge speed for forward motion enforcement
        self.target_surge_speed = nu[0] if nu[0] > 0.1 else 2.0  # Use current speed or default 2 m/s
        self.enforce_forward_motion = True
        
        # Check if we should replan (use cached control if available)
        if self.execution_count % self.replan_interval != 0 and len(self.u_seq_mean) > 0:
            idx = self.execution_count % self.replan_interval
            if idx < len(self.u_seq_mean):
                control = np.array(self.u_seq_mean[idx].copy(), dtype=float)
                # Ensure forward motion even in cached controls
                if self.enforce_forward_motion and self.target_surge_speed is not None:
                    control[0] = max(600.0, float(control[0]))
                self.execution_count += 1
                return control
        
        # Replanning: run diffusion
        self.execution_count = 0
        
        # Prepare local path
        if local_path is None:
            local_path = np.tile(setpoint, (self.horizon, 1))
        else:
            local_path = np.array(local_path)
            # Ensure it matches horizon
            if len(local_path) != self.horizon:
                # Interpolate or repeat
                if len(local_path) < self.horizon:
                    # Repeat last point
                    last_point = local_path[-1:]
                    local_path = np.vstack([local_path, np.tile(last_point, (self.horizon - len(local_path), 1))])
                else:
                    # Take first horizon points
                    local_path = local_path[:self.horizon]
        
        # Warm start: shift previous plan
        shift = self.replan_interval
        if shift < self.horizon:
            self.u_seq_mean[:-shift] = self.u_seq_mean[shift:]
            self.u_seq_mean[-shift:] = self.u_seq_mean[-1]
        else:
            self.u_seq_mean[:] = self.u_seq_mean[-1] if len(self.u_seq_mean) > 0 else np.zeros((self.horizon, 3))
        
        # Initialize YN (noise at final step)
        YN = np.zeros((self.horizon, self.nu_dim))
        # Start from warm start instead of pure noise
        Ybar_i = self.u_seq_mean.copy()
        
        # Reverse diffusion process
        # For speed, skip progress bar in control loop
        Ybars = []
        for i in range(self.num_diffusion_steps - 1, 0, -1):
            Ybar_i, rew = self.reverse_diffusion_step(
                i, Ybar_i, pose, nu, local_path, dt, costmap, goal_y=goal_y
            )
            Ybars.append(Ybar_i)
        
        # Update mean control sequence
        self.u_seq_mean = Ybars[-1] if len(Ybars) > 0 else Ybar_i
        
        # ENFORCE FORWARD MOTION in final result
        if self.enforce_forward_motion and self.target_surge_speed is not None:
            self.u_seq_mean[:, 0] = np.maximum(500.0, self.u_seq_mean[:, 0])
        
        # Store predicted trajectory for visualization
        self.predicted_trajectory = self.simulate_trajectory(
            self.u_seq_mean, pose, nu, dt
        )
        
        # Store some sampled trajectories for visualization (from final step) - skip for speed
        # self.sampled_trajectories_viz = []
        
        self.execution_count += 1
        
        # Always return valid control (first control in sequence)
        if len(self.u_seq_mean) > 0 and self.u_seq_mean.shape[0] > 0:
            control = np.array(self.u_seq_mean[0].copy(), dtype=float)
            # Ensure minimum forward control
            if self.enforce_forward_motion and self.target_surge_speed is not None:
                control[0] = max(1500.0, float(control[0]))  # Higher minimum for full scale
            return control
        else:
            # Fallback: return basic forward control
            if self.enforce_forward_motion and self.target_surge_speed is not None:
                return np.array([2000.0, 0.0, 0.0], dtype=float)  # Very strong forward control
            return np.array([0.0, 0.0, 0.0], dtype=float)
    
    def simulate_trajectory(self, u_seq: np.ndarray, eta_start: np.ndarray, nu_start: np.ndarray, dt: float) -> np.ndarray:
        """Helper to simulate a trajectory for visualization"""
        traj = np.zeros((self.horizon + 1, 3))
        traj[0] = eta_start
        eta = eta_start.copy()
        nu = nu_start.copy()
        
        for t in range(self.horizon):
            u_control = u_seq[t]
            nu_deg = [nu[0], nu[1], np.rad2deg(nu[2])]
            nu_next_deg = self.dynamics(nu_deg[0], nu_deg[1], nu_deg[2], u_control)
            nu_next = [nu_next_deg[0], nu_next_deg[1], np.deg2rad(nu_next_deg[2])]
            
            u_g, v_g = Rxy(eta[2]) @ [nu_next[0], nu_next[1]]
            eta[0] += dt * u_g
            eta[1] += dt * v_g
            eta[2] = (eta[2] + dt * nu_next[2]) % (2 * np.pi)
            
            nu = nu_next
            traj[t+1] = eta
            
        return traj

