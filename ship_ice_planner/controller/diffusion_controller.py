import numpy as np
from ship_ice_planner.controller.NRC_supply import NrcSupply
from ship_ice_planner.geometry.utils import Rxy

class DiffusionNrcSupply(NrcSupply):
    def __init__(self):
        super().__init__()
        
        # Diffusion/MPPI parameters
        self.horizon = 20        # Prediction horizon steps
        self.num_samples = 30    # Number of trajectories to sample
        self.diffusion_steps = 10 # Number of refinement steps (annealing)
        self.temperature = 0.1   # Softmax temperature
        
        # Annealing schedule for noise (sigmas)
        # Increased significantly because u_control is in RPS (can be 100s or 1000s)
        self.sigmas = np.linspace(500.0, 10.0, self.diffusion_steps)
        
        # Override limits for full scale simulation (approximate)
        # NrcSupply is 1:45 scale. Full scale is faster.
        # Surge: 5 m/s, Sway: 1 m/s, Yaw rate: 10 deg/s
        self.input_lims = [5.0, 1.0, 10.0]
        
        # Warm start for control sequence
        self.u_seq_mean = np.zeros((self.horizon, 3))
        
        # Execution tracking
        self.execution_count = 0
        self.replan_interval = 5 # Replan every 5 steps (0.5s if dt=0.1)

    def objective_function(self, u_seq_batch, eta_start, nu_start, local_path, dt, costmap=None):
        """
        Evaluate costs for a batch of control sequences.
        u_seq_batch: (num_samples, horizon, 3)
        eta_start: (3,) [x, y, psi]
        nu_start: (3,) [u, v, r]
        local_path: (horizon, 3) [x, y, psi] reference points
        costmap: CostMap object (optional)
        """
        costs = np.zeros(self.num_samples)
        
        for i in range(self.num_samples):
            cost = 0.0
            eta = eta_start.copy()
            nu = nu_start.copy()
            
            for t in range(self.horizon):
                u_control = u_seq_batch[i, t]
                
                # Dynamics step
                nu_deg = [nu[0], nu[1], np.rad2deg(nu[2])]
                nu_next_deg = self.dynamics(nu_deg[0], nu_deg[1], nu_deg[2], u_control)
                nu_next = [nu_next_deg[0], nu_next_deg[1], np.deg2rad(nu_next_deg[2])]
                
                # Integrate position
                u_g, v_g = Rxy(eta[2]) @ [nu_next[0], nu_next[1]]
                eta[0] += dt * u_g
                eta[1] += dt * v_g
                eta[2] = (eta[2] + dt * nu_next[2]) % (2 * np.pi)
                
                # Update nu
                nu = nu_next
                
                # Calculate cost (distance to reference at time t)
                ref_idx = min(t, len(local_path) - 1)
                target_eta = local_path[ref_idx]
                
                pos_err = (eta[0] - target_eta[0])**2 + (eta[1] - target_eta[1])**2
                ang_err = (eta[2] - target_eta[2])
                ang_err = (ang_err + np.pi) % (2 * np.pi) - np.pi
                ang_err = ang_err**2
                
                # Control effort cost (Reduced to allow more movement)
                # Normalized by large control values
                ctrl_cost = np.sum((u_control/1000.0)**2) * 0.01
                
                # Collision cost from CostMap
                coll_cost = 0.0
                if costmap is not None:
                    # Convert to grid coordinates
                    c_x = int(eta[0] * costmap.scale)
                    c_y = int(eta[1] * costmap.scale)
                    
                    # Check bounds
                    if 0 <= c_x < costmap.shape[1] and 0 <= c_y < costmap.shape[0]:
                        # Add cost from map (high cost for obstacles)
                        map_val = costmap.cost_map[c_y, c_x]
                        if map_val > 0:
                            coll_cost = map_val * 100.0 # Weight for collision
                    else:
                        # Out of bounds cost
                        coll_cost = 1000.0

                cost += pos_err + ang_err * 50.0 + ctrl_cost + coll_cost
            
            costs[i] = cost
            
        return costs

    def DPcontrol(self, pose, setpoint, dt, nu=None, local_path=None, costmap=None):
        """
        Calculate control input using annealed MPPI / Reverse SDE.
        pose: [x, y, psi]
        setpoint: [x_ref, y_ref, psi_ref] (used if local_path is None)
        dt: timestep
        nu: [u, v, r] (velocities)
        local_path: (N, 3) array of future reference points
        costmap: CostMap object
        """
        if nu is None:
            nu = [0.0, 0.0, 0.0]
            
        # Check if we should replan
        if self.execution_count % self.replan_interval != 0:
            # Execute next step from previous plan
            idx = self.execution_count % self.replan_interval
            if idx < len(self.u_seq_mean):
                self.execution_count += 1
                return self.u_seq_mean[idx]
            
        # Replanning
        self.execution_count = 0 # Reset count
            
        # Prepare local path
        if local_path is None:
            local_path = np.tile(setpoint, (self.horizon, 1))
        else:
            local_path = np.array(local_path)
            
        # Shift mean sequence (warm start)
        # We shift by replan_interval since we executed that many steps
        shift = self.replan_interval
        if shift < self.horizon:
            self.u_seq_mean[:-shift] = self.u_seq_mean[shift:]
            self.u_seq_mean[-shift:] = self.u_seq_mean[-1]
        else:
            self.u_seq_mean[:] = self.u_seq_mean[-1]
        
        current_mean = self.u_seq_mean.copy()
        
        # Initialize final samples/weights to avoid attribute error if loop doesn't run (unlikely)
        self.final_samples = np.zeros((self.num_samples, self.horizon, 3))
        self.final_weights = np.ones(self.num_samples) / self.num_samples

        # Annealed MPPI / Diffusion Loop
        for k in range(self.diffusion_steps):
            sigma = self.sigmas[k]
            
            # 1. Generate noise
            noise = np.random.normal(0, sigma, size=(self.num_samples, self.horizon, 3))
            
            # 2. Generate samples around current mean
            u_samples = noise + current_mean 
            
            # Clip samples to limits (RPS)
            # Increased limits to ensure ship moves (approx +/- 1500 RPM/RPS scale)
            u_samples = np.clip(u_samples, -2000, 2000) 
            
            # 3. Evaluate samples
            costs = self.objective_function(u_samples, pose, nu, local_path, dt, costmap=costmap)
            
            # 4. Weighting (Softmax)
            min_cost = np.min(costs)
            weights = np.exp(-(costs - min_cost) / self.temperature)
            weights /= np.sum(weights)
            
            # 5. Update mean
            weighted_samples = u_samples * weights[:, None, None]
            current_mean = np.sum(weighted_samples, axis=0)
            
            # Store samples from the LAST iteration to show the final distribution.
            if k == self.diffusion_steps - 1:
                 # Store all samples for viz
                 self.final_samples = u_samples
                 self.final_weights = weights
            
        # Update internal mean
        self.u_seq_mean = current_mean
        
        # Store sampled trajectories for visualization
        self.predicted_trajectory = self.simulate_trajectory(current_mean, pose, nu, dt)
        
        # Visualize diverse samples
        # Pick top K samples
        top_k_indices = np.argsort(self.final_weights)[-10:] # Top 10
        self.sampled_trajectories_viz = []
        for idx in top_k_indices:
             self.sampled_trajectories_viz.append(
                 self.simulate_trajectory(self.final_samples[idx], pose, nu, dt)
             )
        
        self.execution_count += 1
        return current_mean[0]
            
        # Update internal mean
        self.u_seq_mean = current_mean
        
        # Store sampled trajectories for visualization (re-simulate best mean or all?)
        # Let's store the predicted trajectory of the mean control
        self.predicted_trajectory = self.simulate_trajectory(current_mean, pose, nu, dt)
        
        # Also store all samples (or a subset) for multi-modal viz
        # We'll simulate them to get x,y paths
        self.sampled_trajectories_viz = []
        # Visualize a few samples
        for i in range(min(5, self.num_samples)):
             self.sampled_trajectories_viz.append(
                 self.simulate_trajectory(u_samples[i], pose, nu, dt)
             )
        
        return current_mean[0]

    def simulate_trajectory(self, u_seq, eta_start, nu_start, dt):
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
