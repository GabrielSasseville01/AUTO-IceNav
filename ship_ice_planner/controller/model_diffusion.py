import numpy as np
import math

class ModelBasedDiffusionController:
    """
    Controller based on model-based diffusion (Annealed MPPI).
    Simulates a reverse diffusion process to optimize control inputs.
    """
    def __init__(self, 
                 num_timesteps=20, 
                 num_samples=50, 
                 horizon=20, 
                 dt=0.5,
                 temperature=0.1,
                 u_min=np.array([-1.0, -1.0, -0.5]), # surge, sway, yaw_rate
                 u_max=np.array([1.0, 1.0, 0.5])):
        
        self.num_timesteps = num_timesteps # Diffusion steps
        self.num_samples = num_samples     # Monte Carlo samples
        self.horizon = horizon             # Prediction horizon
        self.dt = dt
        self.temperature = temperature
        
        self.u_min = u_min
        self.u_max = u_max
        self.u_dim = 3 # surge, sway, yaw_rate
        
        # Noise schedule (decreasing variance)
        # Similar to the script: alphas = 1.0 - np.linspace(0.0001, 0.01, num_timesteps + 1)
        # But we want sigmas directly for sampling.
        # We'll use a geometric decay or linear decay for sigma.
        self.sigma_start = 1.0
        self.sigma_end = 0.1
        self.sigmas = np.linspace(self.sigma_start, self.sigma_end, num_timesteps)
        
        # Initialize mean control sequence
        self.U_mean = np.zeros((self.horizon, self.u_dim))

    def objective_function(self, trajectories, goal, obstacles):
        """
        Calculate cost for a batch of trajectories.
        trajectories: (num_samples, horizon, 3) [x, y, psi]
        goal: [x, y]
        obstacles: list of polygon vertices
        """
        # Distance to goal cost (terminal and running)
        # trajectories[:, :, :2] is position
        dist_to_goal = np.linalg.norm(trajectories[:, :, :2] - goal, axis=2)
        
        # Terminal cost (weight higher)
        terminal_cost = dist_to_goal[:, -1] * 10.0
        
        # Running cost
        running_cost = np.mean(dist_to_goal, axis=1)
        
        # Collision cost (simplified: distance to obstacles)
        # This is expensive to compute exactly for all samples. 
        # We'll use a simple circular approximation for the ship and obstacles for now.
        collision_cost = np.zeros(self.num_samples)
        
        # Simple collision check (can be improved)
        # For demo purposes, we might just penalize being close to obstacle centers if available
        # or just rely on goal seeking if obstacles are sparse.
        # Let's add a basic check if we have obstacle info.
        if obstacles is not None:
             for i in range(self.num_samples):
                 traj = trajectories[i]
                 for obs in obstacles:
                     # obs is list of vertices. Compute centroid.
                     obs_center = np.mean(obs, axis=0)
                     # Check min dist in trajectory
                     dists = np.linalg.norm(traj[:, :2] - obs_center, axis=1)
                     if np.any(dists < 20.0): # Threshold
                         collision_cost[i] += 1000.0

        total_cost = terminal_cost + running_cost + collision_cost
        return total_cost

    def simple_ship_dynamics(self, state, U):
        """
        Predict trajectory given initial state and control sequence.
        state: [x, y, psi, u, v, r]
        U: (horizon, 3) [surge_cmd, sway_cmd, yaw_rate_cmd]
        Returns: trajectory (horizon, 3) [x, y, psi]
        """
        # Simplified kinematics for prediction
        # Assumes control inputs are velocity commands (or proportional to them)
        # x_dot = u * cos(psi) - v * sin(psi)
        # y_dot = u * sin(psi) + v * cos(psi)
        # psi_dot = r
        
        traj = np.zeros((self.horizon, 3))
        
        curr_x, curr_y, curr_psi = state[0], state[1], state[2]
        # curr_u, curr_v, curr_r = state[3], state[4], state[5] 
        
        # Simple model: assume perfect tracking of velocity commands for prediction
        # or first order lag. Let's use perfect tracking for simplicity in this demo.
        
        for t in range(self.horizon):
            u_cmd = U[t, 0]
            v_cmd = U[t, 1]
            r_cmd = U[t, 2]
            
            # Kinematics
            x_dot = u_cmd * np.cos(curr_psi) - v_cmd * np.sin(curr_psi)
            y_dot = u_cmd * np.sin(curr_psi) + v_cmd * np.cos(curr_psi)
            psi_dot = r_cmd
            
            curr_x += x_dot * self.dt
            curr_y += y_dot * self.dt
            curr_psi += psi_dot * self.dt
            
            traj[t, 0] = curr_x
            traj[t, 1] = curr_y
            traj[t, 2] = curr_psi
            
        return traj

    def get_control(self, state_obj, setpoint, dt, obstacles=None):
        """
        Main control loop.
        state_obj: SimShipDynamics State object
        setpoint: current target [x, y] (or full path, but we use local target)
        dt: timestep
        """
        # Extract state
        state_vec = np.array([state_obj.x, state_obj.y, state_obj.psi, 
                              state_obj.u, state_obj.v, state_obj.r])
        
        # Goal: setpoint is likely [x, y] or [x, y, psi] or similar.
        # SimShipDynamics.setpoint is usually [x_d, y_d, psi_d] or similar from TrajectorySetpoint
        # But let's assume setpoint is the immediate target.
        # If setpoint is a list/array, take first 2 as x,y
        goal = np.array(setpoint[:2])
        
        # Shift U_mean (warm start)
        self.U_mean[:-1] = self.U_mean[1:]
        self.U_mean[-1] = self.U_mean[-2] # Duplicate last
        
        # Diffusion / Annealing loop
        for k in range(self.num_timesteps):
            sigma = self.sigmas[k]
            
            # 1. Sample perturbations
            # noise: (num_samples, horizon, u_dim)
            noise = np.random.normal(0, sigma, (self.num_samples, self.horizon, self.u_dim))
            
            # U_samples: (num_samples, horizon, u_dim)
            U_samples = self.U_mean + noise
            
            # Clip controls
            np.clip(U_samples, self.u_min, self.u_max, out=U_samples)
            
            # 2. Evaluate trajectories
            trajectories = np.zeros((self.num_samples, self.horizon, 3))
            for i in range(self.num_samples):
                trajectories[i] = self.simple_ship_dynamics(state_vec, U_samples[i])
            
            # 3. Calculate costs
            costs = self.objective_function(trajectories, goal, obstacles)
            
            # 4. Weight samples
            # Softmax weights: w = exp(-cost / lambda)
            min_cost = np.min(costs)
            weights = np.exp(-(costs - min_cost) / self.temperature)
            weights /= np.sum(weights)
            
            # 5. Update mean
            # Weighted average of samples
            # U_mean = sum(w_i * U_i)
            self.U_mean = np.sum(weights[:, None, None] * U_samples, axis=0)
            
        # Return first control action
        return self.U_mean[0]
