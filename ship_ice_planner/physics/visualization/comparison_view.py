"""
Side-by-side comparison visualization for Pymunk vs Chrono backends.

This module provides tools for visually comparing simulation results
between the existing Pymunk backend and the new Chrono backend.
"""

from typing import Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon


class BackendComparisonView:
    """
    Side-by-side visualization of Pymunk vs Chrono simulations.
    
    Used for regression testing to visually verify that the Chrono
    backend produces similar results to Pymunk for non-fracturing cases.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize comparison view.
        
        Args:
            config: Configuration dictionary with xlim, ylim, etc.
        """
        self.config = config or {}
        
        # Create side-by-side figure
        self.fig, (self.ax_pymunk, self.ax_chrono) = plt.subplots(1, 2, figsize=(16, 8))
        self.ax_pymunk.set_title('Pymunk Backend')
        self.ax_chrono.set_title('Chrono Backend')
        
        # Trajectory storage
        self.pymunk_trajectory = []
        self.chrono_trajectory = []
        
        # Ice floe storage
        self.pymunk_floes = []
        self.chrono_floes = []
        
    def run_comparison(self,
                       pymunk_sim,
                       chrono_sim,
                       duration: float = 30.0,
                       dt: float = 0.02,
                       save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run both simulations and visualize side-by-side.
        
        Args:
            pymunk_sim: Pymunk simulation adapter
            chrono_sim: Chrono simulation adapter
            duration: Simulation duration in seconds
            dt: Time step
            save_path: Optional path to save figure
            
        Returns:
            Dictionary with comparison metrics
        """
        self.pymunk_trajectory = []
        self.chrono_trajectory = []
        
        steps = int(duration / dt)
        
        for step in range(steps):
            # Step both simulations
            pymunk_sim.step(dt)
            chrono_sim.step(dt)
            
            # Record ship positions
            pymunk_pos = pymunk_sim.get_ship_position()
            chrono_state = chrono_sim.get_ship_state()
            chrono_pos = (chrono_state[0], chrono_state[1])
            
            self.pymunk_trajectory.append(pymunk_pos)
            self.chrono_trajectory.append(chrono_pos)
            
            # Update visualization periodically
            if step % 10 == 0:
                self._update_display(step)
                
        if save_path:
            self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return self._compute_metrics()
        
    def _update_display(self, step: int):
        """Update both subplot displays."""
        for ax, traj, name in [
            (self.ax_pymunk, self.pymunk_trajectory, 'Pymunk'),
            (self.ax_chrono, self.chrono_trajectory, 'Chrono')
        ]:
            ax.clear()
            
            # Draw trajectory
            if len(traj) > 1:
                traj_arr = np.array(traj)
                ax.plot(traj_arr[:, 0], traj_arr[:, 1], 'b-', linewidth=2, alpha=0.7)
                
            # Draw current ship position
            if traj:
                ax.plot(traj[-1][0], traj[-1][1], 'r^', markersize=15)
                
            ax.set_title(f'{name} | Step {step}')
            ax.set_aspect('equal')
            
            # Set axis limits if provided
            if 'xlim' in self.config:
                ax.set_xlim(self.config['xlim'])
            if 'ylim' in self.config:
                ax.set_ylim(self.config['ylim'])
                
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.grid(True, alpha=0.3)
            
        # Add difference metric in title
        if len(self.pymunk_trajectory) > 0 and len(self.chrono_trajectory) > 0:
            diff = np.linalg.norm(
                np.array(self.pymunk_trajectory[-1]) - 
                np.array(self.chrono_trajectory[-1])
            )
            self.fig.suptitle(f'Position difference: {diff:.2f} m', fontsize=14)
            
        plt.pause(0.01)
        
    def _compute_metrics(self) -> Dict[str, Any]:
        """Compute comparison metrics."""
        if not self.pymunk_trajectory or not self.chrono_trajectory:
            return {}
            
        pymunk_traj = np.array(self.pymunk_trajectory)
        chrono_traj = np.array(self.chrono_trajectory)
        
        # Compute position differences
        min_len = min(len(pymunk_traj), len(chrono_traj))
        position_diff = np.linalg.norm(
            pymunk_traj[:min_len] - chrono_traj[:min_len],
            axis=1
        )
        
        return {
            'pymunk_trajectory': pymunk_traj,
            'chrono_trajectory': chrono_traj,
            'position_diff': position_diff,
            'mean_diff': np.mean(position_diff),
            'max_diff': np.max(position_diff),
            'final_diff': position_diff[-1] if len(position_diff) > 0 else 0,
            'std_diff': np.std(position_diff),
        }
        
    def create_trajectory_comparison_plot(self,
                                           pymunk_traj: np.ndarray,
                                           chrono_traj: np.ndarray,
                                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Create static plot comparing trajectories.
        
        Args:
            pymunk_traj: Numpy array of Pymunk trajectory (N x 2)
            chrono_traj: Numpy array of Chrono trajectory (N x 2)
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Top left: XY trajectory overlay
        ax = axes[0, 0]
        ax.plot(pymunk_traj[:, 0], pymunk_traj[:, 1], 'b-', label='Pymunk', linewidth=2)
        ax.plot(chrono_traj[:, 0], chrono_traj[:, 1], 'r--', label='Chrono', linewidth=2)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Ship Trajectory Comparison')
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Top right: Position difference over time
        ax = axes[0, 1]
        min_len = min(len(pymunk_traj), len(chrono_traj))
        diff = np.linalg.norm(pymunk_traj[:min_len] - chrono_traj[:min_len], axis=1)
        ax.plot(diff, 'k-', linewidth=2)
        ax.axhline(y=diff.mean(), color='r', linestyle='--', 
                   label=f'Mean: {diff.mean():.2f}m')
        ax.fill_between(range(len(diff)), 0, diff, alpha=0.3)
        ax.set_xlabel('Time step')
        ax.set_ylabel('Position difference (m)')
        ax.set_title('Trajectory Divergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Bottom left: X position over time
        ax = axes[1, 0]
        ax.plot(pymunk_traj[:, 0], 'b-', label='Pymunk', linewidth=2)
        ax.plot(chrono_traj[:, 0], 'r--', label='Chrono', linewidth=2)
        ax.set_xlabel('Time step')
        ax.set_ylabel('X position (m)')
        ax.set_title('X Position Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Bottom right: Y position over time
        ax = axes[1, 1]
        ax.plot(pymunk_traj[:, 1], 'b-', label='Pymunk', linewidth=2)
        ax.plot(chrono_traj[:, 1], 'r--', label='Chrono', linewidth=2)
        ax.set_xlabel('Time step')
        ax.set_ylabel('Y position (m)')
        ax.set_title('Y Position Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
        
    @classmethod
    def from_config(cls, config_path: str) -> 'BackendComparisonView':
        """
        Create comparison view from config file.
        
        Args:
            config_path: Path to YAML config file
            
        Returns:
            BackendComparisonView instance
        """
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Extract relevant settings
        map_shape = config.get('map_shape', [600, 600])
        view_config = {
            'xlim': (0, map_shape[0]),
            'ylim': (0, map_shape[1]),
        }
        
        return cls(config=view_config)
        
    def close(self):
        """Close the figure."""
        plt.close(self.fig)
