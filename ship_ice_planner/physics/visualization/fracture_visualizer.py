"""
Fracture visualization for bonded DEM simulation.

This module provides tools for visualizing and animating crack propagation
through bonded ice floes during simulation.
"""

from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

from .particle_renderer import ParticleRenderer


class FractureVisualizer:
    """
    Visualize fracture propagation in bonded ice floes.
    
    Records fracture state over time and creates animations showing
    crack formation and growth.
    """
    
    def __init__(self, assembly):
        """
        Initialize fracture visualizer.
        
        Args:
            assembly: BondedAssembly object to track
        """
        self.assembly = assembly
        self.fracture_history: List[tuple] = []  # List of (time, broken_bond_indices)
        
    def record_state(self, time: float):
        """
        Record current fracture state for animation.
        
        Args:
            time: Current simulation time
        """
        broken_indices = [i for i, b in enumerate(self.assembly.bonds) if b.broken]
        self.fracture_history.append((time, broken_indices.copy()))
        
    def create_animation(self, 
                         output_path: str = 'fracture_propagation.mp4',
                         fps: int = 30,
                         figsize: tuple = (10, 10)) -> FuncAnimation:
        """
        Create animation of fracture propagation.
        
        Args:
            output_path: Path to save animation
            fps: Frames per second
            figsize: Figure size
            
        Returns:
            FuncAnimation object
        """
        fig, ax = plt.subplots(figsize=figsize)
        renderer = ParticleRenderer(ax=ax)
        
        # Store original bond states
        original_states = [b.broken for b in self.assembly.bonds]
        
        def init():
            ax.clear()
            return []
            
        def update(frame):
            ax.clear()
            time, broken_at_frame = self.fracture_history[frame]
            
            # Set bond states to match this frame
            for i, bond in enumerate(self.assembly.bonds):
                bond.broken = i in broken_at_frame
                
            renderer.draw_assembly(
                self.assembly,
                show_broken_bonds=True,
                color_by='stress',
                title=f'Time: {time:.2f}s | Broken bonds: {len(broken_at_frame)}'
            )
            ax.set_aspect('equal')
            return []
            
        anim = FuncAnimation(
            fig, update,
            init_func=init,
            frames=len(self.fracture_history),
            interval=1000/fps,
            blit=False
        )
        
        # Save animation
        try:
            anim.save(output_path, writer='ffmpeg', fps=fps)
            print(f'Animation saved to {output_path}')
        except Exception as e:
            print(f'Could not save animation: {e}')
            print('Trying pillow writer...')
            try:
                anim.save(output_path.replace('.mp4', '.gif'), writer='pillow', fps=fps)
            except Exception as e2:
                print(f'Could not save with pillow either: {e2}')
                
        # Restore original bond states
        for i, state in enumerate(original_states):
            self.assembly.bonds[i].broken = state
            
        return anim
        
    def plot_crack_pattern(self, ax: Optional[plt.Axes] = None):
        """
        Plot final crack pattern with crack tips highlighted.
        
        Args:
            ax: Axes to plot on (creates new figure if None)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
            
        renderer = ParticleRenderer(ax=ax)
        renderer.draw_crack_pattern(self.assembly, ax=ax)
        
    def plot_fracture_timeline(self, ax: Optional[plt.Axes] = None):
        """
        Plot number of broken bonds over time.
        
        Args:
            ax: Axes to plot on (creates new figure if None)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        if not self.fracture_history:
            ax.text(0.5, 0.5, 'No fracture data recorded',
                   transform=ax.transAxes, ha='center', va='center')
            return
            
        times = [t for t, _ in self.fracture_history]
        n_broken = [len(broken) for _, broken in self.fracture_history]
        
        ax.plot(times, n_broken, 'r-', linewidth=2)
        ax.fill_between(times, 0, n_broken, alpha=0.3, color='red')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Number of Broken Bonds')
        ax.set_title('Fracture Progression')
        ax.grid(True, alpha=0.3)
        
    def plot_fracture_rate(self, ax: Optional[plt.Axes] = None, window: int = 5):
        """
        Plot rate of bond breaking over time.
        
        Args:
            ax: Axes to plot on
            window: Smoothing window size
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        if len(self.fracture_history) < 2:
            ax.text(0.5, 0.5, 'Insufficient fracture data',
                   transform=ax.transAxes, ha='center', va='center')
            return
            
        times = np.array([t for t, _ in self.fracture_history])
        n_broken = np.array([len(broken) for _, broken in self.fracture_history])
        
        # Compute rate
        dt = np.diff(times)
        dn = np.diff(n_broken)
        rate = dn / (dt + 1e-6)
        
        # Smooth with moving average
        if len(rate) >= window:
            rate_smooth = np.convolve(rate, np.ones(window)/window, mode='valid')
            times_smooth = times[window//2:-window//2+1][:len(rate_smooth)]
        else:
            rate_smooth = rate
            times_smooth = times[:-1]
            
        ax.plot(times_smooth, rate_smooth, 'b-', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Fracture Rate (bonds/s)')
        ax.set_title('Bond Breaking Rate')
        ax.grid(True, alpha=0.3)
        
    def get_fracture_statistics(self) -> dict:
        """
        Get statistics about the fracture process.
        
        Returns:
            Dictionary with fracture statistics
        """
        if not self.fracture_history:
            return {
                'total_broken': 0,
                'fracture_duration': 0,
                'peak_rate': 0,
            }
            
        times = [t for t, _ in self.fracture_history]
        n_broken = [len(broken) for _, broken in self.fracture_history]
        
        # Find when fracture started and ended
        fracture_started = next((t for t, n in zip(times, n_broken) if n > 0), times[-1])
        
        # Compute rate
        if len(n_broken) >= 2:
            dt = np.diff(times)
            dn = np.diff(n_broken)
            rate = dn / (dt + 1e-6)
            peak_rate = np.max(rate) if len(rate) > 0 else 0
        else:
            peak_rate = 0
            
        return {
            'total_broken': max(n_broken),
            'total_bonds': len(self.assembly.bonds),
            'fraction_broken': max(n_broken) / len(self.assembly.bonds) if self.assembly.bonds else 0,
            'fracture_duration': times[-1] - fracture_started if len(times) > 0 else 0,
            'peak_rate': peak_rate,
            'first_fracture_time': fracture_started,
        }
        
    @classmethod
    def from_history(cls, assembly, history: List[tuple]) -> 'FractureVisualizer':
        """
        Create visualizer from recorded history.
        
        Args:
            assembly: BondedAssembly object
            history: List of (time, broken_indices) tuples
            
        Returns:
            FractureVisualizer instance
        """
        viz = cls(assembly)
        viz.fracture_history = history.copy()
        return viz
