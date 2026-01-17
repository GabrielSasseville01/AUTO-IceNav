"""
Analysis plots for fracture simulation results.

This module provides publication-quality figures for analyzing
fracture simulation results, including fracture statistics,
bond failure modes, and energy evolution.
"""

from typing import Optional, List, Dict
import numpy as np
import matplotlib.pyplot as plt


class FractureAnalysisPlots:
    """
    Generate analysis plots for fracture simulation results.
    
    Provides methods for creating various analysis figures suitable
    for publications and debugging.
    """
    
    @staticmethod
    def plot_fracture_statistics(fracture_events: List[Dict],
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot statistics about fracture events.
        
        Args:
            fracture_events: List of fracture event dictionaries with
                           keys: time, x, y, impulse/impact_force, n_fragments
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        if not fracture_events:
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No fracture events',
                       transform=ax.transAxes, ha='center', va='center')
            return fig
            
        # Extract data
        times = [e.get('time', 0) for e in fracture_events]
        forces = [e.get('impact_force', e.get('impulse', 0)) for e in fracture_events]
        n_fragments = [e.get('n_fragments', 1) for e in fracture_events]
        locations = [(e.get('x', 0), e.get('y', 0)) for e in fracture_events]
        
        # Fracture events over time
        ax = axes[0, 0]
        ax.hist(times, bins=30, edgecolor='black', color='steelblue', alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Fracture events')
        ax.set_title('Fracture Events Distribution')
        ax.grid(True, alpha=0.3)
        
        # Impact force vs fragments
        ax = axes[0, 1]
        ax.scatter(forces, n_fragments, alpha=0.6, c='red', edgecolors='darkred')
        ax.set_xlabel('Impact Force (N)')
        ax.set_ylabel('Number of Fragments')
        ax.set_title('Impact Force vs Fragmentation')
        ax.grid(True, alpha=0.3)
        
        # Fracture locations
        ax = axes[1, 0]
        if locations:
            locs = np.array(locations)
            scatter = ax.scatter(locs[:, 0], locs[:, 1], c=times, 
                               cmap='viridis', alpha=0.7, edgecolors='black')
            fig.colorbar(scatter, ax=ax, label='Time (s)')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title('Fracture Locations (colored by time)')
            ax.set_aspect('equal')
            
        # Fragment size distribution
        ax = axes[1, 1]
        if n_fragments:
            bins = range(1, max(n_fragments) + 2)
            ax.hist(n_fragments, bins=bins, edgecolor='black', 
                   color='green', alpha=0.7, align='left')
        ax.set_xlabel('Number of Fragments')
        ax.set_ylabel('Frequency')
        ax.set_title('Fragment Count Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
        
    @staticmethod
    def plot_bond_failure_modes(bonds: List,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Analyze how bonds failed (tensile vs shear).
        
        Args:
            bonds: List of Bond objects
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        if not bonds:
            for ax in axes:
                ax.text(0.5, 0.5, 'No bond data',
                       transform=ax.transAxes, ha='center', va='center')
            return fig
            
        broken_bonds = [b for b in bonds if b.broken]
        
        if not broken_bonds:
            for ax in axes:
                ax.text(0.5, 0.5, 'No broken bonds',
                       transform=ax.transAxes, ha='center', va='center')
            return fig
            
        # Count failure modes
        tensile = sum(1 for b in broken_bonds if b.break_mode == 'tensile')
        shear = sum(1 for b in broken_bonds if b.break_mode == 'shear')
        mixed = sum(1 for b in broken_bonds if b.break_mode == 'mixed')
        other = len(broken_bonds) - tensile - shear - mixed
        
        # Pie chart of failure modes
        ax = axes[0]
        sizes = [tensile, shear, mixed, other]
        labels = ['Tensile', 'Shear', 'Mixed', 'Other']
        colors = ['#ff6b6b', '#4ecdc4', '#95a5a6', '#f0f0f0']
        
        # Filter out zero values
        non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
        if non_zero:
            sizes, labels, colors = zip(*non_zero)
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors,
                  explode=[0.02] * len(sizes))
        ax.set_title('Bond Failure Modes')
        
        # Spatial distribution of failure modes
        ax = axes[1]
        color_map = {'tensile': 'red', 'shear': 'blue', 'mixed': 'gray', None: 'black'}
        for bond in broken_bonds:
            if hasattr(bond, 'pos_a') and hasattr(bond, 'pos_b'):
                midpoint = (bond.pos_a + bond.pos_b) / 2
            else:
                continue
            color = color_map.get(bond.break_mode, 'black')
            ax.plot(midpoint[0], midpoint[1], 'o', color=color, alpha=0.5, markersize=3)
            
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Failure Mode by Location')
        ax.set_aspect('equal')
        
        # Create legend
        for mode, color in [('Tensile', 'red'), ('Shear', 'blue'), ('Mixed', 'gray')]:
            ax.plot([], [], 'o', color=color, label=mode)
        ax.legend()
        
        # Failure stress distribution
        ax = axes[2]
        stresses = [b.failure_stress for b in broken_bonds if b.failure_stress > 0]
        if stresses:
            ax.hist(stresses, bins=30, edgecolor='black', color='orange', alpha=0.7)
            ax.axvline(np.mean(stresses), color='r', linestyle='--',
                      label=f'Mean: {np.mean(stresses):.2e} Pa')
        ax.set_xlabel('Failure Stress (Pa)')
        ax.set_ylabel('Count')
        ax.set_title('Bond Failure Stress Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
        
    @staticmethod
    def plot_energy_evolution(sim_history: Dict,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot energy components over simulation time.
        
        Args:
            sim_history: Dictionary with keys: time, kinetic_energy_ship,
                        kinetic_energy_ice, bond_strain_energy (optional)
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if not sim_history or 'time' not in sim_history:
            ax.text(0.5, 0.5, 'No simulation history data',
                   transform=ax.transAxes, ha='center', va='center')
            return fig
            
        time = np.array(sim_history['time'])
        
        # Get energy components (with defaults)
        ke_ship = np.array(sim_history.get('kinetic_energy_ship', np.zeros_like(time)))
        ke_ice = np.array(sim_history.get('kinetic_energy_ice', np.zeros_like(time)))
        bond_energy = np.array(sim_history.get('bond_strain_energy', np.zeros_like(time)))
        
        # Stack plot
        ax.fill_between(time, 0, ke_ship, alpha=0.7, label='Ship KE', color='blue')
        ax.fill_between(time, ke_ship, ke_ship + ke_ice, alpha=0.7, 
                       label='Ice KE', color='cyan')
        if np.any(bond_energy > 0):
            ax.fill_between(time, ke_ship + ke_ice, ke_ship + ke_ice + bond_energy,
                           alpha=0.7, label='Bond Strain Energy', color='orange')
        
        # Total energy line
        total = ke_ship + ke_ice + bond_energy
        ax.plot(time, total, 'k--', linewidth=2, label='Total Energy')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Energy (J)')
        ax.set_title('Energy Evolution During Simulation')
        ax.legend(loc='upper right')
        ax.set_xlim(time[0], time[-1])
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
        
    @staticmethod
    def plot_impulse_history(impulses: List[Dict],
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot collision impulse history.
        
        Args:
            impulses: List of impulse dictionaries with time and magnitude
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        if not impulses:
            for ax in axes:
                ax.text(0.5, 0.5, 'No impulse data',
                       transform=ax.transAxes, ha='center', va='center')
            return fig
            
        times = [imp.get('time', 0) for imp in impulses]
        magnitudes = [imp.get('magnitude', np.linalg.norm(imp.get('impulse', [0, 0]))) 
                     for imp in impulses]
        
        # Time series
        ax = axes[0]
        ax.plot(times, magnitudes, 'b-', alpha=0.7)
        ax.scatter(times, magnitudes, c='red', s=20, alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Impulse Magnitude (N·s)')
        ax.set_title('Collision Impulse Over Time')
        ax.grid(True, alpha=0.3)
        
        # Distribution
        ax = axes[1]
        ax.hist(magnitudes, bins=30, edgecolor='black', color='steelblue', alpha=0.7)
        ax.axvline(np.mean(magnitudes), color='r', linestyle='--',
                  label=f'Mean: {np.mean(magnitudes):.0f} N·s')
        ax.set_xlabel('Impulse Magnitude (N·s)')
        ax.set_ylabel('Frequency')
        ax.set_title('Impulse Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
        
    @staticmethod
    def create_summary_figure(fracture_events: List[Dict],
                               bonds: List,
                               energy_history: Dict,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive summary figure with all analysis plots.
        
        Args:
            fracture_events: List of fracture event dictionaries
            bonds: List of Bond objects
            energy_history: Energy history dictionary
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Fracture locations
        ax1 = fig.add_subplot(gs[0, 0])
        if fracture_events:
            locs = np.array([(e.get('x', 0), e.get('y', 0)) for e in fracture_events])
            times = [e.get('time', 0) for e in fracture_events]
            ax1.scatter(locs[:, 0], locs[:, 1], c=times, cmap='viridis', alpha=0.7)
            ax1.set_aspect('equal')
        ax1.set_title('Fracture Locations')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        
        # Fracture timeline
        ax2 = fig.add_subplot(gs[0, 1])
        if fracture_events:
            times = [e.get('time', 0) for e in fracture_events]
            ax2.hist(times, bins=20, edgecolor='black', color='steelblue')
        ax2.set_title('Fracture Events Over Time')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Count')
        
        # Failure modes pie chart
        ax3 = fig.add_subplot(gs[0, 2])
        if bonds:
            broken = [b for b in bonds if b.broken]
            if broken:
                modes = {}
                for b in broken:
                    mode = b.break_mode or 'unknown'
                    modes[mode] = modes.get(mode, 0) + 1
                ax3.pie(modes.values(), labels=modes.keys(), autopct='%1.1f%%')
        ax3.set_title('Failure Modes')
        
        # Energy evolution
        ax4 = fig.add_subplot(gs[1, :])
        if energy_history and 'time' in energy_history:
            time = energy_history['time']
            ke_total = energy_history.get('kinetic_energy', np.zeros_like(time))
            ax4.plot(time, ke_total, 'b-', linewidth=2)
            ax4.fill_between(time, 0, ke_total, alpha=0.3)
        ax4.set_title('Total Kinetic Energy')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Energy (J)')
        ax4.grid(True, alpha=0.3)
        
        # Statistics text box
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        stats_text = "Simulation Summary\n"
        stats_text += "=" * 40 + "\n"
        stats_text += f"Total fracture events: {len(fracture_events)}\n"
        if bonds:
            broken_count = sum(1 for b in bonds if b.broken)
            stats_text += f"Total bonds: {len(bonds)}\n"
            stats_text += f"Broken bonds: {broken_count} ({100*broken_count/len(bonds):.1f}%)\n"
        if fracture_events:
            n_frags = [e.get('n_fragments', 1) for e in fracture_events]
            stats_text += f"Average fragments per event: {np.mean(n_frags):.1f}\n"
            
        ax5.text(0.1, 0.5, stats_text, transform=ax5.transAxes,
                fontfamily='monospace', fontsize=12, verticalalignment='center')
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
