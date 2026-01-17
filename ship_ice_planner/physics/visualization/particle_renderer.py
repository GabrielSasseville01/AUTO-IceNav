"""
Particle renderer for visualizing bonded DEM assemblies.

This module provides visualization tools for inspecting the internal
structure of bonded ice floes, including particles, bonds, and stress states.
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib import patches


class ParticleRenderer:
    """
    Visualize bonded DEM particle assemblies.
    
    Provides detailed visualization of particle positions, bond connectivity,
    stress distributions, and crack patterns for debugging and analysis.
    """
    
    def __init__(self, ax: Optional[plt.Axes] = None, figsize: tuple = (12, 10)):
        """
        Initialize the particle renderer.
        
        Args:
            ax: Matplotlib axes to draw on (creates new figure if None)
            figsize: Figure size if creating new figure
        """
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            self.fig = ax.figure
            self.ax = ax
            
        self._ice_patches = []
        self._bond_collections = []
        
    def draw_assembly(self, 
                      assembly,
                      show_particles: bool = True,
                      show_bonds: bool = True,
                      show_broken_bonds: bool = True,
                      color_by: str = 'stress',
                      title: Optional[str] = None):
        """
        Draw a bonded particle assembly.
        
        Args:
            assembly: BondedAssembly object to visualize
            show_particles: Draw particle circles
            show_bonds: Draw intact bonds as lines
            show_broken_bonds: Draw broken bonds as red dashed lines
            color_by: Color scheme: 'stress', 'velocity', 'fragment', or 'none'
            title: Optional title for the plot
        """
        self.ax.clear()
        self._ice_patches.clear()
        self._bond_collections.clear()
        
        if show_bonds:
            self._draw_bonds(assembly, show_broken_bonds)
            
        if show_particles:
            self._draw_particles(assembly, color_by)
            
        self._draw_boundary(assembly)
        
        # Set axes properties
        self.ax.set_aspect('equal')
        self.ax.autoscale_view()
        
        if title:
            self.ax.set_title(title)
        else:
            self.ax.set_title(
                f'Bonded Assembly: {assembly.n_particles} particles, '
                f'{assembly.n_bonds} bonds ({assembly.n_broken_bonds} broken)'
            )
            
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        
    def _draw_particles(self, assembly, color_by: str):
        """Draw particles as circles with optional coloring."""
        if not assembly.particles:
            return
            
        positions = np.array([p.pos for p in assembly.particles])
        radii = np.array([p.radius for p in assembly.particles])
        
        # Determine colors
        if color_by == 'stress':
            colors = assembly.get_all_particle_stresses()
            if colors.max() > 0:
                colors = colors / colors.max()
            cmap = plt.cm.YlOrRd
            cbar_label = 'Stress (normalized)'
        elif color_by == 'velocity':
            speeds = np.array([np.linalg.norm(p.vel) for p in assembly.particles])
            colors = speeds / (speeds.max() + 1e-6)
            cmap = plt.cm.viridis
            cbar_label = 'Speed (normalized)'
        elif color_by == 'fragment':
            fragments = assembly.get_fragments()
            colors = np.zeros(len(assembly.particles))
            for frag_idx, frag in enumerate(fragments):
                for particle_idx in frag:
                    colors[particle_idx] = frag_idx
            colors = colors / (len(fragments) + 1e-6)
            cmap = plt.cm.Set1
            cbar_label = 'Fragment ID'
        else:
            colors = None
            cmap = None
            cbar_label = None
            
        # Create circle patches
        circles = [plt.Circle(pos, r, alpha=0.7) for pos, r in zip(positions, radii)]
        collection = PatchCollection(circles, alpha=0.7)
        
        if cmap is not None and colors is not None:
            collection.set_cmap(cmap)
            collection.set_array(colors)
            self.ax.add_collection(collection)
            cbar = self.fig.colorbar(collection, ax=self.ax, label=cbar_label, shrink=0.8)
        else:
            collection.set_facecolor('lightblue')
            collection.set_edgecolor('blue')
            self.ax.add_collection(collection)
            
        self._ice_patches.append(collection)
        
    def _draw_bonds(self, assembly, show_broken: bool):
        """Draw bonds as line segments."""
        if not assembly.bonds:
            return
            
        intact_segments = []
        broken_segments = []
        
        for bond in assembly.bonds:
            pi = assembly.particles[bond.i]
            pj = assembly.particles[bond.j]
            segment = [pi.pos.tolist(), pj.pos.tolist()]
            
            if bond.broken:
                broken_segments.append(segment)
            else:
                intact_segments.append(segment)
        
        # Draw intact bonds
        if intact_segments:
            intact_lc = LineCollection(
                intact_segments, 
                colors='steelblue',
                linewidths=1, 
                alpha=0.6,
                label='Intact bonds'
            )
            self.ax.add_collection(intact_lc)
            self._bond_collections.append(intact_lc)
        
        # Draw broken bonds (cracks)
        if show_broken and broken_segments:
            broken_lc = LineCollection(
                broken_segments,
                colors='red',
                linewidths=2,
                linestyles='dashed',
                alpha=0.8,
                label='Broken bonds (cracks)'
            )
            self.ax.add_collection(broken_lc)
            self._bond_collections.append(broken_lc)
            
    def _draw_boundary(self, assembly):
        """Draw original floe boundary."""
        boundary = plt.Polygon(
            assembly.vertices,
            fill=False,
            edgecolor='black',
            linewidth=2,
            linestyle='-',
            label='Floe boundary'
        )
        self.ax.add_patch(boundary)
        
    def draw_crack_pattern(self, assembly, ax: Optional[plt.Axes] = None):
        """
        Draw only the crack pattern (broken bonds) with crack tips highlighted.
        
        Args:
            assembly: BondedAssembly object
            ax: Optional axes (uses self.ax if None)
        """
        if ax is None:
            ax = self.ax
        else:
            ax.clear()
            
        # Draw floe boundary (dashed)
        boundary = plt.Polygon(
            assembly.vertices,
            fill=False,
            edgecolor='gray',
            linewidth=1,
            linestyle='--'
        )
        ax.add_patch(boundary)
        
        # Collect crack segments
        crack_segments = []
        for bond in assembly.bonds:
            if bond.broken:
                pi = assembly.particles[bond.i]
                pj = assembly.particles[bond.j]
                crack_segments.append([pi.pos.tolist(), pj.pos.tolist()])
                
        # Draw crack lines
        if crack_segments:
            lc = LineCollection(crack_segments, colors='red', linewidths=2)
            ax.add_collection(lc)
            
        # Find and highlight crack tips
        crack_tips = self._find_crack_tips(assembly)
        if crack_tips:
            tips = np.array(crack_tips)
            ax.scatter(
                tips[:, 0], tips[:, 1],
                c='yellow', s=100, marker='*',
                edgecolors='red', linewidths=2,
                label='Crack tips', zorder=5
            )
            
        ax.set_aspect('equal')
        ax.autoscale_view()
        ax.legend()
        ax.set_title(f'Crack Pattern | {len(crack_segments)} broken bonds')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
    def _find_crack_tips(self, assembly) -> list:
        """
        Find crack tips (broken bonds adjacent to intact bonds).
        
        Returns list of (x, y) positions of crack tips.
        """
        if not assembly.bonds:
            return []
            
        # Build adjacency from particle index to bonds
        particle_bonds = {i: [] for i in range(len(assembly.particles))}
        for bond in assembly.bonds:
            particle_bonds[bond.i].append(bond)
            particle_bonds[bond.j].append(bond)
            
        crack_tips = []
        
        for bond in assembly.bonds:
            if not bond.broken:
                continue
                
            # Check if this broken bond is adjacent to intact bonds
            for particle_idx in [bond.i, bond.j]:
                has_intact = any(not b.broken for b in particle_bonds[particle_idx] if b != bond)
                has_broken = any(b.broken for b in particle_bonds[particle_idx] if b != bond)
                
                if has_intact and has_broken:
                    # This is a crack tip
                    crack_tips.append(assembly.particles[particle_idx].pos.tolist())
                    
        return crack_tips
        
    def save(self, filepath: str, dpi: int = 150):
        """Save current figure to file."""
        self.fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        
    def show(self):
        """Display the figure."""
        plt.show()
