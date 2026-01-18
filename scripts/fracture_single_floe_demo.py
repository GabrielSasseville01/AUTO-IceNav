#!/usr/bin/env python3
"""
Demo: Ship fracturing ONE LARGE ice floe using PROGRESSIVE crack growth.
With proper physics - ship is a kinematic body that pushes ice.
Shows crack growing over time before full fracture.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D
from PIL import Image
import pymunk
from pymunk import Vec2d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ship_ice_planner.utils.sim_utils import ICE_THICKNESS, create_polygon
from ship_ice_planner.utils.ice_fracture_ridge import (
    IceFloeState, fracture_ice_floe_progressive, SIGMA_C
)

def create_large_floe(space, center_x, center_y, size=80):
    """Create a single large irregular ice floe."""
    n_sides = 8
    angles = np.linspace(0, 2*np.pi, n_sides, endpoint=False)
    np.random.seed(42)
    radii = size * (0.8 + 0.4 * np.random.rand(n_sides))
    vertices = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    poly = create_polygon(space, vertices.tolist(), center_x, center_y)
    poly.collision_type = 2  # Ice
    poly.friction = 0.3
    
    return poly, vertices

def create_ship_body(space, x, y, heading):
    """Create ship as a KINEMATIC body that can push ice."""
    ship_verts = [
        (25, 0), (20, 5), (10, 7), (-20, 7),
        (-20, -7), (10, -7), (20, -5)
    ]
    
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    rotated = [(v[0]*cos_h - v[1]*sin_h, v[0]*sin_h + v[1]*cos_h) for v in ship_verts]
    
    body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    body.position = (x, y)
    
    shape = pymunk.Poly(body, rotated)
    shape.collision_type = 1  # Ship
    shape.friction = 0.5
    shape.elasticity = 0.2
    
    space.add(body, shape)
    return body, shape

def get_world_vertices(poly):
    """Get polygon vertices in world coordinates."""
    verts = poly.get_vertices()
    return np.array([[poly.body.local_to_world(v).x, poly.body.local_to_world(v).y] for v in verts])

def main():
    print("=" * 60)
    print("Progressive Crack Fracture Demo")
    print("=" * 60)
    
    # Setup physics
    space = pymunk.Space()
    space.gravity = (0, 0)
    space.damping = 0.95
    
    # Create ONE large ice floe
    floe_x, floe_y = 100, 200
    large_floe, _ = create_large_floe(space, floe_x, floe_y, size=60)
    large_floe.idx = 0
    
    # Calculate area
    verts = get_world_vertices(large_floe)
    area = 0.5 * abs(sum(verts[i,0] * verts[(i+1)%len(verts),1] - 
                         verts[(i+1)%len(verts),0] * verts[i,1] 
                         for i in range(len(verts))))
    
    ice_floe_states = {0: IceFloeState(large_floe, area, thickness=ICE_THICKNESS)}
    all_polys = [large_floe]
    next_idx = 1
    
    print(f"Created large floe: area={area:.0f} m²")
    
    # Create ship as physics body
    ship_x = 100
    ship_y_start = 50
    ship_heading = np.pi / 2
    ship_speed = 4.0
    
    ship_body, ship_shape = create_ship_body(space, ship_x, ship_y_start, ship_heading)
    print(f"Created ship at ({ship_x}, {ship_y_start})")
    
    # Track collisions for stress computation
    collision_data = {}  # idx -> list of (force, contact_pos)
    
    def post_solve_handler(arbiter, space, data):
        """Called after collision - track force and contact point."""
        ice_shape = None
        for shape in arbiter.shapes:
            if shape.collision_type == 2:
                ice_shape = shape
                break
        
        if ice_shape and hasattr(ice_shape, 'idx'):
            # Get impulse as force (divide by dt)
            impulse = arbiter.total_impulse
            dt = 1.0 / 60.0
            force = (impulse.x / dt, impulse.y / dt)
            
            # Get contact point
            contact_set = arbiter.contact_point_set
            if len(contact_set.points) > 0:
                contact_pos = contact_set.points[0].point_a
                
                if ice_shape.idx not in collision_data:
                    collision_data[ice_shape.idx] = []
                collision_data[ice_shape.idx].append((force, (contact_pos.x, contact_pos.y)))
    
    space.on_collision(
        collision_type_a=1,
        collision_type_b=2,
        post_solve=post_solve_handler
    )
    
    # Animation
    n_frames = 120
    frames_dir = 'fracture_frames_tmp'
    os.makedirs(frames_dir, exist_ok=True)
    
    fractured = False
    fracture_frame = -1
    crack_initiated_frame = -1
    
    print(f"\nGenerating {n_frames} frames with progressive cracking...")
    print("Watch for: 1) Crack initiation  2) Crack growth  3) Full fracture")
    
    dt = 1.0 / 60.0
    
    for frame in range(n_frames):
        # Move ship
        ship_body.velocity = (0, ship_speed * 20)
        
        # Physics substeps
        for _ in range(3):
            space.step(dt)
        
        # Update stress from collisions
        for poly in all_polys:
            if not hasattr(poly, 'idx') or poly.idx not in ice_floe_states:
                continue
            
            state = ice_floe_states[poly.idx]
            
            # Add collision interactions to state
            if poly.idx in collision_data:
                for force, contact_pos in collision_data[poly.idx]:
                    state.add_interaction(force, contact_pos)
            
            # Update stress tensor
            center = poly.body.position
            state.update_stress_from_interactions(center.x, center.y)
        
        collision_data.clear()
        
        # Progressive crack logic
        to_fracture = []
        for poly in all_polys[:]:
            if not hasattr(poly, 'idx') or poly.idx not in ice_floe_states:
                continue
            
            state = ice_floe_states[poly.idx]
            
            if state.has_active_crack():
                # Grow existing crack
                ready = state.update_crack(dt=dt * 3)  # 3 substeps
                if ready and not fractured:
                    to_fracture.append((poly, state))
                    print(f"\n>>> Frame {frame}: Crack spans floe - FULL FRACTURE!")
            else:
                # Check if should initiate crack
                if state.should_fracture_stress(yield_type='mohr'):
                    # Get contact point for crack origin
                    contact_point = None
                    if len(state.interactions) > 0:
                        _, contact_pos = state.interactions[0]
                        contact_point = contact_pos
                    elif state.center_position:
                        # Offset toward ship
                        ship_pos = ship_body.position
                        cx, cy = state.center_position
                        direction = np.array([ship_pos.x - cx, ship_pos.y - cy])
                        if np.linalg.norm(direction) > 0:
                            direction = direction / np.linalg.norm(direction)
                            contact_point = (cx + direction[0] * 30, cy + direction[1] * 30)
                    
                    state.initiate_crack(contact_point)
                    crack_initiated_frame = frame
                    print(f"\n>>> Frame {frame}: CRACK INITIATED! (stress exceeded yield)")
        
        # Apply fractures
        for poly, state in to_fracture:
            new_polys, next_idx = fracture_ice_floe_progressive(
                space, poly, state, ice_floe_states, next_idx
            )
            
            if new_polys:
                print(f"    Split into {len(new_polys)} pieces along crack line")
                if poly in all_polys:
                    all_polys.remove(poly)
                all_polys.extend(new_polys)
                fractured = True
                fracture_frame = frame
        
        # Draw frame
        fig, ax = plt.subplots(figsize=(8, 10), dpi=100)
        ax.set_facecolor('#1a3d5c')
        
        ship_y = ship_body.position.y
        
        # Draw ice floes
        for poly in all_polys:
            if not hasattr(poly, 'idx') or poly.idx not in ice_floe_states:
                continue
            
            state = ice_floe_states[poly.idx]
            
            try:
                verts = get_world_vertices(poly)
            except:
                continue
            if len(verts) < 3:
                continue
            
            # Color based on state
            if fractured and poly.idx > 0:
                color = '#b8d4e8'
                edge = '#4a90b8'
            elif state.has_active_crack():
                color = '#ffe4e1'  # Light red - cracking
                edge = '#8b0000'
            else:
                color = '#e8f4fc'
                edge = '#2c5f7c'
            
            patch = MplPolygon(verts, closed=True,
                              facecolor=color, edgecolor=edge,
                              linewidth=2, alpha=0.9)
            ax.add_patch(patch)
            
            # Draw crack line if active
            if state.has_active_crack() and not fractured:
                crack_line = state.get_crack_line()
                if crack_line:
                    (x1, y1), (x2, y2) = crack_line
                    # Convert from local to world coordinates
                    center = poly.body.position
                    wx1, wy1 = center.x + x1, center.y + y1
                    wx2, wy2 = center.x + x2, center.y + y2
                    
                    # Draw crack as thick red line
                    ax.plot([wx1, wx2], [wy1, wy2], 
                           color='#ff0000', linewidth=3, zorder=5,
                           solid_capstyle='round')
                    
                    # Draw crack tips as circles
                    ax.scatter([wx1, wx2], [wy1, wy2], 
                              c='#ff0000', s=50, zorder=6, marker='o')
        
        # Draw ship
        ship_verts = get_world_vertices(ship_shape)
        ship_patch = MplPolygon(ship_verts, closed=True,
                               facecolor='#c41e3a', edgecolor='#8b0000',
                               linewidth=2, zorder=10)
        ax.add_patch(ship_patch)
        
        # Camera follows ship
        ax.set_xlim(0, 200)
        ax.set_ylim(ship_y - 80, ship_y + 120)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Status text
        if frame == fracture_frame:
            status = "SPLIT!"
        elif fractured:
            status = "Fractured"
        elif crack_initiated_frame >= 0 and frame >= crack_initiated_frame:
            # Show crack length
            for poly in all_polys:
                if hasattr(poly, 'idx') and poly.idx in ice_floe_states:
                    state = ice_floe_states[poly.idx]
                    if state.has_active_crack():
                        crack_len = state.active_crack.length
                        floe_diam = 2 * np.sqrt(state.initial_area / np.pi)
                        pct = 100 * crack_len / floe_diam
                        status = f"Crack growing: {pct:.0f}%"
                        break
            else:
                status = "Cracking..."
        else:
            status = "Approaching..."
        
        ax.set_title(f'Progressive Crack Demo - {status}', 
                    fontsize=14, color='white', pad=10)
        
        fig.patch.set_facecolor('#1a3d5c')
        plt.tight_layout()
        
        frame_path = os.path.join(frames_dir, f'{frame:03d}.png')
        plt.savefig(frame_path, facecolor='#1a3d5c', edgecolor='none')
        plt.close()
    
    # Create GIF
    print("\nAssembling GIF...")
    frames = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.png')])
    images = [Image.open(f) for f in frames]
    
    gif_path = 'fracture_demo.gif'
    images[0].save(gif_path, save_all=True, append_images=images[1:],
                   duration=50, loop=0, optimize=False)
    
    print(f"✓ Saved: {gif_path}")
    
    # Cleanup
    for f in frames:
        os.remove(f)
    os.rmdir(frames_dir)
    
    return gif_path

if __name__ == '__main__':
    main()
