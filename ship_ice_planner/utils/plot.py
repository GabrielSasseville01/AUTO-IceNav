"""
Aggregates all plotting objects into a single class
"""
import os
import time
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches, colors, ticker as tkr
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ship_ice_planner.geometry.utils import Rxy_3d, Rxy

# file names
ANIM_MOVIE_FILE = 'sim.mp4'
DEFAULT_SAVE_FIG_FORMAT = 'pdf'
# colors
ICE_PATCH_COLOR         = 'lightgrey'  # 'white'
SHIP_PATCH_COLOR        = 'white'      # 'lightgrey'
OPEN_WATER_COLOR        = 'lightblue'      # 'lightblue'

# Gradient settings for ice patches
ICE_GRADIENT_LAYERS     = 8
ICE_EDGE_COLOR          = (0.75, 0.75, 0.75)  # light grey at edges
ICE_CENTER_COLOR        = (1.0, 1.0, 1.0)     # white at center


def add_gradient_polygon(ax, vertices, n_layers=ICE_GRADIENT_LAYERS, 
                         edge_color=ICE_EDGE_COLOR, center_color=ICE_CENTER_COLOR):
    """Add a polygon with gradient from edge_color to center_color directly to axes.
    Uses clipping to handle concave polygons correctly."""
    vertices = np.array(vertices)
    centroid = np.mean(vertices, axis=0)
    
    # Create clip patch from original polygon
    clip_patch = patches.Polygon(vertices, closed=True, transform=ax.transData)
    
    added_patches = []
    
    # Create layers from outside (grey) to inside (white)
    for i in range(n_layers):
        scale = 1.0 - (i / n_layers) * 0.8
        t = i / (n_layers - 1) if n_layers > 1 else 0
        color = tuple(edge_color[j] + t * (center_color[j] - edge_color[j]) for j in range(3))
        scaled_vertices = centroid + scale * (vertices - centroid)
        patch = patches.Polygon(scaled_vertices, closed=True, fill=True, fc=color, ec='none', zorder=10)
        ax.add_patch(patch)
        # Clip each layer to the original polygon boundary
        patch.set_clip_path(clip_patch)
        added_patches.append(patch)
    
    # Add outline
    outline = patches.Polygon(vertices, closed=True, fill=False, ec=(0.5, 0.5, 0.5), linewidth=0.5, zorder=10)
    ax.add_patch(outline)
    added_patches.append(outline)
    
    return added_patches


SWATH_COLOR             = 'white'
PLANNED_PATH_COLOR      = 'red'
SHIP_ACTUAL_PATH_COLOR  = 'blue'
GOAL_COLOR              = 'green'
DEFAULT_COLOR_MAP       = 'viridis'


class Plot:

    def __init__(
            self,
            costmap: np.ndarray = None,
            obstacles: List = None,
            path: np.ndarray = None,
            ship_pos: Union[Tuple, np.ndarray] = None,
            ship_vertices: np.ndarray = None,
            horizon: float = None,
            goal: float = None,
            y_axis_limit: Union[float, Tuple] = None,
            legend=False,
            save_fig_dir: str = None,
            map_shape: Tuple[int, int] = None,
            # ---- map/planner plot params ----
            map_figsize=(10, 10),
            turning_radius: float = None,
            path_nodes: Tuple[List, List] = tuple(),
            nodes_expanded: dict = None,
            swath: np.ndarray = None,
            scale: float = 1,
            costmap_min_max: Tuple[float, float] = None,
            global_path: np.ndarray = None,  # can also just be the path used to warm start optim step
            sea_currents: np.ndarray = None,  # sea current vector field (H, W, 2) or (T, H, W, 2)
            sea_currents_subsample: int = 20,  # subsampling factor for quiver plot
            # ---- sim plot params ----
            sim_figsize=(10, 10),
            target: Tuple[float, float] = None,
            inf_stream=False,
            remove_sim_ticks=True,
            track_fps=False,
            save_animation=False,
            anim_fps=50,
            show=True,
    ):
        assert bool(map_figsize) != bool(sim_figsize), 'Use one Plot instance for either map or sim plotting, not both.'
        self.map = bool(map_figsize)
        self.sim = bool(sim_figsize)

        self.path = path  # shape is 3 x n
        self.horizon = horizon if horizon != np.inf else None
        self.goal = goal
        self.inf_stream = inf_stream
        self.save_fig_dir = save_fig_dir

        if len(obstacles) and type(obstacles[0]) is dict:
            # we have a list of obstacles
            obstacles = [obs['vertices'] for obs in obstacles]

        if self.map:
            ##########################################
            # --- initialize the map plot --- #
            self.map_artists = []
            self.sea_currents_subsample = sea_currents_subsample

            # create the figure with appropriate number of subplots
            n_extra_plots = int(bool(nodes_expanded)) + int(sea_currents is not None)
            if n_extra_plots > 0:
                n_subplots = 1 + n_extra_plots
                self.map_fig, axes = plt.subplots(1, n_subplots,
                                                  figsize=(map_figsize[0] * n_subplots / 2, map_figsize[1]),
                                                  sharex='all', sharey='all')
                
                ax_idx = 0
                # assign axes based on what's provided
                if nodes_expanded:
                    self.node_ax = axes[ax_idx]
                    ax_idx += 1
                    # plot the nodes that were expanded
                    self.node_scat = None
                    self.create_node_plot(nodes_expanded)
                    self.map_artists.extend([self.node_scat, self.node_ax.yaxis])
                
                if sea_currents is not None:
                    self.sea_ax = axes[ax_idx]
                    ax_idx += 1
                    # plot the sea current vector field
                    self.sea_quiver = None
                    self.create_sea_currents_plot(sea_currents)
                    self.map_artists.extend([self.sea_quiver, self.sea_ax.yaxis])
                
                self.map_ax = axes[ax_idx]
                # show the ticks for the map plot
                self.map_ax.xaxis.set_tick_params(labelbottom=True)
                self.map_ax.yaxis.set_tick_params(labelleft=True)

            else:
                self.map_fig, self.map_ax = plt.subplots(1, 1, figsize=map_figsize)
            self.map_artists.append(self.map_ax.yaxis)

            # plot the costmap
            if costmap is not None:
                if costmap.sum() > 0:
                    costmap[costmap == np.max(costmap)] = np.nan  # set the max to nan
                    cmap = DEFAULT_COLOR_MAP
                else:
                    cmap = 'Greys'
                if costmap_min_max:
                    self.costmap_image = self.map_ax.imshow(costmap, origin='lower', cmap=cmap,
                                                            vmin=costmap_min_max[0], vmax=costmap_min_max[1])
                else:
                    self.costmap_image = self.map_ax.imshow(costmap, origin='lower', cmap=cmap)
                divider = make_axes_locatable(self.map_ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = self.map_fig.colorbar(self.costmap_image, cax=cax)
                cbar.set_label('cost')
                self.map_artists.append(self.costmap_image)
                self.map_ax.set_title('Costmap')

            if swath is not None:
                # init swath image
                swath_im = np.zeros(swath.shape + (4,))  # init RGBA array
                # fill in the RGB values
                swath_im[:] = colors.to_rgba(SWATH_COLOR)
                swath_im[:, :, 3] = swath  # set pixel transparency to 0 if pixel value is 0
                # plot the full swath
                self.swath_image = self.map_ax.imshow(swath_im, origin='lower', alpha=0.2)
                self.map_artists.append(self.swath_image)

            # add the patches for the ice
            if len(obstacles):
                self.obs_patches = [patches.Polygon(obs, True, fill=False, ec='k', linewidth=0.5) for
                                    obs in obstacles]
                self.obs_patch_collection = self.map_ax.add_collection(
                    PatchCollection(self.obs_patches, match_original=True)
                )
                self.map_artists.append(self.obs_patch_collection)

            # show the path
            if self.path is not None:
                if global_path is not None:
                    self.global_path_line, = self.map_ax.plot(global_path[0], global_path[1], 'c--', label='global')
                    self.path_line, = self.map_ax.plot(self.path[0], self.path[1], PLANNED_PATH_COLOR, label='optimized')
                    self.map_artists.append(self.global_path_line)
                else:
                    self.path_line, = self.map_ax.plot(self.path[0], self.path[1], PLANNED_PATH_COLOR)
                self.map_artists.append(self.path_line)

            # show the ship position
            if ship_pos is not None:
                self.ship_state_line, = self.map_ax.plot(ship_pos[0], ship_pos[1], SHIP_ACTUAL_PATH_COLOR,
                                                         linewidth=1, label='Actual path')
                self.map_artists.append(self.ship_state_line)

            # plot the nodes along the path
            if len(path_nodes) != 0:
                self.nodes_line, = self.map_ax.plot(*path_nodes, 'bx')
                self.map_artists.append(self.nodes_line)

            # plot the goal line segment
            if self.horizon:
                self.goal_line = self.map_ax.axhline(y=self.horizon + self.path[1, 0], color=GOAL_COLOR,
                                                     linestyle='--', linewidth=1.0)
                self.map_artists.append(self.goal_line)
            if self.goal:
                self.goal_line = self.map_ax.axhline(y=self.goal, color=GOAL_COLOR, linestyle='-', linewidth=1.0)
                self.map_artists.append(self.goal_line)

            if legend:
                self.map_ax.legend()

            if ship_vertices is not None:
                assert ship_pos is not None
                if len(np.shape(ship_pos)) > 1:
                    ship_pos = ship_pos[:, 0]

                self.ship_patch = self.map_ax.add_patch(
                    patches.Polygon(ship_vertices @ Rxy(ship_pos[2]).T + ship_pos[:2], True, fill=False, color='m')
                )
                self.map_artists.append(self.ship_patch)

                if turning_radius is not None:
                    x = np.arange(0, 2 * np.pi, 0.01)
                    self.map_ax.plot(
                        (ship_pos[0] - turning_radius * np.sin(ship_pos[2]) + turning_radius * np.cos(x)).tolist(),
                        (ship_pos[1] + turning_radius * np.cos(ship_pos[2]) + turning_radius * np.sin(x)).tolist(), 'g'
                    )
                    self.map_ax.plot(
                        (ship_pos[0] + turning_radius * np.sin(ship_pos[2]) + turning_radius * np.cos(x)).tolist(),
                        (ship_pos[1] - turning_radius * np.cos(ship_pos[2]) + turning_radius * np.sin(x)).tolist(), 'g'
                    )

            self.map_ax.set_aspect('equal')
            if y_axis_limit is not None:
                self.map_ax.axis([0, self.map_ax.get_xlim()[1], 0, y_axis_limit])

            if scale != 1:
                self.scale_axis_labels(self.map_ax, scale)

            self.map_ax.set_xlabel('x (m)')
            self.map_ax.set_ylabel('y (m)')

            if show:
                plt.show(block=False)
                plt.pause(0.1)
            else:
                plt.draw()

            if self.save_fig_dir:
                if not os.path.isdir(self.save_fig_dir):
                    os.makedirs(self.save_fig_dir)
                self.save()

        elif self.sim:
            ##########################################
            # --- initialize the simulation plot --- #

            # follows discussion on faster rendering using blitting from
            # https://matplotlib.org/stable/users/explain/animations/blitting.html
            self.sim_fig, self.sim_ax = plt.subplots(figsize=sim_figsize)
            self._bg = None
            self.sim_artists = []
            self.sea_currents_subsample = sea_currents_subsample

            # keeps track of how far ship has traveled in subsequent steps
            self.prev_ship_pos = ship_pos

            # initialize sea currents background (before obstacles so it's behind)
            # NOTE: Do NOT add to sim_artists - it should stay as static background
            if sea_currents is not None:
                self.sea_quiver = None
                self._create_sim_sea_currents_plot(sea_currents)

            # initialize artist for ice polygons with gradient effect
            if len(obstacles):
                self.obs_patches = []
                for obs in obstacles:
                    self.obs_patches.extend(add_gradient_polygon(self.sim_ax, obs))
            

            # initialize artist for ship
            if ship_vertices is not None:
                self.ship_patch = self.sim_ax.add_patch(
                    patches.Polygon(ship_vertices @ Rxy(ship_pos[2]).T + ship_pos[:2], True, fill=True,
                                    edgecolor='black', facecolor=SHIP_PATCH_COLOR, linewidth=1, alpha=0.8)
                )
                self.add_artist(self.ship_patch)

            # initialize artist for path
            if self.path is not None:
                self.path_line, = self.sim_ax.plot(self.path[0], self.path[1], PLANNED_PATH_COLOR, label='Planned path')
                self.add_artist(self.path_line)

            # initialize artist for ship actual path
            if ship_pos is not None:
                self.ship_state_line, = self.sim_ax.plot(ship_pos[0], ship_pos[1], SHIP_ACTUAL_PATH_COLOR,
                                                         linewidth=1, label='Actual path')
                self.add_artist(self.ship_state_line)

            # initialize artist for target
            if target:
                self.target, = self.sim_ax.plot(*target, 'xm', label='target')
                self.add_artist(self.target)

            # initialize artist for horizon line
            if self.horizon:
                self.horizon_line = self.sim_ax.axhline(y=self.horizon + self.path[1, 0], color=GOAL_COLOR,
                                                        linestyle='--', linewidth=3.0, label='intermediate goal')
                self.add_artist(self.horizon_line)

            # initialize artist for goal line
            if self.goal:
                self.goal_line = self.sim_ax.axhline(y=self.goal, color=GOAL_COLOR, linestyle='-',
                                                     linewidth=3.0, label='final goal')
                self.add_artist(self.goal_line)

            # --- add more artists here --- #

            # initialize artist for title
            self.title_text = self.sim_ax.set_title('')
            self.add_artist(self.title_text)

            # initialize artist for legend
            if legend:
                self.legend_plt = self.sim_ax.legend(loc='upper right')
                self.add_artist(self.legend_plt)

            if remove_sim_ticks:
                # remove axes ticks and labels to speed up animation
                self.sim_ax.set_xlabel('')
                self.sim_ax.set_xticks([])
                self.sim_ax.set_ylabel('')
                self.sim_ax.set_yticks([])

            self.add_artist(self.sim_ax.yaxis)

            # set background color
            self.sim_ax.patch.set_facecolor(OPEN_WATER_COLOR)
            self.sim_ax.patch.set_alpha(0.7)

            # grab the background on every draw
            self.cid = self.sim_fig.canvas.mpl_connect('draw_event', self.on_draw)

            # set the axes limits
            if y_axis_limit is not None:
                if type(y_axis_limit) is tuple:
                    self.sim_ax.set_ylim(y_axis_limit)
                else:
                    self.sim_ax.set_ylim([self.sim_ax.get_ylim()[0], y_axis_limit])

            if map_shape is not None:
                self.sim_ax.set_xlim(0, map_shape[1])
                if y_axis_limit is None:
                    self.sim_ax.set_ylim(0, map_shape[0])
            self.sim_ax.set_aspect('equal')

            self.track_fps = track_fps
            self.anim_fps = anim_fps
            self.save_animation = save_animation

            if self.track_fps and not self.save_animation:
                self.fps_counter = []

            if self.save_animation:
                if self.save_fig_dir is None:
                    self.save_fig_dir = '.'
                elif not os.path.isdir(self.save_fig_dir):
                    os.makedirs(self.save_fig_dir)

                # fps affects the relative speed of video playback, which may be different from the actual simulation speed
                # fps should match the animation fps in the simulation
                self.moviewriter = FFMpegWriter(fps=self.anim_fps)
                self.moviewriter.setup(self.sim_fig, os.path.join(self.save_fig_dir, ANIM_MOVIE_FILE), dpi=200)

                self.sim_fig.canvas.draw()

            elif show:  # cannot show and save at the same time
                plt.show(block=False)
                plt.pause(0.1)

    def update_map(self, cost_map: np.ndarray = None) -> None:
        self.costmap_image.set_data(cost_map)

    def update_path(
            self,
            path: np.ndarray,
            swath: np.ndarray = None,
            path_nodes: Tuple[List, List] = None,
            nodes_expanded: dict = None,
            target: Tuple[float, float] = None,
            ship_state: Tuple[List, List] = None,
            global_path: np.ndarray = None,
    ) -> None:

        if self.map:
            if self.horizon:
                self.goal_line.set_ydata(path[1][0] + self.horizon)

            if global_path is not None:
                self.global_path_line.set_data(global_path[0], global_path[1])
            self.path_line.set_data(path[0], path[1])

            if ship_state is not None:
                self.ship_state_line.set_data(ship_state[0], ship_state[1])

            if nodes_expanded:
                self.create_node_plot(nodes_expanded)

            if path_nodes is not None:
                self.nodes_line.set_data(path_nodes[0], path_nodes[1])

            if swath is not None:
                swath_im = np.zeros(swath.shape + (4,))  # init RGBA array
                # fill in the RGB values
                swath_im[:] = colors.to_rgba(SWATH_COLOR)
                swath_im[:, :, 3] = swath  # set pixel transparency to 0 if pixel value is 0
                # update the swath image
                self.swath_image.set_data(swath_im)

        if self.sim:
            # update goal line segment
            if self.horizon:
                start_y = path[1][0]
                if self.horizon + start_y < self.goal_line.get_ydata()[0]:
                    self.horizon_line.set_ydata(self.horizon + start_y)
                else:
                    self.horizon_line.set_visible(False)

            # update planned path
            self.path_line.set_data(path[0], path[1])

            # update ship actual path
            if ship_state is not None:
                self.ship_state_line.set_data(ship_state[0], ship_state[1])

            # update target
            if target is not None:
                self.target.set_data(*target)

    def update_ship(self, vertices, x, y, psi, move_yaxis_threshold=0) -> None:
        R = Rxy(psi)
        vs = vertices @ R.T + [x, y]
        self.ship_patch.set_xy(vs)

        # update y axis if necessary
        if (
                self.inf_stream and
                y > move_yaxis_threshold and
                y + move_yaxis_threshold < self.goal
        ):
            ymin, ymax = self.sim_ax.get_ylim()
            # compute how much ship has moved in the y direction since last step
            offset = np.array([0, y - self.prev_ship_pos[1]])
            self.sim_ax.set_ylim([ymin + offset[1], ymax + offset[1]])

        self.prev_ship_pos = (x, y)  # update prev ship position

    def update_obstacles(self, obstacles: List = None, patch_fill: str = None) -> None:
        if len(obstacles) and type(obstacles[0]) is dict:
            # we have a list of obstacles
            obstacles = [obs['vertices'] for obs in obstacles]

        if self.sim:
            # Update gradient patches for each obstacle
            n_layers = ICE_GRADIENT_LAYERS
            patches_per_obs = n_layers + 1  # layers + outline
            
            for obs_idx, ob in enumerate(obstacles):
                vertices = np.array(ob)
                centroid = np.mean(vertices, axis=0)
                base_idx = obs_idx * patches_per_obs
                
                # Update clip path for all gradient layers
                clip_patch = patches.Polygon(vertices, closed=True, transform=self.sim_ax.transData)
                
                for i in range(n_layers):
                    scale = 1.0 - (i / n_layers) * 0.8
                    scaled_vertices = centroid + scale * (vertices - centroid)
                    self.obs_patches[base_idx + i].set_xy(scaled_vertices)
                    self.obs_patches[base_idx + i].set_clip_path(clip_patch)
                
                # Update outline
                self.obs_patches[base_idx + n_layers].set_xy(vertices)
        else:
            for i, ob in enumerate(obstacles):
                self.obs_patches[i].set_xy(ob)
            if hasattr(self, 'obs_patch_collection'):
                self.obs_patch_collection.set_paths(self.obs_patches)

        if patch_fill and hasattr(self, 'obs_patch_collection'):
            self.obs_patch_collection.set_facecolor(patch_fill)
    
    def animate_map(self, save_fig_dir=None, suffix=0):
        # draw artists for map plot
        for artist in self.map_artists:
            self.map_ax.draw_artist(artist)
        self.map_fig.canvas.flush_events()
        self.save(save_fig_dir, suffix)

    def animate_sim(self, save_fig_dir=None, suffix=0):
        cv = self.sim_fig.canvas
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            cv.restore_region(self._bg)
            # draw artists
            self._draw_animated()
            # update the gui state
            cv.blit(self.sim_fig.bbox)
        cv.flush_events()
        # self.save(save_fig_dir, suffix)

        if self.save_animation:
            self.moviewriter.grab_frame()

        else:
            plt.pause(0.001)  # this slows down animation... but can fix issue with hanging anim

    # ---- helper functions for fast rendering of animations ---- #
    def _draw_animated(self):
        for a in self.sim_artists:
            self.sim_ax.draw_artist(a)

    def add_artist(self, art):
        if art.figure != self.sim_fig:
            raise RuntimeError
        art.set_animated(True)
        self.sim_artists.append(art)

    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        cv = self.sim_fig.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        try:
            self._bg = cv.copy_from_bbox(cv.figure.bbox)
        except AttributeError:
            pass  # this seems to happen when trying to save a pdf
        self._draw_animated()
    # ----------------------------------------------------------- #

    def close(self):
        if self.sim and self.save_animation:
            self.moviewriter.finish()

        if self.sim:
            plt.close(self.sim_fig)
        else:
            plt.close(self.map_fig)

    def update_fps(self):
        if self.save_animation:
            return self.anim_fps

        self.fps_counter.append(time.time())
        fps = 0
        if len(self.fps_counter) > 50:
            fps = len(self.fps_counter) / (self.fps_counter[-1] - self.fps_counter[0])
            self.fps_counter.pop(0)

        return fps

    def save(self, save_fig_dir=None, suffix=0, im_format=DEFAULT_SAVE_FIG_FORMAT):
        if save_fig_dir is None:
            save_fig_dir = self.save_fig_dir
        if save_fig_dir is not None:
            if not os.path.isdir(save_fig_dir):
                os.makedirs(save_fig_dir)
            if '.' not in str(suffix):
                suffix = str(suffix) + '.' + im_format
            fp = os.path.join(save_fig_dir, suffix)  # pdf is useful in inkscape
            if self.sim:
                self.sim_fig.savefig(fp, dpi=200)
            else:
                self.map_fig.savefig(fp, dpi=200)
            return fp

    def create_node_plot(self, nodes_expanded: dict):
        c, data = self.aggregate_nodes(nodes_expanded)
        if self.node_scat is None:
            self.node_scat = self.node_ax.scatter(data[:, 0], data[:, 1], s=2, c=c, cmap=DEFAULT_COLOR_MAP)
            self.node_ax.set_title('Expanded node plot (total {})'.format(len(nodes_expanded)))
            self.node_ax.set_aspect('equal')
            divider = make_axes_locatable(self.node_ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = self.map_fig.colorbar(self.node_scat, cax=cax)
            cbar.locator = MaxNLocator(integer=True)
            cbar.update_ticks()
            cbar.set_label('Number of Headings')
        else:
            # set x and y data
            self.node_scat.set_offsets(data)
            # set colors
            self.node_scat.set_array(np.array(c))
            # update title
            self.node_ax.set_title('Expanded node plot (total {})'.format(len(nodes_expanded)))

    def create_sea_currents_plot(self, sea_currents: np.ndarray):
        """
        Create a quiver plot showing sea current vector field.
        
        Args:
            sea_currents: numpy array with shape (H, W, 2) or (T, H, W, 2)
                         where the last dimension contains (u, v) velocity components
        """
        # Handle different input shapes
        if sea_currents.ndim == 4:
            # (T, H, W, 2) - take the first time step
            field = sea_currents[0]
        elif sea_currents.ndim == 3:
            # (H, W, 2)
            field = sea_currents
        else:
            raise ValueError(f"Expected sea_currents shape (H, W, 2) or (T, H, W, 2), got {sea_currents.shape}")
        
        H, W, _ = field.shape
        
        # Subsample for visualization (vector fields are hard to see at full resolution)
        step = self.sea_currents_subsample
        Y, X = np.mgrid[0:H:step, 0:W:step]
        U = field[::step, ::step, 0]  # u component
        V = field[::step, ::step, 1]  # v component
        
        # Compute magnitude for coloring
        magnitude = np.sqrt(U**2 + V**2)
        
        if self.sea_quiver is None:
            self.sea_quiver = self.sea_ax.quiver(
                X, Y, U, V, magnitude,
                cmap=DEFAULT_COLOR_MAP,
                scale=None,  # auto-scale
                alpha=0.8
            )
            self.sea_ax.set_title('Sea Current Vector Field')
            self.sea_ax.set_aspect('equal')
            divider = make_axes_locatable(self.sea_ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = self.map_fig.colorbar(self.sea_quiver, cax=cax)
            cbar.set_label('Current magnitude')
        else:
            # Update quiver plot
            self.sea_quiver.set_UVC(U, V, magnitude)

    def update_sea_currents(self, sea_currents: np.ndarray):
        """Update the sea current vector field plot."""
        if hasattr(self, 'sea_quiver') and self.sea_quiver is not None:
            if self.map:
                self.create_sea_currents_plot(sea_currents)
            elif self.sim:
                self._create_sim_sea_currents_plot(sea_currents)

    def _create_sim_sea_currents_plot(self, sea_currents: np.ndarray):
        """
        Create a quiver plot showing sea current vector field for simulation plot.
        
        Args:
            sea_currents: numpy array with shape (H, W, 2) or (T, H, W, 2)
                         where the last dimension contains (u, v) velocity components
        """
        # Handle different input shapes
        if sea_currents.ndim == 4:
            # (T, H, W, 2) - take the first time step
            field = sea_currents[0]
        elif sea_currents.ndim == 3:
            # (H, W, 2)
            field = sea_currents
        else:
            raise ValueError(f"Expected sea_currents shape (H, W, 2) or (T, H, W, 2), got {sea_currents.shape}")
        
        H, W, _ = field.shape
        
        # Subsample for visualization
        step = self.sea_currents_subsample
        Y, X = np.mgrid[0:H:step, 0:W:step]
        U = field[::step, ::step, 0]  # u component
        V = field[::step, ::step, 1]  # v component
        
        # Compute magnitude for coloring
        magnitude = np.sqrt(U**2 + V**2)
        
        if self.sea_quiver is None:
            self.sea_quiver = self.sim_ax.quiver(
                X, Y, U, V, magnitude,
                cmap=DEFAULT_COLOR_MAP,
                scale=None,  # auto-scale
                alpha=0.5,   # more transparent for sim background
                zorder=-10   # behind ice floes
            )
        else:
            # Update quiver plot
            self.sea_quiver.set_UVC(U, V, magnitude)

    @staticmethod
    def aggregate_nodes(nodes_expanded):
        c = {(k[0], k[1]): 0 for k in nodes_expanded}
        xy = c.copy()
        for k, val in nodes_expanded.items():
            key = (k[0], k[1])
            c[key] += 1
            if not xy[key]:
                x, y, _ = val
                xy[key] = [x, y]
        c = list(c.values())
        data = np.asarray(list(xy.values()))
        return c, data

    @staticmethod
    def show_prims(ax, x, y, psi, prim_paths):
        R = Rxy_3d(psi)
        for path in prim_paths:
            if type(prim_paths) is dict:
                path = prim_paths[path]
            xs, ys, _ = R @ path
            ax.plot([i + x for i in xs],
                    [j + y for j in ys], 'b-', linewidth=0.2)

    @staticmethod
    def show_prims_from_nodes_edges(ax, prim, nodes, edges):
        for n, e in zip(nodes[:-1], edges):
            paths = [prim.paths[(e[0], k)] for k in prim.edge_set_dict[e[0]]]
            Plot.show_prims(ax, n[0], n[1], n[2] - e[0][2] * prim.spacing, paths)

    @staticmethod
    def add_ship_patch(ax, vertices, x, y, psi, ec='black', fc=SHIP_PATCH_COLOR):
        R = Rxy(psi)
        ax.add_patch(
            patches.Polygon(vertices @ R.T + [x, y], True, fill=True, edgecolor=ec, facecolor=fc, alpha=0.5)
        )

    @staticmethod
    def scale_axis_labels(ax, scale):
        # divide axis labels by scale
        # thank you, https://stackoverflow.com/a/27575514/13937378
        def numfmt(x, pos):
            s = '{}'.format(x / scale)
            return s

        yfmt = tkr.FuncFormatter(numfmt)  # create your custom formatter function

        ax.yaxis.set_major_formatter(yfmt)
        ax.xaxis.set_major_formatter(yfmt)
