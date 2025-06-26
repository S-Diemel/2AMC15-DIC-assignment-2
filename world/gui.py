import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Patch, Polygon
from matplotlib.lines import Line2D
from world.utils.action_mapping import orientation_to_directions


def render_gui(self, mode="human", show_full_legend=True, show_difficulty_region=False):
    """This function renders the GUI of the environement allowing us to visually inspect the agents behaviour."""
    # Return nothing if GUI is turned off
    if self.no_gui:
        return
    
    # Initializing the environement.
    plt.clf()
    ax = plt.gca()
    ax.set_xlim(0, self.width)
    ax.set_ylim(0, self.height)
    ax.set_aspect('equal', adjustable='box')

    # Turn of the ticks and number on the axes, because they are not relevant for the GUI
    ax.set_xticks([])
    ax.set_yticks([])

    # Give room for a side legend
    fig = ax.get_figure()
    if show_full_legend:
        fig.subplots_adjust(left=0.2, right=0.8)
    else:
        fig.subplots_adjust(right=0.8)

    # Draw Areas
    # Item Spawn (Yellow)
    for (xmin, ymin, xmax, ymax) in self.item_spawn:
        ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color="#fdeeac", alpha=0.75))

    # Delivery Zones (Grey)
    for (xmin, ymin, xmax, ymax) in self.delivery_zones:
            # Clip zones to be within warehouse boundaries for rendering
        draw_xmin = max(xmin, 0)
        draw_ymin = max(ymin, 0)
        draw_xmax = min(xmax, self.width)
        draw_ymax = min(ymax, self.height)
        if draw_xmax > draw_xmin and draw_ymax > draw_ymin:
                ax.add_patch(Rectangle((draw_xmin, draw_ymin), draw_xmax - draw_xmin, draw_ymax - draw_ymin, color="#eeeded"))

    # Charger (Green)
    xmin, ymin, xmax, ymax = self.charger
    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color="#81fd8b", alpha=0.75))

    # Racks (Blue)
    for (xmin, ymin, xmax, ymax) in self.racks:
        ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color="#7881ff"))

    # Extra obstacles (Dark grey)
    for (xmin, ymin, xmax, ymax) in self.extra_obstacles:
        ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color="#636363"))

    # Draw Delivery Points --> with numbers and correct dynamics
    for i, point in enumerate(self.delivery_points):
        if self.carrying == i:  # Make delivery point more clearly visible when carrying the item for that delivery point
            ax.add_patch(Circle(point, self.delivery_radius, color='darkred', alpha=0.85))
        else:
            ax.add_patch(Circle(point, self.delivery_radius, color='darkred', alpha=0.3))
        ax.text(
            point[0], point[1],       # x, y
            str(i),                   # number itself
            color='white',            # text color
            ha='center', va='center', 
            fontsize=7,              
            fontweight='bold',
            zorder=10                 # make sure it overlays the delivery point patch
        )

    # Draw Items (Packages)
    for i, point in enumerate(self.items):
        if not self.delivered[i] or self.carrying == i:
            ax.add_patch(Circle(point, self.item_radius, color="orange"))
            ax.text(
                point[0], point[1],       # x, y
                str(i),                   # number itself
                color='white',            # text color
                ha='center', va='center', 
                fontsize=7,              
                fontweight='bold',
                zorder=10                 # make sure it overlays the delivery point patch
            )

    # Potentially show difficulty region for sampling with curriculum learning      
    if show_difficulty_region:
        if self.difficulty_region is not None:
            xmin, ymin, xmax, ymax = self.difficulty_region
            ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color="green", alpha=0.2))

    # Vision triangle
    tri_coords = np.array(self.vision_triangle)
    vision_tri = Polygon(
        tri_coords,
        closed=True,
        facecolor="yellow",
        edgecolor=None,
        alpha=0.4,
    )
    ax.add_patch(vision_tri)

    # Draw Agent
    if self.carrying > -1:  # Give orange edge to agent when carrying an item
        ax.add_patch(Circle(self.agent_pos, self.agent_radius, facecolor="#00A800", edgecolor="orange", linewidth=2))
    else:
        ax.add_patch(Circle(self.agent_pos, self.agent_radius, color="#00A800"))
    
    # Compute location of white dot expressing agent's orientation 
    # For 45 degree orientations, use unit circle to ensure white circle is on agent 
    dir_vec = np.array(orientation_to_directions(self.orientation), dtype=float)
    dir_unit = dir_vec / np.linalg.norm(dir_vec)
    dot_offset = self.agent_radius * 0.75 
    dot_pos = self.agent_pos + dir_unit * dot_offset
    ax.add_patch(Circle(dot_pos, self.agent_radius * 0.15, color="white"))

    # Add a legend to the GUI for better understanding
    legend_stats = [
        Line2D([0], [0], linestyle="None", label=f"Battery: {self.battery:.1f}%"),
        Line2D([0], [0], linestyle="None", label=f"Total Steps: {self.total_nr_steps:.0f}"),
        Line2D([0], [0], linestyle="None", label=f"Cumulative Reward: {self.cumulative_reward:.0f}"),
        Line2D([0], [0], linestyle="None", label=f"Total Collisions: {self.total_nr_collisions:.0f}")
    ]
    legend1 = ax.legend(
        handles=legend_stats,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        title="Episode statistics:",               
        title_fontsize="medium",      
        borderaxespad=0,
        handlelength=0,  
        handletextpad=0  
    )
    ax.add_artist(legend1)

    if show_full_legend:
        legend_env_info = [
            Line2D([0], [0],
                marker='o', markersize=10,
                markerfacecolor="#00A800",
                markeredgecolor="orange" if self.carrying > -1 else "#00A800",
                linestyle="None",
                label="Agent"
            ),
            Patch(facecolor="#fdeeac", edgecolor="none", alpha=0.75, label="Item Spawn"),
            Patch(facecolor="#eeeded", edgecolor="none", label="Delivery Zone"),
            Patch(facecolor="#81fd8b", edgecolor="none", alpha=0.75, label="Charger"),
            Patch(facecolor="#7881ff", edgecolor="none", label="Storage racks"),
            Patch(facecolor="darkred", edgecolor="none", alpha=0.3, label="Delivery Point"),
            Patch(facecolor="orange", edgecolor="none", label="Item"),
        ]
        if len(self.extra_obstacles) > 0:
            legend_env_info += [Patch(facecolor="#636363", edgecolor="none", label="Extra Obstacle")]
        legend2 = ax.legend(
            handles=legend_env_info,
            loc="upper right",
            bbox_to_anchor=(-0.02, 1),   
            title="Environment info:",               
            title_fontsize="medium",       
            frameon=True,
            borderaxespad=0,
            labelspacing=0.5
        )
        ax.add_artist(legend2)

    plt.title("Warehouse Simulation")
    plt.pause(1 / self.metadata["render_fps"])
    if mode == "rgb_array":
        return np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8).reshape(
            plt.gcf().canvas.get_width_height()[::-1] + (3,)
        )
