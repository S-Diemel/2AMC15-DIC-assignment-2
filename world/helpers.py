from pathlib import Path
from warnings import warn

def action_to_values(action):
    values = {
        0: (1, 0),   # go
        1: (0, 0),  # stop
        2: (0, -1),  # Left 45
        3: (0, 1),   # Right 45
        4: (0, 0), # pick-up
    }
    return values[action]

def orientation_to_directions(orientation):
    directions = {
        0: (0, -1),       # Up
        45: (1, -1),      # Up-Right
        90: (1, 0),       # Right
        135: (1, 1),      # Down-Right
        180: (0, 1),      # Down
        225: (-1, 1),     # Down-Left
        270: (-1, 0),     # Left
        315: (-1, -1),    # Up-Left
    }
    return directions[orientation]


def save_results(file_name, world_stats, path_image, show_images):
    out_dir = Path("results/")
    if not out_dir.exists():
        warn("Evaluation output directory does not exist. Creating the "
             "directory.")
        out_dir.mkdir(parents=True, exist_ok=True)

    # Print evaluation results
    print("Evaluation complete. Results:")
    # Text file
    out_fp = out_dir / f"{file_name}.txt"
    with open(out_fp, "w") as f:
        for key, value in world_stats.items():
            f.write(f"{key}: {value}\n")
            print(f"{key}: {value}")
    
    # Image file
    out_fp = out_dir / f"{file_name}.png"
    path_image.save(out_fp)
    if show_images:
        path_image.show(f"Path Frequency")