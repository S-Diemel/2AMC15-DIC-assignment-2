from pathlib import Path

from world.environment import WarehouseEnv


GRID_CONFIGS_FP = Path(__file__).parents[1].resolve() / Path("grid_configs")
GRID_CONFIGS_FP.mkdir(parents=True, exist_ok=True)

__all__ = ["GRID_CONFIGS_FP", "WarehouseEnv"]

