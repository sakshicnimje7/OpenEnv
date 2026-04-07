"""Minimal server entry point for OpenEnv multi-mode deployment checks."""

from env import WarehouseLogisticsEnvironment


def main() -> None:
    """Start a lightweight OpenEnv-compatible local runner."""
    env = WarehouseLogisticsEnvironment(task_difficulty="easy")
    obs = env.reset()
    print("OpenEnv server bootstrap successful")
    print(f"Initial orders: {len(obs.orders)}")


if __name__ == "__main__":
    main()
