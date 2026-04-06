"""
Quick test/demo script for the warehouse logistics environment.

Run without OpenAI API key to test environment mechanics.
"""

from env import WarehouseLogisticsEnvironment, Action, ActionType


def test_easy_task():
    """Test easy task: single order stock check."""
    print("\n" + "="*60)
    print("TESTING EASY TASK: Single Order Stock Check")
    print("="*60)
    
    env = WarehouseLogisticsEnvironment(task_difficulty="easy")
    obs = env.reset()
    
    print(f"\nTask Info: {env.get_task_info()}")
    print(f"Orders: {len(obs.orders)}")
    print(f"Warehouses: {len(obs.warehouses)}")
    
    # Step 1: Check stock
    print("\n[Step 1] Checking stock...")
    action1 = Action(
        action_type=ActionType.CHECK_STOCK,
        order_id="ORD-001",
        warehouse=None
    )
    obs, reward, done, info = env.step(action1)
    print(f"Reward: {reward.value:.3f} | Reason: {reward.reason}")
    print(f"Order status: {obs.orders[0].status.value}")
    
    if obs.orders[0].allocated_warehouse:
        # Step 2: Allocate
        print("\n[Step 2] Allocating order...")
        action2 = Action(
            action_type=ActionType.ALLOCATE,
            order_id="ORD-001",
            warehouse=obs.orders[0].allocated_warehouse
        )
        obs, reward, done, info = env.step(action2)
        print(f"Reward: {reward.value:.3f} | Reason: {reward.reason}")
        print(f"Order status: {obs.orders[0].status.value}")
        
        # Step 3: Ship
        print("\n[Step 3] Shipping order...")
        action3 = Action(
            action_type=ActionType.SHIP,
            order_id="ORD-001"
        )
        obs, reward, done, info = env.step(action3)
        print(f"Reward: {reward.value:.3f} | Reason: {reward.reason}")
        print(f"Order status: {obs.orders[0].status.value}")
        print(f"Done: {done}")
    
    summary = env.get_episode_summary()
    print(f"\nEpisode Summary:")
    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Total reward: {summary['episode_reward']:.3f}")
    print(f"  Completed orders: {summary['completed_orders']}/{summary['total_orders']}")
    
    return summary


def test_medium_task():
    """Test medium task: batch with address validation."""
    print("\n" + "="*60)
    print("TESTING MEDIUM TASK: Batch Processing with Validation")
    print("="*60)
    
    env = WarehouseLogisticsEnvironment(task_difficulty="medium")
    obs = env.reset()
    
    print(f"\nTask Info: {env.get_task_info()}")
    print(f"Orders: {len(obs.orders)}")
    print(f"Warehouses: {len(obs.warehouses)}")
    
    # Validate addresses for first 3 orders
    for i, order in enumerate(obs.orders[:3]):
        print(f"\n[Step {i+1}] Validating address for {order.order_id}...")
        action = Action(
            action_type=ActionType.VALIDATE_ADDRESS,
            order_id=order.order_id
        )
        obs, reward, done, info = env.step(action)
        print(f"Address: {order.address[:30]}... | Valid: {obs.orders[i].status.value}")
        print(f"Reward: {reward.value:.3f}")
    
    # Try to allocate valid order
    valid_order = next((o for o in obs.orders if o.status.value == "address_validated"), None)
    if valid_order:
        print(f"\n[Step 4] Allocating valid order {valid_order.order_id}...")
        action = Action(
            action_type=ActionType.ALLOCATE,
            order_id=valid_order.order_id,
            warehouse=env.warehouses[0].warehouse_id
        )
        obs, reward, done, info = env.step(action)
        print(f"Reward: {reward.value:.3f}")
        print(f"Order status: {valid_order.status.value}")
    
    summary = env.get_episode_summary()
    print(f"\nEpisode Summary:")
    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Total reward: {summary['episode_reward']:.3f}")
    print(f"  Completed orders: {summary['completed_orders']}/{summary['total_orders']}")
    
    return summary


def test_hard_task():
    """Test hard task: multi-warehouse with rerouting."""
    print("\n" + "="*60)
    print("TESTING HARD TASK: Multi-Warehouse Rerouting")
    print("="*60)
    
    env = WarehouseLogisticsEnvironment(task_difficulty="hard")
    obs = env.reset()
    
    print(f"\nTask Info: {env.get_task_info()}")
    print(f"Orders: {len(obs.orders)}")
    print(f"Warehouses: {len(obs.warehouses)}")
    
    # Check stock at primary warehouse for first order
    print(f"\n[Step 1] Checking stock at {env.warehouses[0].warehouse_id}...")
    action = Action(
        action_type=ActionType.CHECK_STOCK,
        order_id=obs.orders[0].order_id,
        warehouse=env.warehouses[0].warehouse_id
    )
    obs, reward, done, info = env.step(action)
    print(f"Warehouse: {env.warehouses[0].warehouse_id} ({env.warehouses[0].city})")
    print(f"SKU requested: {obs.orders[0].sku} x{obs.orders[0].quantity}")
    print(f"Reward: {reward.value:.3f} | {reward.reason}")
    
    # Try alternate warehouse if needed
    if obs.orders[0].status.value != "stock_checked":
        print(f"\n[Step 2] Trying alternate warehouse {env.warehouses[1].warehouse_id}...")
        action = Action(
            action_type=ActionType.CHECK_STOCK,
            order_id=obs.orders[0].order_id,
            warehouse=env.warehouses[1].warehouse_id
        )
        obs, reward, done, info = env.step(action)
        print(f"Warehouse: {env.warehouses[1].warehouse_id} ({env.warehouses[1].city})")
        print(f"Reward: {reward.value:.3f} | {reward.reason}")
        
        # Allocate from alternate
        if obs.orders[0].status.value == "stock_checked":
            print(f"\n[Step 3] Allocating from {env.warehouses[1].warehouse_id}...")
            action = Action(
                action_type=ActionType.ALLOCATE,
                order_id=obs.orders[0].order_id,
                warehouse=env.warehouses[1].warehouse_id
            )
            obs, reward, done, info = env.step(action)
            print(f"Reward: {reward.value:.3f} | {reward.reason}")
    
    summary = env.get_episode_summary()
    print(f"\nEpisode Summary:")
    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Total reward: {summary['episode_reward']:.3f}")
    print(f"  Completed orders: {summary['completed_orders']}/{summary['total_orders']}")
    
    return summary


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("WAREHOUSE LOGISTICS ENVIRONMENT - DEMO TESTS")
    print("="*60)
    
    results = {}
    
    try:
        results['easy'] = test_easy_task()
    except Exception as e:
        print(f"ERROR in easy task: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        results['medium'] = test_medium_task()
    except Exception as e:
        print(f"ERROR in medium task: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        results['hard'] = test_hard_task()
    except Exception as e:
        print(f"ERROR in hard task: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for difficulty, summary in results.items():
        print(f"\n{difficulty.upper()}:")
        print(f"  Total reward: {summary['episode_reward']:.3f}")
        print(f"  Completed: {summary['completed_orders']}/{summary['total_orders']}")
        print(f"  Steps: {summary['total_steps']}")
    
    print("\n[OK] All tests completed!")


if __name__ == "__main__":
    main()
