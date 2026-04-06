"""
Task definitions for the logistics environment.

Contains easy, medium, and hard tasks with deterministic setup.
"""

from typing import List, Dict, Any
from env.models import (
    Order, WarehouseLocation, TaskConfig, OrderStatus
)


class TaskFactory:
    """Factory for creating tasks."""
    
    @staticmethod
    def create_easy_task() -> tuple[List[Order], List[WarehouseLocation], TaskConfig]:
        """
        Create easy task: single SKU stock check.
        
        Scenario:
        - 1 order for 10 units of SKU-001
        - 2 warehouses: NYC (has stock), LA (no stock)
        - Agent must check stock and allocate from NYC
        
        Returns:
            Tuple of (orders, warehouses, config)
        """
        # Create warehouses
        nyc_warehouse = WarehouseLocation(
            warehouse_id="NYC-01",
            city="New York",
            stock_level={"SKU-001": 50, "SKU-002": 30},
            capacity=1000
        )
        
        la_warehouse = WarehouseLocation(
            warehouse_id="LA-01",
            city="Los Angeles",
            stock_level={"SKU-001": 0, "SKU-002": 20},
            capacity=1000
        )
        
        # Create orders
        order = Order(
            order_id="ORD-001",
            sku="SKU-001",
            quantity=10,
            address="123 Main St, New York, NY",
            status=OrderStatus.PENDING
        )
        
        # Task config
        config = TaskConfig(
            task_id="easy_001",
            difficulty="easy",
            order_count=1,
            warehouse_count=2,
            description="Single order stock check from available warehouse"
        )
        
        return [order], [nyc_warehouse, la_warehouse], config
    
    @staticmethod
    def create_medium_task() -> tuple[List[Order], List[WarehouseLocation], TaskConfig]:
        """
        Create medium task: batch processing with address validation.
        
        Scenario:
        - 5 orders with mixed valid/invalid addresses
        - 3 warehouses with varying stock levels
        - Agent must validate addresses and allocate
        
        Returns:
            Tuple of (orders, warehouses, config)
        """
        # Create warehouses
        warehouses = [
            WarehouseLocation(
                warehouse_id="EAST-01",
                city="New York",
                stock_level={
                    "SKU-001": 100,
                    "SKU-002": 50,
                    "SKU-003": 30
                },
                capacity=1000
            ),
            WarehouseLocation(
                warehouse_id="CENTRAL-01",
                city="Chicago",
                stock_level={
                    "SKU-001": 80,
                    "SKU-002": 40,
                    "SKU-003": 60
                },
                capacity=1000
            ),
            WarehouseLocation(
                warehouse_id="WEST-01",
                city="San Francisco",
                stock_level={
                    "SKU-001": 60,
                    "SKU-002": 80,
                    "SKU-003": 40
                },
                capacity=1000
            ),
        ]
        
        # Create orders with mixed valid/invalid addresses
        orders = [
            Order(
                order_id="ORD-001",
                sku="SKU-001",
                quantity=5,
                address="100 Broadway, New York, NY 10001",
                status=OrderStatus.PENDING
            ),
            Order(
                order_id="ORD-002",
                sku="SKU-002",
                quantity=3,
                address="@@INVALID@@",  # Invalid address
                status=OrderStatus.PENDING
            ),
            Order(
                order_id="ORD-003",
                sku="SKU-003",
                quantity=8,
                address="456 Michigan Ave, Chicago, IL 60611",
                status=OrderStatus.PENDING
            ),
            Order(
                order_id="ORD-004",
                sku="SKU-001",
                quantity=4,
                address="789 Market St, San Francisco, CA 94102",
                status=OrderStatus.PENDING
            ),
            Order(
                order_id="ORD-005",
                sku="SKU-002",
                quantity=6,
                address="XYZ",  # Too short, invalid
                status=OrderStatus.PENDING
            ),
        ]
        
        config = TaskConfig(
            task_id="medium_001",
            difficulty="medium",
            order_count=5,
            warehouse_count=3,
            description="Batch order processing with address validation"
        )
        
        return orders, warehouses, config
    
    @staticmethod
    def create_hard_task() -> tuple[List[Order], List[WarehouseLocation], TaskConfig]:
        """
        Create hard task: multi-warehouse routing with stock constraints.
        
        Scenario:
        - 4 orders with varying SKUs
        - First order out-of-stock at primary warehouse
        - Agent must detect and reroute to alternate warehouse
        - Complex routing decisions required
        
        Returns:
            Tuple of (orders, warehouses, config)
        """
        # Create warehouses with limited stock
        warehouses = [
            WarehouseLocation(
                warehouse_id="HQ-01",
                city="Atlanta",
                stock_level={
                    "SKU-PREMIUM": 5,  # Limited stock
                    "SKU-STANDARD": 100,
                    "SKU-BUDGET": 50
                },
                capacity=500
            ),
            WarehouseLocation(
                warehouse_id="BACKUP-01",
                city="Dallas",
                stock_level={
                    "SKU-PREMIUM": 20,  # Overflow capacity
                    "SKU-STANDARD": 30,
                    "SKU-BUDGET": 80
                },
                capacity=500
            ),
            WarehouseLocation(
                warehouse_id="REGIONAL-01",
                city="Miami",
                stock_level={
                    "SKU-PREMIUM": 10,
                    "SKU-STANDARD": 50,
                    "SKU-BUDGET": 40
                },
                capacity=500
            ),
        ]
        
        # Create orders with one triggering out-of-stock
        orders = [
            Order(
                order_id="ORD-101",
                sku="SKU-PREMIUM",
                quantity=10,  # More than HQ-01 has
                address="1000 Peachtree St, Atlanta, GA 30309",
                status=OrderStatus.PENDING
            ),
            Order(
                order_id="ORD-102",
                sku="SKU-STANDARD",
                quantity=25,
                address="2000 Main St, Houston, TX 77002",
                status=OrderStatus.PENDING
            ),
            Order(
                order_id="ORD-103",
                sku="SKU-BUDGET",
                quantity=15,
                address="3000 Biscayne Blvd, Miami, FL 33132",
                status=OrderStatus.PENDING
            ),
            Order(
                order_id="ORD-104",
                sku="SKU-STANDARD",
                quantity=50,
                address="4000 Airport Blvd, Dallas, TX 75235",
                status=OrderStatus.PENDING
            ),
        ]
        
        config = TaskConfig(
            task_id="hard_001",
            difficulty="hard",
            order_count=4,
            warehouse_count=3,
            description="Multi-warehouse routing with out-of-stock handling and rerouting"
        )
        
        return orders, warehouses, config


def get_task(
    difficulty: str
) -> tuple[List[Order], List[WarehouseLocation], TaskConfig]:
    """
    Get task by difficulty level.
    
    Args:
        difficulty: Task difficulty level (easy, medium, hard)
        
    Returns:
        Tuple of (orders, warehouses, config)
        
    Raises:
        ValueError: If difficulty not recognized
    """
    tasks = {
        'easy': TaskFactory.create_easy_task,
        'medium': TaskFactory.create_medium_task,
        'hard': TaskFactory.create_hard_task,
    }
    
    if difficulty not in tasks:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    
    return tasks[difficulty]()
