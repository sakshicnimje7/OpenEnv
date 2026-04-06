"""
Pydantic models for the E-Commerce Logistics AI Environment.

Defines core data structures for observations, actions, and rewards.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from enum import Enum


class OrderStatus(str, Enum):
    """Order status enumeration."""
    PENDING = "pending"
    STOCK_CHECKED = "stock_checked"
    ADDRESS_VALIDATED = "address_validated"
    ALLOCATED = "allocated"
    SHIPPED = "shipped"
    FAILED = "failed"


class WarehouseLocation(BaseModel):
    """Warehouse location model."""
    warehouse_id: str = Field(..., description="Unique warehouse identifier")
    city: str = Field(..., description="Warehouse city location")
    stock_level: Dict[str, int] = Field(
        default_factory=dict,
        description="Current stock levels by SKU"
    )
    capacity: int = Field(default=1000, description="Warehouse capacity")

    class Config:
        frozen = False


class Order(BaseModel):
    """Order model."""
    order_id: str = Field(..., description="Unique order identifier")
    sku: str = Field(..., description="Product SKU")
    quantity: int = Field(..., description="Quantity required")
    address: str = Field(..., description="Delivery address")
    status: OrderStatus = Field(
        default=OrderStatus.PENDING,
        description="Current order status"
    )
    allocated_warehouse: Optional[str] = Field(
        default=None,
        description="Warehouse from which order is allocated"
    )

    class Config:
        frozen = False


class ActionType(str, Enum):
    """Action type enumeration."""
    CHECK_STOCK = "check_stock"
    VALIDATE_ADDRESS = "validate_address"
    ALLOCATE = "allocate"
    SHIP = "ship"
    REROUTE = "reroute"


class Action(BaseModel):
    """Action model for agent decisions."""
    action_type: ActionType = Field(..., description="Type of action to perform")
    order_id: str = Field(..., description="Target order ID")
    warehouse: Optional[str] = Field(
        default=None,
        description="Target warehouse for allocation/reroute"
    )
    message: Optional[str] = Field(
        default="",
        description="Optional reasoning for the action"
    )

    class Config:
        frozen = True


class Observation(BaseModel):
    """Observation model returned to agent."""
    orders: List[Order] = Field(..., description="Current orders in system")
    warehouses: List[WarehouseLocation] = Field(
        ...,
        description="Warehouse information"
    )
    step_count: int = Field(..., description="Current step number")
    processed_orders: int = Field(
        ...,
        description="Number of successfully processed orders"
    )
    failed_orders: int = Field(
        ...,
        description="Number of failed orders"
    )
    last_action_result: Optional[Dict] = Field(
        default=None,
        description="Result of last action"
    )

    class Config:
        frozen = False


class Reward(BaseModel):
    """Reward model."""
    value: float = Field(..., description="Reward value")
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Reward component breakdown"
    )
    reason: str = Field(..., description="Reason for reward")

    class Config:
        frozen = False


class TaskConfig(BaseModel):
    """Task configuration model."""
    task_id: str = Field(..., description="Unique task ID")
    difficulty: str = Field(..., description="Task difficulty level")
    order_count: int = Field(..., description="Number of orders in task")
    warehouse_count: int = Field(..., description="Number of warehouses")
    description: str = Field(..., description="Task description")

    class Config:
        frozen = False
