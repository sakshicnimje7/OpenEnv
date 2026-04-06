"""
Task graders for evaluating agent performance.

Each grader returns a score between 0.0 and 1.0.
"""

from typing import Dict, List, Any
from env.models import Order, OrderStatus, WarehouseLocation


class TaskGrader:
    """Base class for task graders."""
    
    def grade(
        self,
        orders: List[Order],
        warehouses: List[WarehouseLocation],
        action_history: List[Dict[str, Any]],
        step_count: int
    ) -> float:
        """
        Grade task performance.
        
        Args:
            orders: Final list of orders
            warehouses: Final warehouse states
            action_history: History of actions taken
            step_count: Total steps taken
            
        Returns:
            Score between 0.0 and 1.0
        """
        raise NotImplementedError


class EasyTaskGrader(TaskGrader):
    """Grader for easy task: single SKU stock check."""
    
    def grade(
        self,
        orders: List[Order],
        warehouses: List[WarehouseLocation],
        action_history: List[Dict[str, Any]],
        step_count: int
    ) -> float:
        """
        Grade easy task.
        
        Target: agent checks stock and correctly identifies availability.
        
        Scoring:
        - Order status STOCK_CHECKED: +0.5
        - Order status SHIPPED: +1.0
        - Efficiency bonus: -(step_count - 2) * 0.05
        
        Args:
            orders: Final list of orders
            warehouses: Final warehouse states
            action_history: History of actions taken
            step_count: Total steps taken
            
        Returns:
            Score between 0.0 and 1.0
        """
        score = 0.0
        
        if not orders:
            return 0.0
        
        order = orders[0]
        
        # Check if stock was validated
        check_stock_actions = [
            a for a in action_history
            if a.get('action_type') == 'CHECK_STOCK'
        ]
        
        if check_stock_actions:
            score += 0.5
        
        # Check if order was shipped
        if order.status == OrderStatus.SHIPPED:
            score += 0.5
        elif order.status == OrderStatus.ALLOCATED:
            score += 0.3
        elif order.status == OrderStatus.STOCK_CHECKED:
            score += 0.2
        
        # Efficiency bonus (prefer fewer steps, but at least 2)
        efficiency_penalty = max(0, (step_count - 2) * 0.05)
        score = max(0.0, score - efficiency_penalty)
        
        return min(1.0, score)


class MediumTaskGrader(TaskGrader):
    """Grader for medium task: batch processing with address validation."""
    
    def grade(
        self,
        orders: List[Order],
        warehouses: List[WarehouseLocation],
        action_history: List[Dict[str, Any]],
        step_count: int
    ) -> float:
        """
        Grade medium task.
        
        Target: agent processes batch of orders with address validation.
        
        Scoring:
        - Each ADDRESS_VALIDATED action: +0.1
        - Each ALLOCATED order: +0.15
        - Correctly rejected invalid address: +0.1
        - Penalty for processing too slowly: -(step_count - 20) * 0.01
        
        Args:
            orders: Final list of orders
            warehouses: Final warehouse states
            action_history: History of actions taken
            step_count: Total steps taken
            
        Returns:
            Score between 0.0 and 1.0
        """
        score = 0.0
        
        if not orders:
            return 0.0
        
        # Count address validations
        validate_actions = [
            a for a in action_history
            if a.get('action_type') == 'VALIDATE_ADDRESS'
        ]
        score += len(validate_actions) * 0.1
        
        # Count allocated orders
        allocated_orders = sum(
            1 for o in orders
            if o.status == OrderStatus.ALLOCATED
        )
        score += allocated_orders * 0.15
        
        # Count shipped orders
        shipped_orders = sum(
            1 for o in orders
            if o.status == OrderStatus.SHIPPED
        )
        score += shipped_orders * 0.2
        
        # Efficiency penalty
        efficiency_penalty = max(0, (step_count - 20) * 0.01)
        score = max(0.0, score - efficiency_penalty)
        
        return min(1.0, score)


class HardTaskGrader(TaskGrader):
    """Grader for hard task: multi-warehouse routing with stock constraints."""
    
    def grade(
        self,
        orders: List[Order],
        warehouses: List[WarehouseLocation],
        action_history: List[Dict[str, Any]],
        step_count: int
    ) -> float:
        """
        Grade hard task.
        
        Target: agent handles out-of-stock scenarios with rerouting.
        
        Scoring:
        - Each REROUTE action: +0.15
        - Each SHIPPED order: +0.2
        - Correct allocation from alternate warehouse: +0.25
        - Penalty for failed orders: -0.1 per failed order
        - Efficiency bonus for quick resolution: -(step_count - 30) * 0.005
        
        Args:
            orders: Final list of orders
            warehouses: Final warehouse states
            action_history: History of actions taken
            step_count: Total steps taken
            
        Returns:
            Score between 0.0 and 1.0
        """
        score = 0.0
        
        if not orders:
            return 0.0
        
        # Count reroute actions
        reroute_actions = [
            a for a in action_history
            if a.get('action_type') == 'REROUTE'
        ]
        score += len(reroute_actions) * 0.15
        
        # Count shipped orders (main goal)
        shipped_orders = sum(
            1 for o in orders
            if o.status == OrderStatus.SHIPPED
        )
        score += shipped_orders * 0.2
        
        # Bonus for correct multi-warehouse allocation
        allocated_from_alt = sum(
            1 for o in orders
            if o.status in [OrderStatus.ALLOCATED, OrderStatus.SHIPPED]
            and o.allocated_warehouse is not None
        )
        score += min(allocated_from_alt * 0.25, 0.5)  # Cap at 0.5
        
        # Penalty for failed orders
        failed_orders = sum(
            1 for o in orders
            if o.status == OrderStatus.FAILED
        )
        score -= failed_orders * 0.1
        
        # Efficiency bonus
        efficiency_bonus = max(0, (50 - step_count) * 0.01)
        score += efficiency_bonus
        
        return max(0.0, min(1.0, score))


def get_grader(task_difficulty: str) -> TaskGrader:
    """
    Get appropriate grader for task difficulty.
    
    Args:
        task_difficulty: Task difficulty level
        
    Returns:
        TaskGrader instance
        
    Raises:
        ValueError: If difficulty not recognized
    """
    graders = {
        'easy': EasyTaskGrader(),
        'medium': MediumTaskGrader(),
        'hard': HardTaskGrader(),
    }
    
    if task_difficulty not in graders:
        raise ValueError(f"Unknown difficulty: {task_difficulty}")
    
    return graders[task_difficulty]
