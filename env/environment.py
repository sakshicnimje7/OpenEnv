"""
Main environment implementation following OpenEnv specification.

Implements the warehouse logistics simulator with step-based interaction.
"""

from typing import Tuple, Dict, Any, Optional, List
from copy import deepcopy
import json

from env.models import (
    Order, Action, ActionType, Observation, Reward, WarehouseLocation,
    OrderStatus, TaskConfig
)
from env.tasks import get_task
from env.grader import get_grader
from env.utils import (
    validate_address, check_warehouse_stock, deduct_stock,
    calculate_routing_score, format_action_reason
)


class WarehouseLogisticsEnvironment:
    """
    E-Commerce Logistics AI Environment (OpenEnv compliant).
    
    Simulates a warehouse management system where an AI agent processes orders,
    validates addresses, checks inventory, and routes shipments.
    """
    
    def __init__(self, task_difficulty: str = "easy"):
        """
        Initialize environment.
        
        Args:
            task_difficulty: Task difficulty level (easy, medium, hard)
        """
        self.task_difficulty = task_difficulty
        self.orders: List[Order] = []
        self.warehouses: List[WarehouseLocation] = []
        self.task_config: Optional[TaskConfig] = None
        self.step_count: int = 0
        self.max_steps: int = 100
        self.action_history: List[Dict[str, Any]] = []
        self.action_results: List[Dict[str, Any]] = []
        self.done: bool = False
        self.episode_reward: float = 0.0
        
    def reset(self) -> Observation:
        """
        Reset environment to initial state.
        
        Returns:
            Initial observation
        """
        # Load task
        self.orders, self.warehouses, self.task_config = get_task(self.task_difficulty)
        
        # Reset state
        self.step_count = 0
        self.action_history = []
        self.action_results = []
        self.done = False
        self.episode_reward = 0.0
        
        return self.state()
    
    def state(self) -> Observation:
        """
        Get current observation/state.
        
        Returns:
            Current observation
        """
        processed_count = sum(
            1 for o in self.orders
            if o.status in [OrderStatus.SHIPPED, OrderStatus.ALLOCATED]
        )
        failed_count = sum(
            1 for o in self.orders
            if o.status == OrderStatus.FAILED
        )
        
        last_result = self.action_results[-1] if self.action_results else None
        
        return Observation(
            orders=deepcopy(self.orders),
            warehouses=deepcopy(self.warehouses),
            step_count=self.step_count,
            processed_orders=processed_count,
            failed_orders=failed_count,
            last_action_result=last_result
        )
    
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one step of the environment.
        
        OpenEnv specification: step(action) -> (observation, reward, done, info)
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        self.step_count += 1
        
        # Initialize reward
        reward = Reward(value=0.0, breakdown={}, reason="")
        
        # Check if environment is done
        if self.done or self.step_count >= self.max_steps:
            self.done = True
            final_reward = self._compute_final_reward()
            info = self._get_step_info(action, reward, done=True)
            return self.state(), final_reward, True, info
        
        # Validate action
        if not self._is_valid_action(action):
            reward.value = -0.1
            reward.reason = "Invalid action"
            reward.breakdown['invalid'] = -0.1
            info = self._get_step_info(action, reward)
            self.action_results.append(reward.dict())
            self.episode_reward += reward.value
            return self.state(), reward, False, info
        
        # Execute action
        reward = self._execute_action(action)
        
        # Check if task is complete
        if self._is_task_complete():
            self.done = True
        
        # Build info dict
        info = self._get_step_info(action, reward)
        
        # Update tracking
        self.action_results.append(reward.dict())
        self.episode_reward += reward.value
        
        return self.state(), reward, self.done, info
    
    def _is_valid_action(self, action: Action) -> bool:
        """
        Validate if action is well-formed and possible.
        
        Args:
            action: Action to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check if order exists
        order_ids = [o.order_id for o in self.orders]
        if action.order_id not in order_ids:
            return False
        
        # Check if warehouse exists (if needed)
        if action.warehouse:
            warehouse_ids = [w.warehouse_id for w in self.warehouses]
            if action.warehouse not in warehouse_ids:
                return False
        
        return True
    
    def _execute_action(self, action: Action) -> Reward:
        """
        Execute action and compute reward.
        
        Args:
            action: Action to execute
            
        Returns:
            Reward for this action
        """
        order = next(o for o in self.orders if o.order_id == action.order_id)
        
        if action.action_type == ActionType.CHECK_STOCK:
            return self._action_check_stock(order, action)
        elif action.action_type == ActionType.VALIDATE_ADDRESS:
            return self._action_validate_address(order, action)
        elif action.action_type == ActionType.ALLOCATE:
            return self._action_allocate(order, action)
        elif action.action_type == ActionType.SHIP:
            return self._action_ship(order, action)
        elif action.action_type == ActionType.REROUTE:
            return self._action_reroute(order, action)
        else:
            return Reward(
                value=-0.1,
                breakdown={'unknown_action': -0.1},
                reason=f"Unknown action: {action.action_type}"
            )
    
    def _action_check_stock(self, order: Order, action: Action) -> Reward:
        """
        Check stock for order.
        
        Args:
            order: Order to check
            action: Action with warehouse
            
        Returns:
            Reward
        """
        if not action.warehouse:
            # Check all warehouses
            available_wh = None
            for wh in self.warehouses:
                has_stock, _ = check_warehouse_stock(wh, order.sku, order.quantity)
                if has_stock:
                    available_wh = wh
                    break
            
            if available_wh:
                order.status = OrderStatus.STOCK_CHECKED
                order.allocated_warehouse = available_wh.warehouse_id
                reward_value = 0.2
                reason = f"Stock found in {available_wh.warehouse_id}"
            else:
                order.status = OrderStatus.FAILED
                reward_value = -0.2
                reason = "No stock available in any warehouse"
        else:
            warehouse = next(w for w in self.warehouses if w.warehouse_id == action.warehouse)
            has_stock, info = check_warehouse_stock(warehouse, order.sku, order.quantity)
            
            if has_stock:
                order.status = OrderStatus.STOCK_CHECKED
                order.allocated_warehouse = warehouse.warehouse_id
                reward_value = 0.2
                reason = f"Stock confirmed in {warehouse.warehouse_id}"
            else:
                reward_value = -0.1
                reason = f"Insufficient stock in {warehouse.warehouse_id}"
        
        # Track action
        self.action_history.append({
            'action_type': 'CHECK_STOCK',
            'order_id': order.order_id,
            'warehouse': action.warehouse,
        })
        
        return Reward(
            value=reward_value,
            breakdown={'check_stock': reward_value},
            reason=reason
        )
    
    def _action_validate_address(self, order: Order, action: Action) -> Reward:
        """
        Validate delivery address.
        
        Args:
            order: Order to validate
            action: Action
            
        Returns:
            Reward
        """
        is_valid, reason = validate_address(order.address)
        
        if is_valid:
            order.status = OrderStatus.ADDRESS_VALIDATED
            reward_value = 0.15
        else:
            order.status = OrderStatus.FAILED
            reward_value = -0.15
        
        # Track action
        self.action_history.append({
            'action_type': 'VALIDATE_ADDRESS',
            'order_id': order.order_id,
            'valid': is_valid,
        })
        
        return Reward(
            value=reward_value,
            breakdown={'validate_address': reward_value},
            reason=f"Address validation: {reason}"
        )
    
    def _action_allocate(self, order: Order, action: Action) -> Reward:
        """
        Allocate order to warehouse.
        
        Args:
            order: Order to allocate
            action: Action with warehouse
            
        Returns:
            Reward
        """
        if not action.warehouse:
            return Reward(
                value=-0.1,
                breakdown={'allocate_error': -0.1},
                reason="Warehouse required for allocation"
            )
        
        # Check if can allocate
        if order.status == OrderStatus.FAILED:
            return Reward(
                value=-0.1,
                breakdown={'failed_order': -0.1},
                reason="Cannot allocate failed order"
            )
        
        warehouse = next(w for w in self.warehouses if w.warehouse_id == action.warehouse)
        has_stock, _ = check_warehouse_stock(warehouse, order.sku, order.quantity)
        
        if not has_stock:
            return Reward(
                value=-0.15,
                breakdown={'insufficient_stock': -0.15},
                reason="Insufficient stock for allocation"
            )
        
        # Perform allocation
        if deduct_stock(warehouse, order.sku, order.quantity):
            order.status = OrderStatus.ALLOCATED
            order.allocated_warehouse = warehouse.warehouse_id
            reward_value = 0.25
            reason = f"Order allocated from {warehouse.warehouse_id}"
        else:
            reward_value = -0.2
            reason = "Allocation failed"
        
        # Track action
        self.action_history.append({
            'action_type': 'ALLOCATE',
            'order_id': order.order_id,
            'warehouse': action.warehouse,
        })
        
        return Reward(
            value=reward_value,
            breakdown={'allocate': reward_value},
            reason=reason
        )
    
    def _action_ship(self, order: Order, action: Action) -> Reward:
        """
        Ship order.
        
        Args:
            order: Order to ship
            action: Action
            
        Returns:
            Reward
        """
        if order.status != OrderStatus.ALLOCATED:
            return Reward(
                value=-0.1,
                breakdown={'not_allocated': -0.1},
                reason="Order must be allocated before shipping"
            )
        
        order.status = OrderStatus.SHIPPED
        reward_value = 0.3
        reason = f"Order shipped from {order.allocated_warehouse}"
        
        # Track action
        self.action_history.append({
            'action_type': 'SHIP',
            'order_id': order.order_id,
        })
        
        return Reward(
            value=reward_value,
            breakdown={'ship': reward_value},
            reason=reason
        )
    
    def _action_reroute(self, order: Order, action: Action) -> Reward:
        """
        Reroute order to different warehouse.
        
        Args:
            order: Order to reroute
            action: Action with new warehouse
            
        Returns:
            Reward
        """
        if not action.warehouse:
            return Reward(
                value=-0.1,
                breakdown={'reroute_error': -0.1},
                reason="Warehouse required for reroute"
            )
        
        if order.status == OrderStatus.SHIPPED:
            return Reward(
                value=-0.2,
                breakdown={'already_shipped': -0.2},
                reason="Cannot reroute shipped order"
            )
        
        warehouse = next(w for w in self.warehouses if w.warehouse_id == action.warehouse)
        has_stock, _ = check_warehouse_stock(warehouse, order.sku, order.quantity)
        
        if not has_stock:
            return Reward(
                value=-0.2,
                breakdown={'no_stock_at_reroute': -0.2},
                reason=f"No stock at reroute warehouse {action.warehouse}"
            )
        
        # Restore stock to previous warehouse if allocated
        if order.allocated_warehouse and order.status == OrderStatus.ALLOCATED:
            for wh in self.warehouses:
                if wh.warehouse_id == order.allocated_warehouse:
                    wh.stock_level[order.sku] = wh.stock_level.get(order.sku, 0) + order.quantity
                    break
        
        # Perform new allocation
        if deduct_stock(warehouse, order.sku, order.quantity):
            order.allocated_warehouse = warehouse.warehouse_id
            order.status = OrderStatus.ALLOCATED
            reward_value = 0.2
            reason = f"Order rerouted to {action.warehouse}"
        else:
            reward_value = -0.15
            reason = "Reroute failed"
        
        # Track action
        self.action_history.append({
            'action_type': 'REROUTE',
            'order_id': order.order_id,
            'old_warehouse': order.allocated_warehouse,
            'new_warehouse': action.warehouse,
        })
        
        return Reward(
            value=reward_value,
            breakdown={'reroute': reward_value},
            reason=reason
        )
    
    def _is_task_complete(self) -> bool:
        """
        Check if task is complete.
        
        All non-failed orders must be shipped for completion.
        
        Returns:
            True if task complete
        """
        if not self.orders:
            return False
        
        for order in self.orders:
            if order.status not in [OrderStatus.SHIPPED, OrderStatus.FAILED]:
                return False
        
        return True
    
    def _compute_final_reward(self) -> Reward:
        """
        Compute final reward based on task grader.
        
        Returns:
            Final reward
        """
        grader = get_grader(self.task_difficulty)
        score = grader.grade(
            self.orders,
            self.warehouses,
            self.action_history,
            self.step_count
        )
        
        return Reward(
            value=score,
            breakdown={'final_score': score},
            reason=f"Final task score: {score:.2f}"
        )
    
    def _get_step_info(
        self,
        action: Action,
        reward: Reward,
        done: bool = False
    ) -> Dict[str, Any]:
        """
        Build info dictionary for step.
        
        Args:
            action: Executed action
            reward: Reward received
            done: Whether episode is done
            
        Returns:
            Info dictionary
        """
        return {
            'step': self.step_count,
            'action_type': action.action_type.value,
            'reward': reward.value,
            'done': done,
            'episode_reward': self.episode_reward,
            'task_difficulty': self.task_difficulty,
        }
    
    def get_task_info(self) -> Dict[str, Any]:
        """
        Get current task information.
        
        Returns:
            Task information
        """
        if not self.task_config:
            return {}
        
        return {
            'task_id': self.task_config.task_id,
            'difficulty': self.task_config.difficulty,
            'order_count': self.task_config.order_count,
            'warehouse_count': self.task_config.warehouse_count,
            'description': self.task_config.description,
        }
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """
        Get episode summary.
        
        Returns:
            Episode summary
        """
        completed_orders = sum(1 for o in self.orders if o.status == OrderStatus.SHIPPED)
        failed_orders = sum(1 for o in self.orders if o.status == OrderStatus.FAILED)
        
        return {
            'task_difficulty': self.task_difficulty,
            'total_steps': self.step_count,
            'total_orders': len(self.orders),
            'completed_orders': completed_orders,
            'failed_orders': failed_orders,
            'episode_reward': self.episode_reward,
            'actions_taken': len(self.action_history),
            'final_warehouse_states': [
                {
                    'warehouse_id': w.warehouse_id,
                    'stock_level': dict(w.stock_level)
                }
                for w in self.warehouses
            ]
        }
