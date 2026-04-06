"""
Utility functions for the logistics environment.
"""

import re
from typing import Tuple, Dict
from env.models import Order, WarehouseLocation


def validate_address(address: str) -> Tuple[bool, str]:
    """
    Validate if an address is well-formed.
    
    Args:
        address: Address string to validate
        
    Returns:
        Tuple of (is_valid, reason)
    """
    if not address or len(address) < 5:
        return False, "Address too short"
    
    # Check for required address components
    required_patterns = [
        (r'\d+', 'street number'),
        (r'[A-Za-z]', 'street name'),
    ]
    
    for pattern, component in required_patterns:
        if not re.search(pattern, address):
            return False, f"Missing {component}"
    
    # Check for common invalid patterns
    if '@@' in address or '##' in address:
        return False, "Invalid special characters"
    
    return True, "Valid address"


def check_warehouse_stock(
    warehouse: WarehouseLocation,
    sku: str,
    quantity: int
) -> Tuple[bool, Dict]:
    """
    Check if warehouse has stock for a SKU.
    
    Args:
        warehouse: Warehouse location object
        sku: Product SKU
        quantity: Quantity required
        
    Returns:
        Tuple of (has_stock, info_dict)
    """
    available = warehouse.stock_level.get(sku, 0)
    has_stock = available >= quantity
    
    return has_stock, {
        'sku': sku,
        'required': quantity,
        'available': available,
        'warehouse_id': warehouse.warehouse_id
    }


def calculate_routing_score(
    warehouse_city: str,
    delivery_address: str
) -> float:
    """
    Calculate a simple routing score (lower is better).
    
    Args:
        warehouse_city: Warehouse city location
        delivery_address: Delivery address
        
    Returns:
        Routing score (0.0-1.0)
    """
    # Simple heuristic: same city = 0.1, different = 0.9
    address_lower = delivery_address.lower()
    city_lower = warehouse_city.lower()
    
    if city_lower in address_lower:
        return 0.1  # Local delivery
    
    return 0.9  # Cross-city delivery


def deduct_stock(
    warehouse: WarehouseLocation,
    sku: str,
    quantity: int
) -> bool:
    """
    Deduct stock from warehouse inventory.
    
    Args:
        warehouse: Warehouse location object
        sku: Product SKU
        quantity: Quantity to deduct
        
    Returns:
        True if successful, False if insufficient stock
    """
    current = warehouse.stock_level.get(sku, 0)
    if current < quantity:
        return False
    
    warehouse.stock_level[sku] = current - quantity
    return True


def format_action_reason(action_type: str, order_id: str, details: str) -> str:
    """
    Format action reason message.
    
    Args:
        action_type: Type of action
        order_id: Order ID
        details: Additional details
        
    Returns:
        Formatted message
    """
    return f"{action_type} for order {order_id}: {details}"
