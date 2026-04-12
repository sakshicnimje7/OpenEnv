"""
Baseline inference script using the OpenAI client.

The script runs all three OpenEnv tasks and prints only the required stdout
records: [START], [STEP], and [END].
"""

import json
import os
from typing import Any, Dict, List, Optional, Sequence

from dotenv import load_dotenv
from openai import OpenAI

from env import Action, ActionType, WarehouseLogisticsEnvironment

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
BENCHMARK_NAME = "openenv"
TASKS = ("easy", "medium", "hard")
MAX_STEPS = 50


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_reward(value: float) -> str:
    return f"{value:.2f}"


def _format_rewards(values: Sequence[float]) -> str:
    return ",".join(_format_reward(value) for value in values)


def _log_start(task_name: str) -> None:
    print(f"[START] task={task_name} env={BENCHMARK_NAME} model={MODEL_NAME}", flush=True)


def _log_step(step_num: int, action_text: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_text = error if error is not None else "null"
    print(
        f"[STEP] step={step_num} action={action_text} reward={_format_reward(reward)} "
        f"done={_format_bool(done)} error={error_text}",
        flush=True,
    )


def _log_end(success: bool, steps: int, rewards: Sequence[float]) -> None:
    print(
        f"[END] success={_format_bool(success)} steps={steps} rewards={_format_rewards(rewards)}",
        flush=True,
    )


def _serialize_action(action: Action) -> str:
    if action.warehouse is None:
        return f"{action.action_type.value}('{action.order_id}')"
    return f"{action.action_type.value}('{action.order_id}','{action.warehouse}')"


def _get_actionable_order(observation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for order in observation["orders"]:
        if order["status"] not in {"shipped", "failed"}:
            return order
    return None


def _find_available_warehouse(observation: Dict[str, Any], sku: str, quantity: int) -> Optional[str]:
    for candidate in observation["warehouses"]:
        if candidate["stock_level"].get(sku, 0) >= quantity:
            return candidate["warehouse_id"]
    return None


def _build_pending_action(order: Dict[str, Any], task_difficulty: str) -> Action:
    if task_difficulty == "medium":
        return Action(
            action_type=ActionType.VALIDATE_ADDRESS,
            order_id=order["order_id"],
            warehouse=None,
            message="validate address",
        )
    return Action(
        action_type=ActionType.CHECK_STOCK,
        order_id=order["order_id"],
        warehouse=None,
        message="check stock",
    )


def _build_progressed_action(order: Dict[str, Any], observation: Dict[str, Any]) -> Action:
    warehouse = order.get("allocated_warehouse")
    if not warehouse:
        warehouse = _find_available_warehouse(observation, order["sku"], order["quantity"])

    if warehouse:
        return Action(
            action_type=ActionType.ALLOCATE,
            order_id=order["order_id"],
            warehouse=warehouse,
            message="allocate available stock",
        )

    return Action(
        action_type=ActionType.CHECK_STOCK,
        order_id=order["order_id"],
        warehouse=None,
        message="recheck stock",
    )


def _build_local_action(order: Dict[str, Any], observation: Dict[str, Any], task_difficulty: str) -> Action:
    status = order["status"]
    if status == "allocated":
        return Action(
            action_type=ActionType.SHIP,
            order_id=order["order_id"],
            warehouse=None,
            message="ship allocated order",
        )
    if status in {"stock_checked", "address_validated"}:
        return _build_progressed_action(order, observation)
    if status == "pending":
        return _build_pending_action(order, task_difficulty)
    return Action(
        action_type=ActionType.CHECK_STOCK,
        order_id=order["order_id"],
        warehouse=None,
        message="default",
    )


def _local_policy(observation: Dict[str, Any], task_difficulty: str) -> Action:
    order = _get_actionable_order(observation)
    if order is None:
        fallback_order_id = observation["orders"][0]["order_id"] if observation["orders"] else "ORD-001"
        return Action(
            action_type=ActionType.CHECK_STOCK,
            order_id=fallback_order_id,
            warehouse=None,
            message="fallback",
        )
    return _build_local_action(order, observation, task_difficulty)


def _build_prompt(observation: Dict[str, Any], task_difficulty: str) -> str:
    orders_summary = []
    for order in observation["orders"][:5]:
        orders_summary.append(
            f"{order['order_id']}|{order['sku']}|{order['quantity']}|{order['status']}|{order['address']}"
        )
    warehouses_summary = []
    for warehouse in observation["warehouses"]:
        stock_total = sum(warehouse["stock_level"].values())
        warehouses_summary.append(f"{warehouse['warehouse_id']}|{warehouse['city']}|{stock_total}")

    return (
        f"task={task_difficulty}\n"
        f"orders={orders_summary}\n"
        f"warehouses={warehouses_summary}\n"
        f"processed={observation['processed_orders']} failed={observation['failed_orders']} step={observation['step_count']}\n"
        "Return JSON only with keys action_type, order_id, warehouse, reason."
    )


def _parse_model_action(response_text: str, observation: Dict[str, Any], task_difficulty: str) -> Action:
    try:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start >= 0 and end > start:
            payload = json.loads(response_text[start:end])
            action_text = str(payload.get("action_type", "check_stock")).strip().lower()
            if action_text not in {action.value for action in ActionType}:
                action_text = "check_stock"
            order_id = payload.get("order_id") or observation["orders"][0]["order_id"]
            warehouse = payload.get("warehouse") or None
            return Action(
                action_type=ActionType(action_text),
                order_id=order_id,
                warehouse=warehouse,
                message=str(payload.get("reason", "")),
            )
    except Exception:
        pass

    return _local_policy(observation, task_difficulty)


def _model_action(observation: Dict[str, Any], task_difficulty: str) -> Action:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You choose the next OpenEnv logistics action."},
            {"role": "user", "content": _build_prompt(observation, task_difficulty)},
        ],
        temperature=0.0,
    )
    content = response.choices[0].message.content or ""
    if not content.strip():
        return _local_policy(observation, task_difficulty)
    return _parse_model_action(content, observation, task_difficulty)


def run_episode(task_difficulty: str) -> None:
    env = WarehouseLogisticsEnvironment(task_difficulty=task_difficulty)
    observation = env.reset()
    rewards: List[float] = []
    steps_taken = 0
    success = False

    _log_start(task_difficulty)

    try:
        for step_num in range(1, MAX_STEPS + 1):
            if env.done:
                break

            observation_dict = observation.model_dump()
            try:
                action = _model_action(observation_dict, task_difficulty)
            except Exception:
                action = _local_policy(observation_dict, task_difficulty)

            try:
                next_observation, reward, done, _info = env.step(action)
                error = None
            except Exception as exc:
                next_observation = env.state()
                reward = env._compute_final_reward()
                done = True
                error = str(exc)

            rewards.append(reward.value)
            steps_taken = step_num
            _log_step(step_num, _serialize_action(action), reward.value, done, error)

            observation = next_observation
            if done:
                break

        success = bool(rewards) and bool(observation.orders) and all(
            order.status.value in {"shipped", "failed"} for order in observation.orders
        )
    finally:
        try:
            env.close()
        except Exception:
            pass
        _log_end(success=success, steps=steps_taken, rewards=rewards)


def main() -> None:
    for task_difficulty in TASKS:
        run_episode(task_difficulty)


if __name__ == "__main__":
    main()
