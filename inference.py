"""
Baseline inference script using OpenAI-compatible chat API.

Executes the warehouse logistics environment with an AI agent powered by an
OpenAI-compatible endpoint (for example, Hugging Face Inference API).
Follows exact logging format: [START], [STEP], [END]
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from env import Action, ActionType, WarehouseLogisticsEnvironment

# Load environment variables from .env
load_dotenv()


def _debug(message: str) -> None:
    """Write diagnostic logs to stderr so stdout stays parser-safe."""
    if os.getenv("ENABLE_DEBUG_LOGS", "false").lower() == "true":
        print(message, file=sys.stderr, flush=True)


class WarehouseAgentInference:
    """AI agent for warehouse logistics using an OpenAI-compatible client."""

    def __init__(
        self,
        task_difficulty: str = "easy",
        model_name: Optional[str] = None,
        api_base_url: Optional[str] = None,
        model_temperature: Optional[float] = None,
        request_timeout_seconds: Optional[float] = None,
        max_steps: int = 50,
        max_api_failures: int = 5,
        non_progress_threshold: int = 3,
    ) -> None:
        """
        Initialize inference agent.

        Args:
            task_difficulty: Task difficulty (easy, medium, hard)
            model_name: Model name override
            api_base_url: API base URL override
            model_temperature: Sampling temperature override
            request_timeout_seconds: API timeout override (seconds)
            max_steps: Maximum steps in one episode
            max_api_failures: Number of API failures before switching to local policy
            non_progress_threshold: Number of consecutive non-progress steps before
                forcing local policy
        """
        self.task_difficulty = task_difficulty
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-3.5-turbo")
        self.api_base_url = api_base_url or os.getenv("API_BASE_URL", "")
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN", "")

        temperature_raw = model_temperature
        if temperature_raw is None:
            temperature_raw = os.getenv("MODEL_TEMPERATURE", "0.0")
        try:
            self.model_temperature = max(0.0, min(2.0, float(temperature_raw)))
        except (TypeError, ValueError):
            self.model_temperature = 0.0

        timeout_raw = request_timeout_seconds
        if timeout_raw is None:
            timeout_raw = os.getenv("API_TIMEOUT_SECONDS", "30")
        try:
            self.request_timeout_seconds = max(1.0, float(timeout_raw))
        except (TypeError, ValueError):
            self.request_timeout_seconds = 30.0

        self.max_steps = max_steps
        self.max_api_failures = max_api_failures
        self.non_progress_threshold = non_progress_threshold

        # Keep OpenAI client usage intact and compatible with custom base URLs.
        client_kwargs: Dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.request_timeout_seconds,
        }
        if self.api_base_url:
            client_kwargs["base_url"] = self.api_base_url
        self.client = OpenAI(**client_kwargs)

        self.env = WarehouseLogisticsEnvironment(task_difficulty=task_difficulty)

        # Runtime tracking
        self.history: List[Dict[str, Any]] = []
        self.total_reward = 0.0
        self.action_count = 0
        self.api_failure_count = 0
        self.disable_api = False
        self.non_progress_steps = 0
        self.last_progress_marker: Optional[tuple[int, int]] = None

    def log_start(self, task: str, env_name: str, model: str) -> None:
        """Log episode start in required structured format."""
        message = (
            f"[START]\n"
            f"Task: {task}\n"
            f"Env: {env_name}\n"
            f"Model: {model}\n"
        )
        print(message, flush=True)

    def log_step(
        self,
        step_num: int,
        action: str,
        reward: float,
        done: bool,
        error: Optional[str],
    ) -> None:
        """Log step details in required structured format."""
        error_text = error if error is not None else ""
        message = (
            f"[STEP]\n"
            f"Step: {step_num}\n"
            f"Action: {action}\n"
            f"Reward: {reward:.3f}\n"
            f"Done: {done}\n"
            f"Error: {error_text}\n"
        )
        print(message, flush=True)

    def log_end(self, success: bool, steps: int, score: float, rewards: List[float]) -> None:
        """Log episode end in required structured format."""
        message = (
            f"[END]\n"
            f"Success: {success}\n"
            f"Steps: {steps}\n"
            f"Score: {score:.4f}\n"
            f"Rewards: {rewards}\n"
        )
        print(message, flush=True)

    def _build_prompt(self, observation: Dict[str, Any], step_num: int) -> str:
        """
        Build prompt for model inference.

        Args:
            observation: Current environment observation
            step_num: Current step number

        Returns:
            Prompt text
        """
        orders_summary = "\n".join(
            [
                (
                    f"- Order {o['order_id']}: {o['sku']} x{o['quantity']}, "
                    f"status={o['status']}, address={o['address'][:30]}..."
                )
                for o in observation["orders"][:5]
            ]
        )

        warehouses_summary = "\n".join(
            [
                (
                    f"- {w['warehouse_id']}: {w['city']}, "
                    f"stock={sum(w['stock_level'].values())} units"
                )
                for w in observation["warehouses"]
            ]
        )

        return f"""
Step {step_num}: Warehouse Logistics Task

Current Orders ({len(observation['orders'])}):
{orders_summary}

Available Warehouses:
{warehouses_summary}

Processed: {observation['processed_orders']}/{len(observation['orders'])}
Failed: {observation['failed_orders']}

Choose one next best action and respond with JSON only:
{{
  "action_type": "check_stock|validate_address|allocate|ship|reroute",
  "order_id": "ORD-XXX",
  "warehouse": "WAREHOUSE-ID (optional)",
  "reason": "brief explanation"
}}
""".strip()

    def _parse_action_response(self, response_text: str, observation: Dict[str, Any]) -> Action:
        """
        Parse model response into Action.

        Args:
            response_text: Text content from model output
            observation: Observation for safe fallback values

        Returns:
            Parsed Action object
        """
        try:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                payload = json.loads(response_text[start:end])
                action_raw = str(payload.get("action_type", "check_stock")).strip().lower()
                # Some models return combined choices like "check_stock|validate_address".
                if "|" in action_raw:
                    action_raw = action_raw.split("|")[0].strip()
                if action_raw not in {item.value for item in ActionType}:
                    action_raw = "check_stock"
                return Action(
                    action_type=ActionType(action_raw),
                    order_id=payload.get("order_id", observation["orders"][0]["order_id"]),
                    warehouse=payload.get("warehouse"),
                    message=payload.get("reason", ""),
                )
        except Exception as parse_err:
            _debug(f"[DEBUG] Failed to parse model response: {parse_err}")
            _debug(f"[DEBUG] Raw model content: {response_text}")

        return self._get_local_policy_action(observation)

    def _find_actionable_order(self, observation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Find next order that is not finished.

        Args:
            observation: Current observation

        Returns:
            Order dictionary or None
        """
        for order in observation["orders"]:
            if order["status"] not in ["shipped", "failed"]:
                return order
        return None

    def _find_warehouse_with_stock(self, observation: Dict[str, Any], sku: str, quantity: int) -> Optional[str]:
        """
        Select first warehouse with enough stock.

        Args:
            observation: Current observation
            sku: SKU to check
            quantity: Required quantity

        Returns:
            Warehouse ID or None
        """
        for wh in observation["warehouses"]:
            if wh["stock_level"].get(sku, 0) >= quantity:
                return wh["warehouse_id"]
        return None

    def _get_local_policy_action(self, observation: Dict[str, Any]) -> Action:
        """
        Deterministic fallback policy to avoid infinite no-op loops.

        Args:
            observation: Current observation

        Returns:
            Action chosen by local policy
        """
        order = self._find_actionable_order(observation)
        if not order:
            # Episode should end soon; send safe action on first order if present.
            fallback_order_id = observation["orders"][0]["order_id"] if observation["orders"] else "ORD-001"
            return Action(
                action_type=ActionType.CHECK_STOCK,
                order_id=fallback_order_id,
                warehouse=None,
                message="No actionable order found",
            )

        status = order["status"]
        order_id = order["order_id"]
        sku = order["sku"]
        quantity = order["quantity"]

        if status == "allocated":
            return Action(
                action_type=ActionType.SHIP,
                order_id=order_id,
                warehouse=None,
                message="Local policy: ship allocated order",
            )

        if status == "stock_checked" or status == "address_validated":
            wh = order.get("allocated_warehouse") or self._find_warehouse_with_stock(
                observation, sku, quantity
            )
            if wh:
                return Action(
                    action_type=ActionType.ALLOCATE,
                    order_id=order_id,
                    warehouse=wh,
                    message="Local policy: allocate from available warehouse",
                )
            return Action(
                action_type=ActionType.CHECK_STOCK,
                order_id=order_id,
                warehouse=None,
                message="Local policy: re-check stock",
            )

        if status == "pending":
            # Medium tasks emphasize address validation first.
            if self.task_difficulty == "medium":
                return Action(
                    action_type=ActionType.VALIDATE_ADDRESS,
                    order_id=order_id,
                    warehouse=None,
                    message="Local policy: validate address first for medium task",
                )
            return Action(
                action_type=ActionType.CHECK_STOCK,
                order_id=order_id,
                warehouse=None,
                message="Local policy: check stock first",
            )

        return Action(
            action_type=ActionType.CHECK_STOCK,
            order_id=order_id,
            warehouse=None,
            message="Local policy: default action",
        )

    def _get_next_action(self, observation: Dict[str, Any], step_num: int) -> Action:
        """
        Get next action from API, with safe fallback.

        Args:
            observation: Current observation
            step_num: Step number

        Returns:
            Next action
        """
        if self.disable_api:
            return self._get_local_policy_action(observation)

        prompt = self._build_prompt(observation, step_num)
        messages = [
            {"role": "system", "content": "You are a warehouse logistics AI agent."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.model_temperature,
            )
            response_text = response.choices[0].message.content
            if not response_text:
                raise ValueError("Empty response content")
            return self._parse_action_response(response_text, observation)
        except Exception as api_err:
            self.api_failure_count += 1
            _debug(
                "[DEBUG] API call failed "
                f"({self.api_failure_count}/{self.max_api_failures}): {api_err}"
            )
            _debug(f"[DEBUG] base_url={self.api_base_url}, model={self.model_name}")

            if self.api_failure_count >= self.max_api_failures:
                self.disable_api = True
                _debug("[DEBUG] API disabled for this episode; switching to local policy.")

            return self._get_local_policy_action(observation)

    def _update_progress_guard(self, observation: Dict[str, Any]) -> None:
        """
        Track progress and switch to local policy when the episode stalls.

        Progress is measured as changes in processed/failed order counts.

        Args:
            observation: Current observation as dict
        """
        marker = (observation["processed_orders"], observation["failed_orders"])

        if self.last_progress_marker is None:
            self.last_progress_marker = marker
            self.non_progress_steps = 0
            return

        if marker == self.last_progress_marker:
            self.non_progress_steps += 1
        else:
            self.non_progress_steps = 0

        self.last_progress_marker = marker

        if not self.disable_api and self.non_progress_steps >= self.non_progress_threshold:
            self.disable_api = True
            _debug(
                "[DEBUG] Non-progress guard triggered "
                f"({self.non_progress_steps} stalled steps). "
                "Switching to local policy."
            )

    def run_episode(self) -> Dict[str, Any]:
        """
        Run one complete episode.

        Returns:
            Episode summary
        """
        observation = self.env.reset()
        task_info = self.env.get_task_info()
        self.log_start(
            task=task_info.get("difficulty", self.task_difficulty),
            env_name="warehouse-logistics-env",
            model=self.model_name,
        )

        self.history = []
        self.total_reward = 0.0
        self.action_count = 0
        self.api_failure_count = 0
        self.disable_api = False
        self.non_progress_steps = 0
        self.last_progress_marker = None

        # Initialize progress marker from reset observation.
        self._update_progress_guard(observation.model_dump())

        rewards: List[float] = []
        done = False
        while not done and self.action_count < self.max_steps:
            obs_dict = observation.model_dump()
            action = self._get_next_action(obs_dict, self.action_count + 1)

            error: Optional[str] = None
            try:
                observation, reward, done, _info = self.env.step(action)
            except Exception as exc:
                error = str(exc)
                done = True
                reward_value = -0.2
                self.log_step(
                    step_num=self.action_count + 1,
                    action=action.action_type.value,
                    reward=reward_value,
                    done=done,
                    error=error,
                )
                rewards.append(reward_value)
                break

            self._update_progress_guard(observation.model_dump())

            self.action_count += 1
            self.total_reward += reward.value
            rewards.append(reward.value)

            self.log_step(
                step_num=self.action_count,
                action=action.action_type.value,
                reward=reward.value,
                done=done,
                error=error,
            )

            self.history.append(
                {
                    "step": self.action_count,
                    "action": action.action_type.value,
                    "reward": reward.value,
                    "processed": observation.processed_orders,
                }
            )

        completed = sum(1 for order in self.env.orders if order.status.value == "shipped")
        total_orders = len(self.env.orders)
        score = completed / total_orders if total_orders > 0 else 0.0
        score = max(0.0, min(1.0, score))
        success = score >= 0.8

        summary = {
            "total_reward": self.total_reward,
            "completed": completed,
            "total": total_orders,
            "steps": self.action_count,
            "task_difficulty": self.task_difficulty,
            "api_failures": self.api_failure_count,
            "rewards": rewards,
            "score": score,
            "success": success,
        }

        self.log_end(
            success=summary["success"],
            steps=summary["steps"],
            score=summary["score"],
            rewards=summary["rewards"],
        )
        return summary


def run_baseline_all_tasks(max_steps: int = 50) -> Dict[str, Any]:
    """Run baseline evaluation across easy/medium/hard tasks."""
    task_levels = ["easy", "medium", "hard"]
    task_results: Dict[str, Dict[str, Any]] = {}
    scores: List[float] = []

    for level in task_levels:
        agent = WarehouseAgentInference(task_difficulty=level, max_steps=max_steps)
        result = agent.run_episode()
        task_results[level] = result
        scores.append(result["score"])

    aggregate = sum(scores) / len(scores) if scores else 0.0
    aggregate = max(0.0, min(1.0, aggregate))

    return {
        "tasks": task_results,
        "aggregate_score": aggregate,
        "all_success": all(r["success"] for r in task_results.values()),
    }


def main() -> int:
    """Main entry point for running baseline inference."""
    task_difficulty = os.getenv("TASK_DIFFICULTY", "easy")

    run_all = os.getenv("RUN_ALL_TASKS", "true").lower() == "true"
    if run_all:
        _ = run_baseline_all_tasks(max_steps=50)
    else:
        agent = WarehouseAgentInference(task_difficulty=task_difficulty, max_steps=50)
        _ = agent.run_episode()

    # Exit 0 when inference completes and scores are produced for all tasks.
    return 0


if __name__ == "__main__":
    sys.exit(main())
