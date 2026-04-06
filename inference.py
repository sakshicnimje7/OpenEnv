"""
Baseline inference script using OpenAI client.

Executes the warehouse logistics environment with an AI agent powered by OpenAI LLM.
Follows exact logging format: [START], [STEP], [END]
"""

import asyncio
import json
import os
import sys
from typing import Optional
from dotenv import load_dotenv
from openai import AsyncOpenAI

from env import WarehouseLogisticsEnvironment, Action, ActionType

# Load environment variables
load_dotenv()


class WarehouseAgentInference:
    """AI agent for warehouse logistics using OpenAI."""
    
    def __init__(
        self,
        task_difficulty: str = "easy",
        model_name: Optional[str] = None,
        api_base_url: Optional[str] = None,
        hf_token: Optional[str] = None,
        max_steps: int = 50,
    ):
        """
        Initialize inference agent.
        
        Args:
            task_difficulty: Task difficulty (easy, medium, hard)
            model_name: Model name (default: gpt-3.5-turbo)
            api_base_url: API base URL (default: OpenAI official)
            hf_token: Hugging Face token (for compatibility)
            max_steps: Maximum steps per episode
        """
        self.task_difficulty = task_difficulty
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-3.5-turbo")
        self.api_base_url = api_base_url or os.getenv("API_BASE_URL")
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.max_steps = max_steps
        
        # Initialize OpenAI client
        if self.api_base_url:
            self.client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY", "default-key"),
                base_url=self.api_base_url
            )
        else:
            self.client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
        
        # Initialize environment
        self.env = WarehouseLogisticsEnvironment(task_difficulty=task_difficulty)
        
        # Tracking
        self.history = []
        self.total_reward = 0.0
        self.action_count = 0
    
    def log_start(self, task_info: dict) -> None:
        """
        Log episode start.
        
        Args:
            task_info: Task information
        """
        message = f"[START]\nTask: {task_info['difficulty']}\nOrders: {task_info['order_count']}\nWarehouses: {task_info['warehouse_count']}\n"
        print(message)
    
    def log_step(
        self,
        step_num: int,
        action: Action,
        reward: float,
        observation_summary: dict
    ) -> None:
        """
        Log step execution.
        
        Args:
            step_num: Step number
            action: Action taken
            reward: Reward received
            observation_summary: Summary of observation
        """
        message = f"[STEP]\nStep: {step_num}\nAction: {action.action_type.value}\nReward: {reward:.3f}\nProcessed: {observation_summary['processed']}\n"
        print(message)
    
    def log_end(self, summary: dict) -> None:
        """
        Log episode end.
        
        Args:
            summary: Episode summary
        """
        message = f"[END]\nTotal Reward: {summary['total_reward']:.3f}\nCompleted: {summary['completed']}/{summary['total']}\nSteps: {summary['steps']}\n"
        print(message)
    
    async def _get_next_action(
        self,
        observation: dict,
        step_num: int
    ) -> Action:
        """
        Get next action from agent using OpenAI.
        
        Args:
            observation: Current observation
            step_num: Current step number
            
        Returns:
            Action to execute
        """
        # Build prompt
        prompt = self._build_prompt(observation, step_num)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a warehouse logistics AI agent. Provide actions in JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=200,
            )
            
            response_text = response.choices[0].message.content
            action = self._parse_action_response(response_text)
            return action
        
        except Exception as e:
            # Fallback action
            print(f"Warning: LLM call failed ({e}), using fallback action")
            return self._get_fallback_action(observation)
    
    def _build_prompt(self, observation: dict, step_num: int) -> str:
        """
        Build prompt for LLM.
        
        Args:
            observation: Current observation
            step_num: Current step number
            
        Returns:
            Prompt string
        """
        orders_summary = "\n".join([
            f"- Order {o['order_id']}: {o['sku']} x{o['quantity']}, status={o['status']}, address={o['address'][:30]}..."
            for o in observation['orders'][:3]
        ])
        
        warehouses_summary = "\n".join([
            f"- {w['warehouse_id']}: {w['city']}, stock={sum(w['stock_level'].values())} units"
            for w in observation['warehouses']
        ])
        
        prompt = f"""
Step {step_num}: Warehouse Logistics Task

Current Orders ({len(observation['orders'])}):
{orders_summary}

Available Warehouses:
{warehouses_summary}

Processed: {observation['processed_orders']}/{len(observation['orders'])}
Failed: {observation['failed_orders']}

Choose an action. Respond with JSON:
{{
    "action_type": "check_stock|validate_address|allocate|ship|reroute",
    "order_id": "ORD-XXX",
    "warehouse": "WAREHOUSE-ID (optional)",
    "reason": "brief explanation"
}}
"""
        return prompt
    
    def _parse_action_response(self, response_text: str) -> Action:
        """
        Parse LLM response to Action.
        
        Args:
            response_text: LLM response text
            
        Returns:
            Parsed Action
        """
        try:
            # Extract JSON
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response_text[start:end]
                data = json.loads(json_str)
                
                return Action(
                    action_type=ActionType(data.get('action_type', 'check_stock')),
                    order_id=data.get('order_id', 'ORD-001'),
                    warehouse=data.get('warehouse'),
                    message=data.get('reason', '')
                )
        except Exception as e:
            print(f"Parse error: {e}")
        
        return self._get_fallback_action(None)
    
    def _get_fallback_action(self, observation: Optional[dict]) -> Action:
        """
        Get fallback action when parsing fails.
        
        Args:
            observation: Current observation or None
            
        Returns:
            Fallback Action
        """
        if observation and observation['orders']:
            order_id = observation['orders'][0]['order_id']
        else:
            order_id = "ORD-001"
        
        return Action(
            action_type=ActionType.CHECK_STOCK,
            order_id=order_id,
            warehouse=None,
            message="Fallback action"
        )
    
    async def run_episode(self) -> dict:
        """
        Run one complete episode.
        
        Returns:
            Episode summary
        """
        # Reset environment
        observation = self.env.reset()
        task_info = self.env.get_task_info()
        
        # Log start
        self.log_start(task_info)
        
        self.history = []
        self.total_reward = 0.0
        self.action_count = 0
        done = False
        
        while not done and self.action_count < self.max_steps:
            # Get action from agent
            obs_dict = observation.model_dump()
            action = await self._get_next_action(obs_dict, self.action_count + 1)
            
            # Execute action
            observation, reward, done, info = self.env.step(action)
            
            self.action_count += 1
            self.total_reward += reward.value
            
            # Log step
            obs_summary = {
                'processed': observation.processed_orders,
                'total': len(observation.orders)
            }
            self.log_step(self.action_count, action, reward.value, obs_summary)
            
            # Track history
            self.history.append({
                'step': self.action_count,
                'action': action.action_type.value,
                'reward': reward.value,
                'processed': observation.processed_orders,
            })
        
        # Get episode summary
        episode_summary = self.env.get_episode_summary()
        completed = sum(1 for o in self.env.orders if o.status.value == 'shipped')
        
        summary = {
            'total_reward': self.total_reward,
            'completed': completed,
            'total': len(self.env.orders),
            'steps': self.action_count,
            'task_difficulty': self.task_difficulty,
        }
        
        # Log end
        self.log_end(summary)
        
        return {**summary, 'episode_details': episode_summary}


async def main():
    """Main entry point for inference."""
    # Parse arguments
    task_difficulty = os.getenv("TASK_DIFFICULTY", "easy")
    
    print(f"Starting warehouse logistics inference...")
    print(f"Task difficulty: {task_difficulty}")
    print(f"Model: {os.getenv('MODEL_NAME', 'gpt-3.5-turbo')}")
    print()
    
    # Create agent
    agent = WarehouseAgentInference(
        task_difficulty=task_difficulty,
        max_steps=50
    )
    
    # Run episode
    result = await agent.run_episode()
    
    # Print final results
    print(f"\n=== INFERENCE COMPLETE ===")
    print(f"Final Reward: {result['total_reward']:.3f}")
    print(f"Orders Completed: {result['completed']}/{result['total']}")
    print(f"Total Steps: {result['steps']}")
    
    return result


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result['completed'] > 0 else 1)
