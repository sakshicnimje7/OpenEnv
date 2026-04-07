"""Sample test cases for reward shaping behavior."""

import unittest

from env import Action, ActionType, WarehouseLogisticsEnvironment


class RewardShapingTests(unittest.TestCase):
    """Validate intermediate reward shaping mechanics."""

    def setUp(self) -> None:
        """Create fresh easy environment for each test."""
        self.env = WarehouseLogisticsEnvironment(task_difficulty="easy")
        self.obs = self.env.reset()
        self.order_id = self.obs.orders[0].order_id

    def test_step_penalty_applied_every_step(self) -> None:
        """First valid action should include step penalty in breakdown."""
        _obs, reward, _done, _info = self.env.step(
            Action(action_type=ActionType.CHECK_STOCK, order_id=self.order_id)
        )
        self.assertIn("step_penalty", reward.breakdown)
        self.assertAlmostEqual(reward.breakdown["step_penalty"], -0.01, places=6)

    def test_repeated_action_gets_lower_reward(self) -> None:
        """Repeating same action should reduce reward due to repeat penalty."""
        _obs, reward_first, _done, _info = self.env.step(
            Action(action_type=ActionType.CHECK_STOCK, order_id=self.order_id)
        )
        _obs, reward_second, _done, _info = self.env.step(
            Action(action_type=ActionType.CHECK_STOCK, order_id=self.order_id)
        )
        self.assertLess(reward_second.value, reward_first.value)
        self.assertIn("repeat_penalty", reward_second.breakdown)

    def test_out_of_order_action_penalized(self) -> None:
        """Shipping before allocation should include out-of-order penalty."""
        _obs, reward, _done, _info = self.env.step(
            Action(action_type=ActionType.SHIP, order_id=self.order_id)
        )
        self.assertIn("out_of_order_penalty", reward.breakdown)
        self.assertLessEqual(reward.value, -0.14)

    def test_progressive_actions_have_positive_bonus(self) -> None:
        """Correct sequence should include progress bonuses and complete order."""
        obs, reward_check, done, _info = self.env.step(
            Action(action_type=ActionType.CHECK_STOCK, order_id=self.order_id)
        )
        self.assertFalse(done)
        self.assertIn("progress_bonus", reward_check.breakdown)

        obs, reward_alloc, done, _info = self.env.step(
            Action(
                action_type=ActionType.ALLOCATE,
                order_id=self.order_id,
                warehouse=obs.orders[0].allocated_warehouse,
            )
        )
        self.assertFalse(done)
        self.assertIn("progress_bonus", reward_alloc.breakdown)

        _obs, reward_ship, done, _info = self.env.step(
            Action(action_type=ActionType.SHIP, order_id=self.order_id)
        )
        self.assertTrue(done)
        self.assertIn("progress_bonus", reward_ship.breakdown)

    def test_invalid_order_has_higher_penalty(self) -> None:
        """Invalid order IDs should be strongly penalized."""
        _obs, reward, _done, _info = self.env.step(
            Action(action_type=ActionType.CHECK_STOCK, order_id="ORD-DOES-NOT-EXIST")
        )
        self.assertLessEqual(reward.value, -0.2)
        self.assertIn("invalid", reward.breakdown)


if __name__ == "__main__":
    unittest.main()
