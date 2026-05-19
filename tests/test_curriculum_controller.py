import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "aerial_gym"
    / "rl_training"
    / "cleanrl"
    / "curriculum"
    / "curriculum_controller.py"
)
SPEC = importlib.util.spec_from_file_location("pursuit_curriculum_controller", MODULE_PATH)
curriculum_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(curriculum_module)

CurriculumController = curriculum_module.CurriculumController


def make_args(**overrides):
    values = {
        "curriculum_stable_success_rate": 0.95,
        "curriculum_stable_success_updates": 3,
        "curriculum_stable_success_min_episodes": 64,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_task_config():
    reward = SimpleNamespace(
        success_threshold=99.0,
        visibility_reward_weight_scale=1.0,
    )
    return SimpleNamespace(reward=reward)


def assert_reward_state(testcase, task_config, threshold, visibility_weight):
    testcase.assertAlmostEqual(task_config.reward.success_threshold, threshold)
    testcase.assertAlmostEqual(
        task_config.reward.visibility_reward_weight_scale,
        visibility_weight,
    )


class CurriculumControllerTest(unittest.TestCase):
    def test_initial_stage_applies_fixed_capture_threshold(self):
        task_config = make_task_config()
        controller = CurriculumController(task_config, make_args())

        self.assertEqual(controller.stage_idx, 0)
        self.assertFalse(controller.is_on_final_stage())
        assert_reward_state(self, task_config, 5.0, 0.0)

    def test_only_stable_success_advances_stage(self):
        args = make_args(curriculum_stable_success_updates=2)
        task_config = make_task_config()
        controller = CurriculumController(task_config, args)

        transition = controller.update(
            succ_rate=0.94,
            episode_rewards_summary={"count": 128},
        )
        self.assertIsNone(transition)
        self.assertEqual(controller.stage_idx, 0)
        self.assertEqual(controller.stage_success_streak, 0)

        transition = controller.update(
            succ_rate=0.99,
            episode_rewards_summary={"count": 10},
        )
        self.assertIsNone(transition)
        self.assertEqual(controller.stage_idx, 0)
        self.assertEqual(controller.stage_success_streak, 0)

        transition = controller.update(
            succ_rate=0.99,
            episode_rewards_summary={"count": 128},
        )
        self.assertIsNone(transition)
        self.assertEqual(controller.stage_idx, 0)
        self.assertEqual(controller.stage_success_streak, 1)

        transition = controller.update(
            succ_rate=0.96,
            episode_rewards_summary={"count": 128},
        )
        self.assertIsNotNone(transition)
        self.assertEqual(controller.stage_idx, 1)
        self.assertEqual(controller.stage_success_streak, 0)
        assert_reward_state(self, task_config, 3.0, 0.0)

    def test_fixed_five_stages_and_final_stage_guard(self):
        args = make_args(curriculum_stable_success_updates=1)
        task_config = make_task_config()
        controller = CurriculumController(task_config, args)

        expected_stages = (
            (0, 5.0, 0.0),
            (1, 3.0, 0.0),
            (2, 3.0, 0.20),
            (3, 3.0, 0.40),
            (4, 3.0, 0.60),
        )
        for stage_idx, threshold, visibility_weight in expected_stages:
            self.assertEqual(controller.stage_idx, stage_idx)
            assert_reward_state(
                self,
                task_config,
                threshold,
                visibility_weight,
            )
            if stage_idx == expected_stages[-1][0]:
                continue
            transition = controller.update(
                succ_rate=0.99,
                episode_rewards_summary={"count": 128},
            )
            self.assertIsNotNone(transition)

        self.assertTrue(controller.is_on_final_stage())
        transition = controller.update(
            succ_rate=0.99,
            episode_rewards_summary={"count": 128},
        )
        self.assertIsNone(transition)
        self.assertEqual(controller.stage_idx, 4)
        assert_reward_state(self, task_config, 3.0, 0.60)

    def test_load_state_restores_progress_and_reapplies_fixed_stage(self):
        task_config = make_task_config()
        controller = CurriculumController(task_config, make_args())

        controller.load_state_dict(
            {
                "stage_idx": 3,
                "stage_success_streak": 2,
            }
        )

        self.assertEqual(controller.stage_idx, 3)
        self.assertEqual(controller.stage_success_streak, 2)
        assert_reward_state(self, task_config, 3.0, 0.40)


if __name__ == "__main__":
    unittest.main()
