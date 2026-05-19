"""Fixed five-stage success-gated curriculum for pursuit PPO training."""


class CurriculumController:
    """Advance fixed curriculum stages after stable per-update success."""

    STAGES = (
        (5.0, 0.0),
        (3.0, 0.0),
        (3.0, 0.20),
        (3.0, 0.40),
        (3.0, 0.60),
    )

    def __init__(self, task_config, args):
        self.task_config = task_config
        self.stable_success_rate = float(args.curriculum_stable_success_rate)
        self.stable_success_updates = int(args.curriculum_stable_success_updates)
        self.stable_success_min_episodes = int(
            args.curriculum_stable_success_min_episodes
        )
        if self.stable_success_rate < 0.0 or self.stable_success_rate > 1.0:
            raise ValueError("curriculum_stable_success_rate must be in [0, 1].")
        if self.stable_success_updates <= 0:
            raise ValueError("curriculum_stable_success_updates must be positive.")
        if self.stable_success_min_episodes <= 0:
            raise ValueError("curriculum_stable_success_min_episodes must be positive.")

        self.stage_idx = 0
        self.stage_success_streak = 0
        self.current_threshold = 0.0
        self.visibility_reward_weight_scale = 0.0

        self._apply_stage(self.stage_idx)

    def _apply_stage(self, stage_idx):
        threshold, visibility_weight = self.STAGES[int(stage_idx)]
        self.current_threshold = float(threshold)
        self.visibility_reward_weight_scale = float(visibility_weight)
        self.task_config.reward.success_threshold = self.current_threshold
        self.task_config.reward.visibility_reward_weight_scale = (
            self.visibility_reward_weight_scale
        )

    def is_on_final_stage(self):
        return self.stage_idx >= len(self.STAGES) - 1

    def update(self, succ_rate, episode_rewards_summary):
        """Return transition metadata when a stage transition occurs."""
        if self.is_on_final_stage():
            return None

        finished_episodes = int(episode_rewards_summary.get("count", 0))
        success_ready = (
            succ_rate is not None
            and finished_episodes >= self.stable_success_min_episodes
            and float(succ_rate) >= self.stable_success_rate
        )
        if success_ready:
            self.stage_success_streak += 1
        else:
            self.stage_success_streak = 0

        if self.stage_success_streak < self.stable_success_updates:
            return None

        return self._execute_transition(succ_rate, finished_episodes)

    def _execute_transition(self, succ_rate, finished_episodes):
        old_stage_idx = self.stage_idx
        old_threshold = self.current_threshold
        old_visibility_scale = self.visibility_reward_weight_scale

        self.stage_idx += 1
        self.stage_success_streak = 0
        self._apply_stage(self.stage_idx)

        return {
            "stage_idx": self.stage_idx,
            "old_threshold": old_threshold,
            "new_threshold": self.current_threshold,
            "old_visibility_reward_weight_scale": old_visibility_scale,
            "new_visibility_reward_weight_scale": self.visibility_reward_weight_scale,
            "succ_rate": None if succ_rate is None else float(succ_rate),
            "finished_episodes": finished_episodes,
            "stable_success_rate": self.stable_success_rate,
            "stable_success_updates": self.stable_success_updates,
            "completed_stage_idx": old_stage_idx,
        }

    def log_curriculum_metrics(self, writer, global_step):
        """Log curriculum diagnostics to TensorBoard."""
        try:
            writer.add_scalar("curriculum/stage_idx", float(self.stage_idx), global_step)
            writer.add_scalar("curriculum/success_threshold", self.current_threshold, global_step)
            writer.add_scalar(
                "curriculum/visibility_reward_weight_scale",
                float(self.visibility_reward_weight_scale),
                global_step,
            )
            writer.add_scalar(
                "curriculum/stable_success_streak",
                float(self.stage_success_streak),
                global_step,
            )
            writer.add_scalar(
                "curriculum/stable_success_rate",
                float(self.stable_success_rate),
                global_step,
            )
        except Exception:
            pass

    def get_state_dict(self):
        return {
            "stage_idx": self.stage_idx,
            "stage_success_streak": self.stage_success_streak,
        }

    def load_state_dict(self, state_dict):
        self.stage_idx = int(state_dict["stage_idx"])
        self.stage_success_streak = int(state_dict["stage_success_streak"])
        self._apply_stage(self.stage_idx)
