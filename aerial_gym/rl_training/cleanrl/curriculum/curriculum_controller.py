"""Threshold-only fixed-step curriculum for pursuit-evasion PPO training."""

import os
import torch


class CurriculumController:
    """Advance success thresholds on fixed global-step budgets."""

    def __init__(self, task_config, args, checkpoint_fn=None):
        self.task_config = task_config
        self.args = args
        self._checkpoint_fn = checkpoint_fn

        self.thresholds = [float(value) for value in args.curriculum_thresholds]
        self.stage_step_budgets = [int(value) for value in args.curriculum_stage_step_budgets]
        if len(self.thresholds) < 1:
            raise ValueError("curriculum_thresholds must contain at least one threshold.")
        if len(self.stage_step_budgets) != max(len(self.thresholds) - 1, 0):
            raise ValueError(
                "curriculum_stage_step_budgets must have exactly one entry for every non-final stage."
            )
        if any(budget <= 0 for budget in self.stage_step_budgets):
            raise ValueError("curriculum_stage_step_budgets must be positive.")

        self.stage_idx = 0
        self.current_threshold = self.thresholds[0]
        self.stage_start_step = 0
        self.stage_start_update = 0
        self.stage_history = []

        self._apply_threshold(self.current_threshold)

    def _apply_threshold(self, threshold_m):
        self.task_config.reward.success_threshold = float(threshold_m)
        self.current_threshold = float(threshold_m)

    def is_on_final_stage(self):
        return self.stage_idx >= len(self.thresholds) - 1

    @property
    def updates_in_stage(self):
        return 0  # retained for checkpoint compatibility

    def update(self, global_step, update, succ_rate, episode_rewards_summary,
               agent, optimizer, run_dir, run_name, wandb_run_id, current_ent_coef=None):
        """
        Called after each PPO update's metrics computation.
        Returns a transition dict if a stage transition occurred, else None.
        """
        if self.is_on_final_stage():
            return None

        stage_budget_steps = self.stage_step_budgets[self.stage_idx]
        stage_local_steps = max(int(global_step) - int(self.stage_start_step), 0)
        if stage_local_steps < stage_budget_steps:
            return None

        transition_info = {
            "type": "scheduled",
            "reason": (
                f"stage_step_budget ({stage_budget_steps}) reached, "
                f"stage_local_steps={stage_local_steps}, update={update}"
            ),
            "stage_budget_steps": stage_budget_steps,
            "stage_local_steps": stage_local_steps,
            "succ_rate": None if succ_rate is None else float(succ_rate),
        }
        return self._execute_transition(
            transition_info, global_step, update,
            agent, optimizer, run_dir, run_name, wandb_run_id,
        )

    def _execute_transition(self, info, global_step, update,
                            agent, optimizer, run_dir, run_name, wandb_run_id):
        old_threshold = self.current_threshold
        next_stage_idx = self.stage_idx + 1
        new_threshold = self.thresholds[next_stage_idx]

        # Record this stage
        stage_record = {
            "stage_idx": self.stage_idx,
            "old_threshold": old_threshold,
            "new_threshold": new_threshold,
            "start_update": self.stage_start_update,
            "end_update": update,
            "start_step": self.stage_start_step,
            "end_step": global_step,
            "final_sr": info["succ_rate"],
            "transition_reason": info["reason"],
            "transition_type": info["type"],
        }
        self.stage_history.append(stage_record)

        completed_stage_idx = self.stage_idx
        self.stage_idx = next_stage_idx
        self._apply_threshold(new_threshold)
        self.stage_start_step = global_step
        self.stage_start_update = update

        # Save stage-completion checkpoint
        stage_ckpt_path = os.path.join(
            run_dir,
            f"stage_{completed_stage_idx}_thr_{old_threshold:.1f}m_upd_{update}.pth",
        )

        stage_ckpt = self._checkpoint_fn(
            agent=agent,
            optimizer=optimizer,
            args=self.args,
            global_step=global_step,
            update=update,
            best_score=None,
            best_update=None,
            no_improve_updates=0,
            run_name=run_name,
            run_dir=run_dir,
            wandb_run_id=wandb_run_id,
        )
        stage_ckpt["curriculum"] = self.get_state_dict()
        stage_ckpt["curriculum_completed_stage"] = completed_stage_idx
        stage_ckpt["curriculum_completed_threshold"] = old_threshold
        stage_ckpt["curriculum_history"] = self.stage_history
        torch.save(stage_ckpt, stage_ckpt_path)

        return {
            "stage_idx": self.stage_idx,
            "old_threshold": old_threshold,
            "new_threshold": new_threshold,
            "stage_ckpt_path": stage_ckpt_path,
            "transition_info": info,
        }

    def log_curriculum_metrics(self, writer, global_step, update, succ_rate):
        """Log curriculum diagnostics to TensorBoard."""
        try:
            writer.add_scalar("curriculum/stage_idx", float(self.stage_idx), global_step)
            writer.add_scalar("curriculum/success_threshold", self.current_threshold, global_step)
            writer.add_scalar(
                "curriculum/updates_in_stage",
                float(update - self.stage_start_update),
                global_step,
            )
            if not self.is_on_final_stage():
                stage_budget_steps = float(self.stage_step_budgets[self.stage_idx])
                stage_local_steps = float(max(int(global_step) - int(self.stage_start_step), 0))
                writer.add_scalar(
                    "curriculum/stage_step_budget",
                    stage_budget_steps,
                    global_step,
                )
                writer.add_scalar(
                    "curriculum/stage_local_steps",
                    stage_local_steps,
                    global_step,
                )
                writer.add_scalar(
                    "curriculum/stage_progress_fraction",
                    min(stage_local_steps / max(stage_budget_steps, 1.0), 1.0),
                    global_step,
                )
        except Exception:
            pass

    def get_state_dict(self):
        return {
            "stage_idx": self.stage_idx,
            "current_threshold": self.current_threshold,
            "stage_start_step": self.stage_start_step,
            "stage_start_update": self.stage_start_update,
            "stage_history": self.stage_history,
            "thresholds": self.thresholds,
            "stage_step_budgets": self.stage_step_budgets,
        }

    def load_state_dict(self, state_dict):
        self.stage_idx = state_dict["stage_idx"]
        self.current_threshold = state_dict["current_threshold"]
        self.stage_start_step = state_dict["stage_start_step"]
        self.stage_start_update = state_dict["stage_start_update"]
        self.stage_history = state_dict["stage_history"]
        self.thresholds = [float(value) for value in state_dict.get("thresholds", self.thresholds)]
        self.stage_step_budgets = [
            int(value) for value in state_dict.get("stage_step_budgets", self.stage_step_budgets)
        ]
        self._apply_threshold(self.current_threshold)
