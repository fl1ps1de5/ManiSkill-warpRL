"""Dynamics-shifted variant of SO101PegInsertionSide.

This task keeps the same observation, reward, success, reset, and geometry as
``SO101PegInsertionSide-v1``. It only changes the control channel/dynamics so
sim-to-sim adaptation can mimic the real deployment setting more closely.
"""

from __future__ import annotations

import numpy as np
import torch

from mani_skill.utils.registration import register_env

from .so101_peg_insertion_side import SO101PegInsertionSideEnv


SO101_PEG_INSERTION_SIDE_DYN_ENV_ID = "SO101PegInsertionSideDyn-v1"


@register_env(SO101_PEG_INSERTION_SIDE_DYN_ENV_ID, max_episode_steps=100)
class SO101PegInsertionSideDynEnv(SO101PegInsertionSideEnv):
    """SO101 side insertion with sim-to-real-style control shifts.

    ``arm_stiffness_scale`` scales the five arm joints' PD stiffness/damping.
    ``action_delay_steps`` applies the action emitted N control steps ago.

    The parent task already scales arm actions in ``_step_action`` via
    ``_action_scale`` and clamps the gripper action. This subclass delays the
    raw policy action before calling the parent so that normal parent handling
    is still applied exactly once.
    """

    _BASE_ARM_STIFFNESS = 1e3
    _BASE_ARM_DAMPING = 1e2
    _BASE_ARM_FORCE_LIMIT = 100.0

    def __init__(
        self,
        *args,
        arm_stiffness_scale: float = 1.0,
        action_delay_steps: int = 0,
        **kwargs,
    ) -> None:
        self.arm_stiffness_scale = float(arm_stiffness_scale)
        self.action_delay_steps = int(action_delay_steps)
        if self.action_delay_steps < 0:
            raise ValueError(
                f"action_delay_steps must be >= 0, got {self.action_delay_steps}"
            )
        self._action_delay_buffer: list | None = None
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self._apply_arm_compliance()

    def _apply_arm_compliance(self) -> None:
        if self.arm_stiffness_scale == 1.0:
            return
        scaled_stiffness = self._BASE_ARM_STIFFNESS * self.arm_stiffness_scale
        scaled_damping = self._BASE_ARM_DAMPING * self.arm_stiffness_scale
        arm_joint_names = set(self.agent.arm_joint_names)
        for joint in self.agent.robot.active_joints:
            if joint.name in arm_joint_names:
                joint.set_drive_properties(
                    stiffness=scaled_stiffness,
                    damping=scaled_damping,
                    force_limit=self._BASE_ARM_FORCE_LIMIT,
                    mode="force",
                )

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        if self.action_delay_steps == 0 or self._action_delay_buffer is None:
            return
        for buf in self._action_delay_buffer:
            buf[env_idx] = 0

    def _step_action(self, action):
        if self.action_delay_steps == 0 or action is None:
            return super()._step_action(action)
        self._ensure_delay_buffer(action)
        if isinstance(action, torch.Tensor):
            self._action_delay_buffer.append(action.clone())
        else:
            self._action_delay_buffer.append(np.asarray(action).copy())
        delayed = self._action_delay_buffer.pop(0)
        return super()._step_action(delayed)

    def _ensure_delay_buffer(self, action) -> None:
        if self._action_delay_buffer is not None:
            return
        if isinstance(action, torch.Tensor):
            zero = torch.zeros_like(action)
            self._action_delay_buffer = [
                zero.clone() for _ in range(self.action_delay_steps)
            ]
        else:
            zero = np.zeros_like(action)
            self._action_delay_buffer = [
                zero.copy() for _ in range(self.action_delay_steps)
            ]


__all__ = [
    "SO101_PEG_INSERTION_SIDE_DYN_ENV_ID",
    "SO101PegInsertionSideDynEnv",
]
