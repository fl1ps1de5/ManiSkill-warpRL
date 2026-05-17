"""Side peg insertion on SO101 with a pre-grasped peg.

Insertion axis is world +X (table-plane). This is the SO101's
strongest control direction (shoulder + elbow translate the gripper
forward without coupling into other axes), unlike the downward
variant where wrist motion couples laterally during descent.

Reward/observation structure mirror ``PegInsertionSide-v1``:

1. ``grasp_reward``         -- keep the peg grasped.
2. ``pre_insertion_reward`` -- combined head+body YZ shaping with a
   ``max()`` term plus a yaw error fold-in. The yaw term is the one
   thing we add beyond PegInsertionSide: our peg is square in cross
   section so square-vs-diamond orientation matters.
3. ``insertion_reward``     -- ``tanh`` of the 3D distance from the
   peg head to the in-hole goal (socket center). Gated by binary
   YZ-alignment and grasped.

Geometry also follows PegInsertionSide: ``socket_depth == peg_length``
and the seated-depth target is half the socket depth, so the success
criterion is "peg head at the socket center" with an X-tolerance band.

Observations are all real-hardware reconstructable: joint state from
encoders, TCP pose from FK, calibrated socket pose, FK-derived peg
head estimate from a rigid grasp model.

Use ``control_mode="pd_joint_target_delta_pos"`` so a zero action
holds the current joint target against gravity.
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
import sapien
import sapien.render
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots.so101 import (
    SO101,
)  # noqa: F401  (registers robot_uids="so101")
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import SimConfig


SO101_PEG_INSERTION_SIDE_ENV_ID = "SO101PegInsertionSide-v1"


def _build_wall_with_hole(
    scene: "ManiSkillScene",
    inner_radius: float,
    outer_radius: float,
    depth: float,
    chamfer_depth: float = 0.0,
    chamfer_extra_radius: float = 0.0,
):
    """Build a box with a square hole going through it along +X.

    The hole opens in the -X direction (entry side). Four wall slabs
    form the hole, plus an optional chamfer funnel on the entry face
    for forgiving entry.
    """
    builder = scene.create_actor_builder()
    thickness = (outer_radius - inner_radius) * 0.5
    half_d = depth * 0.5

    half_sizes = [
        [half_d, thickness, outer_radius],
        [half_d, thickness, outer_radius],
        [half_d, outer_radius, thickness],
        [half_d, outer_radius, thickness],
    ]
    offset = thickness + inner_radius
    poses = [
        sapien.Pose([0, offset, 0]),
        sapien.Pose([0, -offset, 0]),
        sapien.Pose([0, 0, offset]),
        sapien.Pose([0, 0, -offset]),
    ]

    base_mat = sapien.render.RenderMaterial(
        base_color=sapien_utils.hex2rgba("#E59B2A"),
        roughness=0.6,
        specular=0.3,
    )

    for half_size, pose in zip(half_sizes, poses):
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size, material=base_mat)

    if chamfer_depth > 0.0 and chamfer_extra_radius > 0.0:
        chamfer_inner = inner_radius + chamfer_extra_radius
        chamfer_thickness = (outer_radius - chamfer_inner) * 0.5
        if chamfer_thickness > 0.0:
            half_cd = chamfer_depth * 0.5
            chamfer_x = -half_d - half_cd
            chamfer_offset = chamfer_thickness + chamfer_inner
            chamfer_half_sizes = [
                [half_cd, chamfer_thickness, outer_radius],
                [half_cd, chamfer_thickness, outer_radius],
                [half_cd, outer_radius, chamfer_thickness],
                [half_cd, outer_radius, chamfer_thickness],
            ]
            chamfer_poses = [
                sapien.Pose([chamfer_x, chamfer_offset, 0]),
                sapien.Pose([chamfer_x, -chamfer_offset, 0]),
                sapien.Pose([chamfer_x, 0, chamfer_offset]),
                sapien.Pose([chamfer_x, 0, -chamfer_offset]),
            ]
            chamfer_mat = sapien.render.RenderMaterial(
                base_color=sapien_utils.hex2rgba("#C77F1F"),
                roughness=0.6,
                specular=0.3,
            )
            for half_size, pose in zip(chamfer_half_sizes, chamfer_poses):
                builder.add_box_collision(pose, half_size)
                builder.add_box_visual(pose, half_size, material=chamfer_mat)

    return builder


@register_env(SO101_PEG_INSERTION_SIDE_ENV_ID, max_episode_steps=100)
class SO101PegInsertionSideEnv(BaseEnv):
    """Pre-grasped side peg insertion on the SO101 arm.

    The peg starts pre-grasped, held with its long axis along world
    +X. A socket box with a square hole sits in front of the robot;
    the policy commands 5-DOF arm joint deltas to advance the peg
    head into the hole until seated mid-socket.

    State observation (all sim2real-reconstructable):
      qpos (6), qvel (6), tcp_pose (7), socket_pose (7),
      estimated_peg_head_to_socket_face (3)

    Success (sim-to-sim RLPD label):
      Uses oracle sim state so sparse rewards correspond to actual
      insertion: peg head/body aligned with the socket, peg head seated
      along X, peg still grasped, peg horizontal, and square yaw aligned.
      A real-hardware-measurable FK proxy is also reported as
      ``success_real_only`` / ``success_proxy`` for false-positive audits.

    Reward (sparse, RLPD-friendly):
      Sparse by default (``sparse_reward=True``): 0 every step,
      ``_success_bonus`` on the step success fires. RLPD only needs the
      binary task indicator on top of the demo buffer, and the sparse
      label is the oracle sim-to-sim success signal. Pass
      ``sparse_reward=False`` to fall back to the dense PPO-training
      reward (used to produce the initial baseline).
    """

    SUPPORTED_ROBOTS = ["so101"]
    agent: Union[SO101]

    # ---------------- peg geometry ----------------
    # 15 mm square cross-section, 60 mm long.
    _peg_radius = 0.0075
    _peg_length = 0.06
    _peg_density = 1000.0
    # Gripper closure (rad). -0.05 keeps the peg secure without
    # ejecting it at init.
    _gripper_grasp_qpos = -0.05
    # Distance along peg-local +Z from the rear end to the grasp line.
    _peg_grasp_offset_from_top = 0.025
    # Peg-in-TCP orientation (wxyz). Identity puts peg-local -Z along
    # world +X at the side-insertion home pose.
    _peg_initial_orientation_local = (1.0, 0.0, 0.0, 0.0)

    # ---------------- socket geometry ----------------
    # PegInsertionSide ratio: hole depth == peg length, so the peg
    # can be fully inserted. The hole is through-and-through.
    #
    # 1.5mm radial clearance matches what is achievable with 3D-printed
    # parts on real hardware, and gives the SO101 enough mechanical
    # forgiveness during insertion that contact-induced slip stays
    # below the deadband for clean attempts. The RLPD story is then
    # "sim trains on the forgiving forward task, real-world dynamics
    # tighten the clearance budget further" - a clean sim-to-real gap.
    _socket_clearance = 0.001
    _socket_outer_radius = 0.045
    _socket_depth = 0.06  # = peg_length
    # Larger chamfer funnels off-axis approaches into the hole without
    # the peg head ever contacting wall material on the way in.
    _socket_chamfer_depth = 0.0
    _socket_chamfer_extra_radius = 0.0

    # Socket position randomization (table frame). socket_x is the
    # box center; entry face is at socket_x - socket_depth/2.
    _socket_x_low = 0.36
    _socket_x_high = 0.39
    _socket_y_low = -0.02
    _socket_y_high = 0.02
    _socket_z_center = 0.12

    # ---------------- success criterion ----------------
    # Seated depth: peg head at socket center = half the peg inside.
    # PegInsertionSide-equivalent for seated_depth = socket_depth/2.
    _seated_depth = 0.030
    # One-sided X-grace zone matching PegInsertionSide: success fires
    # whenever the peg head is at ``goal_x - tolerance`` or deeper. No
    # upper bound -- overshoot still counts. Removes the precision-phase
    # "policy enters hole then can't stop precisely" failure mode that
    # bites two-sided success bands.
    _success_x_tolerance = 0.015

    # ---------------- reward weights ----------------
    # Four-component additive reward, all gated by grasped, designed
    # so max shaped reward == success bonus (no cliff at success):
    #   grasp        (2.0) - is_grasped  (always-on grasp incentive)
    #   pre_insertion(3.0) - head+body YZ shaping + yaw fold-in
    #   insertion    (5.0) - 3D tanh distance to in-hole goal
    #   max shaped   = 10.0 == success_bonus
    # Success replaces reward with success_bonus (PegInsertionSide
    # ``reward[info["success"]] = 10`` pattern). Because the cap
    # already equals the bonus, success is a *stabiliser* (prevents
    # reward flicker near the goal) rather than a cliff the policy
    # has to chase, removing the "rush to commit" failure mode.
    _grasp_reward_weight = 2.0
    _pre_insertion_reward_weight = 3.0
    _pre_insertion_yaw_error_weight = 4.0
    _insertion_reward_weight = 5.0
    _insertion_tanh_scale = 5.0
    _success_bonus = 10.0

    # YZ thresholds for the binary ``pre_inserted_yz`` insertion gate.
    _pre_inserted_head_yz_threshold = 0.005
    _pre_inserted_body_yz_threshold = 0.007

    # Per-step arm-action scaling. The PD delta controller commands
    # +-0.1 rad/step; with ``_action_scale = 0.3`` realised delta is
    # +-0.03 rad. Sim2real safety knob - keeps trained policy from
    # commanding aggressive joint targets on the real robot.
    _action_scale = 0.3

    # ---------------- exploit / safety guards ----------------
    _peg_grasped_max_dist = 0.02
    _peg_horizontal_min_align = 0.7
    _peg_square_yaw_min_align = 0.9

    # ---------------- real-hardware grasp diagnostic ----------------
    # NOTE: in this env (pd_joint_target_delta_pos + arm_only_action)
    # the gripper joint encoder does NOT cleanly separate "peg held"
    # from "peg lost": under a held peg the qpos drifts from ~-0.048
    # at episode start toward ~-0.033 over an episode (the peg shifts
    # in the gripper), and once the peg is gone the joint coasts at
    # wherever it last was rather than actively closing back to the
    # commanded ``_gripper_grasp_qpos = -0.05`` target. We therefore
    # do NOT expose a boolean ``_peg_grasped_real`` -- the raw
    # ``gripper_qpos`` is reported instead so eval runs can audit the
    # drift / catastrophic opens visually, and the success label is
    # built without any grasp gate (mirroring PegInsertionSide-v1).

    # Slip diagnostics (NOT applied to the reward in the current design).
    # ``estimated_peg_head_pos`` assumes the peg sits at a fixed offset
    # from the TCP (rigid grasp model); slip = how far the peg has
    # drifted from that expected pose. We report:
    #
    #   peg_slipped (bool)  = peg_slip > slip_max_dist
    #   slip_factor [0, 1]  = clamp(1 - max(0, peg_slip - deadband)
    #                              / (zero_dist - deadband), 0, 1)
    #
    # ``slip_factor`` is informational: it tells us how often contact
    # is destabilising the grasp during training. We don't gate reward
    # by it (a prior gating design caused a slip cascade that broke
    # training). Geometry (wider clearance, larger chamfer) handles
    # the underlying issue.
    _peg_slip_max_dist = 0.010
    _peg_slip_zero_dist = 0.020
    _peg_slip_deadband = 0.005

    def __init__(
        self,
        *args,
        robot_uids: str = "so101",
        num_envs: int = 1,
        socket_pose_noise_std: float = 0.0,
        home_pose_noise_std: float = 0.0,
        grasp_offset_noise_std: float = 0.0,
        arm_only_action: bool = True,
        sim_config: Any = None,
        align_only: bool = False,
        socket_x_range: Any = None,
        socket_y_range: Any = None,
        socket_z_center: Any = None,
        sparse_reward: bool = True,
        physical_socket_base: bool = True,
        **kwargs,
    ) -> None:
        self.physical_socket_base = bool(physical_socket_base)
        self.socket_pose_noise_std = float(socket_pose_noise_std)
        self.home_pose_noise_std = float(home_pose_noise_std)
        self.grasp_offset_noise_std = float(grasp_offset_noise_std)
        self.arm_only_action = bool(arm_only_action)
        self._sim_config_override = sim_config
        # Phase-1 curriculum: zero insertion reward and force
        # ``has_peg_inserted`` to False so the policy trains on pure
        # pre-insertion (grasp + YZ + yaw).
        self.align_only = bool(align_only)
        # RLPD-style sparse reward (success bonus on success, 0 else).
        # Default True for the sim-to-sim RLPD target task. Set False
        # for dense-reward PPO from-scratch training.
        self.sparse_reward = bool(sparse_reward)
        # Per-instance socket position overrides for OOD evaluation.
        # When None, fall back to the class-level training-distribution
        # defaults. Pass a (low, high) tuple/list to override.
        if socket_x_range is not None:
            self._socket_x_low = float(socket_x_range[0])
            self._socket_x_high = float(socket_x_range[1])
        if socket_y_range is not None:
            self._socket_y_low = float(socket_y_range[0])
            self._socket_y_high = float(socket_y_range[1])
        if socket_z_center is not None:
            self._socket_z_center = float(socket_z_center)
        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        if self._sim_config_override is None:
            return SimConfig()
        if isinstance(self._sim_config_override, SimConfig):
            return self._sim_config_override
        if isinstance(self._sim_config_override, dict):
            return SimConfig(**self._sim_config_override)
        raise TypeError(
            "sim_config must be a SimConfig, dict, or None; got "
            f"{type(self._sim_config_override).__name__}"
        )

    @property
    def _default_sensor_configs(self):
        return []

    @property
    def _default_human_render_camera_configs(self):
        from mani_skill.sensors.camera import CameraConfig

        # Close side view aimed at the socket entry face.
        pose = sapien_utils.look_at([0.28, -0.18, 0.16], [0.36, 0.0, 0.12])
        return CameraConfig("render_camera", pose, 512, 512, 0.75, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0.0, 0.0, 0.0]))

    def _load_scene(self, options: dict):
        with torch.device(self.device):
            self.table_scene = TableSceneBuilder(self)
            self.table_scene.build()

            # ---- peg actor (dynamic) ----
            peg_builder = self.scene.create_actor_builder()
            peg_body_half = [
                self._peg_radius,
                self._peg_radius,
                self._peg_length * 0.5,
            ]
            peg_mat = sapien.render.RenderMaterial(
                base_color=sapien_utils.hex2rgba("#3070C8"),
                roughness=0.5,
                specular=0.5,
            )
            peg_head_mat = sapien.render.RenderMaterial(
                base_color=sapien_utils.hex2rgba("#D94141"),
                roughness=0.5,
                specular=0.5,
            )
            peg_builder.add_box_collision(
                sapien.Pose(p=[0, 0, 0]),
                peg_body_half,
                density=self._peg_density,
            )
            head_visual_len = 0.015
            tail_visual_len = self._peg_length - head_visual_len
            peg_builder.add_box_visual(
                sapien.Pose(p=[0, 0, 0.5 * head_visual_len]),
                [
                    self._peg_radius,
                    self._peg_radius,
                    0.5 * tail_visual_len,
                ],
                material=peg_mat,
            )
            peg_builder.add_box_visual(
                sapien.Pose(p=[0, 0, -0.5 * self._peg_length + 0.5 * head_visual_len]),
                [
                    self._peg_radius,
                    self._peg_radius,
                    0.5 * head_visual_len,
                ],
                material=peg_head_mat,
            )
            peg_builder.initial_pose = sapien.Pose(p=[0.10, 0.0, 0.3])
            self.peg = peg_builder.build("peg")

            # ---- socket actor (kinematic) ----
            inner_radius = self._peg_radius + self._socket_clearance
            socket_builder = _build_wall_with_hole(
                self.scene,
                inner_radius=inner_radius,
                outer_radius=self._socket_outer_radius,
                depth=self._socket_depth,
                chamfer_depth=self._socket_chamfer_depth,
                chamfer_extra_radius=self._socket_chamfer_extra_radius,
            )
            if self.physical_socket_base:
                # Add integrated base from table surface up to socket body.
                # In the socket's local frame, the body spans Z: [-outer_radius, +outer_radius].
                # The table is at world Z=0, socket center at Z=_socket_z_center,
                # so the base extends from local Z = -_socket_z_center to Z = -outer_radius.
                base_top_z = -self._socket_outer_radius
                base_bot_z = -self._socket_z_center
                base_half_z = (base_top_z - base_bot_z) * 0.5
                base_center_z = (base_top_z + base_bot_z) * 0.5
                half_d = self._socket_depth * 0.5
                base_mat = sapien.render.RenderMaterial(
                    base_color=sapien_utils.hex2rgba("#808080"),
                    roughness=0.8,
                )
                socket_builder.add_box_collision(
                    sapien.Pose(p=[0, 0, base_center_z]),
                    [half_d, self._socket_outer_radius, abs(base_half_z)],
                )
                socket_builder.add_box_visual(
                    sapien.Pose(p=[0, 0, base_center_z]),
                    [half_d, self._socket_outer_radius, abs(base_half_z)],
                    material=base_mat,
                )

            socket_builder.initial_pose = sapien.Pose(
                p=[
                    0.5 * (self._socket_x_low + self._socket_x_high),
                    0.5 * (self._socket_y_low + self._socket_y_high),
                    self._socket_z_center,
                ]
            )
            self.socket = socket_builder.build_kinematic("socket")

            self._socket_inner_radius = float(inner_radius)
            # Entry face is on the -X side; actor origin is at the
            # socket box center, so the offset is -depth/2.
            self._socket_face_x_offset = -self._socket_depth * 0.5
            self._grasp_offset_p = torch.zeros((self.num_envs, 3), device=self.device)
            self._socket_pose_meas_offset_p = torch.zeros(
                (self.num_envs, 3), device=self.device
            )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # ---- socket: random xy inside workspace box ----
            low = torch.tensor([self._socket_x_low, self._socket_y_low])
            high = torch.tensor([self._socket_x_high, self._socket_y_high])
            socket_xy = randomization.uniform(low=low, high=high, size=(b, 2))
            socket_p = torch.zeros((b, 3))
            socket_p[:, :2] = socket_xy
            socket_p[:, 2] = self._socket_z_center
            self.socket.set_pose(
                Pose.create_from_pq(
                    p=socket_p,
                    q=torch.tensor([1.0, 0.0, 0.0, 0.0]).unsqueeze(0).repeat(b, 1),
                )
            )

            if self.socket_pose_noise_std > 0.0:
                noise = torch.randn((b, 3)) * self.socket_pose_noise_std
            else:
                noise = torch.zeros((b, 3))
            self._socket_pose_meas_offset_p[env_idx] = noise

            # ---- robot home pose: gripper horizontal, arm partially
            # retracted so the policy has room to extend in +X ----
            qpos_home = torch.tensor(
                [
                    0.0,
                    -np.pi / 4,
                    np.pi / 2,
                    -np.pi / 4,
                    -np.pi / 2,
                    float(self._gripper_grasp_qpos),
                ],
                dtype=torch.float32,
            )
            qpos_home = qpos_home.unsqueeze(0).repeat(b, 1)
            if self.home_pose_noise_std > 0.0:
                qpos_home = qpos_home + torch.randn_like(qpos_home) * float(
                    self.home_pose_noise_std
                )
            self.agent.robot.set_qpos(qpos_home)
            self.agent.robot.set_pose(sapien.Pose([0, 0, 0]))

            # Sync GPU state so subsequent TCP read reflects home pose.
            if self.device.type == "cuda":
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene._gpu_fetch_all()

            # ---- grasp offset in TCP frame ----
            base_grasp_offset = torch.tensor(
                [
                    0.0,
                    0.0,
                    self._peg_length * 0.5 - self._peg_grasp_offset_from_top,
                ],
                dtype=torch.float32,
            )
            grasp_offset = base_grasp_offset.unsqueeze(0).repeat(b, 1)
            if self.grasp_offset_noise_std > 0.0:
                grasp_offset = grasp_offset + torch.randn_like(grasp_offset) * float(
                    self.grasp_offset_noise_std
                )
            self._grasp_offset_p[env_idx] = grasp_offset.to(self.device)

            # ---- place the peg at the gripper-jaw centre ----
            tcp_q = self.agent.tcp_pose.q[env_idx]
            tcp_p = self.agent.tcp_pose.p[env_idx]
            peg_q_local = (
                torch.tensor(
                    self._peg_initial_orientation_local,
                    dtype=torch.float32,
                    device=self.device,
                )
                .unsqueeze(0)
                .repeat(b, 1)
            )
            peg_q = self._quat_mul(tcp_q, peg_q_local)
            grasp_offset_world = self._rotate_vec(peg_q, grasp_offset.to(self.device))
            peg_p = tcp_p - grasp_offset_world
            self.peg.set_pose(Pose.create_from_pq(p=peg_p, q=peg_q))
            zero_vec = torch.zeros((b, 3), device=self.device)
            self.peg.set_linear_velocity(zero_vec)
            self.peg.set_angular_velocity(zero_vec)

    def _step_action(self, action):
        # Scale arm actions down for smoother sim2real motion. Force
        # gripper action to 0 (closure is set at init and held).
        if action is not None:
            if isinstance(action, torch.Tensor):
                action = action.clone()
                action[..., :5] = action[..., :5] * self._action_scale
                if self.arm_only_action and action.shape[-1] >= 6:
                    action[..., 5] = 0.0
            elif isinstance(action, np.ndarray):
                action = action.copy()
                action[..., :5] = action[..., :5] * self._action_scale
                if self.arm_only_action and action.shape[-1] >= 6:
                    action[..., 5] = 0.0
        return super()._step_action(action)

    # ---------------- quaternion / vector helpers ----------------
    @staticmethod
    def _rotate_vec(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        vx, vy, vz = vec[:, 0], vec[:, 1], vec[:, 2]
        tx = 2 * (y * vz - z * vy)
        ty = 2 * (z * vx - x * vz)
        tz = 2 * (x * vy - y * vx)
        rx = vx + w * tx + (y * tz - z * ty)
        ry = vy + w * ty + (z * tx - x * tz)
        rz = vz + w * tz + (x * ty - y * tx)
        return torch.stack([rx, ry, rz], dim=-1)

    @staticmethod
    def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack([w, x, y, z], dim=-1)

    # ---------------- derived poses ----------------
    @property
    def peg_head_pos(self) -> torch.Tensor:
        # Peg head: peg-local -Z, transformed to world.
        peg_q = self.peg.pose.q
        head_offset_local = (
            torch.tensor(
                [0.0, 0.0, -self._peg_length * 0.5],
                device=self.device,
            )
            .unsqueeze(0)
            .expand(peg_q.shape[0], -1)
        )
        head_offset_world = self._rotate_vec(peg_q, head_offset_local)
        return self.peg.pose.p + head_offset_world

    @property
    def estimated_peg_head_pos(self) -> torch.Tensor:
        # Real-hardware reconstruction: FK + calibrated rigid grasp.
        tcp_pose = self.agent.tcp_pose
        peg_q_local = (
            torch.tensor(
                self._peg_initial_orientation_local,
                dtype=torch.float32,
                device=self.device,
            )
            .unsqueeze(0)
            .expand(tcp_pose.q.shape[0], -1)
        )
        peg_q = self._quat_mul(tcp_pose.q, peg_q_local)
        grasp_z = self._peg_length * 0.5 - self._peg_grasp_offset_from_top
        tcp_to_head_local = torch.tensor(
            [0.0, 0.0, -grasp_z - self._peg_length * 0.5],
            dtype=torch.float32,
            device=self.device,
        ).expand(tcp_pose.q.shape[0], -1)
        return tcp_pose.p + self._rotate_vec(peg_q, tcp_to_head_local)

    @property
    def socket_face_pos(self) -> torch.Tensor:
        return self.socket.pose.p + torch.tensor(
            [self._socket_face_x_offset, 0.0, 0.0],
            device=self.device,
        )

    @property
    def measured_socket_pose(self):
        return Pose.create_from_pq(
            p=self.socket.pose.p + self._socket_pose_meas_offset_p,
            q=self.socket.pose.q,
        )

    @property
    def measured_socket_face_pos(self) -> torch.Tensor:
        return self.measured_socket_pose.p + torch.tensor(
            [self._socket_face_x_offset, 0.0, 0.0],
            device=self.device,
        )

    def _get_obs_extra(self, info: dict):
        return dict(
            tcp_pose=self.agent.tcp_pose.raw_pose,
            socket_pose=self.measured_socket_pose.raw_pose,
            estimated_peg_head_to_socket_face=self.estimated_peg_head_pos
            - self.measured_socket_face_pos,
        )

    # ---------------- grasp / alignment helpers ----------------
    def _peg_grasped(self) -> torch.Tensor:
        # Oracle grasp check (uses sim ground-truth peg pose). Used by
        # the dense reward path and reported in ``evaluate`` as the
        # ``peg_grasped`` diagnostic.
        return (
            torch.linalg.norm(self.peg.pose.p - self.agent.tcp_pose.p, dim=-1)
            < self._peg_grasped_max_dist
        )

    def _peg_expected_pos(self) -> torch.Tensor:
        # Where the peg center should be if the rigid grasp model holds.
        # = TCP - rotate(peg_orientation, per-episode grasp offset).
        # Matches the math used to place the peg at episode init, so
        # this should equal the actual peg center under a perfect grasp.
        tcp_p = self.agent.tcp_pose.p
        tcp_q = self.agent.tcp_pose.q
        peg_q_local = (
            torch.tensor(
                self._peg_initial_orientation_local,
                dtype=torch.float32,
                device=self.device,
            )
            .unsqueeze(0)
            .expand(tcp_q.shape[0], -1)
        )
        peg_q = self._quat_mul(tcp_q, peg_q_local)
        return tcp_p - self._rotate_vec(peg_q, self._grasp_offset_p)

    def _peg_slip(self) -> torch.Tensor:
        # Magnitude of deviation between actual peg center and the
        # rigid-grasp-model expected center.
        return torch.linalg.norm(self.peg.pose.p - self._peg_expected_pos(), dim=-1)

    def _slip_factor(self) -> torch.Tensor:
        # Continuous [0, 1] reward attenuation. Deadband region of
        # ``_peg_slip_deadband`` earns full reward (absorbs sim-settling
        # baseline). Above the deadband, linear ramp to 0.0 at
        # ``_peg_slip_zero_dist``.
        effective_slip = (self._peg_slip() - self._peg_slip_deadband).clamp(min=0.0)
        span = max(float(self._peg_slip_zero_dist - self._peg_slip_deadband), 1e-6)
        return (1.0 - effective_slip / span).clamp(0.0, 1.0)

    def _peg_horizontal_alignment(self) -> torch.Tensor:
        # cos(angle) between peg head direction (peg-local -Z) and +X.
        peg_q = self.peg.pose.q
        head_axis_world = self._rotate_vec(
            peg_q,
            torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(
                peg_q.shape[0], -1
            ),
        )
        return head_axis_world[:, 0].clamp(-1.0, 1.0)

    def _peg_horizontal(self) -> torch.Tensor:
        return self._peg_horizontal_alignment() > self._peg_horizontal_min_align

    def _peg_square_yaw_alignment(self) -> torch.Tensor:
        # Project peg-local +X to YZ. 1.0 = face-up, sqrt(0.5) = diamond.
        peg_q = self.peg.pose.q
        ex_world = self._rotate_vec(
            peg_q,
            torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(
                peg_q.shape[0], -1
            ),
        )
        ex_yz = ex_world[:, 1:]
        ex_yz = ex_yz / torch.linalg.norm(ex_yz, dim=-1, keepdim=True).clamp(min=1e-6)
        return torch.maximum(torch.abs(ex_yz[:, 0]), torch.abs(ex_yz[:, 1]))

    def _peg_square_yaw_score(self) -> torch.Tensor:
        # Normalised to [0, 1]: 0 = diamond, 1 = face-up.
        diamond_alignment = np.sqrt(0.5)
        return (
            (self._peg_square_yaw_alignment() - diamond_alignment)
            / (1.0 - diamond_alignment)
        ).clamp(0.0, 1.0)

    def _peg_square_yaw_aligned(self) -> torch.Tensor:
        return self._peg_square_yaw_alignment() > self._peg_square_yaw_min_align

    def _peg_body_yz_error(self) -> torch.Tensor:
        socket_p = self.socket.pose.p
        return torch.linalg.norm(self.peg.pose.p[:, 1:] - socket_p[:, 1:], dim=-1)

    # ---------------- reward helpers ----------------
    def _insertion_goal(self, socket_p: torch.Tensor) -> torch.Tensor:
        # In-hole goal: entry_face + seated_depth in X, socket center
        # in YZ. With seated_depth = socket_depth/2 this is the socket
        # center.
        socket_face_x = socket_p[:, 0] + self._socket_face_x_offset
        goal_x = socket_face_x + self._seated_depth
        return torch.cat([goal_x.unsqueeze(-1), socket_p[:, 1:]], dim=-1)

    def _pre_insertion_reward(
        self,
        head_yz_err: torch.Tensor,
        body_yz_err: torch.Tensor,
        square_yaw_score: torch.Tensor,
    ) -> torch.Tensor:
        # PegInsertionSide-style head+body YZ shaping with max() term
        # so the worse one dominates. Yaw error folded in additively
        # (we need this; PegInsertionSide doesn't because their peg
        # is rotationally symmetric).
        yaw_err = 1.0 - square_yaw_score
        alignment_error = (
            0.5 * (head_yz_err + body_yz_err)
            + 4.5 * torch.maximum(head_yz_err, body_yz_err)
            + self._pre_insertion_yaw_error_weight * yaw_err
        )
        return self._pre_insertion_reward_weight * (1.0 - torch.tanh(alignment_error))

    def _pre_inserted_yz(
        self,
        head_yz_err: torch.Tensor,
        body_yz_err: torch.Tensor,
    ) -> torch.Tensor:
        return (
            (head_yz_err < self._pre_inserted_head_yz_threshold)
            & (body_yz_err < self._pre_inserted_body_yz_threshold)
        ).float()

    def _insertion_score(
        self,
        head: torch.Tensor,
        socket_p: torch.Tensor,
    ) -> torch.Tensor:
        # PegInsertionSide-style: 1 - tanh(k * ||head - goal||).
        # Symmetric 3D Euclidean - overshoot also reduces reward, so
        # the policy converges to the goal point and stops.
        goal = self._insertion_goal(socket_p)
        head_to_goal = torch.linalg.norm(head - goal, dim=-1)
        return 1.0 - torch.tanh(self._insertion_tanh_scale * head_to_goal)

    # ---------------- success / evaluate / reward ----------------
    def has_peg_inserted(self):
        # Sim-to-sim RLPD success criterion. This intentionally uses
        # oracle sim state so sparse labels mean "the peg is actually
        # inserted", not just "the TCP/FK rigid-grasp estimate reached
        # the target." The real-measurable FK proxy is still reported in
        # ``evaluate`` as ``success_real_only`` / ``success_proxy`` so we
        # can quantify false positives before real deployment.
        head = self.peg_head_pos
        socket_p = self.socket.pose.p
        head_yz_err = torch.linalg.norm(head[:, 1:] - socket_p[:, 1:], dim=-1)
        body_yz_err = self._peg_body_yz_error()

        if self.align_only:
            never = torch.zeros(head.shape[0], dtype=torch.bool, device=head.device)
            return never, head_yz_err, body_yz_err, head[:, 0]

        socket_face_x = socket_p[:, 0] + self._socket_face_x_offset
        goal_x = socket_face_x + self._seated_depth
        x_ok = head[:, 0] >= (goal_x - self._success_x_tolerance)
        inner_radius = self._peg_radius + self._socket_clearance
        head_yz_ok = head_yz_err < inner_radius
        body_yz_ok = body_yz_err < self._pre_inserted_body_yz_threshold
        success = (
            self._peg_grasped()
            & x_ok
            & head_yz_ok
            & body_yz_ok
            & self._peg_horizontal()
            & self._peg_square_yaw_aligned()
        )
        return success, head_yz_err, body_yz_err, head[:, 0]

    def evaluate(self):
        # ``success`` is the oracle sim-to-sim label used for RLPD
        # validation. ``success_real_only`` / ``success_proxy`` preserves
        # the FK rigid-grasp success proxy we can compute on hardware.
        success, head_yz_err, body_yz_err, head_x = self.has_peg_inserted()
        head = self.peg_head_pos
        socket_p = self.socket.pose.p
        socket_face_x = socket_p[:, 0] + self._socket_face_x_offset
        depth_past_face = head_x - socket_face_x
        gap_to_face = socket_face_x - head_x

        estimated_head = self.estimated_peg_head_pos
        measured_socket_p = self.measured_socket_pose.p
        measured_socket_face_x = measured_socket_p[:, 0] + self._socket_face_x_offset
        proxy_goal_x = measured_socket_face_x + self._seated_depth
        proxy_head_yz_err = torch.linalg.norm(
            estimated_head[:, 1:] - measured_socket_p[:, 1:], dim=-1
        )
        proxy_x_ok = estimated_head[:, 0] >= (proxy_goal_x - self._success_x_tolerance)
        proxy_yz_ok = proxy_head_yz_err < (self._peg_radius + self._socket_clearance)
        success_proxy = proxy_x_ok & proxy_yz_ok

        # Oracle diagnostic mirrors the active success label with explicit
        # component names for logging and visual inspection.
        oracle_head_yz_err = torch.linalg.norm(head[:, 1:] - socket_p[:, 1:], dim=-1)
        oracle_goal_x = socket_face_x + self._seated_depth
        oracle_inner_radius = self._peg_radius + self._socket_clearance
        oracle_x_ok = head[:, 0] >= (oracle_goal_x - self._success_x_tolerance)
        oracle_head_yz_ok = oracle_head_yz_err < oracle_inner_radius
        oracle_body_yz_ok = body_yz_err < self._pre_inserted_body_yz_threshold
        oracle_grasped = self._peg_grasped()
        oracle_horizontal = self._peg_horizontal()
        oracle_square_yaw_aligned = self._peg_square_yaw_aligned()
        success_oracle = (
            oracle_grasped
            & oracle_x_ok
            & oracle_head_yz_ok
            & oracle_body_yz_ok
            & oracle_horizontal
            & oracle_square_yaw_aligned
        )

        grasped = oracle_grasped
        # Raw gripper-joint readout. Real-hardware-measurable but
        # NOT a reliable binary grasp indicator in this env (see
        # comment near ``_gripper_grasp_qpos``); logged as a scalar
        # so eval can audit gripper drift / catastrophic opens.
        gripper_qpos = self.agent.robot.get_qpos()[:, 5]
        horizontal_alignment = self._peg_horizontal_alignment()
        square_yaw_alignment = self._peg_square_yaw_alignment()
        square_yaw_score = self._peg_square_yaw_score()

        pre_inserted_head_yz = head_yz_err < self._pre_inserted_head_yz_threshold
        pre_inserted_body_yz = body_yz_err < self._pre_inserted_body_yz_threshold
        pre_inserted_yz = self._pre_inserted_yz(head_yz_err, body_yz_err)

        pre_insertion_raw = self._pre_insertion_reward(
            head_yz_err,
            body_yz_err,
            square_yaw_score,
        )
        pre_insertion_score = (
            pre_insertion_raw / max(float(self._pre_insertion_reward_weight), 1e-6)
        ).clamp(0.0, 1.0)

        insertion_score = self._insertion_score(head, socket_p)
        insertion_distance = torch.linalg.norm(
            head - self._insertion_goal(socket_p), dim=-1
        )
        insertion_depth_fraction = (
            depth_past_face / max(float(self._seated_depth), 1e-6)
        ).clamp(0.0, 1.0)

        # ``peg_inserted`` diagnostic: peg has physically entered the
        # hole (head past entry face AND YZ inside the hole footprint).
        inner_radius = self._peg_radius + self._socket_clearance
        peg_inserted = (depth_past_face > 0.005) & (head_yz_err < inner_radius)

        # Slip diagnostics only. ``slip_factor`` is no longer applied
        # to the reward (see ``compute_dense_reward``); kept here as a
        # monitoring signal to detect grasp degradation during contact.
        # ``peg_slipped`` is the strict binary indicator; the wider
        # clearance + chamfer should keep it firing rarely.
        peg_slip = self._peg_slip()
        peg_slipped = peg_slip > self._peg_slip_max_dist
        slip_factor = self._slip_factor()

        qvel_arm = self.agent.robot.get_qvel()[:, :5]
        qvel_arm_l2 = (qvel_arm**2).sum(dim=-1)

        return dict(
            success=success,
            success_oracle=success_oracle,
            success_proxy=success_proxy,
            success_real_only=success_proxy,
            success_seated=(depth_past_face > self._seated_depth),
            success_grasped=grasped,
            success_head_yz_aligned=oracle_head_yz_ok,
            success_body_yz_aligned=oracle_body_yz_ok,
            success_yz_aligned=oracle_head_yz_ok & oracle_body_yz_ok,
            success_horizontal=oracle_horizontal,
            success_square_yaw_aligned=oracle_square_yaw_aligned,
            peg_yz_err=head_yz_err,
            peg_yz_err_proxy=proxy_head_yz_err,
            peg_yz_err_oracle=oracle_head_yz_err,
            peg_body_yz_err=body_yz_err,
            peg_head_x=head_x,
            peg_head_x_proxy=estimated_head[:, 0],
            peg_gap_to_socket_face=gap_to_face,
            insertion_depth=depth_past_face,
            insertion_depth_margin=depth_past_face - self._seated_depth,
            insertion_depth_fraction=insertion_depth_fraction,
            insertion_distance=insertion_distance,
            peg_grasped=grasped,
            gripper_qpos=gripper_qpos,
            peg_horizontal_alignment=horizontal_alignment,
            peg_square_yaw_alignment=square_yaw_alignment,
            peg_square_yaw_score=square_yaw_score,
            pre_inserted_yz=pre_inserted_yz,
            pre_inserted_head_yz=pre_inserted_head_yz,
            pre_inserted_head_yz_margin=self._pre_inserted_head_yz_threshold
            - head_yz_err,
            pre_inserted_body_yz=pre_inserted_body_yz,
            pre_inserted_body_yz_margin=self._pre_inserted_body_yz_threshold
            - body_yz_err,
            peg_inserted=peg_inserted,
            peg_slipped=peg_slipped,
            peg_slip=peg_slip,
            slip_factor=slip_factor,
            pre_insertion_score=pre_insertion_score,
            insertion_score=insertion_score,
            reward_grasp=self._grasp_reward_weight * grasped.float(),
            reward_pre_insertion=pre_insertion_raw * grasped.float(),
            reward_insertion=self._insertion_reward_weight
            * insertion_score
            * pre_inserted_yz
            * grasped.float(),
            qvel_arm_l2=qvel_arm_l2,
        )

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Default (``sparse_reward=True``) RLPD-style reward: zero
        # everywhere, ``_success_bonus`` on the step success fires.
        # This is the only reward we will use during RLPD; the dense
        # fallback below exists only for reproducing the original PPO
        # baseline (``sparse_reward=False``).
        if self.sparse_reward:
            return self._success_bonus * info["success"].float()

        head = self.peg_head_pos
        socket_p = self.socket.pose.p

        # Gate the shaped reward by ``is_grasped`` only -- PegInsertionSide-
        # style. ``slip_factor`` is computed in ``evaluate`` for diagnostics
        # but is NOT applied to the reward, because the slip cascade it
        # produced when insertion began was destabilising training without
        # preventing the destabilisation it was meant to describe. The
        # wider socket clearance + larger chamfer make contact-induced
        # slip mild enough that the rigid grasp observation stays valid
        # in practice without an explicit gate.
        is_grasped = self._peg_grasped().float()

        head_yz_err = torch.linalg.norm(head[:, 1:] - socket_p[:, 1:], dim=-1)
        body_yz_err = self._peg_body_yz_error()
        square_yaw_score = self._peg_square_yaw_score()

        grasp_reward = self._grasp_reward_weight * is_grasped

        pre_insertion_raw = self._pre_insertion_reward(
            head_yz_err,
            body_yz_err,
            square_yaw_score,
        )
        pre_insertion_reward = pre_insertion_raw * is_grasped

        if self.align_only:
            insertion_reward = torch.zeros_like(is_grasped)
        else:
            pre_inserted_yz = self._pre_inserted_yz(head_yz_err, body_yz_err)
            insertion_reward = (
                self._insertion_reward_weight
                * self._insertion_score(head, socket_p)
                * pre_inserted_yz
                * is_grasped
            )

        reward = grasp_reward + pre_insertion_reward + insertion_reward
        reward[info["success"]] = self._success_bonus
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs, action, info) / self._success_bonus
