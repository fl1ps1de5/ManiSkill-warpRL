"""Downward peg insertion on SO101 with a pre-grasped peg.

The reward design mirrors ``PegInsertionSide-v1`` but adapted to a
downward insertion axis:

1. ``grasp_reward``: keep the dynamic peg grasped.
2. ``pre_insertion_reward``: shape peg head/body XY toward the socket
   center and shape square-yaw alignment. Crucially this term has no
   target on the insertion (Z) axis, so it never fights descent. Once
   the peg head is physically inside the socket hole (head_z below
   the lip by ``_post_insertion_z_threshold`` AND head_xy within
   ``_peg_radius + _socket_clearance``) this term is frozen at its
   max value, because the socket -- not the gripper -- then owns the
   peg's XY, and any gripper-driven XY corrections at that point
   torque the grasp against the socket walls and eject the peg. The
   XY component of the condition is essential: a Z-only freeze lets
   the policy ram the peg past the lip with bad alignment and
   collect free max pre_insertion.
3. ``insertion_reward``: a single tanh distance from the peg head to
   the in-hole goal point ``(socket_xy, socket_top_z - seated_depth)``,
   gated by a binary ``pre_inserted_xy`` threshold + grasped, so the
   policy must align before any descent credit is paid.

Yaw is handled as continuous dense shaping (square peg vs. diamond),
not as a strict gate, since the SO101 wrist tends to twist slightly
during fast XY moves and a hard yaw gate stalls learning.

Per-step arm actions are scaled down inside ``_step_action`` to promote
smaller, smoother motion that is closer to what the real SO101 can
follow.

Observations are intended to be reconstructable on the real robot:
joint state from encoders, TCP pose from FK, a calibrated socket pose,
and a TCP-derived estimate of the peg head relative to the socket top.

Use ``control_mode="pd_joint_target_delta_pos"`` so a zero action holds
the current joint target against gravity.
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


SO101_PEG_INSERTION_ENV_ID = "SO101PegInsertion-v1"


def _build_box_with_hole(
    scene: "ManiSkillScene",
    inner_radius: float,
    outer_radius: float,
    depth: float,
    chamfer_depth: float = 0.004,
    chamfer_extra_radius: float = 0.0025,
):
    """Build a box-shaped socket with a centered, optionally chamfered hole.

    The hole is along the +Z axis (the "downward insertion" axis is -Z, so
    the policy descends into the +Z mouth of the socket from above).

    Geometry: four wall slabs around the hole opening, with an extra short
    chamfer ring sitting on top of the slabs to give insertion a forgiving
    funnel without modifying the cylindrical hole proper.
    """
    builder = scene.create_actor_builder()
    thickness = (outer_radius - inner_radius) * 0.5
    half_d = depth * 0.5

    # Four straight wall slabs forming the cylindrical-ish hole.
    half_sizes = [
        [thickness, outer_radius, half_d],
        [thickness, outer_radius, half_d],
        [outer_radius, thickness, half_d],
        [outer_radius, thickness, half_d],
    ]
    offset = thickness + inner_radius
    poses = [
        sapien.Pose([offset, 0, 0]),
        sapien.Pose([-offset, 0, 0]),
        sapien.Pose([0, offset, 0]),
        sapien.Pose([0, -offset, 0]),
    ]

    base_mat = sapien.render.RenderMaterial(
        base_color=sapien_utils.hex2rgba("#E59B2A"),
        roughness=0.6,
        specular=0.3,
    )

    for half_size, pose in zip(half_sizes, poses):
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size, material=base_mat)

    # Chamfer ring: a slightly wider mouth on top of the slabs. Cheap to
    # simulate (still box geometry) and gives the policy a forgiving funnel.
    if chamfer_depth > 0.0 and chamfer_extra_radius > 0.0:
        chamfer_inner = inner_radius + chamfer_extra_radius
        chamfer_thickness = (outer_radius - chamfer_inner) * 0.5
        if chamfer_thickness > 0.0:
            half_cd = chamfer_depth * 0.5
            chamfer_z = half_d + half_cd  # sits on top of the slabs
            chamfer_offset = chamfer_thickness + chamfer_inner
            chamfer_half_sizes = [
                [chamfer_thickness, outer_radius, half_cd],
                [chamfer_thickness, outer_radius, half_cd],
                [outer_radius, chamfer_thickness, half_cd],
                [outer_radius, chamfer_thickness, half_cd],
            ]
            chamfer_poses = [
                sapien.Pose([chamfer_offset, 0, chamfer_z]),
                sapien.Pose([-chamfer_offset, 0, chamfer_z]),
                sapien.Pose([0, chamfer_offset, chamfer_z]),
                sapien.Pose([0, -chamfer_offset, chamfer_z]),
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


@register_env(SO101_PEG_INSERTION_ENV_ID, max_episode_steps=100)
class SO101PegInsertionEnv(BaseEnv):
    """Pre-grasped downward peg insertion on the SO101 arm.

    **Task description**

    The peg starts pre-grasped in the gripper above a box-with-hole socket
    fixed to the table. The policy commands 5-DOF joint position
    deltas to align the peg over the socket hole and descend until the
    peg head reaches a seated z threshold.

    **Observation** (state-only, proprioception + known socket pose)

    * ``qpos`` (6,)            -- arm + gripper joint positions
    * ``qvel`` (6,)            -- arm + gripper joint velocities
    * ``tcp_pose`` (7,)        -- end-effector pose (xyz + quat)
    * ``socket_pose`` (7,)     -- measured socket pose (with optional
        calibration noise)
    * ``peg_pose_in_tcp`` (7,) -- peg pose expressed in TCP frame
    * ``rel_peg_to_socket`` (3,) -- xyz offset from peg head to socket
        mouth in world frame (handy precomputed feature for small MLPs)

    **Success**: peg head z is below the seated threshold AND the peg
    head xy is within the socket hole radius (so a wildly-misaligned
    descent does not trigger success).
    """

    SUPPORTED_ROBOTS = ["so101"]
    agent: Union[SO101]

    # Physical defaults; the dyn subclass scales these via init kwargs.
    _peg_radius = 0.0075  # 15 mm peg diameter
    _peg_length = 0.06  # 60 mm peg length
    _peg_density = 1000.0  # ~13.5 g for the body
    # How far below the peg's *top* edge the gripper grasps it.
    # 0 = gripper at the very top (entire peg sticks out below);
    # ``peg_length`` = gripper at the head (entire peg above gripper).
    # A value slightly past peg_length/2 places the gripper just
    # below the peg centre. With the vertical start this leaves
    # ~2.5 cm of peg below the gripper, raising the insertion head
    # clear of the socket top during lateral motion while still
    # leaving enough shaft to enter the hole.
    _peg_grasp_offset_from_top = 0.025
    # Gripper qpos commanded at episode start. The SO101 gripper joint
    # is revolute with range [-0.175, 2.094] rad; the URDF "rest"
    # keyframe is at 1.047 rad (open), and *lower* values close the
    # jaws further. We command -0.17 rad (just shy of the lower
    # limit -0.175) so the PD drives the jaw toward maximum closure,
    # the peg between the jaws mechanically prevents the target from
    # being reached, and the resulting persistent error generates a
    # continuous strong compressive torque clamping the peg.
    _gripper_grasp_qpos = -0.05
    # Tight radial clearance (0.5 mm) makes misaligned insertion
    # physically impossible -- a misaligned peg hits the lip and stops
    # rather than squeezing in skew. Pairs with the chamfer to convert
    # the descent-through-lip event from an impulsive yaw-perturbing
    # contact into a guided slide. Successful insertions are then
    # *automatically* well-aligned, which is the property we want.
    _socket_clearance = 0.0025  # 0.5 mm radial clearance
    _socket_outer_radius = 0.045  # half-extent of the box
    _socket_depth = 0.030  # 30 mm hole depth
    # Keep the socket as a plain box with a hole. A chamfer/funnel can
    # make insertion easier, but it also adds extra top geometry that
    # the pre-grasped peg can contact during lateral approach.
    # Small chamfer "funnel" sitting on top of the socket walls. It does
    # not change the cylindrical hole, ``_socket_top_z_offset``, or any
    # reward/success threshold -- only mechanically guides the peg head
    # into the lip during the final mm of descent. This reduces the
    # impulsive lip-contact event that destabilises the gripper.
    _socket_chamfer_depth = 0.003
    _socket_chamfer_extra_radius = 0.002

    # Workspace bounds for socket-pose randomization (table frame).
    # x range deliberately starts at 0.16 so the closest socket edge
    # (at x = low_x - outer_radius = 0.115 m) sits ~11 cm in front
    # of the robot base origin. The SO101 base link in the URDF has
    # only visual geometry (no collision), so a kinematic socket
    # spawned on top of the base passes through it visually -- which
    # was producing the "robot and socket overlapping" reports.
    #
    # y range is shifted to +y so the socket is always offset in +y
    # from the home TCP (which sits at world y ~= -0.008). With
    # outer_radius = 0.045 and y_low = 0.06, the closest socket edge
    # in y is at world y = 0.015, ~2.3 cm clear of the home TCP --
    # so the peg never spawns over the socket walls and the policy
    # has to translate to the socket before descending. This also
    # makes "approach-then-insert" the natural learned behaviour
    # rather than something the policy can short-cut by descending
    # straight from home.
    _socket_xy_low = (0.16, 0.06)
    _socket_xy_high = (0.22, 0.10)
    _socket_z = 0.0  # base of socket sits on the table top

    # Initial peg orientation in tcp-local frame (wxyz quaternion).
    # Identity = peg held vertically along the approach direction,
    # which keeps the base task focused on translate + descend rather
    # than requiring PPO to first solve a non-trivial wrist-orientation
    # curriculum. The grasp point above is set near the peg centre so
    # the insertion head still starts above the socket object during
    # lateral movement.
    _peg_initial_orientation_local = (1.0, 0.0, 0.0, 0.0)

    # Target insertion depth measured downward from socket lip to peg
    # head. Used by the success criterion and by the insertion-reward
    # goal point. The socket hole is deeper (``_socket_depth``) so the
    # bottom of the hole is below the seated target, leaving headroom
    # for the policy to overshoot slightly without colliding with the
    # socket floor.
    _seated_depth = 0.015

    # ---------------- reward design ----------------
    # Max shaped reward is approximately:
    #   grasp (1) + pre_insertion (3) + insertion (5) = 9
    # before small smoothness penalties. On success the reward is
    # overwritten with a fixed ``_success_bonus``, keeping the terminal
    # quality target slightly above the best shaped value.

    # Stage 0: keep the peg grasped.
    _grasp_reward_weight = 1.0

    # Stage 1: pre-insertion alignment. Pull peg head + body XY toward
    # the socket center and continuously shape square-yaw alignment.
    # There is deliberately *no* height target on the insertion axis,
    # so this term cannot compete with the descent of stage 2.
    _pre_insertion_reward_weight = 3.0
    # Yaw error contribution inside the tanh. The SO101 wrist tilts
    # slightly under fast XY motion, so we incentivise yaw through
    # dense shaping rather than a hard gate. With weight 4.0, a fully
    # diamond peg (yaw error = 1.0) contributes the same as ~5 mm of
    # XY error to the tanh argument.
    _pre_insertion_yaw_error_weight = 4.0

    # Stage 2: insertion. PegInsertionSide-style two-condition reward:
    #   - Binary XY gate (``pre_inserted_xy``): paid only when both
    #     head_xy_err < ``_pre_inserted_head_xy_threshold`` and
    #     body_xy_err < ``_pre_inserted_body_xy_threshold``. This
    #     produces strong alignment-first pressure -- the policy gets
    #     ~5 reward by clearing the gate, and 0 otherwise.
    #   - Tanh distance toward the in-hole goal, with the Z component
    #     clipped from below so head_z at or below goal_z contributes
    #     0 (no over-insertion incentive, no pull into the socket
    #     floor). XY contribution comes from the head_xy part of the
    #     Euclidean norm; verticality is shaped by the binary gate
    #     using ``body_xy_err`` rather than continuously in the tanh.
    # ``is_grasped`` is also a hard multiplier; grasp loss is fatal.
    _insertion_reward_weight = 5.0
    _insertion_tanh_scale = 5.0

    # XY thresholds shared by the binary ``pre_inserted_xy`` gate on
    # ``insertion_reward``, the ``success_xy_aligned`` success
    # condition, and the diagnostics. A single source of truth keeps
    # the dense reward, success, and logs aligned.
    _pre_inserted_head_xy_threshold = 0.005
    _pre_inserted_body_xy_threshold = 0.007

    # The peg is considered "inside the socket" -- and therefore
    # geometrically constrained by the socket walls rather than by
    # the gripper -- when its head is below the lip by at least
    # ``_post_insertion_z_threshold`` AND its head XY is within
    # ``_peg_radius + _socket_clearance`` of the socket center
    # (i.e. physically inside the hole, not just past the lip plane).
    # While that condition holds, ``pre_insertion_reward`` is frozen
    # at its max so the policy doesn't keep trying to gripper-correct
    # XY against the socket walls (which torques the grasp out).
    #
    # Both conditions are required. Z-only would allow the policy to
    # ram the peg past the lip without alignment (the head can be
    # below the lip while being radially outside the hole, contacting
    # the chamfer/lip), gaming the freeze for max pre_insertion
    # without ever aligning.
    _post_insertion_z_threshold = 0.005

    # Smoothness shaping. Kept light because per-step actions are also
    # scaled (see ``_action_scale``), which already removes much of the
    # high-frequency motion that motivated heavier penalties before.
    _action_rate_l2_weight = 1e-2
    _joint_vel_l2_weight = 1e-2

    # Per-step arm action scaling. The PD delta controller commands at
    # +-0.1 rad per step; with ``_action_scale = 0.3`` the realised
    # delta is +-0.03 rad. Slower per-step motion reduces the
    # impulsive contact force at the descent-through-lip moment,
    # which was identified as the dominant source of yaw and
    # alignment instability in earlier runs.
    _action_scale = 0.3

    _success_bonus = 10.0

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
        **kwargs,
    ) -> None:
        self.socket_pose_noise_std = float(socket_pose_noise_std)
        self.home_pose_noise_std = float(home_pose_noise_std)
        self.grasp_offset_noise_std = float(grasp_offset_noise_std)
        self.arm_only_action = bool(arm_only_action)
        self._sim_config_override = sim_config
        # Curriculum knob. When ``align_only`` is True:
        #   * ``insertion_reward`` is zeroed out (no descent gradient).
        #   * ``has_peg_inserted`` always returns False, so the
        #     ``_success_bonus`` overwrite never fires.
        # The dense reward then reduces to grasp + pre_insertion +
        # smoothness, so the policy is pushed to a pose that maxes
        # head/body XY centering and square yaw alignment above the
        # socket, with no incentive to attempt descent. Use for
        # phase 1 of a two-phase curriculum, then resume training
        # without ``align_only`` to add the insertion stage.
        self.align_only = bool(align_only)
        # Lazy-initialized buffer for the previous action, used by the
        # action_rate_l2 smoothness penalty. We don't know action_dim
        # at __init__ time (no agent yet), so we materialise on first
        # call to ``compute_dense_reward``.
        self._last_action: torch.Tensor | None = None
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

        # Low, close view across the socket mouth so the peg/socket gap
        # is visible during approach and early insertion.
        pose = sapien_utils.look_at([0.30, 0.18, 0.09], [0.19, 0.08, 0.025])
        return CameraConfig("render_camera", pose, 512, 512, 0.75, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0.0, 0.0, 0.0]))

    def _load_scene(self, options: dict):
        with torch.device(self.device):
            self.table_scene = TableSceneBuilder(self)
            self.table_scene.build()

            # ---- peg actor (dynamic, with collision shapes) ----
            # Clean rectangular prism, same dynamics shape as the
            # peg in ``PegInsertionSide-v1``. Held in the gripper by
            # a friction grasp -- see ``_initialize_episode``.
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
            # Visual-only color split: the insertion head is peg-local -z,
            # matching ``peg_head_pos`` below. Collision remains one box.
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
            peg_builder.initial_pose = sapien.Pose(p=[0.15, 0.0, 0.3])
            self.peg = peg_builder.build("peg")

            # ---- socket actor (kinematic, plain box-with-hole) ----
            inner_radius = self._peg_radius + self._socket_clearance
            socket_builder = _build_box_with_hole(
                self.scene,
                inner_radius=inner_radius,
                outer_radius=self._socket_outer_radius,
                depth=self._socket_depth,
                chamfer_depth=self._socket_chamfer_depth,
                chamfer_extra_radius=self._socket_chamfer_extra_radius,
            )
            socket_builder.initial_pose = sapien.Pose(
                p=[
                    0.5 * (self._socket_xy_low[0] + self._socket_xy_high[0]),
                    0.5 * (self._socket_xy_low[1] + self._socket_xy_high[1]),
                    self._socket_z + self._socket_depth * 0.5,
                ]
            )
            self.socket = socket_builder.build_kinematic("socket")

            # Cached buffers we use during reset / reward / step.
            self._socket_inner_radius = float(inner_radius)
            self._socket_top_z_offset = self._socket_depth * 0.5
            # Per-env state filled in during _initialize_episode.
            self._grasp_offset_p = torch.zeros((self.num_envs, 3), device=self.device)
            self._socket_pose_meas_offset_p = torch.zeros(
                (self.num_envs, 3), device=self.device
            )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # ---- socket: random xy inside workspace box ----
            low = torch.tensor([self._socket_xy_low[0], self._socket_xy_low[1]])
            high = torch.tensor([self._socket_xy_high[0], self._socket_xy_high[1]])
            socket_xy = randomization.uniform(low=low, high=high, size=(b, 2))
            socket_p = torch.zeros((b, 3))
            socket_p[:, :2] = socket_xy
            socket_p[:, 2] = self._socket_z + self._socket_depth * 0.5
            self.socket.set_pose(
                Pose.create_from_pq(
                    p=socket_p,
                    q=torch.tensor([1.0, 0.0, 0.0, 0.0]).unsqueeze(0).repeat(b, 1),
                )
            )

            # Calibration noise on the *measured* socket pose. The true
            # socket pose is the ground-truth above; the policy sees
            # ``socket_pose`` = true + noise.
            if self.socket_pose_noise_std > 0.0:
                noise = torch.randn((b, 3)) * self.socket_pose_noise_std
            else:
                noise = torch.zeros((b, 3))
            self._socket_pose_meas_offset_p[env_idx] = noise

            # ---- robot home pose with optional joint-bias DR ----
            # The gripper qpos is set to ``_gripper_grasp_qpos`` (slightly
            # tighter than the peg diameter) so that once the peg sits
            # between the jaws the gripper PD applies a constant
            # closing torque against the peg.
            qpos_home = torch.tensor(SO101.keyframes["start"].qpos, dtype=torch.float32)
            qpos_home[5] = float(self._gripper_grasp_qpos)
            qpos_home = qpos_home.unsqueeze(0).repeat(b, 1)
            if self.home_pose_noise_std > 0.0:
                qpos_home = qpos_home + torch.randn_like(qpos_home) * float(
                    self.home_pose_noise_std
                )
            self.agent.robot.set_qpos(qpos_home)
            self.agent.robot.set_pose(sapien.Pose([0, 0, 0]))

            # Sync GPU state so the subsequent ``self.agent.tcp_pose`` read
            # sees the applied home pose. On the CPU backend ``set_qpos``
            # is immediate; on physx_cuda the call only queues the change,
            # so without this block the TCP read below would be stale and
            # the peg would be placed at the previous TCP location
            # (the "peg placed from stale TCP, falls to floor" symptom
            # on GPU sim).
            if self.device.type == "cuda":
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene._gpu_fetch_all()

            # ---- grasp offset: peg-in-TCP-frame, with optional noise ----
            # ``peg_p_world = TCP - rotate(tcp_q, [0, 0, grasp_z])``.
            # ``grasp_z = peg_length/2 - peg_grasp_offset_from_top``
            # places the gripper-grasp line ``peg_grasp_offset_from_top``
            # below the peg's top edge, so part of the peg is inside
            # the gripper and the rest sticks out below.
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
            # The robot qpos has already been written, so ``tcp_pose``
            # reflects the home configuration. We place the peg so
            # that its grasp-point (peg-local +z = grasp_offset_z)
            # coincides with the TCP, with the peg oriented in TCP-
            # local frame according to ``_peg_initial_orientation_local``.
            #
            # Note: we rotate ``grasp_offset`` (a peg-local vector) by
            # ``peg_q`` (NOT ``tcp_q``). This makes the placement code
            # correct even if we later reintroduce a non-identity
            # peg-in-TCP orientation: the grasp offset is a peg-local
            # vector and must be rotated by the actual peg orientation.
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
            # Reset linear and angular velocity so the peg starts
            # at rest in the gripper rather than inheriting whatever
            # transient state it was in across episodes.
            zero_vec = torch.zeros((b, 3), device=self.device)
            self.peg.set_linear_velocity(zero_vec)
            self.peg.set_angular_velocity(zero_vec)

            # Reset previous-action buffer for resetting envs so the
            # first step of the new episode doesn't see a spurious
            # large action_rate from comparing to the *previous*
            # episode's last action. Buffer is lazy-initialized in
            # ``compute_dense_reward``; nothing to do here on the
            # very first reset (before any action has been taken).
            if self._last_action is not None:
                self._last_action[env_idx] = 0.0

    # -------------------------------------------------------------------
    # Per-step gripper clamp (peg-grasp is handled by physics, not by
    # any per-step pose override).
    # -------------------------------------------------------------------
    def _step_action(self, action):
        # Two adjustments to the raw policy action:
        #   1. Scale arm dimensions by ``_action_scale`` to cap the
        #      realised joint-target delta below the controller bound.
        #      This produces smaller, smoother motion in sim and is
        #      closer to what the real SO101 can track.
        #   2. Force gripper-action dim to 0 so the policy can't open
        #      the gripper. Matches the "5-DOF arm only, gripper held
        #      closed" contract.
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

    # -------------------------------------------------------------------
    # Helpers and observable derived quantities.
    # -------------------------------------------------------------------
    @staticmethod
    def _rotate_vec(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        # quat: (B, 4) wxyz; vec: (B, 3). Returns rotated vector.
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
        """Multiply two batches of wxyz quaternions: q_out = q1 * q2.

        Composition order is the standard "rotate by q2, then by q1"
        i.e. ``R(q1*q2) v = R(q1) R(q2) v``. We use this to compose
        the world-frame TCP orientation with a tcp-local rotation:
        ``peg_q_world = tcp_q_world * peg_q_in_tcp``.
        """
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack([w, x, y, z], dim=-1)

    @property
    def peg_head_pos(self) -> torch.Tensor:
        # Peg head is the insertion-tip face, located ``peg_length/2``
        # along peg-local -z from the peg center. With a dynamic peg
        # we read the actual peg pose (which physics produced) rather
        # than deriving it from the TCP, so this reflects the true
        # location even if the peg has shifted in the gripper.
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
        # Real-hardware reconstruction of peg-head position from FK and
        # the calibrated rigid grasp. This intentionally does *not* read
        # the dynamic peg actor, since the real robot will not have that
        # privileged state.
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
    def socket_top_pos(self) -> torch.Tensor:
        return self.socket.pose.p + torch.tensor(
            [0.0, 0.0, self._socket_top_z_offset],
            device=self.device,
        )

    @property
    def measured_socket_pose(self):
        true_p = self.socket.pose.p
        return Pose.create_from_pq(
            p=true_p + self._socket_pose_meas_offset_p,
            q=self.socket.pose.q,
        )

    @property
    def measured_socket_top_pos(self) -> torch.Tensor:
        return self.measured_socket_pose.p + torch.tensor(
            [0.0, 0.0, self._socket_top_z_offset],
            device=self.device,
        )

    # -------------------------------------------------------------------
    # Observation, evaluation, reward.
    # -------------------------------------------------------------------
    def _get_obs_extra(self, info: dict):
        tcp_pose = self.agent.tcp_pose
        measured_socket = self.measured_socket_pose
        estimated_head = self.estimated_peg_head_pos

        obs: dict[str, Any] = dict(
            tcp_pose=tcp_pose.raw_pose,
            socket_pose=measured_socket.raw_pose,
            estimated_peg_head_to_socket_top=estimated_head
            - self.measured_socket_top_pos,
        )
        return obs

    # ---------------- exploit guards ----------------
    # Reward / success guards that prevent PPO from "winning" by
    # squeezing the peg out of the gripper and letting it land in the
    # socket xy region. Without these guards, the only success
    # criterion is "peg head is below the socket-top threshold AND
    # within socket xy", which is trivially achievable by ejecting
    # the peg from the gripper above the socket.

    # Tightened from 0.05 to 0.02: nominal grasp distance is ~1cm
    # (peg center is _peg_grasp_offset_from_top below TCP), so 2cm
    # allows ~1cm of slip before failing the grasp check. This catches
    # the "TCP overshot" failure mode where the policy descends past
    # the optimal point and the peg gets squeezed up out of the
    # gripper jaws -- the lenient 5cm threshold previously let success
    # remain True briefly during this overshoot, contributing to the
    # success_at_end << success_once gap.
    _peg_grasped_max_dist = 0.02
    _peg_vertical_min_align = 0.7  # cos > 0.7 -> within ~45 deg of vertical
    # Square peg/socket yaw alignment, modulo 90 degrees. 0.9 allows
    # about 25 degrees of yaw error, matching the approximate geometric
    # clearance of a 15mm square peg in a 20mm square hole.
    _peg_square_yaw_min_align = 0.9

    def _peg_grasped(self) -> torch.Tensor:
        return (
            torch.linalg.norm(self.peg.pose.p - self.agent.tcp_pose.p, dim=-1)
            < self._peg_grasped_max_dist
        )

    def _peg_vertical_alignment(self) -> torch.Tensor:
        peg_q = self.peg.pose.q
        ez_world = self._rotate_vec(
            peg_q,
            torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(
                peg_q.shape[0], -1
            ),
        )
        return ez_world[:, 2].clamp(0.0, 1.0)

    def _peg_vertical(self) -> torch.Tensor:
        return self._peg_vertical_alignment() > self._peg_vertical_min_align

    def _peg_square_yaw_alignment(self) -> torch.Tensor:
        """Square-peg yaw alignment score, invariant to 90deg rotations.

        ``1.0`` means the peg cross-section is square-aligned with the
        axis-aligned socket; ``sqrt(0.5)`` means it is at 45 degrees
        (diamond relative to square). This checks rotation around the
        peg's long axis, while ``_peg_vertical`` checks tilt.
        """
        peg_q = self.peg.pose.q
        ex_world = self._rotate_vec(
            peg_q,
            torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(
                peg_q.shape[0], -1
            ),
        )
        ex_xy = ex_world[:, :2]
        ex_xy = ex_xy / torch.linalg.norm(ex_xy, dim=-1, keepdim=True).clamp(min=1e-6)
        return torch.maximum(torch.abs(ex_xy[:, 0]), torch.abs(ex_xy[:, 1]))

    def _peg_square_yaw_score(self) -> torch.Tensor:
        diamond_alignment = np.sqrt(0.5)
        return (
            (self._peg_square_yaw_alignment() - diamond_alignment)
            / (1.0 - diamond_alignment)
        ).clamp(0.0, 1.0)

    def _peg_square_yaw_aligned(self) -> torch.Tensor:
        return self._peg_square_yaw_alignment() > self._peg_square_yaw_min_align

    def _peg_body_xy_error(self) -> torch.Tensor:
        socket_p = self.socket.pose.p
        return torch.linalg.norm(self.peg.pose.p[:, :2] - socket_p[:, :2], dim=-1)

    def _insertion_goal(
        self,
        socket_p: torch.Tensor,
        socket_top_z: torch.Tensor,
    ) -> torch.Tensor:
        # In-hole goal point for the peg head: directly under the socket
        # center, at the seated depth. This is the *only* target along
        # the insertion (Z) axis used by the reward.
        goal_z = socket_top_z - self._seated_depth
        return torch.cat([socket_p[:, :2], goal_z.unsqueeze(-1)], dim=-1)

    def _pre_insertion_reward(
        self,
        head_xy_err: torch.Tensor,
        body_xy_err: torch.Tensor,
        square_yaw_score: torch.Tensor,
    ) -> torch.Tensor:
        # PegInsertionSide-style XY alignment shaping with yaw error
        # folded into the tanh argument. Verticality is *not* added: it
        # is already implicit because head and body XY are both pulled
        # to the same socket center.
        yaw_err = 1.0 - square_yaw_score
        alignment_error = (
            0.5 * (head_xy_err + body_xy_err)
            + 4.5 * torch.maximum(head_xy_err, body_xy_err)
            + self._pre_insertion_yaw_error_weight * yaw_err
        )
        return self._pre_insertion_reward_weight * (1.0 - torch.tanh(alignment_error))

    def _pre_inserted_xy(
        self,
        head_xy_err: torch.Tensor,
        body_xy_err: torch.Tensor,
    ) -> torch.Tensor:
        return (
            (head_xy_err < self._pre_inserted_head_xy_threshold)
            & (body_xy_err < self._pre_inserted_body_xy_threshold)
        ).float()

    def _insertion_score(
        self,
        head: torch.Tensor,
        socket_p: torch.Tensor,
        socket_top_z: torch.Tensor,
    ) -> torch.Tensor:
        # ``1 - tanh(k * ||head - goal||_asym)``. The Z component is
        # clipped from below so head_z at or below goal_z contributes
        # zero distance (no over-insertion incentive, no gradient
        # pulling the peg into the socket floor). XY is the natural
        # Euclidean component. This term is *paid only* when the
        # binary ``pre_inserted_xy`` gate is satisfied at the call
        # site, so alignment-first pressure comes from the gate rather
        # than from a continuous XY weighting inside this tanh.
        goal = self._insertion_goal(socket_p, socket_top_z)
        head_diff = head - goal
        head_diff_z_clipped = head_diff[:, 2].clamp(min=0.0)
        head_diff_clipped = torch.stack(
            [head_diff[:, 0], head_diff[:, 1], head_diff_z_clipped],
            dim=-1,
        )
        head_to_goal = torch.linalg.norm(head_diff_clipped, dim=-1)
        return 1.0 - torch.tanh(self._insertion_tanh_scale * head_to_goal)

    def has_peg_inserted(self):
        head = self.peg_head_pos
        socket_p = self.socket.pose.p
        socket_top_z = socket_p[:, 2] + self._socket_top_z_offset

        head_xy_err = torch.linalg.norm(head[:, :2] - socket_p[:, :2], dim=-1)
        body_xy_err = self._peg_body_xy_error()
        if self.align_only:
            # Phase-1 curriculum: success never triggers, so the
            # ``_success_bonus`` overwrite never fires and the policy
            # has no incentive to attempt descent.
            never = torch.zeros(head.shape[0], dtype=torch.bool, device=head.device)
            return never, head_xy_err, body_xy_err, head[:, 2]
        seated = head[:, 2] < (socket_top_z - self._seated_depth)
        grasped = self._peg_grasped()
        xy_ok = (head_xy_err < self._pre_inserted_head_xy_threshold) & (
            body_xy_err < self._pre_inserted_body_xy_threshold
        )
        success = seated & grasped & xy_ok
        return success, head_xy_err, body_xy_err, head[:, 2]

    def evaluate(self):
        success, head_xy_err, body_xy_err, head_z = self.has_peg_inserted()
        head = self.peg_head_pos
        socket_p = self.socket.pose.p
        socket_top_z = socket_p[:, 2] + self._socket_top_z_offset
        height_above_lip = head_z - socket_top_z
        insertion_depth = socket_top_z - head_z

        grasped = self._peg_grasped()
        vertical_alignment = self._peg_vertical_alignment()
        square_yaw_alignment = self._peg_square_yaw_alignment()
        square_yaw_score = self._peg_square_yaw_score()

        pre_inserted_head_xy = head_xy_err < self._pre_inserted_head_xy_threshold
        pre_inserted_body_xy = body_xy_err < self._pre_inserted_body_xy_threshold
        pre_inserted_xy = self._pre_inserted_xy(head_xy_err, body_xy_err)

        pre_insertion_raw = self._pre_insertion_reward(
            head_xy_err,
            body_xy_err,
            square_yaw_score,
        )
        pre_insertion_score = (
            pre_insertion_raw / max(float(self._pre_insertion_reward_weight), 1e-6)
        ).clamp(0.0, 1.0)

        insertion_score = self._insertion_score(head, socket_p, socket_top_z)
        insertion_distance = torch.linalg.norm(
            head - self._insertion_goal(socket_p, socket_top_z), dim=-1
        )
        insertion_depth_fraction = (
            insertion_depth / max(float(self._seated_depth), 1e-6)
        ).clamp(0.0, 1.0)

        inner_radius = self._peg_radius + self._socket_clearance
        peg_inserted = (insertion_depth > self._post_insertion_z_threshold) & (
            head_xy_err < inner_radius
        )
        pre_insertion_term_paid = torch.where(
            peg_inserted,
            torch.full_like(
                pre_insertion_raw, float(self._pre_insertion_reward_weight)
            ),
            pre_insertion_raw,
        )

        qvel_arm = self.agent.robot.get_qvel()[:, :5]
        qvel_arm_l2 = (qvel_arm**2).sum(dim=-1)

        return dict(
            success=success,
            # Per-stage success components for diagnosing whether the
            # failure mode is geometric (xy) or kinematic (grasp/depth).
            success_seated=(insertion_depth > self._seated_depth),
            success_grasped=grasped,
            success_xy_aligned=(pre_inserted_xy > 0.5),
            # Pose diagnostics.
            peg_xy_err=head_xy_err,
            peg_body_xy_err=body_xy_err,
            peg_head_z=head_z,
            peg_height_above_socket=height_above_lip,
            insertion_depth=insertion_depth,
            insertion_depth_margin=insertion_depth - self._seated_depth,
            insertion_depth_fraction=insertion_depth_fraction,
            insertion_distance=insertion_distance,
            # Alignment diagnostics.
            peg_grasped=grasped,
            peg_vertical_alignment=vertical_alignment,
            peg_square_yaw_alignment=square_yaw_alignment,
            peg_square_yaw_score=square_yaw_score,
            pre_inserted_xy=pre_inserted_xy,
            pre_inserted_head_xy=pre_inserted_head_xy,
            pre_inserted_head_xy_margin=self._pre_inserted_head_xy_threshold
            - head_xy_err,
            pre_inserted_body_xy=pre_inserted_body_xy,
            pre_inserted_body_xy_margin=self._pre_inserted_body_xy_threshold
            - body_xy_err,
            # Stage diagnostic: True once the peg head is past the lip
            # by ``_post_insertion_z_threshold``. While True, the
            # ``pre_insertion_reward`` paid in compute_dense_reward is
            # frozen at its max, so the policy stops trying to
            # gripper-correct XY against the socket walls.
            peg_inserted=peg_inserted,
            # Reward diagnostics.
            pre_insertion_score=pre_insertion_score,
            insertion_score=insertion_score,
            reward_grasp=self._grasp_reward_weight * grasped.float(),
            reward_pre_insertion=pre_insertion_term_paid * grasped.float(),
            reward_insertion=self._insertion_reward_weight
            * insertion_score
            * pre_inserted_xy
            * grasped.float(),
            qvel_arm_l2=qvel_arm_l2,
        )

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        head = self.peg_head_pos
        socket_p = self.socket.pose.p
        socket_top_z = socket_p[:, 2] + self._socket_top_z_offset

        # Three-term additive reward, PegInsertionSide-style:
        #   grasp         - keep the peg grasped
        #   pre_insertion - shape head+body XY centering and yaw.
        #                   FROZEN at its max value once the peg is
        #                   committed past the lip, because the socket
        #                   (not the gripper) then owns peg XY -- any
        #                   gripper-driven XY corrections at that point
        #                   torque the grasp into the socket walls and
        #                   eject the peg.
        #   insertion     - tanh distance to the in-hole goal (Z asym),
        #                   gated by a binary XY-alignment threshold so
        #                   no insertion credit is paid until the peg
        #                   is centered above the hole.
        is_grasped = self._peg_grasped().float()
        head_xy_err = torch.linalg.norm(head[:, :2] - socket_p[:, :2], dim=-1)
        body_xy_err = self._peg_body_xy_error()
        square_yaw_score = self._peg_square_yaw_score()
        # Physical inside-socket predicate: peg head is past the lip
        # AND within the socket's inner radius. Z-only would let the
        # policy ram the peg past the lip with bad XY and collect the
        # frozen max pre_insertion (the bug observed in the previous
        # run). Both axes are needed for the freeze to be principled.
        inner_radius = self._peg_radius + self._socket_clearance
        inserted = (head[:, 2] < (socket_top_z - self._post_insertion_z_threshold)) & (
            head_xy_err < inner_radius
        )

        grasp_reward = self._grasp_reward_weight * is_grasped
        pre_insertion_raw = self._pre_insertion_reward(
            head_xy_err,
            body_xy_err,
            square_yaw_score,
        )
        # Once inserted, replace shaped pre-insertion with its max so
        # the policy sees "this stage is solved, just hold grasp".
        pre_insertion_term = torch.where(
            inserted,
            torch.full_like(
                pre_insertion_raw, float(self._pre_insertion_reward_weight)
            ),
            pre_insertion_raw,
        )
        pre_insertion_reward = pre_insertion_term * is_grasped
        pre_inserted_xy = self._pre_inserted_xy(head_xy_err, body_xy_err)
        if self.align_only:
            # Phase-1 curriculum: no descent gradient at all.
            insertion_reward = torch.zeros_like(is_grasped)
        else:
            insertion_reward = (
                self._insertion_reward_weight
                * self._insertion_score(head, socket_p, socket_top_z)
                * pre_inserted_xy
                * is_grasped
            )

        # Smoothness penalties. Lazy-init the previous-action buffer the
        # first time we see an action shape.
        if self._last_action is None or self._last_action.shape != action.shape:
            self._last_action = torch.zeros_like(action)
        action_rate = action - self._last_action
        action_rate_penalty = -self._action_rate_l2_weight * (action_rate**2).sum(
            dim=-1
        )
        qvel_arm = self.agent.robot.get_qvel()[:, :5]
        joint_vel_penalty = -self._joint_vel_l2_weight * (qvel_arm**2).sum(dim=-1)
        self._last_action = action.detach().clone()

        reward = (
            grasp_reward
            + pre_insertion_reward
            + insertion_reward
            + action_rate_penalty
            + joint_vel_penalty
        )
        # Match PegInsertionSide-v1: success is a fixed terminal-quality
        # reward, not a bonus stacked on top of shaping terms.
        reward[info["success"]] = self._success_bonus
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs, action, info) / self._success_bonus

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return info["success"].float()


__all__ = ["SO101PegInsertionEnv", "SO101_PEG_INSERTION_ENV_ID"]
