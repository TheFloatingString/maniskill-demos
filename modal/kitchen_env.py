"""
Custom "kitchen" scene for ManiSkill: a single PartNet-Mobility cabinet with a
prismatic drawer, plus a static countertop prop, both placed at fixed,
hand-authored poses (no scene dataset download, no per-episode randomization
of layout). Registers "KitchenCabinet-v1": a fixed-base Panda arm must pull
the drawer open.

This mirrors the structure of ManiSkill's built-in OpenCabinetDrawer-v1 task
(mani_skill/envs/tasks/mobile_manipulation/open_cabinet_drawer.py) but swaps
the randomized multi-cabinet / mobile-Fetch setup for a single fixed model at
a single fixed pose with a fixed-base Panda arm, so the "scene generation" is
just plain, predictable Python instead of dataset sampling.
"""

from typing import Any, Optional, Union

import numpy as np
import sapien
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.geometry.geometry import transform_points
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Articulation, Link, Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig

CABINET_COLLISION_BIT = 29

# ── Hand-authored kitchen layout: fixed asset choice + fixed poses ─────────
# The cabinet model is the first entry of ManiSkill's shipped PartNet-Mobility
# "cabinet_drawer" manifest (deterministic, not randomly sampled per episode).
TRAIN_JSON = PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_cabinet_drawer_train.json"

CABINET_POS = np.array([0.3, 0.0, 0.0])  # z is overwritten to sit on the floor
COUNTER_POS = np.array([0.3, 0.55, 0.35])
COUNTER_HALF_SIZES = np.array([0.25, 0.3, 0.35])
COUNTER_COLOR = [0.55, 0.45, 0.35, 1.0]
ROBOT_BASE_POS = np.array([-0.5, 0.0, 0.0])


@register_env("KitchenCabinet-v1", max_episode_steps=100)
class KitchenCabinetEnv(BaseEnv):
    """
    **Task Description:**
    A hand-built single-cabinet kitchen scene (fixed PartNet-Mobility model,
    fixed pose, plus a fixed static countertop prop next to it). A Panda arm
    with a fixed base must pull the cabinet's drawer open.

    **Scene generation:**
    Unlike ReplicaCAD/AI2THOR scene datasets, this scene is built from
    scratch in `_load_scene` by placing individual parts (one articulated
    cabinet + one static box "counter") at fixed, predefined poses.

    **Success Conditions:**
    - the drawer's joint qpos reaches at least `min_open_frac` of its range
      and the drawer link has settled (near-zero linear/angular velocity).
    """

    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    min_open_frac = 0.75

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        reconfiguration_freq=None,
        num_envs=1,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        train_data = load_json(TRAIN_JSON)
        # deterministic, single fixed model id -- not randomized per episode
        self.cabinet_model_id = list(train_data.keys())[0]
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0
        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            spacing=5,
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**21, max_rigid_patch_count=2**19
            ),
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.6, -0.7, 0.9], target=[0.3, 0.2, 0.4])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=64,
                height=64,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.1, -1.1, 1.3], [0.3, 0.2, 0.4])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=ROBOT_BASE_POS))

    def _load_scene(self, options: dict):
        self.ground = build_ground(self.scene)
        self.ground.set_collision_group_bit(
            group=2, bit_idx=CABINET_COLLISION_BIT, bit=1
        )

        # static countertop prop at a fixed, hand-picked pose next to the cabinet
        self.counter = actors.build_box(
            self.scene,
            half_sizes=COUNTER_HALF_SIZES,
            color=COUNTER_COLOR,
            name="counter",
            body_type="static",
            initial_pose=sapien.Pose(p=COUNTER_POS),
        )

        # temporarily silence the big red warnings about oblong collision meshes
        sapien.set_log_level("off")
        self._load_cabinets()
        sapien.set_log_level("warn")

    def _load_cabinets(self):
        self._cabinets: list[Articulation] = []
        handle_links: list[Link] = []
        handle_links_mesh_centers: list[np.ndarray] = []

        for i in range(self.num_envs):
            cabinet_builder = articulations.get_articulation_builder(
                self.scene, f"partnet-mobility:{self.cabinet_model_id}"
            )
            cabinet_builder.set_scene_idxs(scene_idxs=[i])
            cabinet_builder.initial_pose = sapien.Pose(p=CABINET_POS, q=[1, 0, 0, 0])
            cabinet = cabinet_builder.build(name=f"cabinet-{i}")
            self.remove_from_state_dict_registry(cabinet)
            for link in cabinet.links:
                link.set_collision_group_bit(
                    group=2, bit_idx=CABINET_COLLISION_BIT, bit=1
                )
            self._cabinets.append(cabinet)

            # deterministically take the first prismatic (drawer) joint/link
            drawer_link = None
            for link, joint in zip(cabinet.links, cabinet.joints):
                if joint.type[0] == "prismatic":
                    drawer_link = link
                    break
            assert drawer_link is not None, (
                f"Cabinet model {self.cabinet_model_id} has no prismatic "
                "(drawer) joint"
            )
            handle_links.append(drawer_link)
            handle_mesh = drawer_link.generate_mesh(
                filter=lambda _, render_shape: "handle" in render_shape.name,
                mesh_name="handle",
            )[0]
            handle_links_mesh_centers.append(handle_mesh.bounding_box.center_mass)

        self.cabinet = Articulation.merge(self._cabinets, name="cabinet")
        self.add_to_state_dict_registry(self.cabinet)
        self.handle_link = Link.merge(handle_links, name="handle_link")
        self.handle_link_pos = common.to_tensor(
            np.array(handle_links_mesh_centers), device=self.device
        )

        self.handle_link_goal = actors.build_sphere(
            self.scene,
            radius=0.02,
            color=[0, 1, 0, 1],
            name="handle_link_goal",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
        )

    def _after_reconfigure(self, options):
        # shift each cabinet so its collision-mesh bottom sits exactly at z=0
        self.cabinet_zs = []
        for cabinet in self._cabinets:
            collision_mesh = cabinet.get_first_collision_mesh()
            self.cabinet_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.cabinet_zs = common.to_tensor(self.cabinet_zs, device=self.device)

        target_qlimits = self.handle_link.joint.limits  # [b, 1, 2]
        qmin, qmax = target_qlimits[..., 0], target_qlimits[..., 1]
        self.target_qpos = qmin + (qmax - qmin) * self.min_open_frac

    def handle_link_positions(self, env_idx: Optional[torch.Tensor] = None):
        if env_idx is None:
            return transform_points(
                self.handle_link.pose.to_transformation_matrix().clone(),
                common.to_tensor(self.handle_link_pos, device=self.device),
            )
        return transform_points(
            self.handle_link.pose[env_idx].to_transformation_matrix().clone(),
            common.to_tensor(self.handle_link_pos[env_idx], device=self.device),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.tensor(CABINET_POS[:2])
            xyz[:, 2] = self.cabinet_zs[env_idx]
            self.cabinet.set_pose(Pose.create_from_pq(p=xyz))

            # close all drawers -- lower qlimit means "closed" for these assets
            qlimits = self.cabinet.get_qlimits()  # [b, max_dof, 2]
            self.cabinet.set_qpos(qlimits[env_idx, :, 0])
            self.cabinet.set_qvel(self.cabinet.qpos[env_idx] * 0)

            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene.px.step()
                self.scene._gpu_fetch_all()

            self.handle_link_goal.set_pose(
                Pose.create_from_pq(p=self.handle_link_positions(env_idx))
            )

            # fixed-base panda reset (same keyframe/noise pattern TableSceneBuilder uses)
            qpos = np.array(
                [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4, 0.04, 0.04]
            )
            qpos = (
                self._episode_rng.normal(0, self.robot_init_qpos_noise, (b, len(qpos)))
                + qpos
            )
            qpos[:, -2:] = 0.04
            self.agent.reset(qpos)
            self.agent.robot.set_pose(sapien.Pose(p=ROBOT_BASE_POS))

    def evaluate(self):
        open_enough = self.handle_link.joint.qpos >= self.target_qpos
        handle_link_pos = self.handle_link_positions()
        link_is_static = (
            torch.linalg.norm(self.handle_link.angular_velocity, axis=1) <= 1
        ) & (torch.linalg.norm(self.handle_link.linear_velocity, axis=1) <= 0.1)
        return {
            "success": open_enough & link_is_static,
            "handle_link_pos": handle_link_pos,
            "open_enough": open_enough,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                tcp_to_handle_pos=info["handle_link_pos"] - self.agent.tcp.pose.p,
                target_link_qpos=self.handle_link.joint.qpos,
                target_handle_pos=info["handle_link_pos"],
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        tcp_to_handle_dist = torch.linalg.norm(
            self.agent.tcp.pose.p - info["handle_link_pos"], axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_handle_dist)
        amount_to_open_left = torch.div(
            self.target_qpos - self.handle_link.joint.qpos, self.target_qpos
        )
        open_reward = 2 * (1 - amount_to_open_left)
        reaching_reward[amount_to_open_left < 0.999] = 2
        open_reward[info["open_enough"]] = 3
        reward = reaching_reward + open_reward
        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        max_reward = 5.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
