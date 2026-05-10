"""
SAC with RGBD Observations — PushCube-v1 on Modal

Prerequisites:
  1. modal secret create wandb-secret WANDB_API_KEY=<your-key>
  2. modal run modal/sac_pushcube.py
"""

import modal

app = modal.App("sac-pushcube")

volume = modal.Volume.from_name("maniskill-runs", create_if_missing=True)
RUNS_DIR = "/runs"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install([
        "libvulkan-dev", "libvulkan1", "wget",
        "libgl1", "libglib2.0-0", "libglfw3",
        "clang",  # toppra (mani_skill dep) requires clang for Cython extension
    ])
    .run_commands(
        "mkdir -p /etc/vulkan/icd.d",
        # libEGL_nvidia.so.0 is the headless NVIDIA Vulkan interface present in CUDA
        # containers. GLX (libGLX_nvidia.so.0) is not available without a display server.
        "echo '{\"file_format_version\":\"1.0.0\",\"ICD\":{\"library_path\":\"libEGL_nvidia.so.0\",\"api_version\":\"1.3.277\"}}'"
        " > /etc/vulkan/icd.d/nvidia_icd.json",
    )
    .pip_install(["mani_skill", "tyro", "wandb", "tensorboard", "tqdm"])
    .run_commands(
        "mkdir -p /opt/sac",
        "wget -q -O /opt/sac/sac_rgbd.py"
        " https://raw.githubusercontent.com/haosulab/ManiSkill/main/examples/baselines/sac/sac_rgbd.py",
    )
)

# ── Configuration ─────────────────────────────────────────────────────────────
ENV_ID          = "PushCube-v1"
OBS_MODE        = "rgb"
INCLUDE_STATE   = True
CONTROL_MODE    = "pd_ee_delta_pos"
CAMERA_WIDTH    = 64
CAMERA_HEIGHT   = 64

NUM_ENVS        = 8
NUM_EVAL_ENVS   = 16
TOTAL_TIMESTEPS = 1_000_000
BUFFER_SIZE     = 300_000
BATCH_SIZE      = 512
LEARNING_STARTS = 4_000
UTD             = 0.5
TRAINING_FREQ   = 64
EVAL_FREQ       = 10_000
NUM_EVAL_STEPS  = 250
LOG_FREQ        = 1_000
VIDEO_LOG_FREQ  = 10_000

GAMMA           = 0.8
TAU             = 0.01
POLICY_LR       = 3e-4
Q_LR            = 3e-4
AUTOTUNE        = True
ALPHA           = 0.2
BUFFER_DEVICE   = "cuda"
SEED            = 1
CAPTURE_VIDEO   = True
SAVE_MODEL      = True

TRACK           = True
WANDB_PROJECT   = "ManiSkill"
WANDB_ENTITY    = None
WANDB_GROUP     = "SAC"


@app.function(
    image=image,
    gpu="L4",
    timeout=6 * 3600,
    volumes={RUNS_DIR: volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train():
    import os
    import sys
    import random
    import time
    import glob
    from collections import defaultdict

    os.environ["MANI_SKILL_DATA_DIR"] = "/root/.maniskill/data"
    sys.path.insert(0, "/opt/sac")

    import tqdm
    import gymnasium as gym
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter
    import wandb

    from mani_skill.utils import gym_utils
    from mani_skill.utils.wrappers.flatten import (
        FlattenActionSpaceWrapper,
        FlattenRGBDObservationWrapper,
    )
    from mani_skill.utils.wrappers.record import RecordEpisode
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    import mani_skill.envs

    import sac_rgbd
    sac_rgbd.wandb = wandb
    from sac_rgbd import Actor, SoftQNetwork, ReplayBuffer, Logger

    # Set VK_ICD_FILENAMES after all imports so sapien's _vulkan_tricks.py
    # (which runs on import and sets its own ICD) cannot override our choice.
    # libEGL_nvidia.so.0 is the headless NVIDIA Vulkan EGL interface present in CUDA
    # containers; it provides the real L4 GPU device with CUDA-Vulkan interop, which
    # RenderSystemGroup requires for batched RGB observations with sim_backend="gpu".
    import sapien.render as _sr
    os.environ["VK_ICD_FILENAMES"] = "/etc/vulkan/icd.d/nvidia_icd.json"
    print(f"[vulkan] VK_ICD_FILENAMES -> {os.environ['VK_ICD_FILENAMES']}")

    _OrigRS = _sr.RenderSystem
    def _rs_cpu_factory(device="", *args, **kwargs):
        # Pass no device (nullptr → findBestRenderDevice) instead of "cuda:0".
        # findDevice("cuda:0") matches by cudaId; findDevice("") unconditionally throws.
        # With the EGL ICD, findBestRenderDevice picks the real L4 GPU (highest priority:
        # DISCRETE_GPU + CUDA match), enabling CUDA-Vulkan interop in RenderSystemGroup.
        print(f"[sapien] RenderSystem({device!r}) -> using findBestRenderDevice()")
        return _OrigRS()
    _sr.RenderSystem = _rs_cpu_factory

    # ── Derived config ────────────────────────────────────────────────────────
    GRAD_STEPS       = int(TRAINING_FREQ * UTD)
    STEPS_PER_ENV    = TRAINING_FREQ // NUM_ENVS
    RUN_NAME         = f"{ENV_ID}__sac_rgbd__{SEED}__{int(time.time())}"
    run_dir          = f"{RUNS_DIR}/{RUN_NAME}"
    eval_output_dir  = f"{run_dir}/videos"
    os.makedirs(eval_output_dir, exist_ok=True)
    print(f"Run: {RUN_NAME}  |  grad_steps/iter: {GRAD_STEPS}  |  steps_per_env: {STEPS_PER_ENV}")

    # ── Seeding + device ──────────────────────────────────────────────────────
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env_kwargs = dict(
        obs_mode=OBS_MODE,
        render_mode="all",
        sim_backend="gpu",
        control_mode=CONTROL_MODE,
        sensor_configs=dict(width=CAMERA_WIDTH, height=CAMERA_HEIGHT),
    )

    envs      = gym.make(ENV_ID, num_envs=NUM_ENVS, **env_kwargs)
    eval_envs = gym.make(
        ENV_ID, num_envs=NUM_EVAL_ENVS,
        human_render_camera_configs=dict(shader_pack="default"),
        **env_kwargs,
    )

    use_rgb   = OBS_MODE in ("rgb", "rgbd")
    use_depth = OBS_MODE in ("depth", "rgbd")
    envs      = FlattenRGBDObservationWrapper(envs,      rgb=use_rgb, depth=use_depth, state=INCLUDE_STATE)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=use_rgb, depth=use_depth, state=INCLUDE_STATE)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs      = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    eval_envs = RecordEpisode(
        eval_envs, output_dir=eval_output_dir,
        save_trajectory=False, save_video=CAPTURE_VIDEO,
        trajectory_name="trajectory",
        max_steps_per_video=NUM_EVAL_STEPS, video_fps=30,
    )

    envs      = ManiSkillVectorEnv(envs,      NUM_ENVS,      ignore_terminations=True, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, NUM_EVAL_ENVS, ignore_terminations=True, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box)

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    print(f"Max episode steps: {max_episode_steps}")

    # ── Networks ──────────────────────────────────────────────────────────────
    obs, _      = envs.reset(seed=SEED)
    eval_obs, _ = eval_envs.reset(seed=SEED)

    actor   = Actor(envs, sample_obs=obs).to(device)
    qf1     = SoftQNetwork(envs, actor.encoder).to(device)
    qf2     = SoftQNetwork(envs, actor.encoder).to(device)
    qf1_tgt = SoftQNetwork(envs, actor.encoder).to(device)
    qf2_tgt = SoftQNetwork(envs, actor.encoder).to(device)
    qf1_tgt.load_state_dict(qf1.state_dict())
    qf2_tgt.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(
        list(qf1.mlp.parameters()) + list(qf2.mlp.parameters()) + list(qf1.encoder.parameters()),
        lr=Q_LR,
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=POLICY_LR)

    if AUTOTUNE:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha      = torch.zeros(1, requires_grad=True, device=device)
        alpha          = log_alpha.exp().item()
        a_optimizer    = optim.Adam([log_alpha], lr=Q_LR)
    else:
        alpha     = ALPHA
        log_alpha = None

    # ── WandB + TensorBoard ───────────────────────────────────────────────────
    config = dict(
        env_id=ENV_ID, obs_mode=OBS_MODE, control_mode=CONTROL_MODE,
        camera_width=CAMERA_WIDTH, camera_height=CAMERA_HEIGHT,
        num_envs=NUM_ENVS, total_timesteps=TOTAL_TIMESTEPS,
        buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
        learning_starts=LEARNING_STARTS, utd=UTD,
        gamma=GAMMA, tau=TAU, policy_lr=POLICY_LR, q_lr=Q_LR,
        autotune=AUTOTUNE, seed=SEED, env_horizon=max_episode_steps,
    )
    if TRACK:
        wandb.init(
            project=WANDB_PROJECT, entity=WANDB_ENTITY,
            sync_tensorboard=False, config=config,
            name=RUN_NAME, save_code=True,
            group=WANDB_GROUP, tags=["sac", "rgbd"],
        )

    writer = SummaryWriter(f"{run_dir}/tb")
    logger = Logger(log_wandb=TRACK, tensorboard=writer)

    # ── Replay buffer ─────────────────────────────────────────────────────────
    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        env=envs, num_envs=NUM_ENVS, buffer_size=BUFFER_SIZE,
        storage_device=torch.device(BUFFER_DEVICE), sample_device=device,
    )
    print(f"Actor params: {sum(p.numel() for p in actor.parameters()):,}")

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step           = 0
    global_update         = 0
    learning_has_started  = False
    cumulative_times      = defaultdict(float)
    global_steps_per_iter = NUM_ENVS * STEPS_PER_ENV
    pbar = tqdm.tqdm(total=TOTAL_TIMESTEPS, desc="Training")

    while global_step < TOTAL_TIMESTEPS:

        # Evaluation
        if EVAL_FREQ > 0 and (global_step - TRAINING_FREQ) // EVAL_FREQ < global_step // EVAL_FREQ:
            actor.eval()
            stime    = time.perf_counter()
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            for _ in range(NUM_EVAL_STEPS):
                with torch.no_grad():
                    eval_obs, _, _, _, eval_infos = eval_envs.step(actor.get_eval_action(eval_obs))
                if "final_info" in eval_infos:
                    for k, v in eval_infos["final_info"]["episode"].items():
                        eval_metrics[k].append(v)
            eval_means = {k: torch.stack(v).float().mean() for k, v in eval_metrics.items()}
            for k, v in eval_means.items():
                logger.add_scalar(f"eval/{k}", v, global_step)
            eval_time = time.perf_counter() - stime
            cumulative_times["eval_time"] += eval_time
            logger.add_scalar("time/eval_time", eval_time, global_step)
            pbar.set_postfix(
                success=f"{eval_means.get('success_once', torch.tensor(0)):.2f}",
                ret=f"{eval_means.get('return', torch.tensor(0)):.2f}",
            )
            actor.train()

            if TRACK and VIDEO_LOG_FREQ > 0 and (global_step - TRAINING_FREQ) // VIDEO_LOG_FREQ < global_step // VIDEO_LOG_FREQ:
                videos = sorted(glob.glob(f"{eval_output_dir}/*.mp4"))
                if videos:
                    wandb.log({"eval/video": wandb.Video(videos[-1], fps=30, format="mp4")}, step=global_step)

            if SAVE_MODEL:
                torch.save(
                    {"actor": actor.state_dict(), "qf1": qf1_tgt.state_dict(),
                     "qf2": qf2_tgt.state_dict(), "log_alpha": log_alpha},
                    f"{run_dir}/ckpt_{global_step}.pt",
                )
                volume.commit()

        # Rollout
        t_rollout = time.perf_counter()
        for _ in range(STEPS_PER_ENV):
            global_step += NUM_ENVS
            if not learning_has_started:
                actions = 2 * torch.rand(envs.action_space.shape, device=device) - 1
            else:
                with torch.no_grad():
                    actions, _, _, _ = actor.get_action(obs)
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            logger.add_scalar("train/reward", rewards.mean().item(), global_step)

            real_next_obs = {k: v.clone() for k, v in next_obs.items()}
            need_final_obs = truncations | terminations
            stop_bootstrap = torch.zeros_like(terminations, dtype=torch.bool)
            if "final_info" in infos:
                for k in real_next_obs:
                    real_next_obs[k][need_final_obs] = infos["final_observation"][k][need_final_obs].clone()
                done_mask = infos["_final_info"]
                for k, v in infos["final_info"]["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
            rb.add(obs, real_next_obs, actions, rewards, stop_bootstrap)
            obs = next_obs
        rollout_time = time.perf_counter() - t_rollout
        cumulative_times["rollout_time"] += rollout_time
        pbar.update(NUM_ENVS * STEPS_PER_ENV)

        if global_step < LEARNING_STARTS:
            continue

        # Gradient updates
        t_update = time.perf_counter()
        learning_has_started = True
        for _ in range(GRAD_STEPS):
            global_update += 1
            data = rb.sample(BATCH_SIZE)
            o, no, act, rew, done = data.obs, data.next_obs, data.actions, data.rewards, data.dones

            with torch.no_grad():
                na, nlogpi, _, vf = actor.get_action(no)
                next_q = rew.flatten() + (1 - done.flatten()) * GAMMA * (
                    torch.min(qf1_tgt(no, na, vf), qf2_tgt(no, na, vf)) - alpha * nlogpi
                ).view(-1)

            vf_obs  = actor.encoder(o)
            q1_val  = qf1(o, act, vf_obs).view(-1)
            q2_val  = qf2(o, act, vf_obs).view(-1)
            qf1_loss = F.mse_loss(q1_val, next_q)
            qf2_loss = F.mse_loss(q2_val, next_q)
            q_optimizer.zero_grad()
            (qf1_loss + qf2_loss).backward()
            q_optimizer.step()

            pi, log_pi, _, vf_obs = actor.get_action(o)
            actor_loss = (
                (alpha * log_pi) - torch.min(
                    qf1(o, pi, vf_obs, detach_encoder=True),
                    qf2(o, pi, vf_obs, detach_encoder=True),
                ).view(-1)
            ).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            if AUTOTUNE:
                with torch.no_grad():
                    _, log_pi, _, _ = actor.get_action(o)
                alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                a_optimizer.zero_grad()
                alpha_loss.backward()
                a_optimizer.step()
                alpha = log_alpha.exp().item()

            for p, tp in zip(qf1.parameters(), qf1_tgt.parameters()):
                tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)
            for p, tp in zip(qf2.parameters(), qf2_tgt.parameters()):
                tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

        update_time = time.perf_counter() - t_update
        cumulative_times["update_time"] += update_time

        # Logging
        if (global_step - TRAINING_FREQ) // LOG_FREQ < global_step // LOG_FREQ:
            for tag, val in [
                ("losses/qf1_values", q1_val.mean().item()),
                ("losses/qf2_values", q2_val.mean().item()),
                ("losses/qf1_loss",   qf1_loss.item()),
                ("losses/qf2_loss",   qf2_loss.item()),
                ("losses/qf_loss",    (qf1_loss + qf2_loss).item() / 2),
                ("losses/actor_loss", actor_loss.item()),
                ("losses/alpha",      alpha),
                ("time/update_time",  update_time),
                ("time/rollout_time", rollout_time),
                ("time/rollout_fps",  global_steps_per_iter / rollout_time),
            ]:
                logger.add_scalar(tag, val, global_step)
            for k, v in cumulative_times.items():
                logger.add_scalar(f"time/total_{k}", v, global_step)
            if AUTOTUNE:
                logger.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    pbar.close()

    # ── Save final checkpoint ─────────────────────────────────────────────────
    if SAVE_MODEL:
        model_path = f"{run_dir}/final_ckpt.pt"
        torch.save(
            {"actor": actor.state_dict(), "qf1": qf1_tgt.state_dict(),
             "qf2": qf2_tgt.state_dict(), "log_alpha": log_alpha},
            model_path,
        )
        print(f"Model saved to {model_path}")
        volume.commit()

    logger.close()
    if TRACK:
        wandb.finish()
    envs.close()
    eval_envs.close()


@app.local_entrypoint()
def main():
    train.remote()
