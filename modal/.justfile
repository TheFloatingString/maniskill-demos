# Modal training recipes for ManiSkill SAC baselines.
# Requires: `modal token set` configured, and `modal secret create wandb-secret WANDB_API_KEY=<key>`.

set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

# List available recipes.
default:
    @just --list

# Train a single-task SAC policy (see sac_pushcube.py for available task names).
pushcube task="PushCube-v1":
    modal run sac_pushcube.py --task {{task}}

# Train a single SAC policy alternating between a pair of tasks (see TASK_PAIRS in sac_alternate.py).
alternate pair="push-pull":
    modal run sac_alternate.py --pair {{pair}}

# List logs / tail a running Modal app.
logs app:
    modal app logs {{app}}
