# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#


from env import JackalEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo import MultiInputPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
import torch as th

log_dir = "./multiinput_policy/policy_test"
# set headless to false to visualize training
my_env = JackalEnv(headless=True)


policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[16, dict(pi=[128, 64], vf=[128, 64])]) # 64 32 -> 128 64
policy = MultiInputPolicy
total_timesteps = 2500000

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="jackal_policy_checkpoint")
model = PPO(
    policy,
    my_env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    n_steps=10000,
    batch_size=1000,
    learning_rate=0.00025,
    gamma=0.9995,
    device="cuda",
    ent_coef=0,
    vf_coef=0.5,
    max_grad_norm=10,
    tensorboard_log=log_dir,
)
model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback])

model.save(log_dir + "/jackal_policy")

my_env.close()
