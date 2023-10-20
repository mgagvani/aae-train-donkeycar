import gym
from gym import envs
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import gym_donkeycar
from stable_baselines3.ppo import *
import glob
import os
import time

conf = {
        "body_style": "cybertruck",
        "body_rgb": (64, 224, 117),
        "car_name": "PPO",
        "font_size": 100,
        "racer_name": "Manav Gagvani",
        "country": "USA",
        "bio": ":)",
        "max_cte": 10,
    }    

timesteps = 10_000

if __name__ == "__main__":
    env = gym.make("donkey-mountain-track-v0", conf=conf)

    # Train the agent
    # model = PPO("CnnPolicy", env, verbose=1)

    # Load most recent checkpoint
    model_path = max(glob.glob('./logs/*.zip'), key=(timestamp:=os.path.getctime))
    model = PPO.load(model_path, env=env, verbose=1)

    # TODO
    # print("Loaded model from: ", model_path, " saved at: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp)))
    
    # callback to save model
    trained_model = model.learn(total_timesteps=timesteps, progress_bar=True, callback=CheckpointCallback(save_freq=500, save_path='./logs/'))

    # Save the agent
    # trained_model.save("ppo_donkey")
