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
        "max_cte": 7.5,

        "throttle_min": 0.0,
    }    

timesteps = 1_000_000

if __name__ == "__main__":
    env = gym.make("donkey-mountain-track-v0", conf=conf)

    # Train the agent
    model = PPO("CnnPolicy", env, verbose=1)

    # Load most recent checkpoint
    model_path = max(glob.glob('./logs/*.zip'), key=os.path.getctime)
    # model = PPO.load(model_path, env=env, verbose=1)

    # find date and time of saved model
    save_time = os.path.getctime(model_path)
    # convert to human readable time
    save_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(save_time))
    assert type(save_time) == str, type(save_time)
    print("Loaded model from", model_path, "saved at:", save_time)
    
    # callback to save model
    trained_model = model.learn(total_timesteps=timesteps, progress_bar=True, callback=CheckpointCallback(save_freq=15000, save_path='./logs/'))

    # Save the agent
    # trained_model.save("ppo_donkey")
