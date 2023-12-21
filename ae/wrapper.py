import os
from typing import Any, Dict, Tuple

import cv2
import gym
import numpy as np

from ae.autoencoder import load_ae


class AutoencoderWrapper(gym.Wrapper):
    """
    Wrapper to encode input image using the pre-trained autoencoder

    :param env: Gym environment
    :param ae_path: Path to the pre-trained autoencoder
    """
    def __init__(self, env: gym.Env, ae_path: str = os.environ["AE_PATH"]):
        super().__init__(env)
        self.autoencoder = load_ae(ae_path)
        z_size = self.autoencoder.z_size
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(z_size + 1,), 
            dtype=np.float32,
        )

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        # Convert to BGR
        encoded_image = self.autoencoder.encode_from_raw_image(obs[:,:,::-1])
        new_obs = np.concatenate([encoded_image.flatten(), [0.0]])
        return new_obs.flatten()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        # Encode with the pre-trained autoencoder
        encoded_image = self.autoencoder.encode_from_raw_image(obs[:,:,::-1])
        # reconstructed_image = self.autoencoder.decode(encoded_image)[0]
        # cv2.imshow("original", obs[:,:,::-1])
        # cv2.imshow("reconstructed_image", reconstructed_image)
        # k = cv2.waitKey(0) & 0xFF
        # if k == 27:
        #     pass
        speed = info["speed"]
        new_obs = np.concatenate([encoded_image.flatten(), [speed]])
        assert new_obs.shape == self.autoencoder.z_size + 1

        return new_obs.flatten(), reward, done, info
