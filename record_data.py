"""
Simple script to manually record data
by driving around.
"""
import argparse
import math
import os
import threading
from typing import Tuple

import cv2
import gym
import gym_donkeycar  # noqa: F401
import numpy as np
import pygame
from inputs import get_gamepad  # noqa: F401
from pygame.locals import *  # noqa: F403

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", help="Path to folder where images will be saved", type=str, required=True)
parser.add_argument("-n", "--max-steps", help="Max number of steps", type=int, default=10000)
args = parser.parse_args()

UP = (1, 0)
LEFT = (0, 1)
RIGHT = (0, -1)
DOWN = (-1, 0)

MAX_TURN = 1
MAX_THROTTLE = 0.5
# Smoothing constants
STEP_THROTTLE = 0.8
STEP_TURN = 0.8

frame_skip = 2
total_frames = args.max_steps
render = True
output_folder = args.folder

# Create folder if needed
os.makedirs(output_folder, exist_ok=True)

# from: https://github.com/mgagvani/Xbox-Game-AI/blob/141611a02cea857c7ca5e06c3c0cb234cb9cc356/utils.py#L107
class XboxController(object):
    MAX_TRIG_VAL = math.pow(2, 8)
    MAX_JOY_VAL = math.pow(2, 15)

    def __init__(self):

        self.LeftJoystickY = 0
        self.LeftJoystickX = 0
        self.RightJoystickY = 0
        self.RightJoystickX = 0
        self.LeftTrigger = 0
        self.RightTrigger = 0
        self.LeftBumper = 0
        self.RightBumper = 0
        self.A = 0
        self.X = 0
        self.Y = 0
        self.B = 0
        self.LeftThumb = 0
        self.RightThumb = 0
        self.Back = 0
        self.Start = 0
        self.LeftDPad = 0
        self.RightDPad = 0
        self.UpDPad = 0
        self.DownDPad = 0

        self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()


    def read(self):
        L_X = self.LeftJoystickX
        L_Y = self.LeftJoystickY
        R_X = self.RightJoystickX
        R_Y = self.RightJoystickY
        LT = self.LeftTrigger
        RT = self.RightTrigger
        LB = self.LeftBumper
        RB = self.RightBumper
        A = self.A
        X = self.X
        Y = self.Y
        B = self.B
        LTh = self.LeftThumb
        RTh = self.RightThumb
        Back = self.Back
        Start = self.Start
        # dpad does not work
        DP_L = self.LeftDPad
        DP_R = self.RightDPad
        DP_U = self.UpDPad
        DP_D = self.DownDPad

        # return [L_X, L_Y, R_X, R_Y, RT]
        return [L_X, L_Y, R_X, R_Y, LT, RT, LB, RB, A, X, Y, B, LTh, RTh, Back, Start]
        # return [L_X, L_Y, R_X, R_Y, RT]


    def _monitor_controller(self):
        while True:
            events = get_gamepad()
            for event in events:
                if event.code == 'ABS_Y':
                    self.LeftJoystickY = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_X':
                    self.LeftJoystickX = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_RY':
                    self.RightJoystickY = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_RX':
                    self.RightJoystickX = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_Z':
                    self.LeftTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1
                elif event.code == 'ABS_RZ':
                    self.RightTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1
                elif event.code == 'BTN_TL':
                    self.LeftBumper = event.state
                elif event.code == 'BTN_TR':
                    self.RightBumper = event.state
                elif event.code == 'BTN_SOUTH':
                    self.A = event.state
                elif event.code == 'BTN_NORTH':
                    self.X = event.state
                elif event.code == 'BTN_WEST':
                    self.Y = event.state
                elif event.code == 'BTN_EAST':
                    self.B = event.state
                elif event.code == 'BTN_THUMBL':
                    self.LeftThumb = event.state
                elif event.code == 'BTN_THUMBR':
                    self.RightThumb = event.state
                elif event.code == 'BTN_SELECT':
                    self.Back = event.state
                elif event.code == 'BTN_START':
                    self.Start = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY1':
                    self.LeftDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY2':
                    self.RightDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY3':
                    self.UpDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY4':
                    self.DownDPad = event.state


def control(
    x,
    theta: float,
    control_throttle: float,
    control_steering: float,
) -> Tuple[float, float]:
    """
    Smooth control.

    :param x:
    :param theta:
    :param control_throttle:
    :param control_steering:
    :return:
    """
    target_throttle = x * MAX_THROTTLE
    target_steering = MAX_TURN * theta
    if target_throttle > control_throttle:
        control_throttle = min(target_throttle, control_throttle + STEP_THROTTLE)
    elif target_throttle < control_throttle:
        control_throttle = max(target_throttle, control_throttle - STEP_THROTTLE)
    else:
        control_throttle = target_throttle

    if target_steering > control_steering:
        control_steering = min(target_steering, control_steering + STEP_TURN)
    elif target_steering < control_steering:
        control_steering = max(target_steering, control_steering - STEP_TURN)
    else:
        control_steering = target_steering
    return control_throttle, control_steering


# pytype: disable=name-error
moveBindingsGame = {K_UP: UP, K_LEFT: LEFT, K_RIGHT: RIGHT, K_DOWN: DOWN}  # noqa: F405
WHITE = (230, 230, 230)
pygame.font.init()
FONT = pygame.font.SysFont("Open Sans", 25)

pygame.init()
window = pygame.display.set_mode((400, 400), RESIZABLE)
# pytype: enable=name-error

control_throttle, control_steering = 0, 0

conf = {
        "body_style": "cybertruck",
        "body_rgb": (64, 224, 117),
        "car_name": "Manual Drive",
        "font_size": 100,
        "racer_name": "Manav Gagvani",
        "country": "USA",
        "bio": ":)",
        "max_cte": 7.5,

        "throttle_min": 0.0,
} 

env = gym.make("donkey-mountain-track-v0", conf=conf)
obs = env.reset()

controller = XboxController()

for frame_num in range(total_frames):
    x, theta = 0, 0
    # Record pressed keys
    # keys = pygame.key.get_pressed()
    # for keycode in moveBindingsGame.keys():
    #     if keys[keycode]:
    #         x_tmp, th_tmp = moveBindingsGame[keycode]
    #         x += x_tmp
    #         theta += th_tmp   
        
    # Read controller
    controller_input = controller.read()
    L_X, L_Y, R_X, R_Y, LT, RT, LB, RB, A, X, Y, B, LTh, RTh, Back, Start = controller_input
    x = RT
    theta = L_X

    if B == 1:
        env.reset()
        control_throttle, control_steering = 0, 0

    # Smooth control for teleoperation
    control_throttle, control_steering = control(x, theta, control_throttle, control_steering)

    window.fill((0, 0, 0))
    pygame.display.flip()
    # Limit FPS
    # pygame.time.Clock().tick(1 / TELEOP_RATE)
    for event in pygame.event.get():
        if (event.type == QUIT or event.type == KEYDOWN) and event.key in [  # pytype: disable=name-error
            K_ESCAPE,  # pytype: disable=name-error
            K_q,  # pytype: disable=name-error
        ]:
            env.close()
            exit()

    window.fill((0, 0, 0))
    text = "Control ready"
    text = FONT.render(text, True, WHITE)
    window.blit(text, (100, 100))
    pygame.display.flip()

    # steer, throttle
    action = np.array([control_steering, control_throttle])

    for _ in range(frame_skip):
        obs, _rew, done, _info = env.step(action)
        # print(_rew, done, _info)
        if done:
            break
    if render:
        env.render()
    path = os.path.join(output_folder, f"{frame_num}.jpg")
    # Convert to BGR
    cv2.imwrite(path, obs[:, :, ::-1])
    if done:
        obs = env.reset()
        control_throttle, control_steering = 0, 0
