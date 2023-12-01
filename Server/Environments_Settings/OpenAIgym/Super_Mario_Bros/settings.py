from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

Episode_Timeout = 5000  # Must be either None\False\0 or a positive number.
Screen_Buffer_Pixel_Values_Scaling = True
Env_ID = "SuperMarioBros-v0"
RewardScalingEnabled = False  # scales the reward so that they are always between -1 and 1. (theoretically not
#                              necessary, but greatly improves performance with most learning algorithms)
Scenario_Max_Reward_Per_Episode = None
Scenario_Min_Reward_Per_Episode = None
Mean_Normalization_Enabled = False  # reduces the mean reward from each reward
Mean_Step_Reward = None  # The average step reward


def special_env_wrapper_func(env):
    """ The wrapper limits the control options to the seven that are relevant to the game.
        It is originally all all ascii. (reduces action space of environment from 256 possible actions to only 7)"""
    return JoypadSpace(env, SIMPLE_MOVEMENT)
