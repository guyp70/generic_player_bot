import os

curr_dir_path = os.path.dirname(__file__)  # relative to the settings file
Episode_Timeout = None  # Must be either None\False\0 or a positive number.
Screen_Buffer_Pixel_Values_Scaling = True
VizDoom_CFG_File = os.path.join(curr_dir_path, "basic.cfg")
VizDoom_WAD_File = os.path.join(curr_dir_path, "basic.wad")
RewardScalingEnabled = True  # scales the reward so that they are always between -1 and 1. (theoretically not
#                              necessary, but greatly improves performance with most learning algorithms)
Scenario_Max_Reward_Per_Episode = 100.0  # you get 100 points if you kill the monster
Scenario_Min_Reward_Per_Episode = -300.0  # you lose 1 point every step until you kill the monster, episode timeout is
#                                           300 steps.
Mean_Normalization_Enabled = False  # reduces the mean reward from each reward
Mean_Step_Reward = None  # reduces the mean reward from each reward
