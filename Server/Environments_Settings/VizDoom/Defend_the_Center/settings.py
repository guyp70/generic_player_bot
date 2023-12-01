import os

curr_dir_path = os.path.dirname(__file__)  # relative to the settings file
Episode_Timeout = None  # Must be either None\False\0 or a positive number.
Screen_Buffer_Pixel_Values_Scaling = True
VizDoom_CFG_File = os.path.join(curr_dir_path, "defend_the_center.cfg")
VizDoom_WAD_File = os.path.join(curr_dir_path, "defend_the_center.wad")
RewardScalingEnabled = False  # scales the reward so that they are always between -1 and 1. (theoretically not
#                               necessary, but greatly improves performance with most learning algorithms)
Scenario_Max_Reward_Per_Episode = 0  # there is only a penalty for dying (and no other rewards)
Scenario_Min_Reward_Per_Episode = -1  # if you die immediately
Mean_Normalization_Enabled = False  # reduces the mean reward from each reward
Mean_Step_Reward = None  # reduces the mean reward from each reward
