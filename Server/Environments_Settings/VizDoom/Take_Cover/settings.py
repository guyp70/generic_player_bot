import os

curr_dir_path = os.path.dirname(__file__)  # relative to the settings file
Episode_Timeout = 4000  # Must be either None\False\0 or a positive number.
Screen_Buffer_Pixel_Values_Scaling = True
VizDoom_CFG_File = os.path.join(curr_dir_path, "take_cover.cfg")
VizDoom_WAD_File = os.path.join(curr_dir_path, "take_cover.wad")
RewardScalingEnabled = False  # scales the reward so that they are always between -1 and 1. (theoretically not
#                              necessary, but greatly improves performance with most learning algorithms)
Scenario_Max_Reward_Per_Episode = 4000  # gets 1 point for every turn the bot lives. Timeout is at 4000.
Scenario_Min_Reward_Per_Episode = 0
Mean_Normalization_Enabled = True  # reduces the mean reward from each reward
Mean_Step_Reward = 0.99  # reduces the mean reward from each reward
