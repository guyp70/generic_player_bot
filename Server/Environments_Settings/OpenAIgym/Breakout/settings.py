
Episode_Timeout = None  # Must be either None\False\0 or a positive number.
Screen_Buffer_Pixel_Values_Scaling = True
Env_ID = "Breakout-v0"
RewardScalingEnabled = False  # scales the reward so that they are always between -1 and 1. (theoretically not
#                              necessary, but greatly improves performance with most learning algorithms)
Scenario_Max_Reward_Per_Episode = None
Scenario_Min_Reward_Per_Episode = None
Mean_Normalization_Enabled = False  # reduces the mean reward from each reward
Mean_Step_Reward = None  # The average step reward
