from vizdoom import DoomGame
from vizdoom import Button
from vizdoom import GameVariable
from vizdoom import ScreenFormat
from vizdoom import ScreenResolution
from abc import ABC, abstractmethod
import cv2
import random
import time
import numpy as np
import os
from importlib.machinery import SourceFileLoader
import gym
import gym_ple


VisibleScreen = False


class Environment(ABC):
    # A template for all future environments, meant to work with GrayScale screen buffers.

    def __init__(self, a_size):
        """
        Initiates an Environment instance. Meant to be inherited from and not to operate by itself.
        A partially abstract class.
        :param a_size: Number of possible actions in the environment.
        """
        self.a_size = a_size  # The number of possible actions.

    @property
    def possible_actions(self):
        """
        Returns a list of one hot vectors representing the possible actions in our environment.
        :return: a list of one hot vectors representing the possible actions in our environment.
        """
        return np.identity(self.a_size, dtype=np.float32).tolist()

    @abstractmethod
    def step(self, action):
        """
        Makes an action and returns a reward. Also scales the reward if self.reward_scaling_enabled is True.
        :param action: one hot vector detailing the action to take.(must be one of the vectors in self.possible_actions)
        :return: Reward value
        """
        pass

    @abstractmethod
    def get_post_terminal_step_reward(self):
        """
        Returns a reward for the state after the terminal step.
        Technically all steps after the episode is finished are supposed have a V value of zero but mean normalization
        might change what a reward of zero means and so we put zero through all the process a normal reward goes
        through and send the result.
        :return: Terminal reward value
        """
        pass

    @abstractmethod
    def get_state_screen_buffer(self):
        """
        Returns the current state's screen buffer. (Should be used
        :param img_size_to_return: tuple of 2 integers. The size of the image that will be returned in pixels.
        :param scaling: If true scales all pixels values to the range between 0 and 1. Improves performance.
        :return: returns the current state's screen buffer.
        """
        pass

    @abstractmethod
    def start_new_episode(self):
        """
        Starts a new episode.
        :return: None
        """
        pass

    @abstractmethod
    def is_episode_finished(self):
        """
        Returns True if episode is finished, False otherwise.
        :return: True if episode is finished, False otherwise.
        """
        pass

    @abstractmethod
    def get_total_reward(self):
        """
        Returns accumulated total reward from all states since episode start.
        :return: The accumulated total reward since episode start.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Close environment.
        :return: None
        """
        pass


class LocalEnvironment(Environment):
    # A template for all future environments run locally, meant to work with GrayScale screen buffers.

    Max_Pixel_value = 255

    def __init__(self, a_size, screen_buffer_pixel_values_scaling, reward_scaling_enabled, max_episode_reward,
                 min_episode_reward, mean_normalization_enabled, episode_timeout, mean_step_reward):
        """
        Initiates a LocalEnvironment instance. Meant to be inherited from and not to operate by itself.
        Partially abstract.
        :param a_size: Number of possible actions in the environment.
        :param screen_buffer_pixel_values_scaling: True/False, whether we should apply scaling to the screen buffer's
                                                   pixel values.
        :param reward_scaling_enabled: True/False, whether we should apply scaling to the rewards values.
        :param max_episode_reward: Episode's maximum possible total reward.
        :param min_episode_reward: Episode's minimum possible total reward.
        :param mean_normalization_enabled: True/False, whether we should apply mean normalization to the rewards values.
        :param episode_timeout: Must be either None\False\0 or a positive number. If None\False\0 episode never
                                times out. If a positive number limits the episode run on this instance to this number
                                of steps. (is_terminal starts returning True afterwards.)
        :param mean_step_reward: The average reward for a step.
        """
        super(LocalEnvironment, self).__init__(a_size)
        self.episode_timeout = episode_timeout
        self.episode_steps_cnt = 0
        self.screen_buffer_pixel_values_scaling = screen_buffer_pixel_values_scaling
        self.reward_scaling_enabled = reward_scaling_enabled
        self.mean_normalization_enabled = mean_normalization_enabled
        self.max_episode_reward = max_episode_reward
        self.min_episode_reward = min_episode_reward
        self.mean_reward = mean_step_reward

    @property
    def possible_actions(self):
        """
        Returns a list of one hot vectors representing the possible actions in our environment.
        :return: a list of one hot vectors representing the possible actions in our environment.
        """
        return np.identity(self.a_size, dtype=np.int32).tolist()

    def step(self, action):
        """
        Makes an action and returns a reward. Also scales the reward if self.reward_scaling_enabled is True.
        :param action: one hot vector detailing the action to take.(must be one of the vectors in self.possible_actions)
        :return: Reward value
        """
        reward = self._step(action)
        if self.mean_normalization_enabled:
            reward = self.mean_normalize_reward(reward)
        if self.reward_scaling_enabled:
            reward = self.scale_reward(reward)
        self.episode_steps_cnt += 1
        return reward

    @abstractmethod
    def _step(self, action):
        """
        DO NOT use this function!!! Use the .step() function instead!
        Makes an action and returns a reward.
        self.step() is a wrapper to this function so please use self.step().
        :param action: (list) action to do (must be from self.possible_action)
        :return: reward(int) ( R(s,a) )
        """
        pass

    def get_post_terminal_step_reward(self):
        """
        Returns a reward for the state after the terminal step.
        Technically all steps after the episode is finished are supposed have a V value of zero but mean normalization
        might change what a reward of zero means and so we put zero through all the process a normal reward goes
        through and send the result.
        :return: Terminal reward value
        """
        reward = 0  # if the episode ended, we get no reward in the next state
        if self.mean_normalization_enabled:
            reward = self.mean_normalize_reward(reward)
        if self.reward_scaling_enabled:
            reward = self.scale_reward(reward)
        return reward

    def mean_normalize_reward(self, reward):
        """
        Applies mean normalization to the reward. Essentially reduce the reward by the mean reward so as to make the
        mean reward 0. (theoretically not necessary,  but greatly improves performance with most learning algorithms)
        :param reward: (int or float) Reward value to mean normalize.
        :return: mean normalized reward.
        """
        return float(reward) - self.mean_reward

    def scale_reward(self, reward):
        """
        scales the reward so that they are always between -1 and 1. (theoretically not necessary,  but greatly improves
         performance with most learning algorithms)
        :param reward: (int or float) Reward value to scale
        :return: scaled reward in range between -1 and 1 (inclusive).
        """
        return float(reward) / max(abs(self.min_episode_reward), abs(self.max_episode_reward))

    @abstractmethod
    def _get_state_screen_buffer(self):
        """
        DO NOT use this function!!! Use the .get_state_screen_buffer() function instead.
        This is the one you should overwrite when inheriting from this class though, not .get_state_screen_buffer()
        returns the current state's original screen buffer.
        self.get_state_screen_buffer() is a wrapper to this function so please use self.get_state_screen_buffer().
        :return: returns the current state's screen buffer.
        """
        pass

    def get_state_screen_buffer(self, img_size_to_return=None):
        """
        Returns the current state's preprocessed screen buffer.
        :param img_size_to_return: tuple of 2 integers. The size of the image that will be returned in pixels.
        :return: returns the current state's screen buffer.
        """
        screen_buffer = self._get_state_screen_buffer()
        if not img_size_to_return == None:
            screen_buffer = cv2.resize(screen_buffer, dsize=tuple(img_size_to_return))
        if self.screen_buffer_pixel_values_scaling:
            screen_buffer = self.scale_pixel_values(screen_buffer)
        return screen_buffer

    def get_unprocessed_state_screen_buffer(self, img_size_to_return=None):
        """
        Returns the current state's unprocessed screen buffer.
        :param img_size_to_return: tuple of 2 integers. The size of the image that will be returned in pixels.
        :return: returns the current state's screen buffer.
        """
        screen_buffer = self._get_state_screen_buffer()
        if not img_size_to_return == None:
            screen_buffer = cv2.resize(screen_buffer, dsize=tuple(img_size_to_return))
        return screen_buffer


    def scale_pixel_values(self, state_screen_buffer):
        """
        Returns the current state's screen buffer with its values all scaled to the range between 0 and 1.
        (theoretically not necessary,  but greatly improves performance with most learning algorithms)
        :param state_screen_buffer: state_screen_buffer from self.get_state_screen_buffer().
        :return: screen buffer with its values all scaled to the range between 0 and 1
        """
        return state_screen_buffer / float(self.Max_Pixel_value)

    def start_new_episode(self):
        """
        Starts a new episode.
        :return: None
        """
        self._start_new_episode()
        self.episode_steps_cnt = 0

    @abstractmethod
    def _start_new_episode(self):
        """
        DO NOT use this function!!! Use the .start_new_episode() function instead.
        This is the one you should overwrite when inheriting from this class though, not .start_new_episode()

        Starts a new episode.
        :return: None
        """
        pass

    def is_episode_finished(self):
        """
        Returns True if episode is finished, False otherwise.
        :return: True if episode is finished, False otherwise.
        """
        is_finished = self._is_episode_finished()
        if self.episode_timeout:
            is_finished = is_finished or not (self.episode_steps_cnt < self.episode_timeout)
        return is_finished


    @abstractmethod
    def _is_episode_finished(self):
        """
        DO NOT use this function!!! Use the .is_episode_finished() function instead.
        This is the one you should overwrite when inheriting from this class though, not .is_episode_finished()
        returns True if episode is finished, False otherwise"""
        pass

    @abstractmethod
    def get_total_reward(self):
        """
        Returns accumulated total reward from all states since episode start.
        :return: The accumulated total reward since episode start.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Close environment.
        :return: None
        """
        pass


class DoomGameEnvironment(LocalEnvironment):
    def __init__(self, settings_file_path, visible_screen=False, sound_enabled=False):
        """
        Initiates a Local DoomGame environment (from VizDoom). Inherits from LocalEnvironment and therefore adheres to
        the interface defined in said LocalEnvironment class.
        :param settings_file_path: Path to the environment's settings.py file (string).
        :param visible_screen: If true VizDoom will open a window in the machine it runs on and display the game.
            If False, will run environment in the background and will not open such a display window.
        :param sound_enabled: If true VizDoom will enable sound from environment.
            If False, VizDoom will mute said environment.
        """
        self.game = DoomGame()
        settings = SourceFileLoader('settings', settings_file_path).load_module()  # A .py file containing details about
        #                                                                            the environment.

        # Load the correct configuration
        self.game.load_config(settings.VizDoom_CFG_File)

        # Load the correct scenario (in our case basic scenario)
        self.game.set_doom_scenario_path(settings.VizDoom_WAD_File)

        self.game.set_screen_format(ScreenFormat.GRAY8)
        self.game.set_screen_resolution(ScreenResolution.RES_160X120)

        self.game.set_window_visible(visible_screen)  # sets screen visibility
        self.game.set_sound_enabled(sound_enabled)  # enables or disables sound
        self.game.init()

        a_size = len(self.game.get_available_buttons())
        super(DoomGameEnvironment, self).__init__(a_size, settings.Screen_Buffer_Pixel_Values_Scaling,
                                                  settings.RewardScalingEnabled,
                                                  settings.Scenario_Max_Reward_Per_Episode,
                                                  settings.Scenario_Min_Reward_Per_Episode,
                                                  settings.Mean_Normalization_Enabled,
                                                  settings.Episode_Timeout,
                                                  settings.Mean_Step_Reward)

    def _start_new_episode(self):
        """
        DO NOT use this function!!! Use the .start_new_episode() function instead!
        """
        self.game.new_episode()

    def _step(self, action):
        """
        DO NOT use this function!!! Use the .step() function instead!
        """
        reward = self.game.make_action(action)
        return reward

    def _get_state_screen_buffer(self):
        """
        DO NOT use this function!!! Use the .get_state_screen_buffer() function instead!
        """
        return self.game.get_state().screen_buffer

    def _is_episode_finished(self):
        """
        DO NOT use this function!!! Use the .is_episode_finished() function instead!
        """
        return self.game.is_episode_finished()

    def get_total_reward(self):
        """
        Returns accumulated total reward from all states since episode start.
        :return: The accumulated total reward since episode start.
        """
        return self.game.get_total_reward()

    def close(self):
        """
        Close environment.
        :return: None
        """
        self.game.close()


class OpenAIGymEnvironment(LocalEnvironment):
    def __init__(self, settings_file_path, visible_screen=True):
        """
        Initiates a Local OpenAIGym environment. Inherits from LocalEnvironment and therefore adheres to
        the interface defined in said LocalEnvironment class.
        :param settings_file_path: Path to the environment's settings.py file (string).
        :param visible_screen: If true OpenAIGym will open a window in the machine it runs on and display the game.
            If False, will run environment in the background and will not open such a display window.
        """
        self.visible_screen = visible_screen

        settings = SourceFileLoader('settings', settings_file_path).load_module()  # A .py file containing details about
        #                                                                            the environment.

        self.env = gym.make(settings.Env_ID)  # loads the environment

        if "special_env_wrapper_func" in dir(settings):
            self.env = settings.special_env_wrapper_func(self.env)

        a_size = self.env.action_space.n

        super(OpenAIGymEnvironment, self).__init__(a_size, settings.Screen_Buffer_Pixel_Values_Scaling,
                                                   settings.RewardScalingEnabled,
                                                   settings.Scenario_Max_Reward_Per_Episode,
                                                   settings.Scenario_Min_Reward_Per_Episode,
                                                   settings.Mean_Normalization_Enabled,
                                                   settings.Episode_Timeout,
                                                   settings.Mean_Step_Reward)

        self.curr_state = None
        self.is_terminal = None
        self.total_episode_reward = None
        self.start_new_episode()

    def _start_new_episode(self):
        """
        DO NOT use this function!!! Use the .start_new_episode() function instead!
        """
        self.curr_state = self.env.reset()
        self.is_terminal = False
        self.total_episode_reward = 0

    def _step(self, action):
        """
        DO NOT use this function!!! Use the .step() function instead!
        """
        action = np.argmax(np.asarray(action))
        self.curr_state, reward, self.is_terminal, info = self.env.step(action)
        self.total_episode_reward += reward

        if self.visible_screen:
            self.env.render()

        return reward

    def _get_state_screen_buffer(self):
        """
        DO NOT use this function!!! Use the .get_state_screen_buffer() function instead!
        """
        return rgb2gray(self.curr_state)

    def _is_episode_finished(self):
        """
        DO NOT use this function!!! Use the .is_episode_finished() function instead!
        """
        return self.is_terminal

    def get_total_reward(self):
        """
        Returns accumulated total reward from all states since episode start.
        :return: The accumulated total reward since episode start.
        """
        return self.total_episode_reward

    def close(self):
        """
        Close environment.
        :return: None
        """
        self.env.close()


class EnvironmentsInitializer(object):
    Icon_Files_Name = "icon.jpg"
    """A class used to initialize environments."""
    def __init__(self, environments_settings_folder="Environments_Settings"):
        """
        A class used to initialize environments.
        :param environments_settings_folder:  Path to the Environments_Settings directory. (string)
        """
        self.environments_settings_folder = os.path.join(os.path.dirname(__file__), environments_settings_folder)

    def get_env(self, env_name):
        """
        Reutrns an instance that complies with the structure of Environment abstract class.
        :param env_name: The request environment identifying string.
        :return: An instance that complies with the structure of Environment abstract class.
        """
        vizdoom_envs_folder_path = os.path.join(self.environments_settings_folder, 'VizDoom')
        vizdoom_envs = os.listdir(vizdoom_envs_folder_path)
        if env_name in vizdoom_envs:
            settings_file_path = os.path.join(vizdoom_envs_folder_path, env_name, "settings.py")
            return DoomGameEnvironment(settings_file_path, visible_screen=VisibleScreen)

        open_ai_gym_envs_folder_path = os.path.join(self.environments_settings_folder, 'OpenAIgym')
        open_ai_gym_envs = os.listdir(open_ai_gym_envs_folder_path)
        if env_name in open_ai_gym_envs:
            settings_file_path = os.path.join(open_ai_gym_envs_folder_path, env_name, "settings.py")
            return OpenAIGymEnvironment(settings_file_path, visible_screen=VisibleScreen)

        raise UnknownEnvironmentRequested()

    def get_available_envs(self):
        """
        Returns the identifying string of all available environments.

        :return: A list of all available environments.
        """
        return os.listdir(os.path.join(self.environments_settings_folder, 'VizDoom')) + \
               os.listdir(os.path.join(self.environments_settings_folder, 'OpenAIgym'))

    def get_available_environments_initialization_strings(self):
        """
        Alias for self.get_available_envs().
        Returns the identifying string of all available environments.

        :return: A list of all available environments.
        """
        return self.get_available_envs()

    def env_init_str_to_env_icon_dict(self, icon_display_size):
        """
        Returns a dict where the keys, the environment initialization strings, map to their environments' corresponding
        icons (stored as numpy arrays).
        :param icon_display_size: The size of the icons to be placed in the dict. (tuple of 2 ints (width, height))
        :return: A dict where the keys, the environment initialization strings, map to their environments' corresponding
            icons (stored as numpy arrays).
        """
        env_init_str_to_icon = dict()
        paths = [os.path.join(self.environments_settings_folder, 'VizDoom'),
                 os.path.join(self.environments_settings_folder, 'OpenAIgym')]
        for path in paths:
            for env_init_str in os.listdir(path):
                env_init_str_to_icon[env_init_str] = cv2.resize(cv2.imread(os.path.join(path, env_init_str,
                                                                                        self.Icon_Files_Name)),
                                                                dsize=tuple(icon_display_size))
        return env_init_str_to_icon


class UnknownEnvironmentRequested(Exception):
    pass


class OpenAIGymNotImplementedYet(Exception):
    pass


def rgb2gray(rgb):
    """
    Converts an RGB image (stored as a numpy array) int a gray scale image (stored as a numpy array as well).
    :param rgb: RGB image (stored as a numpy array).
    :return: gray scale image (stored as a numpy array as well).
    """
    return np.dot(rgb[:, :, :3], [0.2989, 0.5870, 0.1140])


def main():
    """TESTS"""
    SAVE_SCREEN = True
    print("---TEST---")
    env_init = EnvironmentsInitializer()
    env_init.env_init_str_to_env_icon_dict((300, 300))
    print(env_init.get_available_envs())
    env_init_str = env_init.get_available_envs()[4]
    print(env_init_str)
    game = env_init.get_env(env_init_str)
    print(len(game.possible_actions))

    game.start_new_episode()
    if SAVE_SCREEN:
        from PIL import Image
        photos_folder = 'eps'
        if not os.path.isdir(photos_folder):
            os.mkdir(photos_folder)
        cnt = 0

    while not game.is_episode_finished():
        print(game.episode_steps_cnt)  # game.get_state_screen_buffer())
        if SAVE_SCREEN:
            screen = game.get_state_screen_buffer()
            if game.screen_buffer_pixel_values_scaling:
                screen *= 256
            Image.fromarray(screen).convert('RGB').save(os.path.join(photos_folder, r"%d.bmp" % cnt))
            cnt += 1
        game.step(random.choice(game.possible_actions))
        time.sleep(0.2)
        print(game.is_episode_finished())


if __name__ == "__main__":
    main()
