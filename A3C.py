from Utils import Coordinator, Memory, SavesManager, InvalidSavePathException
from Remote_Environment import RemoteEnvironmentsManager
from Server.Environments import EnvironmentsInitializer
from AC_Network import AC_Network
import multiprocessing
import tensorflow as tf
import threading
import numpy as np
import time
import os
from PIL import Image
from Server.Shared_Code import Status


Log_Weights_Histograms_and_Distributions = False


def main():
    """TESTS"""
    ip, port = "10.0.0.3", 5381
    env_init_str = "Take_Cover"  # "Defend_the_Line"
    OFFLINE = True
    if not OFFLINE:
        env_manager = RemoteEnvironmentsManager(ip, port)
    else:
        env_manager = EnvironmentsInitializer()
    try:
        start = time.time()
        if not os.path.exists(A3C_Algorithm.DefaultSavesFolderPath):
            os.mkdir(A3C_Algorithm.DefaultSavesFolderPath)
        A3C_Algorithm(env_manager, env_init_str, num_of_workers=8).train(10)
        A3C_Algorithm(env_manager, env_init_str, num_of_workers=0).play(1)
        #A3C_Algorithm(env_manager, env_init_str, num_of_workers=8).train(8 * 1000)
        #A3C_Algorithm(env_manager, env_init_str, num_of_workers=8).train(30000)
        print("Time: %d seconds." % (time.time() - start))
    except KeyboardInterrupt:
        print("Keyboard Interrupt! Exiting...")
    if not OFFLINE:
        env_manager.shutdown()
        env_manager.join()


class A3C_Algorithm(object):
    Rescale_screen_images_to = [84, 84]
    s_size = Rescale_screen_images_to[0] * Rescale_screen_images_to[1]
    train_network_scope = 'global'  # This is the global network's scope. The network we update using the gradients
    #                                 from all threads.
    Workers_ACNN_Scope_Template = "Worker%d"
    DefaultSavesFolderPath = "Saves"
    SleepIntervalLength = 0.5  # in seconds
    SaveModelEveryNEpisodes = 10  # 100
    DefaultAlphaLearningRate = 1e-4
    DefaultWorkerMemoryBufferSize = 30
    LogsFolder = "Logs"
    def __init__(self, envs_manager, env_init_str, save_folder_path=DefaultSavesFolderPath,
                 Play_Display_Size=[84, 84], Worker_Memory_Buffer_Size=DefaultWorkerMemoryBufferSize,
                 AlphaLearningRate=DefaultAlphaLearningRate, num_of_workers=multiprocessing.cpu_count()):
        """
        A class capable of learning how to maximize a reward function in an environment.
        Uses A3c and is basically the API through which the other parts of the project use machine learning.
        :param env_cls: Environment class from which we will create instances. We use these instances to train and play.
        :param num_of_workers: Defines how many worker threads will be raised to train the models.
        :param envs_manager: An instance of either the EnvironmentsInitializer or the RemoteEnvironmentsManager classes.
        :param env_init_str: An environment initialization string. (get list of available environment initialization
            string from the envs_manager.get_available_environments_initialization_strings() func).
        :param save_folder_path: path to the directory we save our model in.
        :param Play_Display_Size: tuple of two int values representing (width, height) in pixels.
            Determines the size of the images given when the .play() func is called.
        :param Worker_Memory_Buffer_Size: The size of the Workers' Memory Buffer in frames (int).
        :param AlphaLearningRate: Alpha Learning Rate (double)
        :param num_of_workers: num of threads to start in addition to the main thread. (int)
        """
        self.OFFLINE = (not type(envs_manager) is RemoteEnvironmentsManager)
        self.Worker_Memory_Buffer_Size = Worker_Memory_Buffer_Size
        self.AlphaLearningRate = AlphaLearningRate
        self.env_init_str = env_init_str
        self.env_manager = envs_manager
        self.Play_Display_Size = Play_Display_Size
        self.env = self.get_a_new_environment_instance()
        with tf.variable_scope("Optimizer", reuse=tf.AUTO_REUSE):
            self.trainer = tf.train.AdamOptimizer(learning_rate=self.AlphaLearningRate)  # tf.train.RMSPropOptimizer(learning_rate=self.AlphaLearningRate)
        self.train_model = AC_Network(s_size=self.s_size, a_size=self.env.a_size,
                                      scope=self.train_network_scope, trainer=self.trainer,
                                      train_network_scope=self.train_network_scope)

        # Prepare the things needed for the initialization fo the workers.
        self.num_of_workers = num_of_workers
        self.worker_ACNN_Scopes = [self.Workers_ACNN_Scope_Template % i for i in range(self.num_of_workers)]
        self.worker_envs = [self.get_a_new_environment_instance() for _ in range(self.num_of_workers)]
        self.worker_local_AC_networks = [AC_Network(self.s_size, self.env.a_size, self.worker_ACNN_Scopes[i],
                                                    self.trainer, train_network_scope=self.train_network_scope)
                                         for i in range(self.num_of_workers)]
        self.workers = []

        self.save_manger = SavesManager(save_folder_path)  # it's important to only init the saver after all of the
        #                                                        tensorflow graph has been declared.

        # Preparation for the shutdown property. (i.e. The signal system that shuts down an instance.)
        self._shutdown_requested_lock = threading.Lock()
        self._shutdown_requested = False

    def reset_tf_graph(self):
        """
        Resets and redefines the tensorflow graph so prevent problems when calling .train() and .play() funcs more
        than once.
        :return:  None
        """
        tf.reset_default_graph()
        with tf.variable_scope("Optimizer", reuse=tf.AUTO_REUSE):
            self.trainer = tf.train.RMSPropOptimizer(learning_rate=self.AlphaLearningRate)
        self.train_model = AC_Network(s_size=self.s_size, a_size=self.env.a_size,
                                      scope=self.train_network_scope, trainer=self.trainer,
                                      train_network_scope=self.train_network_scope)
        self.worker_local_AC_networks = [AC_Network(self.s_size, self.env.a_size, self.worker_ACNN_Scopes[i],
                                                    self.trainer, train_network_scope=self.train_network_scope)
                                         for i in range(self.num_of_workers)]
        self.save_manger = SavesManager(self.save_manger.saves_folder_path)  # it's important to only init the saver
        #                                                                      after all of the tensorflow graph has
        #                                                                      been declared.

    @property
    def shutdown_requested(self):
        """
        shutdown_requested Property getter. Safely returns the value of self._shutdown_requested.
        :return: The value of self._shutdown_requested.
        """
        with self._shutdown_requested_lock:
            return self._shutdown_requested

    @shutdown_requested.setter
    def shutdown_requested(self, value):
        """
        shutdown_requested Property setter. Safely sets the value of self._shutdown_requested.
        :param value: The value to which we set self._shutdown_requested
        :return: None
        """
        with self._shutdown_requested_lock:
            self._shutdown_requested = value

    def shutdown(self):
        """
        Terminates operation of instance
        :return: None
        """
        self.shutdown_requested = True
        for w in self.workers:
            if w.is_alive():
                w.join()
        if self.env_manager.status == Status.Running:
            self.env.close()
            for env in self.worker_envs:
                env.close()

    def change_save_folder_path(self, new_path):
        """
        Change save folder path.
        :param new_path: path to the new directory we will save our model to.
        :return: None
        """
        self.save_manger.saves_folder_path = new_path

    def get_a_new_environment_instance(self):
        """
        Gets a new environment instance from self.env_manager.
        :return: An object that inherits from the abstract Environment class.
        """
        if not self.OFFLINE:
            return self.env_manager.get_new_remote_environment_terminal(self.env_init_str, self.Rescale_screen_images_to)
        else:
            return self.env_manager.get_env(self.env_init_str)

    def get_state_screen_buffer(self):
        """
        Gets a state screen buffer from self.env_manager.
        :return: A state screen buffer (numpy array)
        """
        if not self.OFFLINE:
            return self.env.get_state_screen_buffer()
        else:
            return self.env.get_state_screen_buffer(img_size_to_return=self.Rescale_screen_images_to)

    def play(self, n_episodes=1, print_func=print, frame_update_func=None, get_stop_flag=lambda *args: False):
        """
        Plays n_episodes without learning. No threads are put to work learning. It only allows us to how well our model
        preforms.
        :param n_episodes: episodes to play
        :param print_func: function with which to print our textual output. (must be able to take a single argument)
        :param frame_update_func: function with which to output our frame every turn. (must be able to take a single
            argument that is an instance of the PIL.Image.Image class)
        :param get_stop_flag: A function that will be called every some time. if it returns True the .play() will exit.
            (Must be able to be called with no arguments given)
        :return: None
        """
        self.reset_tf_graph()
        with tf.Session() as sess:
            try:
                self.init_tensorflow_variables(sess)
            except (tf.errors.InvalidArgumentError, PermissionError) as e:
                if type(e) is tf.errors.InvalidArgumentError:
                    print_func("It seems like the save you have loaded is incompatible with your current environment. "
                               "(either the a_size or the s_size are different)\r\nPlaying Aborted.")
                else:
                    print_func("OS Denied access to save files. Please check that no program is already making use of"
                               " said files.\r\nPlaying Aborted.")
                return
            start_time = time.time()
            episode_cnt = 0
            stop_flag_bool = False
            for i in range(n_episodes):
                if get_stop_flag() or stop_flag_bool:
                    break
                episode_cnt += 1
                episode_reward = 0
                episode_length = 0
                print_func("Starting Episode %d" % i)
                self.env.start_new_episode()
                context = sess.run(self.train_model.init_context)
                while (not self.env.is_episode_finished()) and (not self.shutdown_requested):
                    if get_stop_flag():
                        stop_flag_bool = True
                        break
                    frame = Image.fromarray(self.env.get_unprocessed_state_screen_buffer(self.Play_Display_Size)).convert('RGB')
                    if frame_update_func:
                        frame_update_func(frame)
                    state = self.get_state_screen_buffer()
                    feed_dict = {self.train_model.inputs: np.reshape(state, newshape=[1, -1]),
                                 self.train_model.context_in: context}
                    policy, context = sess.run([self.train_model.policy, self.train_model.context_out],
                                                feed_dict=feed_dict)
                    policy = policy.flatten()
                    action2take = (np.random.choice(policy, p=policy) == policy).astype(dtype=np.int32).flatten()
                    # print(policy, action2take)
                    episode_reward += self.env.step(list(action2take))
                    episode_length += 1
                    time.sleep(1.0/60)
                if self.env.is_episode_finished():
                    episode_reward += self.env.get_post_terminal_step_reward()
                print_func("Episode %d Ended. Episode reward: %f. Episode Length: %d" % (i, episode_reward,
                                                                                         episode_length))
            end_time = time.time()
            try:
                avg_episode_time = float(end_time - start_time) / episode_cnt
            except ZeroDivisionError:
                avg_episode_time = 0
            print_func("Finished Playing.\r\n"
                       "Overall Time: %f (secs)\r\nEpisodes run: %s\r\nAvg Time per Episode: %f (secs)" %
                       (end_time - start_time, episode_cnt, avg_episode_time))

    def train(self, n_episodes=10,  print_func=print, get_stop_flag=lambda *args: False):
        """
        Trains for n_episodes.
        :param n_episodes: episodes to play
        :param print_func: function with which to print our textual output. (must be able to take a single argument)
        :param get_stop_flag: A function that will be called every some time. if it returns True the .play() will exit.
            (Must be able to be called with no arguments given)
        :return: None
        """
        self.reset_tf_graph()
        self.workers = [Worker(self.worker_envs[i], self.worker_local_AC_networks[i], self.s_size, self.env.a_size,
                               self.worker_ACNN_Scopes[i], self.Worker_Memory_Buffer_Size,
                               train_network_scope=self.train_network_scope, OFFLINE=self.OFFLINE)
                        for i in range(self.num_of_workers)]
        with tf.Session(graph=tf.get_default_graph()) as sess:
            try:
                # coord is an object of the Coordinator class and is used to keep rack of the global step count and
                # over how many episodes we've trained through.
                coord = self.init_tensorflow_variables_and_coord(sess, n_episodes)
            except (tf.errors.InvalidArgumentError, PermissionError) as e:
                if type(e) is tf.errors.InvalidArgumentError:
                    print_func("It seems like the save you have loaded is incompatible with your current environment. "
                               "(either the a_size or the s_size are different)\r\nPlaying Aborted.")
                else:
                    print_func("OS Denied access to save files. Please check that no program is already making use of"
                               " said files.\r\nTraining Aborted.")
                return
            logs_path = os.path.join(self.save_manger.saves_folder_path, self.LogsFolder)
            self.summary_writer = tf.summary.FileWriter(logs_path, sess.graph)
            [w.start(sess, coord, summary_file_writer=self.summary_writer) for w in self.workers]
            episode_cnt_at_last_save = 0
            start_time = time.time()
            while (not coord.should_stop_training(inc_episode_cnt=False)) and (not self.shutdown_requested) and (not get_stop_flag()):
                time.sleep(self.SleepIntervalLength)
                if Log_Weights_Histograms_and_Distributions:
                    # Logs all global tensorflow vars (basically the train model's weighs) to pretty histograms and
                    # distribution charts
                    self.summary_writer.add_summary(sess.run(self.train_model.log_histograms_for_all_global_vars),
                                                    global_step=coord.get_global_step_cnt())
                    self.summary_writer.flush()

                curr_ep_cnt = coord.get_episode_cnt()
                episodes_since_last_save = curr_ep_cnt - episode_cnt_at_last_save
                if episodes_since_last_save >= self.SaveModelEveryNEpisodes:
                    self.save(sess, coord)
                    episode_cnt_at_last_save = curr_ep_cnt
                    print_func("Episode %d Started! Model Saved." % curr_ep_cnt)
            coord.shutdown()
            [w.join() for w in self.workers]
            self.save(sess, coord)
            self.summary_writer.close()
        self.workers = []
        end_time = time.time()
        try:
            avg_episode_time = float(end_time - start_time) / coord.get_episode_cnt()
        except ZeroDivisionError:
            avg_episode_time = 0
        print_func("Finished Training. Model Saved.\r\n"
                   "Overall Time: %f (secs)\r\nEpisodes run: %s\r\nAvg Time per Episode: %f (secs)" %
                   (end_time - start_time, coord.get_episode_cnt(), avg_episode_time))

    def init_tensorflow_variables_and_coord(self, sess, n_episodes):
        """
        Use only if you also need a Coordinator class object. Otherwise, use the init_tensorflow_variables() function.
        Initialize all tensorflow vars and if save files exist, load from them all tensorflow vars.
        Initializes a Cooordinator class object. If save files exits, continues the global step count from where it was
        left. If no save files are to be found, initializes global step count to 0.
        :param sess: tensorflow session object
        :param n_episodes: (int) the number of episodes we want to train
        :return: Coordinator class object
        """
        sess.run(tf.global_variables_initializer())
        global_step = 0

        try:
            save_file_path, saved_params = self.save_manger.load(sess)  # try to load from save files (will crash if no
            #                                                            save files are to be found)
            global_step = saved_params['global_step_cnt']
            print("Model Loaded! Save File Path: %s" % save_file_path)
            print("Parms Loaded! Saved Params: %s" % str(saved_params))
        except (InvalidSavePathException, ValueError):
            print("No Saves Found! Initialized all values.")
        return Coordinator(n_episodes, init_global_step_cnt=global_step)

    def init_tensorflow_variables(self, sess):
        """
        Initialize all tensorflow vars and if save files exist, load from them all tensorflow vars.
        :param sess: tensorflow session object
        :return: None
        """
        sess.run(tf.global_variables_initializer())
        try:
            save_file_path, saved_params = self.save_manger.load(sess)  # try to load from save files (will crash if no
            #                                                            save files are to be found)
            print("Model Loaded! Save File Path: %s" % save_file_path)
        except (InvalidSavePathException, ValueError):
            print("No Saves Found! Initialized all values.")

    def save(self, sess, coord):
        """
        Handles saving the model's tensorflow vars and the necessary parameters.
        :param sess: tf.Session class object
        :param coord: Coordinator class object
        :return: None
        """
        params_dict = {'global_step_cnt': coord.get_global_step_cnt()}
        self.save_manger.save(sess, params_dict=params_dict)


class Worker(threading.Thread):
    DiscountFactor = 0.99  # discount factor for value approximation

    def __init__(self, env, AC_network, s_size, a_size, scope, Memory_Buffer_Size, train_network_scope='global',
                 rescale_screen_images_to=[84, 84], OFFLINE=False):
        """
        A class that is meant to make training more efficient by allowing parallelism.
        The A3c Create several instances of this class, with each worker playing in his environment,  acquiring
        experience, updating the global model with the produced gradients and then updating themselves with global
        model's new variables.

        :param env: An object that inherits from the abstract Environment class.
        :param AC_network: An instance of the AC_network class.
        :param s_size: Input size, essentially the size of pixels in the screen.(int)
        :param a_size: Action space. The number of actions possible in our environment. (int)
        :param scope: Tensorflow Varables scope. (string)
        :param Memory_Buffer_Size: The size of the Workers' Memory Buffer in frames (int).
        :param train_network_scope: Tensorflow Varables scope of the global model. (string)
        :param rescale_screen_images_to: Size to which to scale the screen image.
        :param OFFLINE: True if working on localy run environments. False if working on Remotely run environments.
        """
        super(Worker, self).__init__()
        self.OFFLINE = OFFLINE
        self.Memory_Buffer_Size = Memory_Buffer_Size
        self.local_ACNN_scope = scope
        self.global_ACNN_scope = train_network_scope
        self.local_ACNN = AC_network
        self.env = env
        self.rescale_screen_images_to = rescale_screen_images_to
        self.s_size, self.a_size = s_size, a_size
        self.update_local_ACNN_weights = update_target_graph(self.global_ACNN_scope, self.local_ACNN_scope)

    def start(self, sess, coord, summary_file_writer=None):
        """
        Starts the Worker thread. The worker will train until coord orders it to stop.
        :param sess: tensorflow session object
        :param coord: Coordinator object
        :return: None
        """
        self.sess = sess
        self.coord = coord
        self.summary_file_writer = summary_file_writer
        super(Worker, self).start()

    def run(self):
        """
        Goes through episodes and trains the global ACNN
        :return: None
        """
        self.train(self.sess, self.coord)

    def train(self, sess, coord):
        """
        Trains for as long as the coord.should_stop_training() returns false.
        :param sess: tensorflow session object
        :param coord: Coordinator object
        :return: None
        """
        mem = Memory(max_size=self.Memory_Buffer_Size)
        while not coord.should_stop_training(inc_episode_cnt=True):
            self.env.start_new_episode()
            context_out = sess.run(self.local_ACNN.init_context)
            episode_reward = 0
            episode_length = 0
            while not self.env.is_episode_finished():
                sess.run(self.update_local_ACNN_weights)  # updates model vars to current global ACNN vars.
                while not mem.is_full() and not self.env.is_episode_finished():  # Gain Experience
                    state = self.get_state_screen_buffer()
                    context_in = context_out  # we save it for the memory add_time_step() func
                    feed_dict = {self.local_ACNN.inputs: np.reshape(state, newshape=[1, -1]),
                                 self.local_ACNN.context_in: context_in}
                    value, policy, context_out = sess.run([self.local_ACNN.value, self.local_ACNN.policy,
                                                       self.local_ACNN.context_out], feed_dict=feed_dict)
                    policy = policy.flatten()
                    action2take = (np.random.choice(policy, p=policy) == policy).astype(dtype=np.int32).flatten()
                    immediate_reward = self.env.step(list(action2take))
                    episode_reward += immediate_reward
                    mem.add_time_step(state, action2take, value, context_in, immediate_reward)
                    episode_length += 1

                """Use the experience to calc grads and apply to global ACNN"""
                # Here we calculate the V value of the state t tag (considering step t is the last state in the mem
                # buffer).
                if not self.env.is_episode_finished():
                    new_state = self.get_state_screen_buffer()
                    feed_dict = {self.local_ACNN.inputs: np.reshape(new_state, newshape=[1, -1]),
                                 self.local_ACNN.context_in: context_out}
                    value_at_new_state = sess.run([self.local_ACNN.value], feed_dict=feed_dict)
                else:
                    value_at_new_state = [self.env.get_post_terminal_step_reward()]

                batch_stats = self.calc_grads_and_update_train_model(sess, mem, value_at_new_state, coord)
                if self.summary_file_writer:  # logs batch's stats to log files
                    # print("Logging Batch Statistics!")
                    log_python_values_to_tensorboard(self.summary_file_writer, batch_stats, coord)
                mem.reset()
            episode_reward += self.env.get_post_terminal_step_reward()
            log_python_values_to_tensorboard(self.summary_file_writer, {'Objective/episode_reward': episode_reward,
                                                                        'Objective/episode_length': episode_length},
                                             coord)

    def calc_grads_and_update_train_model(self, sess, mem, est_value_of_next_state, coord):
        """
        Calculates gradients for the model based on the batch. Applies grads to the global network and returns
        statistics about the batch.
        Be Aware, this function is one hell of a mess!I built it to feed the ACNN grads but it's very technical and in
        all honesty, I'm pretty sure I won't be able to even understand it tomorrow morning.
        :param sess: tensorflow session object
        :param mem: A Memory class object that contains some experience.
        :param est_value_of_next_state: estimated V value of the next state
        :param coord: Coordinator class object
        :return: a dictionary object containing the following entries:
                 'avg_value_loss', 'avg_policy_loss', 'avg_entropy', 'grad_norms', 'var_norms'
        """
        if mem.is_empty():
            raise CannotLearnFromEmptyMemoryException()
        states, actions, values, rnn_states, rewards = mem.get_episode_data_for_grads()
        batch_size = len(actions)
        # put context into proper form
        c_in = []
        h_in = []
        [(h_in.append(h), c_in.append(c)) for h, c in rnn_states]
        h_in = np.reshape(np.asarray(h_in), newshape=[batch_size, -1])  # shape: [batch_size, lstm_n_units]
        c_in = np.reshape(np.asarray(c_in), newshape=[batch_size, -1])  # shape: [batch_size, lstm_n_units]
        context = h_in, c_in
        # put the rest of the vars into proper form
        states = np.reshape(np.asarray(states, dtype=np.float32), newshape=[batch_size, self.s_size])  # shape: [batch_size, s_zise]
        actions = np.reshape(np.asarray(actions, dtype=np.int32), newshape=[batch_size, self.a_size])
        values = np.asarray(values, dtype=np.float32)
        est_value_of_next_state = np.asarray(est_value_of_next_state)
        target_vs = approximate_real_v(rewards, self.DiscountFactor, est_value_of_next_state).flatten()  # shape: [batch_size * a_size]
        advantages = calc_advantage(rewards, values, est_value_of_next_state,
                                    self.DiscountFactor)  # shape: [batch_size]

        actions = np.argmax(actions, axis=1)  # size: [batch_size]
        feed_dict = {self.local_ACNN.inputs: states, self.local_ACNN.context_in: context,
                     self.local_ACNN.actions: actions, self.local_ACNN.target_v: target_vs,
                     self.local_ACNN.advantages: advantages}
        # print(feed_dict)
        # [(print(k, ": ",  v.shape)) for k, v in feed_dict.items() if type(v) != tuple]
        loss, value_loss, policy_loss, entropy, grad_norms, var_norms, _ = \
            sess.run([self.local_ACNN.loss, self.local_ACNN.value_loss, self.local_ACNN.policy_loss,
                      self.local_ACNN.entropy, self.local_ACNN.grad_norms, self.local_ACNN.var_norms,
                      self.local_ACNN.apply_grads], feed_dict=feed_dict)
        # self.summary_file_writer.flush()
        batch_statistics = {'Losses/loss': loss, 'Losses/avg_value_loss': value_loss / batch_size,
                            'Losses/avg_policy_loss': policy_loss / batch_size,
                            'Losses/avg_entropy': entropy / batch_size, 'grad_norms': grad_norms,
                            'var_norms': var_norms}
        coord.inc_global_step_cnt()
        return batch_statistics

    def get_state_screen_buffer(self):
        """
        Gets a state screen buffer from self.env_manager.
        :return: A state screen buffer (numpy array)
        """
        if not self.OFFLINE:
            return self.env.get_state_screen_buffer()
        else:
            return self.env.get_state_screen_buffer(img_size_to_return=A3C_Algorithm.Rescale_screen_images_to)


def log_python_values_to_tensorboard(summary_file_writer, tags_values_dict, coord, scope=""):
    """
    Logs the values to a tensorboard event file.
    :param summary_file_writer: tf.summary.FileWriter() class object
    :param tags_values_dict: the values of the dict will be saved under the names corresponding the the keys.
    :param coord: Coordinator class object
    :param scope: Scope to put the values under in tensorboard
    :return: None
    """
    summary_entries = [tf.Summary.Value(tag="/".join([scope, name]), simple_value=value)
                       for name, value in tags_values_dict.items()]
    summary_file_writer.add_summary(tf.Summary(value=summary_entries),
                                    global_step=coord.get_global_step_cnt())
    summary_file_writer.flush()


class CannotLearnFromEmptyMemoryException(Exception):
    pass


def update_target_graph(from_scope, to_scope):
    """
    Copies a set of variables from one scope to another.
    Used to set worker network parameters to those of global network.
    Taken from https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb
    :param from_scope:  scope to copy all vars from
    :param to_scope:  scope to copy all vars to
    :return: None
    """
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    return tf.group([to_var.assign(from_var) for from_var, to_var in list(zip(from_vars, to_vars))])


def approximate_real_v(rewards, gamma, est_value_in_T_plus_one):
    """
    V(s) is defined as so V(s) = max over possible a (r + gamma * sum over all s' (T(s,a,s')V(s'))).
    T(s, a, s') is the probability that we will get to s' if we make action a in state s. We assume it is 1 for now.
    We also assume the action we did is the best action we could've taken.
    So we can compute V(s) as V(s) = r + gamma * V(s')
    That is what we here.
    :param rewards: rewards given by the environment.
    :param gamma: discount factor.
    :param est_value_in_T_plus_one: the value the V estimator gave at the state that we got after we took the action who
                                    us the last reward.
    :return: approximated V(s) values.
    """
    apx_v_reversed = list()
    apx_v_reversed.append(rewards[-1] + gamma * est_value_in_T_plus_one)
    for r in reversed(rewards[:-1]):
        apx_v_reversed.append(r + gamma * apx_v_reversed[-1])
    return np.asarray(tuple(reversed(apx_v_reversed)))


def calc_advantage(rewards, values, value_in_T_plus_one, gamma):
    """
    Advantage function A(s,a) is defined A(s,a) = Q(s,a) - V(s).
    We can also write as A(s) = r +  (sum over all s' (T(s,a,s')V(s'))) - V(s)
    T(s, a, s') is the probability that we will get to s' if we make action a in state s. We assume it is 1 for now.
    So we can compute A(s) as A(s) = r + V(s') - V(s)
    :param rewards: rewards given by the environment. (shape=[batch_size])
    :param values: The values given by our V estimator at time steps 1 to T. (shape=[batch_size])
    :param value_in_T_plus_one: The value given by our V estimator  at time step T + 1. (shape=scalar/[1])
    :param gamma: Discount Factor (shape=scalar/[1])
    :return: advantage values. for time steps 1 to T. (shape=[batch_size])
    """
    values_s = values
    values_s_tag = np.roll(values, shift=-1)
    values_s_tag[-1] = value_in_T_plus_one
    advantages = rewards + gamma * values_s_tag - values_s
    advantages = discount(advantages, gamma)  # frankly, I have no clue why the advantages are discounted, and I can't
    #                                           seem to find anything in the maths of the algorithm that justify doing
    #                                           so. I've seen it done in Juliani's implementation and if we go by tests
    #                                           I've made it seems to improve sample efficiency by a whopping 100% (i.e.
    #                                           it takes twice as much data without it) and also to reduce oscillation
    #                                           in performance so I'm putting it here none the less. (What's more 
    #                                           important is theoscillation part really, cause before this addition 
    #                                           the algoorithm was quite unstable)
    return advantages


def discount(x, gamma):
    """
    Discounts the values of x by a factor gamma meaning that by the end disc_x[i] = x[i] + gamma * disc_x[i + 1].
    if i == len(x) - 1 (meaning it is the last value in array x),  disc_x[i] = x[i].
    :param x: Values to discount. (numpy array)
    :param gamma: Discount Factor (double)
    :return: An array of the discounted values.
    """
    x = x.copy()
    for i in range(len(x) - 2, -1, -1):
        x[i] += gamma * x[i + 1]
    return np.asarray(x)































if __name__ == '__main__':
    main()