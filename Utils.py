from threading import Lock
import datetime
import os
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp  # import the inspect_checkpoint library
import pickle

class Coordinator(object):
    """
    A class to coordinate Worker threads.
    """
    def __init__(self, n_episodes, init_global_step_cnt=0):
        """
        A class to coordinate Worker threads.
        :param n_episodes: The number of episodes, we wish to go through overall. (the sum of the number of episodes
                           each thread goes through.
        :param init_global_step_cnt: Initializes the global step count at the given value.
        """
        self.max_episodes = n_episodes
        self._episode_cnt = 0
        self._episode_cnt_lock = Lock()
        self._global_step_cnt = init_global_step_cnt  # Used to count how many batches we've trained our model through.
        #                                               Should be incremented every time we calc gradients and update
        #                                               our model.
        self._global_step_cnt_lock = Lock()

        self._shutdown_requested_lock = Lock()
        self._shutdown_requested = False

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
        Signals the instance to shut down.
        :return: None
        """
        self.shutdown_requested = True

    def should_stop_training(self, inc_episode_cnt=False):
        """
        Returns True if cnt is no smaller than max_episodes.
        :param inc_episode_cnt: If true, increments the episode counter.
        :return: True if cnt is no smaller than max_episodes, False otherwise.
        """
        with self._episode_cnt_lock:
            should_stop = (not (self.max_episodes > self._episode_cnt)) or self.shutdown_requested
            if inc_episode_cnt and not should_stop:
                self._episode_cnt += 1
            return should_stop

    def get_episode_cnt(self):
        """
        Returns self._episodes_cnt.
        :return:  episodes count
        """
        with self._episode_cnt_lock:
            return self._episode_cnt

    def get_global_step_cnt(self):
        """
        Returns self._global_step_cnt
        :return: global step count
        """
        with self._global_step_cnt_lock:
            return self._global_step_cnt

    def inc_global_step_cnt(self):
        """
        Increments the global step count
        :return: None
        """
        with self._global_step_cnt_lock:
            self._global_step_cnt += 1


class Memory(object):
    Default_Max_Size = 30

    def __init__(self, max_size=Default_Max_Size):
        """
        A class meant store experience data for training. Facilitating the easy storing of experience and it's eventual
        recall top be used in gradient calculatin.
        :param max_size: Max size of memory space in time steps (int).
        """
        self.max_size = max_size
        self.states = []
        self.actions = []
        self.values = []
        self.rnn_states = []
        self.rewards = []

    def is_full(self):
        """
        Returns True if no more space is left in memory.
        """
        return not (self.max_size > len(self.states))

    def is_empty(self):
        """
        Returns True if  memory is empty.
        """
        return len(self.states) == 0

    def add_time_step(self, s, a, v, rnn_state_in, r):
        """
        Use this func to add data to memory. Will raise exception if memory is full.
        :param s: state
        :param a: action
        :param v: value
        :param rnn_state_in: the in rnn_state (the one that was used as the in state in the current time step)
        :param r: reward
        :return: None
        """
        if not self.is_full():
            self.states.append(s)
            self.actions.append(a)
            self.values.append(v)
            self.rnn_states.append(rnn_state_in)
            self.rewards.append(r)
        else:
            raise NoSpaceLeftInMemoryException()

    def get_episode_data_for_grads(self):
        """
        Returns all data needed for gradient calculations. Returns data as an iterator of tuples each of the
        following structure:(state, action, value, in_rnn_state, target_v, new_state).
        :return: Returns all data needed for gradient calculations. Returns data as an iterator of tuples each of the
                 following structure:(state, action, value, in_rnn_state, target_v, new_state).
        """
        return self.states, self.actions, self.values, self.rnn_states, self.rewards

    def reset(self):
        """
        Resets the memory.
        :param batch_initial_state: The first in rnn_state in the batch (the out state of the previous batch's last step).
        :return: None
        """
        self.states = []
        self.actions = []
        self.values = []
        self.rnn_states = []
        self.rewards = []


class NoSpaceLeftInMemoryException(Exception):
    pass


class SavesManager(object):
    """simplified saving mechanized, made for distribution build"""
    PickleFileName = "save.pkl"
    ckptFileName = "save.ckpt"

    def __init__(self, saves_folder_path):
        """
        Returns a SaveManager object.
        Uses tf.train.Saver() to save and load tf vars to and from files.
        Make sure to give it a valid path as saves_folder_path for it will crash otherwise.
        :param saves_folder_path: Path to the folder where the save folders will be saved to and loaded from.
        """
        self.saver = tf.train.Saver()
        self.saves_folder_path = saves_folder_path
        if not os.path.exists(self.saves_folder_path):
            print(os.path.exists(self.saves_folder_path))
            raise CannotFindDesignatedSavesFolderException()

    def save(self, sess, params_dict):
        """
        Saves the model's vars to file.
        :param sess: tensorflow session object
        :param params_dict: a dictionary with parameters to save
        :return: path of save file
        """
        print("Model Saved! Save Folder Path: %s" % self.saves_folder_path)
        # print([v.name for v in sess.graph.as_graph_def().node])
        # print(self.saver.saver_def)
        success = self.saver.save(sess, os.path.join(self.saves_folder_path, self.ckptFileName))
        with open(os.path.join(self.saves_folder_path, self.PickleFileName), "wb") as f:
            pickle.dump(params_dict, f)
        return success

    def load(self, sess, print_func=print):
        """
        Loads the model's vars from file.
        :param sess: tensorflow session object
        :return: a tuple (path of save file loaded (str), a dictionary with the saved parameters (dict)).
        """
        try:
            self.saver.restore(sess, os.path.join(self.saves_folder_path, self.ckptFileName))
            with open(os.path.join(self.saves_folder_path, self.PickleFileName), "rb") as f:
                saved_params_dict = pickle.load(f)
            return self.saves_folder_path, saved_params_dict
        except (tf.errors.DataLossError, FileNotFoundError):
            print_func("Failed to load Properly. Please check that the folder contains a valid save.")
            raise InvalidSavePathException()

    @classmethod
    def check_for_save(cls, saves_folder_path):
        """Returns true if the folder contains a save. Returns False otherwise."""
        try:

            with open(os.path.join(saves_folder_path, cls.PickleFileName), "rb") as f:
                pickle.load(f)
            if os.path.exists(os.path.join(saves_folder_path, "checkpoint")):
                return True
        except (FileNotFoundError, pickle.UnpicklingError):
            pass
        return False


class CannotFindDesignatedSavesFolderException(Exception):
    pass


class InvalidSavePathException(Exception):
    pass


class FilesWithNamesThatDoNotConformToSaveNameFormatExistInSavesFolderException(Exception):
    pass


def main():
    """ Tests """
    import os

    # Memory Class Test
    a = Memory(max_size=1)
    a.add_time_step(1, 2, 3, 3, 4)
    print(a.is_full())
    for i in a.get_episode_data_for_grads():
        print(i)

    # Coordinator Class Test
    from threading import Thread
    b = Coordinator(1000)
    def work(coord):
        coord.should_stop_training()
    threads = [Thread(target=work, args=(b,)) for i in range(1000)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    with b._episode_cnt_lock:
        print(b._episode_cnt)

    # SavesManager Class Test
    a = tf.Variable(initial_value=3, dtype=tf.int32, name='a')
    b = tf.Variable(initial_value=4, dtype=tf.int32, name='b')

    if not os.path.exists("Saves"):
        os.mkdir("Saves")
    sm = SavesManager("Saves")
    """for _ in range(10):
        with open(sm.get_new_save_path(), "wb"):
            pass"""
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.print(a))
        sess.run(a.assign(5))
        sm.save(sess, {"heyheyhey":6})
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # not necessary, just here to show that load() overwrites previous
        #                                              values.
        path, params_dict = sm.load(sess)
        sess.run(tf.print(a))
    print("PARAMS START")  # it appears that tensorflow interferes with printing so don't be surprised if you find
    #                            the output of tf.print(a) in between "CKPT PRINT START" and "CKPT PRINT END"
    print(params_dict)
    print("PARAMS END")

if __name__ == "__main__":
    main()
