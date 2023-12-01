from Server.Shared_Code import *
import socket
import queue
import select
from Server.Environments import Environment


class RemoteEnvironmentsManager(Threadable):
    EnvironmentIDFormat = "env%d"
    Select_Timeout = 0.01
    Server_Check_Timeout = 5
    QueueGetTimeout = 5

    def __init__(self, server_ip, server_port, icon_display_size=(250, 250)):
        """
        The instances of these class allow the creation of remote environments run on the server and the interaction
        with said environments via the instances of the RemoteEnvironmentTerminal class it creates and passes to the
        client.
        The user (the programmer using the class, not the user of the project as a whole) only uses this class to create
        environments and seemingly uses the instances of the  RemoteEnvironmentTerminal class returned in the
        self.get_new_remote_environment_terminal() to interact with the environment run on the server.
        Unbeknown to user though, it is the instance of this class that actually handles all communications with the
        server. The RemoteEnvironmentTerminal instances communicating through the RemoteEnvironmentsManager instance
        that created them.
        :param server_ip: Server ip (string).
        :param server_port: Server port (int).
        :param icon_display_size: The size of the icon display widget. (tuple of 2 ints (width, height))
        """
        super(RemoteEnvironmentsManager, self).__init__()
        self.sock_lock = threading.Lock()
        self.sock = socket.socket()
        self._server_ip = server_ip
        self._server_port = server_port
        self.sock.connect((self._server_ip, self._server_port))

        self.cnt_lock = threading.Lock()
        self.cnt = 0

        self.envs_msgs_queues = {}
        self.msgs2send = queue.Queue()

        self.__icon_display_size = icon_display_size
        self.__available_environments_initialization_strings = None
        self.get_available_environments_initialization_strings()

        self.__environments_initialization_strings_to_icons_dict = None
        self.get_environments_initialization_strings_to_icons_dict()

        self.start()  # start operations.

    def run(self):
        """
        The RemoteEnvironmentsManager instance main function. Handles the creation and communications of its child
        environments terminals (RemoteEnvironmentTerminal insstances).
        :return: None
        """
        try:
            self.status = Status.Running
            msg = ""
            while self.status == Status.Running:
                with self.sock_lock:
                    rl, wl, _ = select.select([self.sock], [self.sock], [], self.Select_Timeout)
                    if self.sock in rl:
                        msg = receive_by_size(self.sock)
                        self.handle_if_EROR_msg(msg)
                        self.envs_msgs_queues[self.get_msg_env_id(msg)].put(msg)
                        #print(msg[:4], self.get_msg_env_id(msg))
                    if self.sock in wl and not self.msgs2send.empty():
                        send_by_size(self.sock, self.msgs2send.get())
            self._shutdown_procedure()
        except SocketClosedRemotelyError as ex:
            print("MSG:", msg)
            raise ex

    def _shutdown_procedure(self):
        """
        Shuts down the RemoteEnvironmentsManager instance. Errors will occur should you try to continue and use the
        RemoteEnvironmentTerminal instances created using this manager (by the closed instance).
        :return: None
        """
        with self.sock_lock:
            send_by_size(self.sock, ProtocolHandler.Exit_Msg_Identifier)
            self.sock.close()
        self.status = Status.ShutDown

    def get_server_ip_and_port(self):
        """
        Returns the server's ip and port.
        :return: The server's ip and port. (tuple of (ip (string), port(int) ) )
        """
        return self._server_ip, self._server_port

    def get_available_environments_initialization_strings(self):
        """
        Returns the available environments initialization strings offered by the server.
        :return: The available environments initialization strings offered by the server.
        """
        if self.__available_environments_initialization_strings is None:
            with self.sock_lock:
                send_by_size(self.sock, ProtocolHandler.format_RAES_msg())
                AEIS_msg = receive_by_size(self.sock)
            self.__available_environments_initialization_strings = ProtocolHandler.parse_AEIS_msg(AEIS_msg)
        return self.__available_environments_initialization_strings.copy()

    def get_environments_initialization_strings_to_icons_dict(self):
        """
        Returns the available environments initialization strings to icons dict from the server.
        The dict maps each environment initialzation string to their matching icons (the icon are numpy arrays)
        :return: A dict where the keys, the environment initialization strings, map to their environments' corresponding
            icons (stored as numpy arrays).
        """
        if self.__environments_initialization_strings_to_icons_dict is None:
            with self.sock_lock:
                send_by_size(self.sock, ProtocolHandler.format_GTMN_msg(self.__icon_display_size))
                MENU_msg = receive_by_size(self.sock)
            self.__environments_initialization_strings_to_icons_dict = ProtocolHandler.parse_MENU_msg(MENU_msg)
        return self.__environments_initialization_strings_to_icons_dict.copy()

    def get_new_remote_environment_terminal(self, env_init_str, s_size):
        """
        Get a new remote environment terminal. Creates an environment in the server and returns an instance of the
        RemoteEnvironmentTerminal class.
        :param env_init_str: An environment initialization string. (get list of available environment initialization
            string from the self.get_available_environments_initialization_strings() func).
        :param s_size: Input size, essentially the size of pixels in the screen.(int)
        :return: An instance of the RemoteEnvironmentTerminal class. The RemoteEnvironmentTerminal class inherits from
            the abstract Environment class there by guaranteeing that it is possible to interact with the returned
            instances via the interface defined in the above mentioned Environment class.
        """
        self.crash_if_shutdown()

        env_id = self.get_name_for_new_env()

        self.envs_msgs_queues[env_id] = queue.Queue()
        STNE_msg = ProtocolHandler.format_SEEN_msg(env_id, env_init_str, s_size)
        self.msgs2send.put(STNE_msg)

        NEIN_msg = self.get_msg_from_evn_msg_queue(env_id)
        msg_dict = ProtocolHandler.parse_NEIN_msg(NEIN_msg)

        return RemoteEnvironmentTerminal(env_id, msg_dict['a_size'], self)

    def step(self, env_id, action):
        """
        Makes an action and returns a reward.
        :param env_id: (string) used to identify the environment. See docs in ProtocolHandler class.
        :param action: (list) action to do (must be from self.possible_action)
        :return: reward(float) ( R(s,a) )
        """
        self.crash_if_shutdown()

        SA2M_msg = ProtocolHandler.format_SA2M_msg(env_id, action)
        self.msgs2send.put(SA2M_msg)

        STAT_msg = self.get_msg_from_evn_msg_queue(env_id)
        msg_dict = ProtocolHandler.parse_STAT_msg(STAT_msg)

        state = msg_dict['state']
        reward_for_last_action = msg_dict['reward_for_last_action']
        is_terminal = msg_dict['is_terminal_state']

        return state, reward_for_last_action, is_terminal

    def get_post_terminal_step_reward(self, env_id):
        """
        Returns the post terminal step reward.
        :param env_id: (string) used to identify the environment. See docs in ProtocolHandler class.
        :return: post terminal step reward (float)
        """
        self.crash_if_shutdown()

        GPTR_msg = ProtocolHandler.format_GPTR_msg(env_id)
        self.msgs2send.put(GPTR_msg)

        PTSR_msg = self.get_msg_from_evn_msg_queue(env_id)
        msg_dict = ProtocolHandler.parse_PTSR_msg(PTSR_msg)

        post_terminal_step_reward = msg_dict['post_terminal_step_reward']

        return post_terminal_step_reward

    def get_unprocessed_state_screen_buffer(self, env_id, requested_screen_size):
        """
        Makes an action and returns a reward.
        :param env_id: (string) used to identify the environment. See docs in ProtocolHandler class.
        :param requested_screen_size: requested screen size (tuple of ints, [width, height].
                                                             None will return the screen with it's original size.))
        :return: unprocessed screen. (numpy array)
        """
        self.crash_if_shutdown()

        GUPS_msg = ProtocolHandler.format_GUPS_msg(env_id, requested_screen_size)
        self.msgs2send.put(GUPS_msg)

        UPSC_msg = self.get_msg_from_evn_msg_queue(env_id)
        msg_dict = ProtocolHandler.parse_UPSC_msg(UPSC_msg)

        state = msg_dict['state']

        return state

    def start_new_episode(self, env_id):
        """
        Starts a new episode, returns the initial state and whether or not it is terminal.
        :param env_id: (string) used to identify the environment. See docs in ProtocolHandler class.
        :return: tuple (initial_step (numpy array), is_terminal (bool))
        """
        self.crash_if_shutdown()

        STNE_msg = ProtocolHandler.format_STNE_msg(env_id)
        self.msgs2send.put(STNE_msg)

        NEST_msg = self.get_msg_from_evn_msg_queue(env_id)
        msg_dict = ProtocolHandler.parse_NEST_msg(NEST_msg)

        return msg_dict['initial_state'], msg_dict['is_terminal']

    def close(self, env_id):
        """
        Closes the specified environment.
        :param env_id: (string) used to identify the environment. See docs in ProtocolHandler class.
        :return: None
        """
        self.crash_if_shutdown()

        CLEN_msg = ProtocolHandler.format_CLEN_msg(env_id)
        self.msgs2send.put(CLEN_msg)

    def get_name_for_new_env(self):
        """
        Utility function. Generates env_ids for new environments.
        :return: New env_id for a new environment.
        """
        with self.cnt_lock:
            name = self.EnvironmentIDFormat % self.cnt
            self.cnt += 1
        return name

    def crash_if_shutdown(self):
        """
        Crashes if self is not running.
        Prevents us from trying to communicate with server when sockets have been closed.
        :return: None
        """
        if self.status != Status.Running:
            print("The RemoteEnvironmentsManager must be running for communication with the server to be possible.")
            raise ManagerUsedIsNotRunningError()
        
    def get_msg_from_evn_msg_queue(self, env_id):
        """
        Fetches the top message in the environment's msg_queue. Blocks for RemoteEnvironmentsManager.QueueGetTimeout
        seconds. Crashes if queue is empty and wait time is timed out.
        :param env_id: (string) used to identify the environment. See docs in ProtocolHandler class.
        :return: If msg queue isn't empty, returns the top msg in the queue. Raises an
        TimeoutExceededTryingToGetMsgError error otherwise.
        """
        try:
            return self.envs_msgs_queues[env_id].get(timeout=self.QueueGetTimeout)
        except queue.Empty:
            raise TimeoutExceededTryingToGetMsgError()

    @staticmethod
    def get_msg_env_id(msg):
        """
        Returns the env_id of the environment for which the msg is meant.
        :param msg: msg (string)
        :return: Environment id (string).
        """
        msg_id_to_func = {
                          ProtocolHandler.State_Msg_Identifier: ProtocolHandler.parse_STAT_msg,
                          ProtocolHandler.New_Environment_Initiated_Msg_Identifier: ProtocolHandler.parse_NEIN_msg,
                          ProtocolHandler.New_Episode_Started_Msg_Identifier: ProtocolHandler.parse_NEST_msg,
                          ProtocolHandler.Unprocessed_Screen_Msg_Identifier: ProtocolHandler.parse_UPSC_msg,
                          ProtocolHandler.Post_Terminal_Step_Reward_Msg_Identifier: ProtocolHandler.parse_PTSR_msg
                         }
        parsed = msg_id_to_func[msg[:ProtocolHandler.MSG_IDENTIFIER_LEN]](msg)
        if type(parsed) == dict:
            return parsed['environment_id']
        else:
            return parsed

    @staticmethod
    def handle_if_EROR_msg(msg):
        """
        Checks if the msg is an EROR_msg. If it is, it prints a description and raises an exception.
        :param msg:
        :return:
        """
        if msg[:ProtocolHandler.MSG_IDENTIFIER_LEN] == ProtocolHandler.Error_Msg_Identifier:
            print('An Error has occurred! Description: %s' % ProtocolHandler.parse_EROR_msg(msg))
            raise ERORMsgEncountered()

    @staticmethod
    def check_for_server(ip, port):
        """
        Returns True if an Environments_Server is listening in (ip, port). False otherwise.
        :param ip: IP address of server. Assumes the IP address is a valid one. (string)
        :param port: Server port. Assumes the port number is a valid one. (int)
        :return: True if an Environments_Server is listening in (ip, port). False otherwise.
        """
        sock = socket.socket()
        try:
            try:
                socket.setdefaulttimeout(RemoteEnvironmentsManager.Server_Check_Timeout)
                sock.connect((ip, port))
                socket.setdefaulttimeout(None)
            except socket.timeout:
                socket.setdefaulttimeout(None)
                return False
            send_by_size(sock, ProtocolHandler.format_PING_msg())
            rlist, _, _ = select.select([sock], [], [], RemoteEnvironmentsManager.Server_Check_Timeout)
            if sock in rlist:
                success = receive_by_size(sock)
                send_by_size(sock, ProtocolHandler.format_EXIT_msg())
                return success
        except (ConnectionRefusedError, OSError) as e:
            pass
        return False


class TimeoutExceededTryingToGetMsgError(Exception):
    pass


class ERORMsgEncountered(Exception):
    pass


class ManagerUsedIsNotRunningError(Exception):
    pass


class RemoteEnvironmentTerminal(Environment):
        def __init__(self, env_id, a_size, manager):
            """
            A class representing an environment run on a remote server, allowing interaction with said environment from
            this computer using the abstract Environment class interface.
            The instances of this class do not directly communicate with th server. Instead they communicate with the
            RemoteEnvironmentsManager instance set as their manager and it communicates with server for them.
            :param env_id: (string) used to identify the environment. See docs in ProtocolHandler class.
            :param a_size: Action space. The number of actions possible in our environment. (int)
            :param manager: The instance of the RemoteEnvironmentsManager who created the instance.
            """
            self.env_id = env_id
            super(RemoteEnvironmentTerminal, self).__init__(a_size)
            self.manager = manager
            self.curr_state = None
            self.last_reward = None
            self.is_curr_state_terminal = None
            self.total_reward = None
            self._post_terminal_step_reward = None

        def step(self, action):
            """
            Makes an action and returns a reward.
            self.step() is a wrapper to this function so please use self.step().
            :param action: (list) action to do (must be from self.possible_action)
            :return: reward(int) ( R(s,a) )
            """
            self.curr_state, self.last_reward, self.is_curr_state_terminal = self.manager.step(self.env_id, action)
            self.total_reward += self.last_reward
            return self.last_reward

        def get_post_terminal_step_reward(self):
            """
            Returns the post terminal step reward.
            :return: post terminal step reward (float)
            """
            if self._post_terminal_step_reward is None:
                self._post_terminal_step_reward = self.manager.get_post_terminal_step_reward(self.env_id)
            return self._post_terminal_step_reward

        def get_state_screen_buffer(self):
            """
            DO NOT use this function!!! Use the .get_state_screen_buffer() function instead
            returns the current state's screen buffer.
            self.get_state_screen_buffer() is a wrapper to this function so please use self.get_state_screen_buffer().
            :return: returns the current state's screen buffer.
            """
            return np.copy(self.curr_state)

        def get_unprocessed_state_screen_buffer(self, requested_screen_size):
            """
            returns the unprocessed screen of size requested_screen_size .
            :param requested_screen_size: requested screen size (tuple of ints, [width, height].
                                                                 None will return the screen with it's original size.)
            :return: returns the unprocessed screen. (numpy array)
            """
            return self.manager.get_unprocessed_state_screen_buffer(self.env_id, requested_screen_size)

        def start_new_episode(self):
            """
            Starts a new episode.
            :return: None
            """
            self.curr_state, self.is_curr_state_terminal = self.manager.start_new_episode(self.env_id)
            self.total_reward = 0

        def is_episode_finished(self):
            """
            Returns True if episode is finished, False otherwise.
            :return: True if episode is finished, False otherwise.
            """
            return self.is_curr_state_terminal

        def get_total_reward(self):
            """
            Returns the accumulated total reward from all states since episode start.
            :return: The accumulated total reward from all states since episode start.
            """
            return self.total_reward

        def close(self):
            """
            Close environment.
            :return: None
            """
            self.manager.close(self.env_id)
            self.curr_state = None
            self.last_reward = None
            self.is_curr_state_terminal = None
            self.total_reward = None


def main():
    ip, port = "10.0.0.3", 5381
    env_manager = RemoteEnvironmentsManager(ip, port)
    print(env_manager.get_available_environments_initialization_strings())

    from PIL import Image
    import time
    import os

    def play_episode(thread_name, save_path):
        env = env_manager.get_new_remote_environment_terminal("Defend_the_Center", [84, 84])
        cnt = 0
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        env.start_new_episode()
        while not env.is_episode_finished():
            Image.fromarray(env.get_state_screen_buffer() * 255).convert('RGB').save(os.path.join(save_path,
                                                                                                  "%d.bmp" % cnt))
            cnt += 1

            poss_actions = env.possible_actions
            action = poss_actions[np.random.choice(len(poss_actions))]
            env.step(action)
            print(env.is_episode_finished(), thread_name)
        print("Total Reward:", env.get_total_reward())
        env.close()

    path = r"test"
    if not os.path.isdir(path):
        os.mkdir(path)
    threads = [threading.Thread(target=play_episode, args=(str(i), os.path.join(path, str(i)))) for i in range(10)]
    start = time.time()
    [t.start() for t in threads]
    [t.join() for t in threads]
    print(time.time() - start)

    env_manager.shutdown()
    env_manager.join()


if __name__ == '__main__':
    main()
