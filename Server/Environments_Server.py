from Shared_Code import *
import socket
import select
from Environments import EnvironmentsInitializer
import settings
import time
import datetime

TimeStampFormat = '%H:%M:%S'  # '%Y-%m-%d %H:%M:%S'

class Server(Threadable):
    Listen_Queue_Size = 10
    Select_Timeout = 0.01

    def __init__(self, ip, port):
        """
        A class that handles the set up and operation of an environments server.
        :param ip: Server ip to bind the server socket to (string). (usually "0.0.0.0")
        :param port: Server port to bind the server socket to (int).
        """
        super(Server, self).__init__()
        self.server_sock = socket.socket()
        self.server_sock.bind((ip, port))
        self.server_sock.listen(self.Listen_Queue_Size)
        self.envs_initializer = EnvironmentsInitializer(environments_settings_folder=
                                                        settings.environments_settings_folder_path)

        self.client_handlers_lock = threading.Lock()
        self.client_handlers = []

    def run(self):
        """
        Runs the server. Essentially the server's main function.
        Accepts connections and passes them to ClientHandler instances who handle their requests.
        :return: None
        """
        self.status = Status.Running
        while self.status == Status.Running:
            rl, _, _ = select.select([self.server_sock], [], [], self.Select_Timeout)
            if self.server_sock in rl:
                with self.client_handlers_lock:
                    client_sock, addr = self.server_sock.accept()
                    self.client_handlers.append(ClientHandler(client_sock, addr,  self.envs_initializer))
                    self.client_handlers[-1].start()
        self._shutdown_procedure()

    def _shutdown_procedure(self):
        """
        Shuts down the server and all of its children ClientHandler instances.
        :return: None
        """
        self.server_sock.close()
        with self.client_handlers_lock:
            [ch.shutdown() for ch in self.client_handlers]
            [ch.join() for ch in self.client_handlers]
        self.status = Status.ShutDown


class ClientHandler(Threadable):
    Select_Timeout = 0.01

    def __init__(self, client_sock, addr, envs_initializer):
        """
        A class that handles communications with a single client and reply to their requests.
        :param client_sock: client socket through which communication with the client is conducted (socket.socket
            object).
        :param addr: Client ip address (string).
        :param envs_initializer: An instance of the EnvironmentsInitializer class.
        """
        super(ClientHandler, self).__init__()
        self.client_sock = client_sock
        self.envs_iniatlizer = envs_initializer
        self.environments = {}
        self.environments_s_size = {}
        self.addr = addr
        print(time_stamp("Client %s has established connection." % str(self.addr)))

    def run(self):
        """
        Runs the ClientHandler. Essentially the ClientHandler's thread main function.
        Receives requestsfrom the client and act accordingly and replies.
        :return: None
        """
        self.status = Status.Running
        msg_identifier_to_funtion = {ProtocolHandler.Setup_Environment_Msg_Identifier: self.handle_SEEN_msg,
                                     ProtocolHandler.Close_Environment_Msg_Identifier: self.handle_CLEN_msg,
                                     ProtocolHandler.Start_New_Episode_Msg_Identifier: self.handle_STNE_msg,
                                     ProtocolHandler.Send_Action_Make_Msg_Identifier: self.handle_SA2M_msg,
                                     ProtocolHandler.Get_Post_Terminal_Step_Reward_Msg_Identifier: self.handle_GPTR_msg,
                                     ProtocolHandler.Request_Available_Environment_Initialization_Strings_Msg_Identifier: self.handle_RAES_msg,
                                     ProtocolHandler.Get_Unprocessed_Screen_Msg_Identifier: self.handle_GUPS_msg,
                                     ProtocolHandler.Get_Menu_Msg_Identifier: self.handle_GTMN_msg,
                                     ProtocolHandler.Ping_Msg_Identifier: self.handle_PING_msg,
                                     ProtocolHandler.Exit_Msg_Identifier: self.handle_EXIT_msg}
        while self.status == Status.Running:
            try:
                rl, _, xl = select.select([self.client_sock], [], [], self.Select_Timeout)
                if self.client_sock in rl:
                    msg = self.receive_by_size()
                    msg_identifier_to_funtion[msg[:ProtocolHandler.MSG_IDENTIFIER_LEN]](msg)
            except SocketClosedRemotelyError:
                print(time_stamp("Client %s disconnected unexpectedly!" % str(self.addr)))
                self.shutdown()
        self._shutdown_procedure()

    def handle_GPTR_msg(self, msg):
        """
        Handles a GPTR message (See Shared_Code.ProtocolHandler class documentation).
        Responds with a PTSR message.
        :param msg: A GPTR message (string)
        :return: None
        """
        env_id = ProtocolHandler.parse_GPTR_msg(msg)
        terminal_reward = self.environments[env_id].get_post_terminal_step_reward()

        # Send a PTSR message
        PTSR_msg = ProtocolHandler.format_PTSR_msg(env_id, terminal_reward)
        self.send_by_size(PTSR_msg)

    def handle_GUPS_msg(self, msg):
        """
        Handles a GUPS message (See Shared_Code.ProtocolHandler class documentation).
        Responds with a UPSC message.
        :param msg: A GUPS message (string)
        :return: None
        """
        msg_dict = ProtocolHandler.parse_GUPS_msg(msg)
        env_id = msg_dict['environment_id']
        unprocessed_screen = self.environments[env_id].get_unprocessed_state_screen_buffer(img_size_to_return=msg_dict['requested_screen_size'])

        # Send a UPSC message
        UPSC_msg = ProtocolHandler.format_UPSC_msg(env_id, unprocessed_screen)
        self.send_by_size(UPSC_msg)

    def handle_SEEN_msg(self, msg):
        """
        Handles a SEEN message (See Shared_Code.ProtocolHandler class documentation).
        Responds with a NEIN message.
        :param msg: A SEEN message (string)
        :return: None
        """
        msg_dict = ProtocolHandler.parse_SEEN_msg(msg)
        self.environments[msg_dict['environment_id']] = self.envs_iniatlizer.get_env(msg_dict['environment_initialization_string'])
        self.environments_s_size[msg_dict['environment_id']] = msg_dict['s_size']

        # Send a NEIN message
        a_size = self.environments[msg_dict['environment_id']].a_size
        NEIN_msg = ProtocolHandler.format_NEIN_msg(msg_dict['environment_id'], a_size)
        self.send_by_size(NEIN_msg)

    def handle_STNE_msg(self, msg):
        """
        Handles a STNE message (See Shared_Code.ProtocolHandler class documentation).
        Responds with a NEST message.
        :param msg: A STNE message (string)
        :return: None
        """
        env_id = ProtocolHandler.parse_STNE_msg(msg)
        if env_id in self.environments:
            self.environments[env_id].start_new_episode()

            # Send a New Episode Started msg
            init_state = self.environments[env_id].get_state_screen_buffer(img_size_to_return=
                                                                           self.environments_s_size[env_id])
            is_terminal = self.environments[env_id].is_episode_finished()
            NEST_msg = ProtocolHandler.format_NEST_msg(env_id, init_state, is_terminal)
            self.send_by_size(NEST_msg)
        else:
            EROR_msg = ProtocolHandler.format_EROR_msg("Unknown env_id was given.")
            self.send_by_size(EROR_msg)
            raise UnknownEnvIDWasSentByClient()

    def handle_CLEN_msg(self, msg):
        """
        Handles a CLEN message (See Shared_Code.ProtocolHandler class documentation).
        :param msg: A CLEN message (string)
        :return: None
        """
        env_id = ProtocolHandler.parse_CLEN_msg(msg)
        if env_id in self.environments:
            self.environments[env_id].close()
            self.environments.pop(env_id)
        else:
            EROR_msg = ProtocolHandler.format_EROR_msg("Unknown env_id was given.")
            self.send_by_size(EROR_msg)
            raise UnknownEnvIDWasSentByClient()

    def handle_SA2M_msg(self, msg):
        """
        Handles a SA2M message (See Shared_Code.ProtocolHandler class documentation).
        Responds with a STAT message.
        :param msg: A SA2M message (string)
        :return: None
        """
        msg_dict = ProtocolHandler.parse_SA2M_msg(msg)
        env_id = msg_dict['environment_id']
        if env_id in self.environments:
            reward = self.environments[env_id].step(msg_dict['action2make'])  # if it crashes here, it may be that
            #                                                                   client tried to make an action after the
            #                                                                   episode ended.

            # Sending a STAT message
            is_terminal = self.environments[env_id].is_episode_finished()
            if not is_terminal:
                state = self.environments[env_id].get_state_screen_buffer(img_size_to_return=self.environments_s_size[env_id])
            else:
                state = np.zeros(self.environments_s_size[env_id], dtype=np.int32)
            STAT_msg = ProtocolHandler.format_STAT_msg(env_id, state, reward, is_terminal)
            self.send_by_size(STAT_msg)
        else:
            EROR_msg = ProtocolHandler.format_EROR_msg("Unknown env_id was given.")
            self.send_by_size(EROR_msg)
            raise UnknownEnvIDWasSentByClient()

    def handle_RAES_msg(self, msg):
        """
        Handles a RAES message (See Shared_Code.ProtocolHandler class documentation).
        Responds with a AEIS message.
        :param msg: A RAES message (string)
        :return: None
        """
        ProtocolHandler.parse_RAES_msg(msg)  # does nothing, here for fun (and consistency)

        # Send an AEIS message
        AEIS_msg = ProtocolHandler.format_AEIS_msg(self.envs_iniatlizer.get_available_envs())
        self.send_by_size(AEIS_msg)

    def handle_GTMN_msg(self, msg):
        """
        Handles a GTMN message (See Shared_Code.ProtocolHandler class documentation).
        Responds with a MENU message.
        :param msg: A GTMN message (string)
        :return: None
        """
        icon_display_size = ProtocolHandler.parse_GTMN_msg(msg)
        env_init_str_to_icon_dict = self.envs_iniatlizer.env_init_str_to_env_icon_dict(icon_display_size)

        # Send an MENU message
        MENU_msg = ProtocolHandler.format_MENU_msg(env_init_str_to_icon_dict)
        self.send_by_size(MENU_msg)

    def handle_PING_msg(self, msg):
        """
        Handles a PING message (See Shared_Code.ProtocolHandler class documentation).
        Responds with a PING message.
        :param msg: A PING message (string)
        :return: None
        """
        if ProtocolHandler.parse_PING_msg(msg):
            self.send_by_size(ProtocolHandler.format_PING_msg())  # return a PING message
            print(time_stamp("Client %s pinged!" % str(self.addr)))

    def handle_EXIT_msg(self, msg):
        """
        Handles a EXIT message (See Shared_Code.ProtocolHandler class documentation).
        Initiates a shutdown of the ClientHandler instance.
        :param msg: A EXIT message (string)
        :return: None
        """
        ProtocolHandler.parse_EXIT_msg(msg)  # does nothing, here for fun (and consistency)
        print(time_stamp("Client %s disconnected." % str(self.addr)))
        self.shutdown()

    def send_by_size(self, data):
        """
        Wrapper for the send_by_size() func made for ease of use.
        Uses the self.client_socket socket.
        :param data: Data to send (bytes object)
        :return: None
        """
        send_by_size(self.client_sock, data)

    def receive_by_size(self):
        """
        Wrapper for the receive_by_size() func made for ease of use.
        Uses the self.client_socket socket.
        :return: Data received (bytes object)
        """
        return receive_by_size(self.client_sock)

    def _shutdown_procedure(self):
        """
        Handle the shut down procedure for the clientHandler instance.
        Closes all open environments before exit.
        :return: None
        """
        self.client_sock.close()
        [env.close() for env in self.environments.values()]
        self.status = Status.ShutDown


def time_stamp(str_to_stamp):
    """
    Adds at the begining of the str an str time of the format set in TimeStampFormat.
    :param str_to_stamp: string to be stamped (string)
    :return: stamped string (string)
    """
    ts = time.time()
    return "%s %s" % (datetime.datetime.fromtimestamp(ts).strftime(TimeStampFormat), str_to_stamp)


class UnknownEnvIDWasSentByClient(Exception):
    pass


def main():
    """Run Server"""
    import time
    import settings
    s = Server(settings.Server_IP, settings.Server_Port)
    s.start()
    try:
        time.sleep(100000)
    except KeyboardInterrupt:
        print("Keyboard Interrupt! Exiting...")
    s.shutdown()
    s.join()


if __name__ == '__main__':
    main()

