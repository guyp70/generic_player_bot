import pickle
import numpy as np
import threading
import socket


MSG_SIZE_HEADER_LENGTH = 10


class ProtocolHandler(object):
    MSG_IDENTIFIER_LEN = 4
    State_Msg_Identifier = b"STAT"
    Send_Action_Make_Msg_Identifier = b"SA2M"
    Get_Post_Terminal_Step_Reward_Msg_Identifier = b"GPTR"
    Post_Terminal_Step_Reward_Msg_Identifier = b"PTSR"
    Setup_Environment_Msg_Identifier = b"SEEN"
    New_Environment_Initiated_Msg_Identifier = b"NEIN"
    Close_Environment_Msg_Identifier = b"CLEN"
    Start_New_Episode_Msg_Identifier = b"STNE"
    New_Episode_Started_Msg_Identifier = b"NEST"
    Request_Available_Environment_Initialization_Strings_Msg_Identifier = b"RAES"
    Available_Environment_Initialization_Strings_Msg_Identifier = b"AEIS"
    Get_Unprocessed_Screen_Msg_Identifier = b"GUPS"
    Unprocessed_Screen_Msg_Identifier = b"UPSC"
    Ping_Msg_Identifier = b"PING"
    Get_Menu_Msg_Identifier = b"GTMN"
    Menu_Msg_Identifier = b"MENU"
    Error_Msg_Identifier = b"EROR"
    Exit_Msg_Identifier = b"EXIT"

    """ 
        STAT - State msg. Data Structure: 4 bytes msg identifier, pickled dict with values for the following keys:  
               environment_id (str), state(numpy array), reward_for_last_action (float), is_terminal_state (boolean).  
        SA2M - Send action to make msg. Data Structure: 4 bytes msg identifier, pickled dict with values for the  
               following keys: environment_id (str), action (int - between 0 and a_size).  
        GPTR - Get post terminal step reward msg. Data Structure: 4 bytes msg identifier, environment_id (str).  
        PTSR - Post Terminal step reward msg. Data Structure: 4 bytes msg identifier, pickled dict with values for the  
               following keys: environment_id (str), post_terminal_step_reward (float).  
        SEEN - Setup environment msg. Data Structure: 4 bytes msg identifier, pickled dict with values for the following  
               keys: environment_id (str), environment_string (string), s_size (int)).  
        NEIN - New environment initiated msg. Data Structure: 4 bytes msg identifier, pickled dict with values for the  
               following keys: new_environment_id (str), a_size (int).  
        CLEN - Close Environment. Data Structure: 4 bytes msg identifier, pickled environment_id (string).  
        STNE - Start a new episode. Data Structure: 4 bytes msg identifier, pickled environment_id (string).  
        NEST - New episode started msg. 4 bytes msg identifier pickled dict with values for the following keys:  
               environment_id, initial state (numpy array), is_terminal (bool).  
        RAES - Request all available environment initialization strings msg. Data Structure: 4 bytes msg identifier.  
        AEIS - All available environment initialization strings msg. Data Structure: 4 bytes msg identifier, pickled  
               list of all available environments strings.  
        GUPS - Get unprocessed screen message. Data Structure: 4 bytes msg identifier, pickled dict with values for the  
               following keys: environment_id (str), requested_screen_size (tuple of ints, [width, height]).  
        UPSC - Unprocessed screen message. Data Structure: 4 bytes msg identifier, pickled dict with values for the  
               following keys: environment_id (str), state(numpy array).  
        GTMN - Get menu message. Data Structure: 4 bytes msg identifier, pickled tuple of integers signifying the size  
                                                 of the icon display (width (int) , height (int)).  
        MENU - Menu message. Data Structure: 4 bytes msg identifier, pickled dict where each key is the initialization  
               string of an available environment while it's value is the environment's icon (numpy array of image the  
               size of icon_display_size as passed in the GTMN message).  
        PING - Ping message. Data Structure: 4 bytes msg identifier.  
        EROR - Error msg. Data Structure: 4 bytes msg identifier, pickled error description string.  
        EXIT - exit msg. Data Structure: 4 bytes msg identifier  
          
        Just to make it clear, environment initialization strings are the string used to describe the type of the 
        environment to initiate and are not to be confused with the environment_id strings which are used to identify 
        and refer to instances of environments already initiated.  
        Environment initialization strings are chosen from the list offered in AEIS messages while environment ids are 
        chosen by the client.  
    """

    @staticmethod
    def format_STAT_msg(environment_id, state, reward_for_last_action, is_terminal_state):
        """
        Format a state msg, sent by the server after receiving a SA2M message.
        :param environment_id: environment id (string)
        :param state: state at time t (numpy array)
        :param reward_for_last_action: reward for action made at time t-1 (float)
        :param is_terminal_state: True if state at time t is terminal state. False otherwise.
        :return: msg string to send
        """
        msg_dict = {'environment_id': environment_id, 'state': state, 'reward_for_last_action': reward_for_last_action,
                    'is_terminal_state': is_terminal_state}
        return ProtocolHandler.State_Msg_Identifier + pickle.dumps(msg_dict)

    @staticmethod
    def parse_STAT_msg(msg):
        """
        Parse a state msg, sent by the server after receiving a SA2M message.
        :param msg: message string
        :return: Returns msg dict containing values for keys: environment_id (string), state (numpy array of s_size), reward_for_last_action (float),
                 is_terminal_state (boolean)
        """
        msg_identifier = msg[:ProtocolHandler.MSG_IDENTIFIER_LEN]
        msg = msg[ProtocolHandler.MSG_IDENTIFIER_LEN:]
        return pickle.loads(msg)

    @staticmethod
    def format_SA2M_msg(environment_id, action2make):
        """
        Format a send action to make msg, sent by the client.
        :param environment_id: environment id (string)
        :param action2make: action to make (one hot vector of action, numpy array)
        :return: msg string to send
        """
        msg_dict = {'environment_id': environment_id, 'action2make': np.argmax(action2make),
                    'a_size': np.size(action2make)}
        return ProtocolHandler.Send_Action_Make_Msg_Identifier + pickle.dumps(msg_dict)

    @staticmethod
    def parse_SA2M_msg(msg):
        """
        Parse a send action to make msg, sent by the client."
        :param msg: message string
        :param a_size: a_size (int)
        :return: Returns msg dict containing values for keys: environment_id (string), action2make (one hot vector of size [a_size], numpy array)
        """
        msg_identifier = msg[:ProtocolHandler.MSG_IDENTIFIER_LEN]
        msg = msg[ProtocolHandler.MSG_IDENTIFIER_LEN:]
        msg_dict = pickle.loads(msg)
        msg_dict['action2make'] = np.identity(msg_dict['a_size'], dtype=np.int32)[:, msg_dict['action2make']].tolist()
        return msg_dict

    @staticmethod
    def format_GPTR_msg(environment_id):
        """
        Format a get post terminal step reward msg, sent by the client.
        :param environment_id: environment id (string)
        :return: msg string to send
        """
        return ProtocolHandler.Get_Post_Terminal_Step_Reward_Msg_Identifier + environment_id.encode()

    @staticmethod
    def parse_GPTR_msg(msg):
        """
        Parse a get post terminal step reward msg, sent by the client."
        :param msg: message string
        :return: Returns environment_id (string)
        """
        msg_identifier = msg[:ProtocolHandler.MSG_IDENTIFIER_LEN]
        msg = msg[ProtocolHandler.MSG_IDENTIFIER_LEN:]
        env_id = msg
        return env_id.decode()

    @staticmethod
    def format_PTSR_msg(environment_id, post_terminal_step_reward):
        """
        Format a post terminal step reward msg, sent by the server after receiving a GPTR message.
        :param environment_id: environment id (string)
        :param post_terminal_step_reward: The reward for a post terminal state.
        :return: msg string to send
        """
        msg_dict = {'environment_id': environment_id, 'post_terminal_step_reward': post_terminal_step_reward}
        return ProtocolHandler.Post_Terminal_Step_Reward_Msg_Identifier + pickle.dumps(msg_dict)

    @staticmethod
    def parse_PTSR_msg(msg):
        """
        Parse a post terminal step reward msg, sent by the server after receiving a GPTR message.
        :param msg: message string
        :return: Returns msg dict containing value for keys: environment_id (string), post_terminal_step_reward (float)
        """
        msg_identifier = msg[:ProtocolHandler.MSG_IDENTIFIER_LEN]
        msg = msg[ProtocolHandler.MSG_IDENTIFIER_LEN:]
        msg_dict = pickle.loads(msg)
        return msg_dict

    @staticmethod
    def format_SEEN_msg(environment_id, environment_initialization_string, s_size):
        """
        Format a setup a new environment message, sent by the client to initiate an environment instance on the server side.
        :param new_environment_id: The environment id for the new environment(string)
        :param environment_initialization_string: A string from the ones offered in th AEIS message detailing the
                                                  requested environment (string).
        :param s_size: a_size (int)
        :return: msg string to send
        """
        msg_dict = {'environment_id': environment_id,
                    'environment_initialization_string': environment_initialization_string, 's_size': s_size}
        return ProtocolHandler.Setup_Environment_Msg_Identifier + pickle.dumps(msg_dict)

    @staticmethod
    def parse_SEEN_msg(msg):
        """
        Parse a setup a new environment message, sent by the client to initiate an environment instance on the server side.
        :param msg: message string
        :return: Returns msg dict containing values for keys: environment_id (string), environment_initialization_string (string), s_size (int).
        """
        msg_identifier = msg[:ProtocolHandler.MSG_IDENTIFIER_LEN]
        msg = msg[ProtocolHandler.MSG_IDENTIFIER_LEN:]
        return pickle.loads(msg)

    @staticmethod
    def format_NEIN_msg(environment_id, a_size):
        """
        Format a new environment initiated msg, sent by the sever after receiving a SEEN message.
        :param environment_id: environment id (string)
        :param a_size: a_size (int)
        :return: msg string to send
        """
        msg_dict = {'environment_id': environment_id, 'a_size': a_size}
        return ProtocolHandler.New_Environment_Initiated_Msg_Identifier + pickle.dumps(msg_dict)

    @staticmethod
    def parse_NEIN_msg(msg):
        """
        Parse a new environment initiated msg, sent by the sever after receiving a SEEN message.
        :param msg: message string
        :return: Returns a pickled dict containing values for keys: environment_id (string), a_size (int)
        """
        msg_identifier = msg[:ProtocolHandler.MSG_IDENTIFIER_LEN]
        msg = msg[ProtocolHandler.MSG_IDENTIFIER_LEN:]
        return pickle.loads(msg)

    @staticmethod
    def format_STNE_msg(environment_id):
        """
        Format a start a new episode message, sent by the client.
        :param environment_id: environment id (string)
        :return: msg string to send
        """
        return ProtocolHandler.Start_New_Episode_Msg_Identifier + pickle.dumps(environment_id)

    @staticmethod
    def parse_STNE_msg(msg):
        """
        Parse a start a new episode message, sent by the client.
        :param msg: message string
        :return: Returns environment_id
        """
        msg_identifier = msg[:ProtocolHandler.MSG_IDENTIFIER_LEN]
        msg = msg[ProtocolHandler.MSG_IDENTIFIER_LEN:]
        return pickle.loads(msg)

    @staticmethod
    def format_CLEN_msg(environment_id):
        """
        Format a close environment message, sent by the client.
        :param environment_id: environment id (string)
        :return: msg string to send
        """
        return ProtocolHandler.Close_Environment_Msg_Identifier + pickle.dumps(environment_id)

    @staticmethod
    def parse_CLEN_msg(msg):
        """
        Parse a close environment message, sent by the client.
        :param msg: message string
        :return: Returns environment_id
        """
        msg_identifier = msg[:ProtocolHandler.MSG_IDENTIFIER_LEN]
        msg = msg[ProtocolHandler.MSG_IDENTIFIER_LEN:]
        return pickle.loads(msg)

    @staticmethod
    def format_NEST_msg(environment_id, initial_state, is_terminal):
        """
        Format a new episode started msg, sent by the server after receiving a STNE message.
        :param environment_id: environment id (string)
        :param initial_state: initial state at time step 0 (numpy array)
        :param is_terminal: is the initial state also the terminal state (bool)
        :return: msg string to send
        """
        msg_dict = {'environment_id': environment_id, 'initial_state': initial_state, 'is_terminal': is_terminal}
        return ProtocolHandler.New_Episode_Started_Msg_Identifier + pickle.dumps(msg_dict)

    @staticmethod
    def parse_NEST_msg(msg):
        """
        Parse a new episode started msg, sent by the server after receiving a STNE message.
        :param msg: message string
        :return: Returns msg dict containing values for keys: environment_id (string),
                 initial_state (numpy array of s_size), is_terminal (bool)
        """
        msg_identifier = msg[:ProtocolHandler.MSG_IDENTIFIER_LEN]
        msg = msg[ProtocolHandler.MSG_IDENTIFIER_LEN:]
        return pickle.loads(msg)

    @staticmethod
    def format_RAES_msg():
        """
        Format a request all available environment initialization strings msg, sent by client.
        :return: msg string to send
        """
        return ProtocolHandler.Request_Available_Environment_Initialization_Strings_Msg_Identifier

    @staticmethod
    def parse_RAES_msg(msg):
        """
        Parse a request all available environment initialization strings msg, sent by client.
        :param msg: message string
        :return: None
        """
        msg_identifier = msg[:ProtocolHandler.MSG_IDENTIFIER_LEN]
        return None

    @staticmethod
    def format_AEIS_msg(all_available_environment_initialization_strings):
        """
        Format a all available environment initialization strings msg, sent by the server after receiving a RAES message.
        :param all_available_environment_initialization_strings: A list of all available environment initialization strings. (list of strings)
        :return: msg string to send
        """
        return ProtocolHandler.Available_Environment_Initialization_Strings_Msg_Identifier + \
               pickle.dumps(all_available_environment_initialization_strings)

    @staticmethod
    def parse_AEIS_msg(msg):
        """
        Parse an all available environment initialization strings msg, sent by the server after receiving a RAES message.
        :return: Returns a list of all available environments strings
        """
        msg_identifier = msg[:ProtocolHandler.MSG_IDENTIFIER_LEN]
        msg = msg[ProtocolHandler.MSG_IDENTIFIER_LEN:]
        return pickle.loads(msg)

    @staticmethod
    def format_GUPS_msg(environment_id, requested_screen_size):
        """
        Format a get unprocessed screen sent by the client.
        :param environment_id: environment id (string)
        :param requested_screen_size: requested screen size (tuple of ints, [width, height])
        :return: msg string to send
        """
        msg_dict = {'environment_id': environment_id, 'requested_screen_size': requested_screen_size}
        return ProtocolHandler.Get_Unprocessed_Screen_Msg_Identifier + pickle.dumps(msg_dict)

    @staticmethod
    def parse_GUPS_msg(msg):
        """
        Parse a get unprocessed screen sent by the client.
        :param msg: message string
        :return: Returns msg dict containing values for keys: environment_id (str),
                                                              requested_screen_size (tuple of ints, [width, height])
        """
        msg_identifier = msg[:ProtocolHandler.MSG_IDENTIFIER_LEN]
        msg = msg[ProtocolHandler.MSG_IDENTIFIER_LEN:]
        return pickle.loads(msg)

    @staticmethod
    def format_UPSC_msg(environment_id, state):
        """
        Format an unprocessed screen message sent by the server as response to a GUPS message.
        :param environment_id: environment id (string)
        :param state: state (numpy array)
        :return: msg string to send
        """
        msg_dict = {'environment_id': environment_id, 'state': state}
        return ProtocolHandler.Unprocessed_Screen_Msg_Identifier + pickle.dumps(msg_dict)

    @staticmethod
    def parse_UPSC_msg(msg):
        """
        Parse an unprocessed screen message sent by the server as response to a GUPS message.
        :param msg: message string
        :return: Returns msg dict containing values for keys: environment_id (str), state(numpy array)
        """
        msg_identifier = msg[:ProtocolHandler.MSG_IDENTIFIER_LEN]
        msg = msg[ProtocolHandler.MSG_IDENTIFIER_LEN:]
        return pickle.loads(msg)

    @staticmethod
    def format_GTMN_msg(icon_display_size):
        """
        Format a GTMN message sent by the client to request a MENU message.
        :param icon_display_size: tupple of integers. ( width (int) , height (int) )
        :return: msg string to send
        """
        return ProtocolHandler.Get_Menu_Msg_Identifier + pickle.dumps(icon_display_size)

    @staticmethod
    def parse_GTMN_msg(msg):
        """
        Parse a GTMN message sent by the client to request a MENU message.
        :return: icon_display_size: tupple of integers. ( width (int) , height (int) )
        """
        msg_identifier = msg[:ProtocolHandler.MSG_IDENTIFIER_LEN]
        msg = msg[ProtocolHandler.MSG_IDENTIFIER_LEN:]
        return pickle.loads(msg)

    @staticmethod
    def format_MENU_msg(env_init_str_to_env_icon_dict):
        """
        Format a menu sent by the server as response to a GTMN message.
        :param env_init_str_to_env_icon_dict: dict of the following structure: {environment_name (string):
                                               environment icon (numpy array of image the size of icon_display_size as
                                               passed in the GTMN message)} for all available environments
        :return: msg string to send
        """
        return ProtocolHandler.Menu_Msg_Identifier + pickle.dumps(env_init_str_to_env_icon_dict)

    @staticmethod
    def parse_MENU_msg(msg):
        """
        Parse a Menu sent by the server as response to a GTMN message.
        :param msg: message string
        :return: Returns msg dict where each key is the initialization string of an available environment while it's
                 value is the environment's icon (numpy array of image the size of icon_display_size as passed in the
                 GTMN message).
        """
        msg_identifier = msg[:ProtocolHandler.MSG_IDENTIFIER_LEN]
        msg = msg[ProtocolHandler.MSG_IDENTIFIER_LEN:]
        return pickle.loads(msg)

    @staticmethod
    def format_PING_msg():
        """
        Format a PING message. If the server receives such a message it returns a Ping message itself.
        :return: msg string to send
        """
        return ProtocolHandler.Ping_Msg_Identifier

    @staticmethod
    def parse_PING_msg(msg):
        """
        Parse a PING message. If the receives such a message it returns a Ping message itself.
        :return: True if msg is a PING msg.
        """
        msg_identifier = msg[:ProtocolHandler.MSG_IDENTIFIER_LEN]
        return ProtocolHandler.Ping_Msg_Identifier == msg_identifier


    @staticmethod
    def format_EROR_msg(error_description_string):
        """
        Format an erroe message.
        :param error_description_string: A string describing the error that has occurred (string)
        :return: msg string to send
        """
        return ProtocolHandler.State_Msg_Identifier + pickle.dumps(error_description_string)

    @staticmethod
    def parse_EROR_msg(msg):
        """
        Parse an error message.
        :param msg: message string
        :return: Returns an error description message (string)
        """
        msg_identifier = msg[:ProtocolHandler.MSG_IDENTIFIER_LEN]
        msg = msg[ProtocolHandler.MSG_IDENTIFIER_LEN:]
        return pickle.loads(msg)

    @staticmethod
    def format_EXIT_msg():
        """
        Format an exit message.
        :return: msg string to send
        """
        return ProtocolHandler.Exit_Msg_Identifier

    @staticmethod
    def parse_EXIT_msg(msg):
        """
        Parse an exit message.
        :return: None
        """
        msg_identifier = msg[:ProtocolHandler.MSG_IDENTIFIER_LEN]
        return None


def read_file(path):
    """
    Returns the contents of the file in the path.
    :param path: Path to the file.
    :return: Data in file. (bytes type)
    """
    with open(path, "rb") as f:
        return f.read()


def write_file(path, data):
    """
    Writes data to the file in the path.
    :param path: Path to the file.
    :param data: Data to save to file.
    :return: None
    """
    with open(path, "wb") as f:
        f.write(data)


class Status:
    """
    Enum class. Standardizes status codes for running objects (ones that inherit from thread or processes).
    """
    ShutDown = "ShutDown"
    Shutting_Down = "Shutting_Down"
    Running = "Running"
    Error_Occurred = "Error_Occurred"


def send_by_size(sock, data):
    """
    A function that sends a msg with a length prefix. Used in tandem with the receive_by_size() func.
    Used together, the two functions make sure each message is received in full and that no left overs remain to
    confuse future readings from the socket.
    :param sock: Socket to send data through. (socket.socket object)
    :param data: Data to send. (bytes object)
    :return: None
    """
    # print(data[:ProtocolHandler.MSG_IDENTIFIER_LEN])
    msg = str(len(data)).zfill(MSG_SIZE_HEADER_LENGTH).encode() + data
    try:
        sock.send(msg)
    except socket.error:
        raise SocketClosedRemotelyError()


def receive_by_size(sock):
    """
    A function that receives a msg with a length prefix. Used in tandem with the send_by_size() func.
    Used together, the two functions make sure each message is received in full and that no left overs remain to
    confuse future readings from the socket.
    :param sock: Socket to send data through. (socket.socket object)
    :return: Data received. (bytes object)
    """
    data = b''
    try:
        len_of_data_to_receive = sock.recv(MSG_SIZE_HEADER_LENGTH)
    except socket.error:
        raise SocketClosedRemotelyError()
    if len_of_data_to_receive == b'':
        raise SocketClosedRemotelyError()
    len_of_data_to_receive = int(len_of_data_to_receive)
    while len_of_data_to_receive > len(data):
        data += sock.recv(len_of_data_to_receive - len(data))
    # print(data[:ProtocolHandler.MSG_IDENTIFIER_LEN])
    return data


class SocketClosedRemotelyError(Exception):
    def __init__(self):
        self.message = "The socket was closed remotely by the other machine"


class Threadable(threading.Thread):
    def __init__(self):
        """
        A class meant to standardize and facilitate the creation of classes meant to run on threads, in parallel.
        Meant to be inherited from.
        """
        super(Threadable, self).__init__()
        self._status_Lock = threading.Lock()
        self.status = Status.ShutDown

    @property
    def status(self):
        """
        status Property getter. Safely returns the value of self._status.
        :return: The value of self._status.
        """
        with self._status_Lock:
            return self._status

    @status.setter
    def status(self, value):
        """
        status Property setter. Safely sets the value of self._status.
        :param value: The value to which we set self._status
        :return: None
        """
        with self._status_Lock:
            self._status = value

    def shutdown(self):
        """
        Call to signal the thread to shutdown.
        :return: None
        """
        self.status = Status.Shutting_Down

    def _protected_run_maker(self, run_func):
        """
        Takes a function and returns a function protected by a try statement.
        Prevents the crash of the thread from affecting the rest of the threads running in parallel and the crash of the
        program as a whole.
        :param run_func: The thread's unprotected run() method.
        :return: Protected run() method.
        """
        self._unprotected_run_func = run_func

        """def protected_run():
            self._unprotected_run_func()"""

        def protected_run():
            debug = False
            if debug:
                self._unprotected_run_func()
            else:
                try:
                    self._unprotected_run_func()
                except Exception as ex:
                    self.status = Status.Error_Occurred
                    raise ex

        return protected_run

    def start(self):
        """
        Overrides the default start func, wrapping the run method with the self._protected_run_maker() method.
        :return: None
        """
        self.run = self._protected_run_maker(self.run)
        super(Threadable, self).start()

    def run(self):
        """
        Example Run method, meant to be overriden.
        When overriding take care to mimic the function's structure so that a change in the self.status property
        actually affects the running of the code. (as in make sure to check every once in a while if self.status isn't
        equals to Status.Running and if it different, exit the loop and shutdown.
        """
        self.status = Status.Running
        while self.status == Status.Running:
            pass  # do work
        self._shutdown_procedure()

    def _shutdown_procedure(self):
        """
        Example _shutdown_procedure method, meant to be overriden.
        Here you should go through the instances shutdown procedure as well as the shutdown of it's components should
        they require such a thing.
        Make sure you set the self.status to Status.ShutDown at the end, to mark that you have completed the necessary
        shutdown procedures.
        """
        self.status = Status.ShutDown


def bytes2str(data):
    """
    Converts bytes like object to strings.
    :param data: bytes type object
    :return: string type object
    """
    return (b"%s" % data).encode("ascii")
