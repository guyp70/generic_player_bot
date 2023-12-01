import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.scrolledtext as tkst
from tkinter import ttk
from A3C import A3C_Algorithm, RemoteEnvironmentsManager, EnvironmentsInitializer
import os
import time
import shutil
import threading
import tensorflow as tf
from Utils import SavesManager
from PIL import ImageTk, Image
import socket
import subprocess
import queue
from Remote_Environment import RemoteEnvironmentsManager, TimeoutExceededTryingToGetMsgError, Status,\
                               ManagerUsedIsNotRunningError, SocketClosedRemotelyError
from decimal import Decimal


Default_IP, Default_Port = "10.0.0.3", 5381
env_init_str = "Defend_the_Line"
SAVE_PLAY = False  # if true will save the frames of episode fcrom play to '.\frames'

def main():
    envs_manager = RemoteEnvironmentsManager(Default_IP, Default_Port)
    try:
        start = time.time()

        root = tk.Tk()
        game_screen = GameScreen(root, envs_manager, env_init_str, env_name="Game")
        #game_screen = GameSelectionScreen(root)
        try:
            game_screen.pack(side="top", fill="both", expand=True)
        except tk.TclError:
            pass
        root.mainloop()
        game_screen.shutdown()
        print("Time: %d seconds." % (time.time() - start))
    except KeyboardInterrupt:
        print("Keyboard Interrupt! Exiting...")



def GamesScreen_threader_decorator(function):
    """
    Designed to make synchronous methods in the GamesScreen class run on threads.
    :param function: any GamesScreen instance bound method.
    :return: a functions that will run on a thread.
    """
    def wrapper(self, *args, **kwargs):
        def prepare_work(func2thread):
            def work(self, *work_args, **work_kwargs):
                res = func2thread(self, *work_args, **work_kwargs)
                self.working = False
                return res
            return work
        if self.working:
            self.print_to_gui_console("Busy handling a previous request. "
                                      "Please wait for it to finish before making any new ones.")
            return

        self.working = True
        self.stop_flag = False
        thread = threading.Thread(target=prepare_work(function), args=(self, *args), kwargs=kwargs)
        self.append_thread_to_threads_activated(thread)
        thread.start()
    return wrapper


class GameScreen(tk.Frame):
    Max_Episode_To_Train_In_One_Go = 10000
    Max_Episode_To_Play_In_One_Go = 100
    Display_Update_Intervals = 1000//60  # in mili seconds, must be int
    Display_Update_Queue_Put_Timeout = 2
    Max_Worker_Memory_Buffer_Size = 1000

    def __init__(self, parent, envs_manager, env_init_str, display_size=(320, 180), env_name=""):
        """
        A class that handles the creation and operation of the GUI that interfaces with A3C_Algorithm.
        :param parent: A tk.Tk instance.
        :param envs_manager: An instance of the RemoteEnvironmentsManager class.
        :param env_init_str: An environment initialization string. (get list of available environment initialization
            string from the envs_manager.get_available_environments_initialization_strings() func).
        :param display_size: The size of the display box in which we will show the screen images of the environment
            (tuple of ints (width, height).
            Note that it is not the size of the GameScreen window, only the size of the display box.
        :param env_name: Used in the windows title and in the label at the top of the window. (string)
        """
        self.parent = parent
        self.parent.protocol("WM_DELETE_WINDOW", self.exit)
        self.env_init_str = env_init_str
        self.envs_manager = envs_manager
        self.env_name = env_name
        self.__display_size = display_size

        # the model's hyper parameters
        self.alpha_learning_rate = A3C_Algorithm.DefaultAlphaLearningRate
        self.worker_memory_buffer_size = A3C_Algorithm.DefaultWorkerMemoryBufferSize

        # handle graphics
        tk.Frame.__init__(self, parent)
        self.parent.title(self.env_name)
        self.init_widgets(display_size)

        self.threads_activated_lock = threading.Lock()
        self.threads_activated = []
        self._working_lock = threading.Lock()
        self._working = False
        self._stop_flag_lock = threading.Lock()
        self._stop_flag = False

        self.tensorboard_process = None

        self._display_frames_queue = None
        self._curr_frame = None  # for it's purpose see documentation in ._update_display_screen() func.
        self._streaming_lock = threading.Lock()
        self._streaming = False

        # handle initial prompt (new/load)
        self.a3c_model = None
        initial_prompt = "Do you want to load a save? (yes - load save, no - create a new one)"
        should_load_save = messagebox.askyesno("Load?", initial_prompt)
        if should_load_save:
            self.load_command()
        else:
            self.new_command()

    @property
    def save_folder_path(self):
        """
        save_folder_path Property getter. Safely returns the value of self.a3c_model.save_manger.saves_folder_path.
        :return: The value of self.a3c_model.save_manger.saves_folder_path.
        """
        if self.a3c_model != None:
            return self.a3c_model.save_manger.saves_folder_path
        return None

    @property
    def working(self):
        """
        working Property getter. Safely returns the value of self._working
        :return: The value of self._working
        """
        with self._working_lock:
            return self._working

    @working.setter
    def working(self, value):
        """
        working Property setter. Safely sets the value of self._working.
        :param value: The value to which we set self._working
        :return: None
        """
        with self._working_lock:
            self._working = value

    @property
    def stop_flag(self):
        """
        stop_flag Property getter. Safely returns the value of self._stop_flag
        :return: The value of self._stop_flag
        """
        with self._stop_flag_lock:
            return self._stop_flag

    @stop_flag.setter
    def stop_flag(self, value):
        """
        stop_flag Property setter. Safely sets the value of self._stop_flag
        :param value: The value to which we set self._stop_flag
        :return: None
        """
        with self._stop_flag_lock:
            self._stop_flag = value

    def get_stop_flag(self):
        return self.stop_flag

    @property
    def streaming(self):
        """
        streaming Property getter. Safely returns the value of self._streaming
        :return: The value of self._streaming
        """
        with self._streaming_lock:
            return self._streaming

    @streaming.setter
    def streaming(self, value):
        """
        streaming Property setter. Safely sets the value of self._streaming
        :param value: The value to which we set self._streaming
        :return: None
        """
        with self._streaming_lock:
            self._streaming = value

    def append_thread_to_threads_activated(self, thread):
        """
        Safely appends the thread to the self.threads_activated list.
        :param thread: An instance that inherits from thread.
        :return: None
        """
        with self.threads_activated_lock:
            self.threads_activated.append(thread)

    def init_widgets(self, display_size):
        """
        Handles the initialization of all tk widgets.
        :param display_size: The size of the display box in which we will show the screen images of the environment
            (tuple of ints (width, height).
        :return: None
        """
        # creating a root menu to insert all the sub menus
        self.root_menu = tk.Menu(self.parent)
        self.parent.config(menu=self.root_menu)

        # creating sub menus in the root menu
        self.file_menu = tk.Menu(self.root_menu)  # it intializes a new su menu in the root menu
        self.root_menu.add_cascade(label="File", menu=self.file_menu)  # it creates the name of the sub menu
        self.file_menu.add_command(label="New", command=self.file_new_command)
        self.file_menu.add_command(label="Load", command=self.file_load_command)
        self.file_menu.add_command(label="Stop", command=self.file_stop_command)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.exit)

        self.tensorboard_menu = tk.Menu(self.root_menu)  # it intializes a new su menu in the root menu
        self.root_menu.add_cascade(label="TensorBoard", menu=self.tensorboard_menu)  # it creates the name of the sub menu
        self.tensorboard_menu.add_command(label="Launch Server...", command=self.tensorboard_start_command)
        self.tensorboard_menu.add_command(label="Stop Server", command=self.tensorboard_stop_command)

        self.lable = tk.Label(master=self, text=self.env_name)
        self.lable.grid(row=0, column=0)  # , columnspan=6)

        self.display = tk.Canvas(master=self, width=display_size[0], height=display_size[1])  # canvas
        self.display.grid(column=0, row=1, rowspan=3)

        self.play_button = tk.Button(master=self, text="Play", command=self.play_button_command)
        self.play_button.grid(column=1, row=1)

        self.play_episodes_spinbox = tk.Spinbox(master=self, from_=1, to=self.Max_Episode_To_Play_In_One_Go,
                                                 width=5)
        self.play_episodes_spinbox.grid(column=2, row=1, sticky='w')

        self.train_button = tk.Button(master=self, text="Train", command=self.train_button_command)
        self.train_button.grid(column=1, row=2)

        self.train_episodes_spinbox = tk.Spinbox(master=self, from_=1, to=self.Max_Episode_To_Train_In_One_Go,
                                                 width=5)
        self.train_episodes_spinbox.grid(column=2, row=2, sticky='w')

        self.update_hyper_parameters_button = tk.Button(master=self, text="Update Hyper Parameters",
                                                      command=self.update_hyper_parameters_button_command)
        self.update_hyper_parameters_button.grid(column=3, row=1, rowspan=2)

        self.alpha_learning_rate_lable = tk.Label(master=self, text="Alpha Learning Rate")
        self.alpha_learning_rate_lable.grid(column=4, row=1)

        self.alpha_learning_rate_entry = tk.Entry(master=self)
        self.alpha_learning_rate_entry.insert(tk.END, '%.E' % Decimal(self.alpha_learning_rate))
        self.alpha_learning_rate_entry.grid(column=5, row=1)

        self.memory_buffer_size_lable = tk.Label(master=self, text="Memory Buffer Size")
        self.memory_buffer_size_lable.grid(column=4, row=2)

        self.memory_buffer_size_entry = tk.Entry(master=self)
        self.memory_buffer_size_entry.insert(tk.END, str(self.worker_memory_buffer_size))
        self.memory_buffer_size_entry.grid(column=5, row=2)

        self.gui_console = tkst.ScrolledText(master=self, width=70, height=10, state=tk.DISABLED)
        self.gui_console.grid(column=1, row=3, columnspan=5)

    @GamesScreen_threader_decorator
    def update_hyper_parameters_button_command(self):
        """
        Updates the model's hyper parameters.
        Decorated with the @GamesScreen_threader_decorator.
        :return: None
        """
        try:
            alpha_learning_rate = float(self.alpha_learning_rate_entry.get())
            if alpha_learning_rate <= 0:
                raise ValueError()
        except ValueError:
            self.print_to_gui_console("Please enter a valid alpha learning rate. (must be greater than 0 and big enough"
                                      "that python doesn't treat it as 0)\r\nUpdate Aborted.")
            return
        try:
            memory_buffer_size = int(self.memory_buffer_size_entry.get())
            if not (0 < memory_buffer_size <= self.Max_Worker_Memory_Buffer_Size):
                raise ValueError()
        except ValueError:
            self.print_to_gui_console("Please enter a valid alpha learning rate. (must be an integer greater than 0 and"
                                      " smaller or equal to %d)\r\nUpdate Aborted." %
                                      self.Max_Worker_Memory_Buffer_Size)
            return
        self.print_to_gui_console("Updating Hyper Parameters...\r\n(Note that this action can not be stopped by the "
                                  "'Stop' command in the 'File' menu) ")
        self.alpha_learning_rate, self.worker_memory_buffer_size = alpha_learning_rate, memory_buffer_size
        self.init_a3c_algorithm()
        self.print_to_gui_console("Hyper Parameters Updated.")

    def reset_graphics(self):
        """
        Resets the widgets.
        :return: None
        """
        self.gui_console.edit_reset()

    def init_a3c_algorithm(self, save_folder_path=None):
        """
        Use after every use of the .train() func and at first when initiating.
        :param save_folder_path: if None retain the previous save_folder_path.
        :return: fresh instance of A3C_Algorithm class
        """
        if save_folder_path is None:
            save_folder_path = self.save_folder_path

        tf.reset_default_graph()
        if self.a3c_model != None:
            self.a3c_model.shutdown()
        self.a3c_model = A3C_Algorithm(self.envs_manager, self.env_init_str, Play_Display_Size=self.__display_size,
                                       save_folder_path=save_folder_path,
                                       Worker_Memory_Buffer_Size=self.worker_memory_buffer_size,
                                       AlphaLearningRate=self.alpha_learning_rate)

    @GamesScreen_threader_decorator
    def file_new_command(self):
        """
        Calls the self.new_command() func.
        Decorated with the @GamesScreen_threader_decorator.
        :return: None
        """
        self.new_command()

    def new_command(self):
        """
        Handles the process of creating a new save.
        :return: None
        """
        folder_path = self.ask4folder(window_title="Choose a directory to create a new save in...")

        # if the user presses 'cancel' (If the user presses 'cancel', self.ask4folder() returns '')
        if folder_path == '':
            prompt = "Do you want to exit then? (yes - exit, no - create a new save)"
            should_exit = messagebox.askyesno("Exit?", prompt)
            if should_exit:
                self.after(1, func=self.exit)   # I don't call it directly because if the exit() func is called before
                                                # the tk mainloop is started (can happan in the initial promp), it will
                                                # have no effect. By using the .after() I insure that the mainloop was
                                                # started.
            else:
                self.new_command()
            return

        if len(os.listdir(folder_path)) > 0:
            warning_msg = "You have cosen a folder that contains file. Should you proceed all said files will be " \
                          "deleted. Proceed?"
            if messagebox.askyesno("Python", warning_msg):
                delete_dir_contents(folder_path)
        self.reset_graphics()
        self.init_a3c_algorithm(folder_path)
        self.print_to_gui_console("New Model initiated.")

    @GamesScreen_threader_decorator
    def file_load_command(self):
        """
        Calls the self.load_command() func.
        Decorated with the @GamesScreen_threader_decorator.
        :return: None
        """
        self.load_command()
    
    def load_command(self):
        """
        Handles the process of loading a save.
        :return: None
        """
        folder_path = self.ask4folder(window_title="Choose a directory to load a save from...")

        if SavesManager.check_for_save(folder_path):
            self.reset_graphics()
            self.init_a3c_algorithm(folder_path)
            self.print_to_gui_console("Model Loaded.")
        else:
            initial_prompt = "No Save Found! Do you want to try and load a differnt save? " \
                             "(yes - load a different save, no - create a new save)"
            should_load_save = messagebox.askyesno("Load?", initial_prompt)
            if should_load_save:
                self.load_command()
            else:
                self.new_command()

    def file_stop_command(self, verbose=True):
        """
        Stops all current operations. Cannot stop the self.update_hyper_parameters_button_command() func.
        :param verbose: Should or should not print textual report to GUI Console (bool).
        :return: None
        """
        self.tensorboard_stop_command(verbose=False)
        self.stop_flag = True
        if verbose:
            self.print_to_gui_console("Stopping operations...")

    def exit(self):
        """
        Exit tk.mainloop(). (see main)
        :return: None
        """
        self.parent.quit()

    def shutdown(self):
        """
        Shutdown this instance and all of it's components.
        :return: None
        """
        self.file_stop_command(verbose=False)
        self.tensorboard_stop_command(verbose=False)
        with self.threads_activated_lock:
            for t in self.threads_activated:
                if t.is_alive():
                    t.join()
        if self.a3c_model != None:
            self.a3c_model.shutdown()
        if self.envs_manager != None and self.envs_manager.status == Status.Running:
            self.envs_manager.shutdown()
            self.envs_manager.join()

    def tensorboard_start_command(self):
        """
        Starts a process that runs a tensorboard server allowing to view the logs data.
        :return: None
        """
        with self._working_lock:
            if self._working:
                self.print_to_gui_console("Busy handling a previous request. "
                                          "Please wait for it to finish before making any new ones.")
                return
            self._working = True

        if self.tensorboard_process == None:
            if not is_port_taken(6006):  # checks that port 6006 isn't taken.
                cmd_line = "tensorboard --logdir \"%s\"" % os.path.join(self.save_folder_path,
                                                                        self.a3c_model.LogsFolder)
                self.tensorboard_process = subprocess.Popen(cmd_line)
                self.print_to_gui_console("TensorBoard server active. Available at 'localhost:6006'")
            else:
                self.print_to_gui_console("It seems that some program is already using port 6006. "
                                          "Aborting TensorBoard activation.")
        else:
            self.print_to_gui_console("TensorBoard is already active.")

    def tensorboard_stop_command(self, verbose=True):
        """
        Terminates the process that runs a tensorboard server allowing to view the logs data.
        :param verbose: Should or should not print textual report to GUI Console (bool).
        :return: None
        """
        if self.tensorboard_process != None:
            self.tensorboard_process.terminate()
            self.tensorboard_process = None
            self.working = False
            if verbose:
                self.print_to_gui_console("TensorBoard server Terminated.")
        else:
            if verbose:
                self.print_to_gui_console("It seems that TensorBoard isn't active. Termination aborted.")


    @GamesScreen_threader_decorator
    def play_button_command(self):
        """
        Handles playing and displaying of the episodes played by the A3C_Algorithm.
        Decorated with the @GamesScreen_threader_decorator.
        :return: None
        """
        if not self.streaming:
            try:
                eps2play = int(self.play_episodes_spinbox.get())
                if eps2play <= 0:
                    raise ValueError()
            except ValueError:
                self.print_to_gui_console("Please enter a valid number of episodes to play.")
                return
            else:
                self.print_to_gui_console("Preparing model to play...")

            self._display_frames_queue = queue.Queue(maxsize=1)

            self.streaming = True

            self.display.after(self.Display_Update_Intervals, func=self._update_display_screen)
            try:
                self.make_sure_envs_manager_is_active()
                self.a3c_model.play(n_episodes=eps2play, print_func=self.print_to_gui_console,
                                    frame_update_func=self._update_display_frames_queue,
                                    get_stop_flag=self.get_stop_flag)
            except (TypeError, RuntimeError, TimeoutExceededTryingToGetMsgError, tf.errors.CancelledError):
                pass  # Might happen when the client disconnects
            except ManagerUsedIsNotRunningError:
                self.print_to_gui_console("It seems that the environment manager shut down.\r\n"
                                          "Please try again.")
            except SocketClosedRemotelyError:
                self.print_to_gui_console(
                    "It seems that connection with the server has been lost.\r\nPlease try again.")
            self.streaming = False
            self.stop_flag = False

        else:
            self.print_to_gui_console("Wait for the current play to end.")

    def _update_display_frames_queue(self, frame):
        """
        A function that takes an image and puts to the  self._display_frames_queue so that the self.display will show it.
        :param frame: An instance of PIL.Image.Image class.
        :return: None
        """
        try:
            self._display_frames_queue.put(frame, block=True, timeout=self.Display_Update_Queue_Put_Timeout)
        except queue.Full:
            pass

    def _update_display_screen(self):
        """
        Safely takes an image from self._display_frames_queue and diplay it in self.display.
        :return: None
        """
        if self.streaming:
            if not self._display_frames_queue.empty():
                img = self._display_frames_queue.get()

                # ---DEBUG START---
                if SAVE_PLAY:
                    if not os.path.exists(".\\frames\\"):
                        os.mkdir(".\\frames\\")
                    img.save(".\\frames\\%s.jpg" % time.time())
                # ---DEBUG END---

                frame_image = ImageTk.PhotoImage(image=img)
                self._curr_frame = frame_image  # There seems to a bug with ImageTk and the create_image func where if
                #                                 we don't deliberately save the ImageTk.PhotoImage() to some variable,
                #                                 the garbage collector collects and deletes it before the image is
                #                                 loaded to the canvas and the image the create_image() func loads is
                #                                 garbage data. That is why we assign it to a variable.
                #                                 For more details see:
                #                                 https://effbot.org/pyfaq/why-do-my-tkinter-images-not-appear.htm
                self.display.create_image((0, 0), image=frame_image, anchor=tk.NW)
            self.display.after(self.Display_Update_Intervals, func=self._update_display_screen)

    @GamesScreen_threader_decorator
    def train_button_command(self):
        """
        Handles the training of the model.
        Decorated with the @GamesScreen_threader_decorator.
        :return: None
        """
        try:
            eps2train = int(self.train_episodes_spinbox.get())
            if eps2train <= 0:
                raise ValueError()
        except ValueError:
            self.print_to_gui_console("Please enter a valid number of episodes to train.")
            return
        else:
            self.print_to_gui_console("Preparing model for training... \r\nThe following Hyper Parameters' values will "
                                      "be used:\r\nAlpha Learning Rate: " "%f\r\nMemory Buffer Size: %d" %
                                      (self.alpha_learning_rate, self.worker_memory_buffer_size))
        try:
            self.make_sure_envs_manager_is_active()
            self.a3c_model.train(eps2train, print_func=self.print_to_gui_console, get_stop_flag=self.get_stop_flag)
        except (TypeError, RuntimeError, TimeoutExceededTryingToGetMsgError, tf.errors.CancelledError):
            pass  # Might happen when the client disconnects
        except ManagerUsedIsNotRunningError:
            self.print_to_gui_console("It seems that the environment manager shut down.\r\n"
                                      "Please try again.")
        except SocketClosedRemotelyError:
            self.print_to_gui_console("It seems that connection with the server has been lost.\r\nPlease try again.")
        self.stop_flag = False

    def make_sure_envs_manager_is_active(self):
        """
        Makes sure that self.envs_manager is active and we can use it. Will fix it if self.envs_manager isn't active.
        :return: None
        """
        if self.envs_manager.status != Status.Running:
            self.envs_manager = RemoteEnvironmentsManager(*self.envs_manager.get_server_ip_and_port())
            self.init_a3c_algorithm()
    
    def print_to_gui_console(self, data):
        """
        Print text to the self.gui_console widget.
        :param data: Text to print to the self.gui_console widget.
        :return: None
        """
        try:
            self.gui_console.config(state=tk.NORMAL)
            self.gui_console.insert('end', '\n>> ' + data)
            self.gui_console.see(tk.END)
            self.gui_console.update()
            self.gui_console.config(state=tk.DISABLED)
        except RuntimeError:  # might happens when functions try to write after the window closed.
            pass              # (while the client closes)

    @staticmethod
    def ask4folder(window_title=""):
        """
        Ask for a path to a folder graphically.
        :param window_title: The title of the browse window that will open. (string)
        :return: path chosen by the user. (string)
        """
        file_path_string = filedialog.askdirectory(title=window_title)
        return file_path_string


def is_port_taken(port):
    """
    Returns True if the port is already in use, False otherwise.
    :param port: port index (int)
    :return: True if the port is already in use, False otherwise.
    """
    try:
        s = socket.socket()
        s.bind(("0.0.0.0", port))
        s.close()
        return False
    except OSError:
        return True


def delete_dir_contents(folder_path):
    """
    Delete the contents of the given folder.
    :param folder_path: The path of the folder to be deleted.
    :return: None
    """
    for name in os.listdir(folder_path):
        path = os.path.join(folder_path, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


class GameSelectionScreen(tk.Frame):
    Icon_Display_Size = [250, 250]

    def __init__(self, parent):
        """
        A class that handles the creation and operation of the GUI window that sets up connection with the server and
        choose an environment to play/train on.
        :param parent: A tk.Tk instance.
        """
        self.parent = parent

        # handle graphics
        tk.Frame.__init__(self, parent)
        self.parent.title("Setup")

        self.envs_manager = None
        self.env_init_str_to_icons_dict = None

        self.chosen_env_init_str_tk_var = tk.StringVar(master=self)
        self.chosen_env_init_str_tk_var.trace('w', self._update_icon_display_screen)  # Will call _update_icon_display
        #                                                                               _screen() func every time the
        #                                                                               value of self.env_init_str
        #                                                                               changes. Note to self: check
        #                                                                               what the 'w' means.

        self._curr_icon = None  # for it's purpose see documentation in ._update_display_screen() func.
        
        self.select_and_launch_button_was_pressesd = False

        self.init_widgets()

    @property
    def chosen_env_init_str(self):
        """
        chosen_env_init_str Property getter. Returns the value of the initialization string chosen by the user.
        (stored in self.chosen_env_init_str_tk_var)
        :return: The value of the initialization string chosen by the user.
        """
        return self.chosen_env_init_str_tk_var.get()

    def init_widgets(self):
        """
        Handles the initialization of all tk widgets.
        :return: None
        """
        # creating a root menu to insert all the sub menus

        self.lable = tk.Label(master=self, text="Please connect to the server and then proceed to choose a game.")
        self.lable.grid(column=0, row=0, columnspan=3, sticky=tk.NW)

        self.ip_entry = tk.Entry(master=self)
        self.ip_entry.insert(tk.END, Default_IP)
        self.ip_entry.grid(column=0, row=1)

        self.port_entry = tk.Entry(master=self)
        self.port_entry.insert(tk.END, Default_Port)
        self.port_entry.grid(column=1, row=1)

        self.connect_button = tk.Button(master=self, text="Connect", command=self.connect_button_command)
        self.connect_button.grid(column=2, row=1)

        self.games_option_menu = ttk.Combobox(self, textvariable=self.chosen_env_init_str_tk_var, state=tk.DISABLED)
        self.games_option_menu.grid(column=0, row=2)

        self.select_and_launch_button = tk.Button(master=self, text="Launch",
                                                  command=self.select_and_launch_button_command, state=tk.DISABLED)
        self.select_and_launch_button.grid(column=1, row=2)

        self.icon_display = tk.Canvas(master=self, width=self.Icon_Display_Size[0], height=self.Icon_Display_Size[1])
        self.icon_display.grid(column=2, row=2, rowspan=3)

    def connect_button_command(self):
        """
        Attempts connection with the server.
        First tries to ping the server, if the server returns a ping:
        Establishes connection and enables interaction with the environment selection related widgets.
        :return: None
        """
        ip = self.ip_entry.get()
        if not is_valid_ipv4_address(ip):
            print("Please enter a valid ip address.")
            return

        try:
            port = int(self.port_entry.get())
            if not (0 <= port < 2 ** 16):
                raise ValueError()
        except ValueError:
            print("Please enter a valid port number.")
            return

        if RemoteEnvironmentsManager.check_for_server(ip, port):
            self.connect2server(ip, port)
            self.ip_entry.config(state=tk.DISABLED)
            self.port_entry.config(state=tk.DISABLED)
            self.connect_button.config(text="Connected", state=tk.DISABLED)
            self.select_and_launch_button.config(state=tk.NORMAL)
            self.games_option_menu.config(state='readonly')
            self.games_option_menu['values'] = list(self.env_init_str_to_icons_dict.keys())
            self.games_option_menu.set(list(self.env_init_str_to_icons_dict.keys())[0])
        else:
            print("Failed to ping the server.")

    def connect2server(self, ip, port):
        """
        Establishes connection with the server, creates an instance of the RemoteEnvironmentsManager class and gets the
        available environments initialization strings as well their icons.
        :param ip: Server's ip (string)
        :param port: Server's port (int)
        :return: None
        """
        self.envs_manager = RemoteEnvironmentsManager(ip, port, icon_display_size=self.Icon_Display_Size)
        self.env_init_str_to_icons_dict = self.envs_manager.get_environments_initialization_strings_to_icons_dict()

    def select_and_launch_button_command(self):
        """
        Selects the environment currently chosen by the user. Exits tk.mainloop() so that the GameScreen will be
        launched.
        :return: None
        """
        self.select_and_launch_button_was_pressesd = True
        self.shutdown(shutdown_envs_manager=False)
        self.exit()
        
    def get_envs_mmanager_and_chosen_init_str(self):
        """
        Returns the envs_mmanager and the environment initialization string chosen.
        :return: Returns (self.envs_manager, self.chosen_env_init_str) if the user pressed the select button,
            returns (None, None) otherwise.
        """
        if self.select_and_launch_button_was_pressesd:
            return self.envs_manager, self.chosen_env_init_str
        else:
            return self.envs_manager, None

    def exit(self):
        """
        Exit the tk.mainloop(). See main.py.
        :return:
        """
        self.parent.quit()

    def shutdown(self, shutdown_envs_manager=False):
        """
        Terminates operation of instance and it's components.
        :param shutdown_envs_manager: If True, shuts down self.envs_manager. Doesn't shut is down if False.
        :return: None
        """
        if shutdown_envs_manager and self.envs_manager is not None and self.envs_manager.status == Status.Running:
            self.envs_manager.shutdown()
            self.envs_manager.join()

    def _update_icon_display_screen(self, *args):
        """
        Updates the icon display screen.
        :param args: Irrelevant. Here so it could be used as a function to be called with widget.trace() func.
        :return: None
        """
        img = Image.fromarray(self.env_init_str_to_icons_dict[self.chosen_env_init_str], mode='RGB')
        frame_image = ImageTk.PhotoImage(image=img)
        self._curr_icon = frame_image  # There seems to a bug with ImageTk and the create_image func where if
        #                                 we don't deliberately save the ImageTk.PhotoImage() to some variable,
        #                                 the garbage collector collects and deletes it before the image is
        #                                 loaded to the canvas and the image the create_image() func loads is
        #                                 garbage data. That is why we assign it to a variable.
        #                                 For more details see:
        #                                 https://effbot.org/pyfaq/why-do-my-tkinter-images-not-appear.htm
        self.icon_display.create_image((0, 0), image=frame_image, anchor=tk.NW)


def is_valid_ipv4_address(address):
    """
    Returns True if the adrress is a valid IP address, False otherwise.
    :param address: ip address. (string)
    :return: True if addres is valid, False otherwise.
    """
    address = address.split('.')
    if len(address) != 4:
        return False
    try:
        for num in address:
            num = int(num)
            if not (0 <= num <= 255):
                return False
    except ValueError:
        return False
    return True




if __name__ == "__main__":
    main()

