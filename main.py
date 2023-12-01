from graphics import GameScreen, GameSelectionScreen
import time
import tkinter as tk


def main():
    """
    The project's main function.
    :return: None
    """
    try:
        start = time.time()

        root = tk.Tk()
        game_selection_screen = GameSelectionScreen(root)
        try:
            game_selection_screen.pack(side="top", fill="both", expand=True)
        except tk.TclError:
            pass
        root.mainloop()
        game_selection_screen.shutdown(shutdown_envs_manager=False)

        envs_manager, env_init_str = game_selection_screen.get_envs_mmanager_and_chosen_init_str()

        if not (envs_manager is None or env_init_str is None):
            game_selection_screen.destroy()
            game_screen = GameScreen(root, envs_manager, env_init_str, env_name=env_init_str)
            try:
                game_screen.pack(side="top", fill="both", expand=True)
            except tk.TclError:
                pass
            root.mainloop()
            game_screen.shutdown()

        if envs_manager is not None:
            envs_manager.shutdown()
        print("Time: %d seconds." % (time.time() - start))
    except KeyboardInterrupt:
        print("Keyboard Interrupt! Exiting...")


if __name__ == '__main__':
    main()
