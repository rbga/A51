import pyglet


################################################################################################
#                  INFORMATION
#
#            File Name  :   videoButton.py
#            Developer  :   Rishi Balasubramanian
#            Call Sign  :   RBGA
#   First Stable Build  :   25th JULY 2024
#             Use Case  :   Custom Live UI Button Class
#                 
#                 Type  :   Class
#               Inputs  :   Pos X (value), Pos Y (value), Button Width (value), 
#                           Button Height (value), videos (list), Batch (Pyglet Object), 
#                           Verbose = True
#
#               Output  :   videoButton Object
#          Description  :   The VideoButton class represents a button in a graphical user 
#                           interface that plays different videos based on its state. 
#                           It handles various button states (e.g., idle, hover, press) 
#                           and transitions between them using videos. It supports resizing, 
#                           mouse interactions, and state management for playing and looping 
#                           videos.
# ------------------------------------------------------------------
#               LAST MODIFICATION
#            Developer  :   Rishi Balasubramanian
#            Call Sign  :   RBGA
# Date of Modification  :   25th JULY 2024
#
#          Description  :   Added Information Block and Code Module 
#                           Block for every Code Module in the file.
#------------------------------------------------------------------
#
################################################################################################


class VideoButton:

    ###----------------------------------------------------------------------
    # Initialize the VideoButton.
    #
    # :param x: X-coordinate of the button's position.
    # :param y: Y-coordinate of the button's position.
    # :param width: Initial width of the button.
    # :param height: Initial height of the button.
    # :param videos: Dictionary mapping states to video file paths.
    # :param batch: Pyglet graphics batch for drawing.
    # :param verbose: Flag for printing debug information.
    ###----------------------------------------------------------------------
    def __init__(self, x, y, width, height, videos, batch, verbose = True):
        self.x = x
        self.y = y
        self.original_width = width
        self.original_height = height
        self.button_width = width
        self.button_height = height
        self.batch = batch
        self.videos = videos
        self.state = 'idle'
        self.current_player = None
        self.verbose = verbose
        self.load_videos()
        self.set_state(self.state)


    ###----------------------------------------------------------------------
    # Load videos for each button state and initialize media players.
    ###----------------------------------------------------------------------
    def load_videos(self):
        self.players = {}
        for state, path in self.videos.items():
            try:
                pyglet.resource.path = ['uiElements/']
                pyglet.resource.reindex()
                source = pyglet.resource.media(path)
                player = pyglet.media.Player()
                player.queue(source)
                self.players[state] = player
                player.on_eos = self.on_eos
            except Exception as e:
                print(f"Error loading video {path}: {e}")


    ###----------------------------------------------------------------------
    # Change the button's state and play the corresponding video.
    #
    # :param new_state: The new state to transition to.
    ###----------------------------------------------------------------------
    def set_state(self, new_state):
        if new_state in self.players:
            if self.current_player:
                if self.verbose:
                    print(f"Stopping {self.state}")
                self.current_player.pause()
                self.current_player.seek(0)
            
            if self.verbose:
                print(f"Switching to {new_state}")
            self.current_player = self.players[new_state]
            self.state = new_state

            if new_state in ['idle', 'hover_idle', 'press_idle']:
                self.current_player.loop = True
            else:
                self.current_player.loop = False

            self.current_player.play()
            if self.verbose:
                print(f"Playing {new_state}, loop={self.current_player.loop}")


    ###----------------------------------------------------------------------
    # Handle end of stream event for the current video.
    ###----------------------------------------------------------------------
    def on_eos(self):
        if self.verbose:
            print(f"End of stream for state {self.state}")
        if self.state in ['hover_transition', 'dehover_transition', 'press_transition', 'unpress_transition']:
            if self.state == 'hover_transition':
                self.set_state('hover_idle')
            elif self.state == 'dehover_transition':
                self.set_state('idle')
            elif self.state == 'press_transition':
                self.set_state('press_idle')
            elif self.state == 'unpress_transition':
                self.set_state('idle')

        if self.state in ['idle', 'hover_idle', 'press_idle'] and not self.current_player.loop:
            self.current_player.loop = True
            if self.verbose:
                print(f"Ensured looping for state {self.state}")
        
        if self.current_player.loop:
            if self.verbose:
                print(f"Restarting loop for state {self.state}")
            self.current_player.seek(0)
            self.current_player.play()


    ###----------------------------------------------------------------------
    # Handle mouse motion events to change button state based on mouse position.
    #
    # :param x: X-coordinate of the mouse.
    # :param y: Y-coordinate of the mouse.
    # :param dx: Change in X-coordinate since the last event.
    # :param dy: Change in Y-coordinate since the last event.
    ###----------------------------------------------------------------------
    def on_mouse_motion(self, x, y, dx, dy):
        if self.is_mouse_over(x, y):
            if self.state == 'idle':
                self.set_state('hover_transition')
        else:
            if self.state == 'hover_idle':
                self.set_state('dehover_transition')


    ###----------------------------------------------------------------------
    # Handle mouse press events to change button state when pressed.
    #
    # :param x: X-coordinate of the mouse.
    # :param y: Y-coordinate of the mouse.
    # :param button: Mouse button pressed.
    # :param modifiers: Keyboard modifiers active during the event.
    ###----------------------------------------------------------------------
    def on_mouse_press(self, x, y, button, modifiers):
        if self.is_mouse_over(x, y):
            if self.state == 'hover_idle':
                self.set_state('press_transition')
            if self.state == "idle":
                self.set_state('press_transition')


    ###----------------------------------------------------------------------
    # Handle mouse release events to transition to appropriate state.
    #
    # :param x: X-coordinate of the mouse.
    # :param y: Y-coordinate of the mouse.
    # :param button: Mouse button released.
    # :param modifiers: Keyboard modifiers active during the event.
    ###----------------------------------------------------------------------
    def on_mouse_release(self, x, y, button, modifiers):
        if self.is_mouse_over(x, y):
            if self.state == 'press_idle':
                self.set_state('unpress_transition')


    ###----------------------------------------------------------------------
    # Check if the mouse cursor is over the button.
    #
    # :param x: X-coordinate of the mouse.
    # :param y: Y-coordinate of the mouse.
    # :return: True if the mouse is over the button, False otherwise.
    ###----------------------------------------------------------------------
    def is_mouse_over(self, x, y):
        return self.x <= x <= self.x + self.button_width and self.y <= y <= self.y + self.button_height


    ###----------------------------------------------------------------------
    # Draw the current video frame on the button.
    ###----------------------------------------------------------------------
    def draw(self):
        if self.current_player and self.current_player.source:
            frame = self.current_player.texture
            if frame:
                frame.blit(self.x, self.y, width=self.button_width, height=self.button_height)


    ###----------------------------------------------------------------------
    # Update logic for the button (e.g., handle state transitions). This method is a placeholder
    # and can be extended to include any per-frame updates.
    #
    # :param dt: Time since last update.
    ###----------------------------------------------------------------------
    def update(self, dt):
        # This can be used to update logic, e.g., handling state transitions, etc.
        pass
    

    ###----------------------------------------------------------------------
    # Adjust the button's size to fit the new window size while maintaining the original aspect ratio.
    #
    # :param width: New width of the window.
    # :param height: New height of the window.
    ###----------------------------------------------------------------------
    def resize(self, width, height):
        # Adjust button size according to new window size
        aspect_ratio = self.original_width / self.original_height
        if width / height > aspect_ratio:
            self.button_height = height
            self.button_width = int(height * aspect_ratio)
        else:
            self.button_width = width
            self.button_height = int(width / aspect_ratio)
        if self.verbose:
            print(f"Button resized to {self.button_width}x{self.button_height}")











###########################################################################################
#                           Example Execution
###########################################################################################

# Main application window
# window = pyglet.window.Window(800, 600, caption='Main Application Window', resizable=True)
# batch = pyglet.graphics.Batch()

# # Example usage of VideoButton
# videos = {
#     'idle': 'detectIdleState_15M.mp4',
#     'hover_transition': 'detectHoverTransition_15M.mp4',
#     'hover_idle': 'detectHoverIdle_15M.mp4',
#     'dehover_transition': 'detectUnhoverTransition_15M.mp4',
#     'press_transition': 'detectPressedTransition_15M.mp4',
#     'press_idle': 'detectPressedIdle_15M.mp4',
#     'unpress_transition': 'detectUnpressedTransition_15M.mp4'
# }

# video_button = VideoButton(1, 1, 300, 300, videos, batch, False)

# @window.event
# def on_draw():
#     window.clear()
#     video_button.draw()

# @window.event
# def on_mouse_motion(x, y, dx, dy):
#     video_button.on_mouse_motion(x, y, dx, dy)

# @window.event
# def on_mouse_press(x, y, button, modifiers):
#     video_button.on_mouse_press(x, y, button, modifiers)

# @window.event
# def on_mouse_release(x, y, button, modifiers):
#     video_button.on_mouse_release(x, y, button, modifiers)

# @window.event
# def on_resize(width, height):
#     video_button.resize(width, height)

# pyglet.clock.schedule_interval(video_button.update, 1/60)  # 60 Hz update rate
# pyglet.app.run()
