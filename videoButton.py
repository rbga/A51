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
#------------------------------------------------------------------
#               LAST MODIFICATION
#            Developer  :   Rishi Balasubramanian
#            Call Sign  :   RBGA
# Date of Modification  :   26th JULY 2024
#
#          Description  :   Video rendered through Sprite instead of Blit
#                           Added CodingGuidelines comments for all wrong.
#                           __init__()
#                           resize()
#                           update_sprite()
#                           is_inside()
#                           draw()
#------------------------------------------------------------------
#
################################################################################################


class VideoButton(pyglet.event.EventDispatcher):

    ###----------------------------------------------------------------------
    #             <__init__>()
    #       Inputs    :     x, y: The initial position of the button.
    #                       width, height: The initial size of the button.
    #                       WINDOW_WIDTH, WINDOW_HEIGHT: The initial dimensions of the window.
    #                       videos: A dictionary mapping states to video file paths.
    #                       batch: The Pyglet graphics batch to which the button belongs.
    #                       group: The Pyglet group for layering (optional).
    #                       on_toggle_callback: A callback function triggered on state changes (optional).
    #                       verbose: A flag for enabling verbose logging.
    #
    #       Output    :     None
    #   Description   :     Initializes the button, setting its position, size, videos, 
    #                       state, and other properties. It also loads the videos and sets the initial state.
    ###----------------------------------------------------------------------
    def __init__(self, x, y, width, height, WINDOW_WIDTH, WINDOW_HEIGHT, videos, batch, group=None, on_toggle_callback=None, verbose=True):
        self.x = x
        self.y = y
        self.original_width = width
        self.original_height = height
        self.button_width = width
        self.button_height = height
        self.batch = batch
        self.group = group if group is not None else pyglet.graphics.OrderedGroup(0)
        self.videos = videos
        self.state = 'idle'
        self.current_player = None
        self.verbose = verbose
        self.on_toggle_callback = on_toggle_callback
        self.sprite = None

        # Store initial window size as reference
        self.initial_window_width = WINDOW_WIDTH
        self.initial_window_height = WINDOW_HEIGHT

        self.load_videos()
        self.set_state(self.state)

        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)

    ###----------------------------------------------------------------------
    #              <resize>()
    #       Inputs    :     window_width: The new width of the window.
    #                       window_height: The new height of the window.
    #
    #       Output    :     None
    #   Description   :     Adjusts the button's size proportionally based 
    #                       on the new window dimensions while maintaining 
    #                       the aspect ratio. It updates the button's size 
    #                       and the corresponding sprite if necessary.
    ###----------------------------------------------------------------------
    def resize(self, window_width, window_height):
        width_factor = window_width / self.initial_window_width
        height_factor = window_height / self.initial_window_height

        new_width = int(self.original_width * width_factor)
        new_height = int(self.original_height * height_factor)

        aspect_ratio = self.original_width / self.original_height
        if new_width / new_height > aspect_ratio:
            new_width = int(new_height * aspect_ratio)
        else:
            new_height = int(new_width / aspect_ratio)

        if self.button_width != new_width or self.button_height != new_height:
            self.button_width = new_width
            self.button_height = new_height
            if self.verbose:
                print(f"Button resized to {self.button_width}x{self.button_height}")

        if self.verbose:
            print(f"Resize called with window_width={window_width}, window_height={window_height}")

        # Update sprite with new dimensions
        if self.sprite:
            self.sprite.scale = min(self.button_width / self.sprite.width, self.button_height / self.sprite.height)




    ###----------------------------------------------------------------------
    #            <load_videos>()
    #       Inputs    :     None
    #
    #       Output    :     None
    #   Description   :     Loads the videos specified for different button 
    #                       states into Pyglet media players, associating each 
    #                       state with its respective player. Handles any errors 
    #                       during loading.
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
                player.on_eos = self.on_eos
                self.players[state] = player
                if self.verbose:
                    print(f"Loaded video {path} for state {state}")
            except Exception as e:
                print(f"Error loading video {path}: {e}")

    ###----------------------------------------------------------------------
    #             <set_state>()
    #       Inputs    :     new_state: The new state to transition to.
    #
    #       Output    :     None
    #   Description   :     Changes the button's state, handling the transition 
    #                       between different video players. It ensures the correct 
    #                       playback behavior (looping or non-looping) for the new 
    #                       state and updates the sprite accordingly.
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

            # Update the sprite with the new texture
            self.update_sprite()

    ###----------------------------------------------------------------------
    #              <on_eos>()
    #       Inputs    :     None
    #
    #       Output    :     None
    #   Description   :     Called when the end of a video stream is reached. 
    #                       It handles state transitions after certain videos 
    #                       finish playing, ensuring proper looping for idle 
    #                       states and resetting the video playback.
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
    #          <update_sprite>()
    #       Inputs    :     None
    #
    #       Output    :     None
    #   Description   :     Updates or creates the sprite used to display the 
    #                       button's video. It scales the sprite based on the 
    #                       current button dimensions and positions it correctly.
    ###----------------------------------------------------------------------
    def update_sprite(self):
        if self.current_player and self.current_player.source:
            texture = self.current_player.texture
            if texture:
                if self.sprite:
                    self.sprite.delete()
                self.sprite = pyglet.sprite.Sprite(texture, x=self.x, y=self.y, batch=self.batch, group=self.group)
                self.sprite.scale = min(self.button_width / texture.width, self.button_height / texture.height)
                if self.verbose:
                    print(f"Sprite updated for {self.state}")


    ###----------------------------------------------------------------------
    #           <on_mouse_motion>()
    #       Inputs    :     x, y: The current mouse coordinates.
    #                       dx, dy: The change in mouse position.
    #
    #       Output    :     None
    #   Description   :     Handles mouse movement events. If the mouse is 
    #                       over the button, it transitions to the hover state; 
    #                       otherwise, it transitions away from the hover state.
    ###----------------------------------------------------------------------
    def on_mouse_motion(self, x, y, dx, dy):
        if self.is_mouse_over(x, y):
            if self.state == 'idle':
                self.set_state('hover_transition')
        else:
            if self.state == 'hover_idle':
                self.set_state('dehover_transition')

    ###----------------------------------------------------------------------
    #           <on_mouse_press>()
    #       Inputs    :     x, y: The coordinates of the mouse when the button is pressed.
    #                       button: The mouse button that was pressed.
    #                       modifiers: Any modifier keys pressed.
    #
    #       Output    :     None
    #   Description   :     Manages the button's response to mouse press events, 
    #                       transitioning to the press state if the button is pressed. 
    #                       It can also trigger a callback function after the state change.
    ###----------------------------------------------------------------------
    def on_mouse_press(self, x, y, button, modifiers):
        if self.is_inside(x, y):
            if self.state in ['hover_idle', 'idle']:
                self.set_state('press_transition')

            # Trigger the callback after the state change to press_transition
            if self.on_toggle_callback:
                self.on_toggle_callback(self.state)
    
    
    
    ###----------------------------------------------------------------------
    #              <is_inside>()
    #       Inputs    :     x, y: The coordinates to check.
    #
    #       Output    :     Boolean
    #   Description   :     Determines if the given coordinates are inside the button's boundaries.
    ###----------------------------------------------------------------------
    def is_inside(self, x, y):
        return (self.x <= x <= self.x + self.button_width and
                self.y <= y <= self.y + self.button_height)

    
    
    
    ###----------------------------------------------------------------------
    #           <on_mouse_release>()
    #       Inputs    :     x, y: The coordinates of the mouse when the button is pressed.
    #                       button: The mouse button that was pressed (e.g., left, right).
    #                       modifiers: Any modifier keys pressed (e.g., Shift, Ctrl).
    #
    #       Output    :     None
    #   Description   :     Handles the button's response to mouse release events, 
    #                       transitioning away from the press state if necessary.
    ###----------------------------------------------------------------------
    def on_mouse_release(self, x, y, button, modifiers):
        if self.is_mouse_over(x, y):
            if self.state == 'press_idle':
                self.set_state('unpress_transition')




    ###----------------------------------------------------------------------
    #           <is_mouse_over>()
    #       Inputs    :     x, y: The coordinates to check.
    #
    #       Output    :     Boolean
    #   Description   :     Checks if the mouse is currently over the button.
    ###----------------------------------------------------------------------
    def is_mouse_over(self, x, y):
        return self.x <= x <= self.x + self.button_width and self.y <= y <= self.y + self.button_height




    ###----------------------------------------------------------------------
    #               <draw>()
    #       Inputs    :     None
    #
    #       Output    :     None
    #   Description   :     Draws the button's sprite on the screen. It updates 
    #                       the sprite's position and renders it.
    ###----------------------------------------------------------------------
    def draw(self):
        if self.sprite:
            self.sprite.update(x=self.x, y=self.y)
            self.sprite.draw()




    ###----------------------------------------------------------------------
    #              <update>()
    #       Inputs    :     dt: The time delta since the last update.
    #
    #       Output    :     None
    #   Description   :     A placeholder function for updating logic, such as 
    #                       handling state transitions or other periodic updates.
    ###----------------------------------------------------------------------
    def update(self, dt):
        # This can be used to update logic, e.g., handling state transitions, etc.
        pass

    










###########################################################################################
#                           Example Execution
###########################################################################################

# # Main application window
# window = pyglet.window.Window(800, 600, caption='Main Application Window', resizable=True)

# batch = pyglet.graphics.Batch()

# # Example usage of VideoButton
# detectButtonStateVisuals = {
#     'idle': 'detectIdleState_15M.mp4',
#     'hover_transition': 'detectHoverTransition_15M.mp4',
#     'hover_idle': 'detectHoverIdle_15M.mp4',
#     'dehover_transition': 'detectUnhoverTransition_15M.mp4',
#     'press_transition': 'detectPressedTransition_15M.mp4',
#     'press_idle': 'detectPressedIdle_15M.mp4',
#     'unpress_transition': 'detectUnpressedTransition_15M.mp4'
# }

# trainButtonStateVisuals = {
#     'idle': 'trainIdleState_15M.mp4',
#     'hover_transition': 'trainHoverTransition_15M.mp4',
#     'hover_idle': 'trainHoverIdle_15M.mp4',
#     'dehover_transition': 'trainUnhoverTransition_15M.mp4',
#     'press_transition': 'trainPressedTransition_15M.mp4',
#     'press_idle': 'trainPressedIdle_15M.mp4',
#     'unpress_transition': 'trainUnpressedTransition_15M.mp4'
# }


# video_button = VideoButton(1, 1, 30, 30, trainButtonStateVisuals, batch, False)

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
#     video_button.resize(width, height, maintain_initial_size=True)
    
# pyglet.clock.schedule_interval(video_button.update, 1/60)  # 60 Hz update rate
# pyglet.app.run()
