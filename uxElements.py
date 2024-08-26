import pyglet
import os
from pyglet.image import ImageData
import numpy as np



################################################################################################
#                  INFORMATION
#
#            File Name  :   uxElements.py
#            Developer  :   Rishi Balasubramanian
#            Call Sign  :   RBGA
#   First Stable Build  :   11th AUG 2024
#             Use Case  :   Custom UI Elements Class.
#                 
#                 Type  :   Classes
#               Inputs  :   Pos X (value), Pos Y (value), Button Width (value), 
#                           Button Height (value), videos (list), Batch (Pyglet Object),
#                           Group (opt), Verbose = True
#
#               Output  :   Ux Objects
#          Description  :   The UX elements consists of classes for
#                           uxLiveButton and uxWindowElements.
# ------------------------------------------------------------------
#               LAST MODIFICATION
#            Developer  :   Rishi Balasubramanian
#            Call Sign  :   RBGA
# Date of Modification  :   11th AUG 2024
#
#          Description  :   Added Information Block and Code Module 
#                           Block for every Code Module in the file.
#------------------------------------------------------------------
#
################################################################################################




class uxWindowElements(pyglet.event.EventDispatcher):
    
    ###----------------------------------------------------------------------
    #             <__init__>()
    #       Inputs    : 
    #           - `x`, `y`: Initial position of the window element.
    #           - `width`, `height`: Initial dimensions of the window element.
    #           - `WINDOW_WIDTH`, `WINDOW_HEIGHT`: The dimensions of the entire window.
    #           - `entrance_videos`, `loop_videos`: Paths to the video folders for entrance and loop states.
    #           - `batch`: Pyglet batch to which the element belongs.
    #           - `group`: Pyglet group for layering (optional).
    #           - `verbose`: Flag for enabling verbose logging.
    #       Output    : None
    #   Description   : 
    #       Initializes the window element, loads the videos for different states, 
    #       sets the initial state, and resizes the element based on the window size.
    ###----------------------------------------------------------------------
    def __init__(self, x, y, width, height, WINDOW_WIDTH, WINDOW_HEIGHT, entrance_videos, loop_videos, batch, group=None, verbose=True):
        self.x = x
        self.y = y
        self.original_width = width
        self.original_height = height
        self.window_width = width
        self.window_height = height
        self.batch = batch
        self.group = group if group is not None else pyglet.graphics.Group(order=0)
        self.entrance_videos = entrance_videos
        self.loop_videos = loop_videos
        self.state = 'entrance'
        self.current_video = None
        self.verbose = verbose
        self.sprite = None

        # Store initial window size as reference
        self.initial_window_width = WINDOW_WIDTH
        self.initial_window_height = WINDOW_HEIGHT

        self.load_videos()
        self.set_state(self.state)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)

    ###----------------------------------------------------------------------
    #             <resize>()
    #       Inputs    : 
    #           - `window_width`, `window_height`: The new dimensions of the window.
    #       Output    : None
    #   Description   : 
    #       Resizes the window element proportionally based on the new window dimensions 
    #       while maintaining the aspect ratio. Updates the sprite's scale accordingly.
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

        if self.window_width != new_width or self.window_height != new_height:
            self.window_width = new_width
            self.window_height = new_height
            if self.verbose:
                print(f"Window frame resized to {self.window_width}x{self.window_height}")

        if self.verbose:
            print(f"Resize called with window_width={window_width}, window_height={window_height}")

        # Update sprite with new dimensions
        if self.sprite:
            self.sprite.scale = min(self.window_width / self.sprite.width, self.window_height / self.sprite.height)

    ###----------------------------------------------------------------------
    #             <load_videos>()
    #       Inputs    : None
    #       Output    : None
    #   Description   : 
    #       Loads the images from the provided video folders for both the entrance 
    #       and loop states and stores them for future use.
    ###----------------------------------------------------------------------
    def load_videos(self):
        self.video_captures = {
            'entrance': self.load_images(self.entrance_videos),
            'loop': self.load_images(self.loop_videos)
        }
        if self.verbose:
            print("Videos loaded for entrance and loop states")


    ###----------------------------------------------------------------------
    #             <load_images>()
    #       Inputs    : 
    #           - `path`: The path to the folder containing the image sequence.
    #       Output    : 
    #           - A list of loaded Pyglet images.
    #   Description   : 
    #       Loads all PNG images from the specified folder, sorts them, and returns 
    #       them as a list of Pyglet image objects.
    ###----------------------------------------------------------------------
    def load_images(self, path):
        image_folder = f'UiElements/{path}'
        image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
        images = [pyglet.image.load(os.path.join(image_folder, f)) for f in image_files]
        return images
    

    ###----------------------------------------------------------------------
    #             <set_state>()
    #       Inputs    : 
    #           - `new_state`: The state to transition to (e.g., 'entrance', 'loop').
    #       Output    : None
    #   Description   : 
    #       Switches the window element to the specified state, updating the current 
    #       video to the corresponding image sequence. Determines whether the video 
    #       should loop based on the state.
    ###----------------------------------------------------------------------
    def set_state(self, new_state):
        if new_state in self.video_captures:
            if self.verbose:
                print(f"Switching to {new_state}")

            self.state = new_state
            self.current_video = self.video_captures[new_state]
            self.current_frame = 0
            self.loop = new_state == 'loop'


    ###----------------------------------------------------------------------
    #             <on_eos>()
    #       Inputs    : None
    #       Output    : None
    #   Description   : 
    #       Handles the end of a video stream (EOS). If the current state is 'entrance', 
    #       it automatically transitions to the 'loop' state.
    ###----------------------------------------------------------------------
    def on_eos(self):
        if self.verbose:
            print(f"End of stream for state {self.state}")
        if self.state == 'entrance':
            self.set_state('loop')


    ###----------------------------------------------------------------------
    #             <update_sprite>()
    #       Inputs    : None
    #       Output    : None
    #   Description   : 
    #       Updates the sprite to display the next frame in the current video sequence. 
    #       Handles end-of-sequence logic and scales the sprite according to the window size.
    ###----------------------------------------------------------------------
    def update_sprite(self):
        if self.current_video:
            # Increment the index and handle end of sequence (EOS)
            self.current_frame += 1
            if self.current_frame >= len(self.current_video):
                self.current_frame = 0
                if not self.loop:
                    self.on_eos()
                    return
            
            # Get the current image data
            image_data = self.current_video[self.current_frame]
            
            if self.sprite:
                self.sprite.delete()

            self.sprite = pyglet.sprite.Sprite(image_data, x=self.x, y=self.y, batch=self.batch, group=self.group)
            self.sprite.scale = min(self.window_width / image_data.width, self.window_height / image_data.height)
            
            if self.verbose:
                print(f"Sprite updated for {self.state}. Index: {self.current_frame}")


    ###----------------------------------------------------------------------
    #             <draw>()
    #       Inputs    : None
    #       Output    : None
    #   Description   : 
    #       Draws the current sprite on the screen, updating its position and rendering it 
    #       within the window element.
    ###----------------------------------------------------------------------
    def draw(self):
        if self.sprite:
            self.sprite.update(x=self.x, y=self.y)
            self.sprite.draw()

    ###----------------------------------------------------------------------
    #             <update>()
    #       Inputs    : 
    #           - `dt`: The time delta since the last update.
    #       Output    : None
    #   Description   : 
    #       Periodically updates the window element, particularly the sprite, by 
    #       advancing to the next frame in the video sequence.
    ###----------------------------------------------------------------------
    def update(self, dt):
        self.update_sprite()



class uxLiveButton(pyglet.event.EventDispatcher):

    ###----------------------------------------------------------------------
    #             <__init__>()
    #       Inputs    : 
    #           - `x`, `y`: Initial position of the button.
    #           - `width`, `height`: Initial dimensions of the button.
    #           - `WINDOW_WIDTH`, `WINDOW_HEIGHT`: The dimensions of the entire window.
    #           - `videos`: A dictionary mapping states to video folders.
    #           - `batch`: Pyglet batch to which the button belongs.
    #           - `group`: Pyglet group for layering (optional).
    #           - `on_toggle_callback`: Callback function to execute when the button is toggled (optional).
    #           - `verbose`: Flag for enabling verbose logging.
    #       Output    : None
    #   Description   : 
    #       Initializes the live button, loads the videos for different states, 
    #       sets the initial state, and resizes the button based on the window size.
    ###----------------------------------------------------------------------
    def __init__(self, x, y, width, height, WINDOW_WIDTH, WINDOW_HEIGHT, videos, batch, group=None, on_toggle_callback=None, verbose=True):
        self.x = x
        self.y = y
        self.original_width = width
        self.original_height = height
        self.button_width = width
        self.button_height = height
        self.batch = batch
        self.group = group if group is not None else pyglet.graphics.Group(order=1)
        self.videos = videos
        self.state = 'idle'
        self.current_video = None
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
    #             <resize>()
    #       Inputs    : 
    #           - `window_width`, `window_height`: The new dimensions of the window.
    #       Output    : None
    #   Description   : 
    #       Resizes the button proportionally based on the new window dimensions 
    #       while maintaining the aspect ratio. Updates the sprite's scale accordingly.
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
    #             <load_videos>()
    #       Inputs    : None
    #       Output    : None
    #   Description   : 
    #       Loads the images from the provided video folders for each state and 
    #       stores them for future use.
    ###----------------------------------------------------------------------
    def load_videos(self):
        self.video_captures = {}
        for state, path in self.videos.items():
            image_folder = f'UiElements/{path}'
            image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
            images = [pyglet.image.load(os.path.join(image_folder, f)) for f in image_files]
            self.video_captures[state] = images
            if self.verbose:
                print(f"Loaded images for state {state}")
                
    ###----------------------------------------------------------------------
    #             <set_state>()
    #       Inputs    : 
    #           - `new_state`: The state to transition to (e.g., 'idle', 'hover_idle').
    #       Output    : None
    #   Description   : 
    #       Switches the button to the specified state, updating the current video 
    #       to the corresponding image sequence. Determines whether the video should 
    #       loop based on the state.
    ###----------------------------------------------------------------------
    def set_state(self, new_state):
        if new_state in self.video_captures:
            if self.verbose:
                print(f"Switching to {new_state}")

            self.state = new_state
            self.current_video = self.video_captures[new_state]
            self.current_frame = 0
            self.loop = new_state in ['idle', 'hover_idle', 'press_idle']

    ###----------------------------------------------------------------------
    #             <on_eos>()
    #       Inputs    : None
    #       Output    : None
    #   Description   : 
    #       Handles the end of a video stream (EOS) for transition states. 
    #       Automatically sets the appropriate idle state after a transition.
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
    
    ###----------------------------------------------------------------------
    #             <update_sprite>()
    #       Inputs    : None
    #       Output    : None
    #   Description   : 
    #       Updates the sprite to display the next frame in the current video sequence. 
    #       Handles end-of-sequence logic and scales the sprite according to the button size.
    ###----------------------------------------------------------------------
    def update_sprite(self):
        if self.current_video:
            # Increment the index and loop back to 0 if necessary
            self.current_frame = (self.current_frame + 1) % len(self.current_video)
            
            # Check if we need to handle end of sequence (EOS)
            if not self.loop and self.current_frame == 0:
                self.on_eos()  # Call end-of-sequence handler
                return
            
            # Get the current image data
            image_data = self.current_video[self.current_frame]
            
            if self.sprite:
                self.sprite.delete()

            self.sprite = pyglet.sprite.Sprite(image_data, x=self.x, y=self.y, batch=self.batch, group=self.group)
            self.sprite.scale = min(self.button_width / image_data.width, self.button_height / image_data.height)
            
            if self.verbose:
                print(f"Sprite updated for {self.state}. Index: {self.current_frame}")


    ###----------------------------------------------------------------------
    #             <on_mouse_motion>()
    #       Inputs    : 
    #           - `x`, `y`: Current mouse coordinates.
    #           - `dx`, `dy`: Change in mouse coordinates.
    #       Output    : None
    #   Description   : 
    #       Checks if the mouse is over the button. If so, it transitions to 
    #       the hover state; otherwise, it transitions back to the idle state.
    ###----------------------------------------------------------------------
    def on_mouse_motion(self, x, y, dx, dy):
        if self.is_mouse_over(x, y):
            if self.state == 'idle':
                self.set_state('hover_transition')
        else:
            if self.state == 'hover_idle':
                self.set_state('dehover_transition')

    ###----------------------------------------------------------------------
    #             <on_mouse_press>()
    #       Inputs    : 
    #           - `x`, `y`: Mouse click coordinates.
    #           - `button`: Mouse button pressed.
    #           - `modifiers`: Modifier keys pressed during the click.
    #       Output    : None
    #   Description   : 
    #       Checks if the mouse press is inside the button. If true, it transitions 
    #       to the press state and triggers the toggle callback, if provided.
    ###----------------------------------------------------------------------
    def on_mouse_press(self, x, y, button, modifiers):
        if self.is_inside(x, y):
            self.set_state('press_transition')
            if self.on_toggle_callback:
                self.on_toggle_callback(self.state)

    ###----------------------------------------------------------------------
    #             <is_inside>()
    #       Inputs    : 
    #           - `x`, `y`: Coordinates to check.
    #       Output    : 
    #           - Boolean: True if the coordinates are inside the button; otherwise, False.
    #   Description   : 
    #       Determines if a given set of coordinates is inside the button's boundaries.
    ###----------------------------------------------------------------------
    def is_inside(self, x, y):
        return (self.x <= x <= self.x + self.button_width and
                self.y <= y <= self.y + self.button_height)

    ###----------------------------------------------------------------------
    #             <on_mouse_release>()
    #       Inputs    : 
    #           - `x`, `y`: Mouse release coordinates.
    #           - `button`: Mouse button released.
    #           - `modifiers`: Modifier keys pressed during the release.
    #       Output    : None
    #   Description   : 
    #       Checks if the mouse release occurs over the button. If so, it 
    #       transitions to the release state.
    ###----------------------------------------------------------------------
    def on_mouse_release(self, x, y, button, modifiers):
        if self.is_mouse_over(x, y):
            self.set_state('press_deTransition')

    ###----------------------------------------------------------------------
    #             <is_mouse_over>()
    #       Inputs    : 
    #           - `x`, `y`: Mouse coordinates to check.
    #       Output    : 
    #           - Boolean: True if the mouse is over the button; otherwise, False.
    #   Description   : 
    #       Determines if the mouse is currently over the button.
    ###----------------------------------------------------------------------
    def is_mouse_over(self, x, y):
        return self.x <= x <= self.x + self.button_width and self.y <= y <= self.y + self.button_height
    
    ###----------------------------------------------------------------------
    #             <on_mouse_over>()
    #       Inputs    : None
    #       Output    : None
    #   Description   : 
    #       Transitions the button to the hover state when the mouse moves over it.
    ###----------------------------------------------------------------------
    def on_mouse_over(self):
        self.set_state('hover_transition')

    ###----------------------------------------------------------------------
    #             <on_mouse_out>()
    #       Inputs    : None
    #       Output    : None
    #   Description   : 
    #       Transitions the button back to the idle state when the mouse moves out.
    ###----------------------------------------------------------------------
    def on_mouse_out(self):
        self.set_state('dehover_transition')

    ###----------------------------------------------------------------------
    #             <draw>()
    #       Inputs    : None
    #       Output    : None
    #   Description   : 
    #       Draws the current sprite on the screen, updating its position and 
    #       rendering it within the button.
    ###----------------------------------------------------------------------
    def draw(self):
        if self.sprite:
            self.sprite.update(x=self.x, y=self.y)
            self.sprite.draw()

    ###----------------------------------------------------------------------
    #             <update>()
    #       Inputs    : 
    #           - `dt`: The time delta since the last update.
    #       Output    : None
    #   Description   : 
    #       Periodically updates the button, particularly the sprite, by 
    #       advancing to the next frame in the video sequence.
    ###----------------------------------------------------------------------
    def update(self, dt):
        self.update_sprite()


class uxAnimation(pyglet.event.EventDispatcher):
    def __init__(self, x, y, width, height, WINDOW_WIDTH, WINDOW_HEIGHT, animation_folder, batch, group=None, verbose=True):
        self.x = x
        self.y = y
        self.original_width = width
        self.original_height = height
        self.window_width = width
        self.window_height = height
        self.batch = batch
        self.group = group if group is not None else pyglet.graphics.Group(order=0)
        self.animation_folder = animation_folder
        self.verbose = verbose
        self.frames = self.load_images(animation_folder)
        self.current_frame = 0
        self.sprite = None
        self.animation_complete = False

        self.initial_window_width = WINDOW_WIDTH
        self.initial_window_height = WINDOW_HEIGHT

        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.update_sprite()

    def load_images(self, path):
        image_folder = f'UiElements/{path}'
        image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
        images = [pyglet.image.load(os.path.join(image_folder, f)) for f in image_files]
        if self.verbose:
            print(f"Loaded {len(images)} frames from {path}")
        return images

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

        self.window_width = new_width
        self.window_height = new_height

        if self.verbose:
            print(f"Animation resized to {self.window_width}x{self.window_height}")

        if self.sprite:
            self.sprite.scale = min(self.window_width / self.sprite.width, self.window_height / self.sprite.height)

    def play_animation(self):
        self.current_frame = 0
        self.animation_complete = False
        self.update_sprite()

    def update_sprite(self):
        if self.current_frame < len(self.frames):
            image_data = self.frames[self.current_frame]

            if self.sprite:
                self.sprite.delete()

            self.sprite = pyglet.sprite.Sprite(image_data, x=self.x, y=self.y, batch=self.batch, group=self.group)
            self.sprite.scale = min(self.window_width / image_data.width, self.window_height / image_data.height)

            if self.verbose:
                print(f"Displaying frame {self.current_frame + 1} of {len(self.frames)}")

            self.current_frame += 1
        else:
            self.animation_complete = True
            if self.verbose:
                print("Animation complete.")

    def update(self, dt):
        if not self.animation_complete:
            self.update_sprite()

    def draw(self):
        if self.sprite:
            self.sprite.update(x=self.x, y=self.y)
            self.sprite.draw()



###########################################################################################
#                           Example Execution
###########################################################################################
# # Define the window
# window = pyglet.window.Window(width=1920, height=1080)

# # Define the batch for drawing all sprites together
# main_batch = pyglet.graphics.Batch()

# # Example usage paths for mainWindowFrame and mainWindowFrameLoop
# entrance_folder = 'mainWindowFrame_frames'  # Folder containing entrance animation PNGs
# loop_folder = 'mainWindowFrameLoop_frames'  # Folder containing loop animation PNGs

# # Create an instance of MainWindowFrame
# main_window_frame = MainWindowFrame(
#     x=0, 
#     y=0, 
#     width=1920, 
#     height=1080, 
#     WINDOW_WIDTH=1920, 
#     WINDOW_HEIGHT=1080, 
#     entrance_videos=entrance_folder, 
#     loop_videos=loop_folder, 
#     batch=main_batch, 
#     verbose=True
# )

# # Define the update function for the pyglet clock
# def update(dt):
#     main_window_frame.update(dt)

# # Schedule the update function
# pyglet.clock.schedule_interval(update, 1/60.0)  # Update at 60 FPS

# @window.event
# def on_draw():
#     window.clear()
#     main_batch.draw()

# @window.event
# def on_resize(width, height):
#     main_window_frame.resize(width, height)

# # Run the application
# pyglet.app.run()


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