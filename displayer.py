import pyglet
from pyglet.gl import *
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import cv2
from pyglet.gl import glViewport
from pyglet.image import ImageData



################################################################################################
#                  INFORMATION
#
#            File Name  :   displayer.py
#            Developer  :   Rishi Balasubramanian
#            Call Sign  :   RBGA
#   First Stable Build  :   27th MAY 2024
#             Use Case  :   Custom Python Library involving most things related to display using
#                           pyglet and OpenGL
#
#                 Type  :   Function(s)
#               Inputs  :   Many
#
#               Output  :   Many
#          Description  :   This file is a part of a graphical application that uses Pyglet
#                           and OpenGL to create an interactive user interface. The UI includes 
#                           buttons, text entry fields, and displays video frames. The functions
#                           in this file handle the rendering and interaction of UI elements, 
#                           including buttons with different states, text entry fields, 
#                           video playback, and OpenGL-based drawing operations.
#
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


labels = []
previous_texts = []

# Button colors (RGB)
BUTTON_COLOR = (0.2, 0.6, 1.0, 1.0)
BUTTON_HOVER_COLOR = (0.4, 0.7, 1.0, 1.0)

# Button label colors (RGB)
LABEL_COLOR = (255, 255, 255, 255)

# Define button dimensions
BUTTON_HEIGHT = 128
BUTTON_WIDTH = 200
BUTTON_MARGIN = 100

# Define window dimensions
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 768  # 640 (video) + 128 (buttons)

# Define button labels
DETECT_LABEL = 'DETECT'
TRAIN_LABEL = 'TRAIN'


###----------------------------------------------------------------------
#              canvas()
#       Inputs    :     None
#
#       Output    :     None
#   Description   :     Draws a grey rectangle on the screen using OpenGL. 
#                       This function sets the color and defines the vertices
#                       of a quadrilateral to render the rectangle. It is 
#                       used to create a background or a placeholder
#                       area on the canvas.
###----------------------------------------------------------------------
def canvas():
    # Draw the rectangle
    glColor3f(0.5, 0.5, 0.5)  # Set the color (RGB)
    glBegin(GL_QUADS)
    glVertex2f(100, 100)  # Bottom-left corner
    glVertex2f(700, 100)  # Bottom-right corner
    glVertex2f(700, 500)  # Top-right corner
    glVertex2f(100, 500)  # Top-left corner
    glEnd()




###----------------------------------------------------------------------
#           render_button()
#       Inputs    :     label: The text label for the button (unused in this implementation).
#                       x: X-coordinate for button position.
#                       y: Y-coordinate for button position.
#                       press: Path to the image used when the button is pressed.
#                       depress: Path to the image used when the button is not pressed.
#                       hover: Path to the image used when the button is hovered over.
#                       batch: Pyglet graphics batch for rendering optimization.
#
#       Output    :     Returns a pyglet.gui.ToggleButton object.
#   Description   :     Creates a toggle button using Pyglet, with different images for 
#                       pressed, unpressed, and hovered states. The button's size is set
#                       to 200x60 pixels, and it is added to the given graphics batch 
#                       for rendering.
###----------------------------------------------------------------------
def render_button(label, x, y, press, depress, hover, batch):
    press_b = pyglet.resource.image(press)
    depre_b = pyglet.resource.image(depress)
    hover_b = pyglet.resource.image(hover)

    press_b.width = depre_b.width = hover_b.width = 200
    press_b.height = depre_b.height = hover_b.height = 60
    pushbutton = pyglet.gui.ToggleButton(x, y, pressed=press_b, depressed=depre_b, hover=hover_b, batch=batch)

    return pushbutton



###----------------------------------------------------------------------
#        create_scaled_sprite()
#       Inputs    :     image_path: The file path of the image to be loaded.
#                       window: The Pyglet window object to determine screen dimensions.
#                       batch: Pyglet graphics batch for rendering.
#                       group: Pyglet graphics group for ordering.
#
#       Output    :     Returns a pyglet.sprite.Sprite object.
#   Description   :     Loads an image and scales it to fit the window while maintaining 
#                       the aspect ratio. The scaled image is then created as a sprite 
#                       and added to the specified batch and group for rendering.
###----------------------------------------------------------------------
def create_scaled_sprite(image_path, window, batch, group):
    
    # Load the image
    image = pyglet.image.load(image_path)

    # Get the window dimensions
    screen_width, screen_height = window.width, window.height

    # Get the image dimensions
    image_width, image_height = image.width, image.height

    # Calculate the scale
    scale_x = screen_width / image_width
    scale_y = screen_height / image_height

    # Choose the smaller scale to maintain aspect ratio
    scale = min(scale_x, scale_y)

    # Create and scale the sprite
    sprite = pyglet.sprite.Sprite(image, x=0, y=0, batch=batch, group=group)
    sprite.update(scale=scale)

    return sprite



###----------------------------------------------------------------------
# 
#        cv2_to_pyglet_image()
#       Inputs    :     cv2_frame: A frame image from OpenCV (in BGR format).
#
#       Output    :     Returns a pyglet.image.ImageData object.
#   Description   :     Converts an OpenCV image (BGR format) to a Pyglet-compatible image
#                       format (RGB). It extracts image data and prepares it for use in 
#                       Pyglet applications, such as for textures or sprites.
###----------------------------------------------------------------------
def cv2_to_pyglet_image(cv2_frame):
    height, width, channels = cv2_frame.shape
    # Convert BGR to RGB
    cv2_frame_rgb = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
    # Create a Pyglet image from the RGB frame
    image_data = cv2_frame_rgb.tobytes()
    pyglet_image = ImageData(width, height, 'RGB', image_data, pitch=-width * 3)
    return pyglet_image




###----------------------------------------------------------------------
#           render_TextEntry()
#       Inputs    :     x: X-coordinate for text entry position.
#                       y: Y-coordinate for text entry position.
#                       batch: Pyglet graphics batch for rendering.
#
#       Output    :     Returns a pyglet.gui.TextEntry object.
#   Description   :     Creates a text entry field at the specified position and adds
#                       it to the given graphics batch. The text entry field allows 
#                       users to input text, which can be used for labeling or other 
#                       interactive purposes.
###----------------------------------------------------------------------
def render_TextEntry(x, y, batch):
    textentry = pyglet.gui.TextEntry("Enter Name", x, y, 100, batch=batch)
    return textentry




###----------------------------------------------------------------------
#           render_frame()
#       Inputs    :     output_frames: A frame image from OpenCV.
#                       texture: The OpenGL texture ID to which the frame will be applied.
#
#       Output    :     None (modifies the OpenGL context).
#   Description   :     Resizes the frame image, converts it to RGB format, and flips it
#                       vertically to align with OpenGL's coordinate system. The image 
#                       data is then bound to an OpenGL texture and rendered onto a defined 
#                       quadrilateral, effectively displaying the video frame within the 
#                       application window.
###----------------------------------------------------------------------
def render_frame(output_frames, texture):
    finsize = (640, 640)
    output_frames = cv2.resize(output_frames, finsize)
    frame_rgb = cv2.cvtColor(output_frames, cv2.COLOR_BGR2RGB)
    frame_rgb = np.flip(frame_rgb, axis=0)

    # Update the texture with new frame data
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, frame_rgb)

    # Define vertices, texture coordinates, and surface
    verts = [
        (-1, -1), (1, -1),
        (1, 1), (-1, 1)
    ]

    vertis = [
    (-0.9, -0.9), (0.9, -0.9),  # Bottom-left and bottom-right corners
    (0.9, 0.8), (-0.9, 0.8)      # Top-right and top-left corners
    ]
    texts = [
        (0, 0), (1, 0),
        (1, 1), (0, 1)
    ]
    surf = (0, 1, 2, 3)

    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture)

    # Draw the quad using vertex arrays
    glBegin(GL_QUADS)
    for i in surf:
        glTexCoord2f(texts[i][0], texts[i][1])
        glVertex2f(vertis[i][0], vertis[i][1] + 0.1)
    glEnd()

    glDisable(GL_TEXTURE_2D)




###----------------------------------------------------------------------
#           render_splash_screen()
#       Inputs    :     None
#
#       Output    :     None (modifies the OpenGL context).
#   Description   :     Draws a splash screen within the OpenGL context, creating a black
#                       rectangle covering the window except for a specified area reserved
#                       for buttons. This function sets up the projection and model view 
#                       matrices for rendering in 2D and is likely used during the initial 
#                       launch of the application or during transitions.
###--------------------------------------------------------------------------------------------------------------------------------------------
def render_splash_screen():
    # Draw splash screen within the OpenGL context
    glPushAttrib(GL_ALL_ATTRIB_BITS)
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, -1, 1)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glBegin(GL_QUADS)
    glColor4f(0.0, 0.0, 0.0, 1.0)
    glVertex2f(0, 0)
    glVertex2f(WINDOW_WIDTH, 0)
    glVertex2f(WINDOW_WIDTH, WINDOW_HEIGHT - BUTTON_HEIGHT)
    glVertex2f(0, WINDOW_HEIGHT - BUTTON_HEIGHT)
    glEnd()

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopAttrib()




"""
textures = glGenTextures(1, [0])
    texture = textures[0]
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

def display_video(output_frames, scsize):
    # Define window dimensions based on the first frame received
    first_frame = output_frames.get()
    window_width, window_height = scsize.shape[1], scsize.shape[0]

    # Create Pyglet window
    window = pyglet.window.Window(width=window_width, height=window_height)
    
    # Set up OpenGL
    glClearColor(0, 0, 0, 0)
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Create OpenGL texture
    texture_id = GLuint()
    glGenTextures(1, ctypes.byref(texture_id))
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, window_width, window_height, 0, GL_BGR, GL_UNSIGNED_BYTE, first_frame)

    @window.event
    def on_draw():
        window.clear()
        glLoadIdentity()
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex2f(0, 0)
        glTexCoord2f(1, 0)
        glVertex2f(window_width, 0)
        glTexCoord2f(1, 1)
        glVertex2f(window_width, window_height)
        glTexCoord2f(0, 1)
        glVertex2f(0, window_height)
        glEnd()

    def update_texture(dt):
        frame = output_frames.get()
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.shape[1], frame.shape[0], 0, GL_BGR, GL_UNSIGNED_BYTE, frame)

    # Schedule the update function to be called every frame
    pyglet.clock.schedule_interval(update_texture, 1 / 30.0)

    pyglet.app.run()
"""