import pyglet
from pyglet.gl import *
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from ctypes import c_uint8, c_bool
import cv2


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

def canvas():
        # Draw the rectangle
    glColor3f(0.5, 0.5, 0.5)  # Set the color (RGB)
    glBegin(GL_QUADS)
    glVertex2f(100, 100)  # Bottom-left corner
    glVertex2f(700, 100)  # Bottom-right corner
    glVertex2f(700, 500)  # Top-right corner
    glVertex2f(100, 500)  # Top-left corner
    glEnd()

def render_button(label, x, y, press, depress, hover, batch):
    press_b = pyglet.resource.image(press)
    depre_b = pyglet.resource.image(depress)
    hover_b = pyglet.resource.image(hover)

    press_b.width = depre_b.width = hover_b.width = 200
    press_b.height = depre_b.height = hover_b.height = 60
    pushbutton = pyglet.gui.ToggleButton(x, y, pressed=press_b, depressed=depre_b, hover=hover_b, batch=batch)

    return pushbutton


def render_frame(output_frames, texture):
    finsize = (640, 640)
    output_frames = cv2.resize(output_frames, finsize)
    frame_rgb = cv2.cvtColor(output_frames, cv2.COLOR_BGR2RGB)
    frame_rgb = np.flip(frame_rgb, axis=0)

    # Update the texture with new frame data
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WINDOW_WIDTH, WINDOW_WIDTH, 0, GL_RGB, GL_UNSIGNED_BYTE, frame_rgb)

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