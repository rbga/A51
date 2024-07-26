import cv2
import torch
from ultralytics import YOLO
import object_detection
import multiprocessing as mp
from queue import Empty
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import displayer
import pyglet
from pyglet.gl import *
import event_support as esp
from logprinter import print_log, print_simple_log, log_std



################################################################################################
#                  INFORMATION
#
#            File Name  :   main.py
#            Developer  :   Rishi Balasubramanian
#            Call Sign  :   RBGA
#   First Stable Build  :   14th MAY 2024
#             Use Case  :   Vision Framework's Main Program
#                 
#                 Type  :   Functions and MODULE MAIN PROGRAM
#               Inputs  :   Many
#
#               Output  :   Many
#          Description  :   The file implements a real-time object detection system using YOLOv8
#                           and OpenGL for visual display. It includes functions for capturing 
#                           video frames, processing frames with a neural network, handling user input, 
#                           and displaying the results using Pyglet and OpenGL.
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




###----------------------------------------------------------------------
#          capture_frames()
#       Inputs    :     video_source: The video source, such as a camera index or video file path.
#                       input_frames: A multiprocessing queue to store captured video frames.
#
#       Output    :     None
#   Description   :     Captures frames from the specified video source and stores them in the input_frames queue. 
#                       Ensures the queue is not full by removing the oldest frames when necessary.
###----------------------------------------------------------------------
def capture_frames(video_source, input_frames):
    cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
    while cap.isOpened():
        ret, frame = cap.read()
        #print("Camera Frames - " + str(frame.shape[:2]))
        if not ret:
            break
        while input_frames.full():
            try:
                input_frames.get_nowait()  # Remove the oldest frame to make space
            except Empty:
                pass
        input_frames.put(frame)
    cap.release()
    input_frames.close()  # Signal workers to stop





###----------------------------------------------------------------------
#          load_model_worker()
#       Inputs    :     input_frames: A multiprocessing queue containing video frames to be processed.
#                       output_frames: A multiprocessing queue to store processed video frames.
#                       model_type: A string indicating the model type ('detect' or 'train').
#                       eventQueue: A multiprocessing queue for event handling.
#                       labelQueue: A multiprocessing queue for label handling.
#
#       Output    :     None
#   Description   :     Loads a YOLO model for object detection or segmentation based on model_type. 
#                       Processes frames from input_frames, performs object detection/segmentation, 
#                       and places the results in output_frames. Uses eventQueue and labelQueue for 
#                       additional processing and handling.
###----------------------------------------------------------------------
def load_model_worker(input_frames, output_frames, model_type, eventQueue, labelQueue):
    log_std('Entered LOAD MODEL WORKER')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    allobj = list(range(1, 80))
    obj = allobj[1:]

    if model_type == "detect":
        print_simple_log('DETECT Loaded')
        model = YOLO('yolov8n.pt').to(device)
        scsize = (640, 480)
    elif model_type == "train":
        print_simple_log('SEGMENT Loaded')
        model = YOLO('yolov8n-seg.pt').to(device)
        #scsize = (256, 256)
        scsize = (640, 480)

    while True:
        frame = input_frames.get()
        if frame is None:
            break
        
        frame = cv2.resize(frame, scsize)  # Resize to expected input size
        frame_processed = frame.transpose((2, 0, 1))  # Channels first
        frame_processed = frame_processed / 255.0
        frame_processed = torch.from_numpy(frame_processed).float().unsqueeze(0).to(device)

        with torch.no_grad():
            if model_type == "detect":
                #print_simple_log('DETECT Predicting')
                results = model.predict(frame_processed, max_det=7, vid_stride=20, imgsz=640, classes=obj, conf=0.3, iou=0.5, verbose=False, device="cuda")
                processed_frame = object_detection.process_frame(frame_processed, frame, False, results, eventQueue, labelQueue)
            
            elif model_type == "train":
                #print_simple_log('TRAIN Predicting')
                results = model(frame_processed, max_det=1, vid_stride=20, imgsz=640, classes=39, conf=0.3, iou=0.5, verbose=False, device="cuda")
                processed_frame = object_detection.process_frame(frame_processed, frame, True, results, eventQueue, labelQueue)
                # Remove ignored class indices from allobj
                obj = [i for i in allobj if i not in esp.rejected_obj.values()]
        
        
        while output_frames.full():
            try:
                output_frames.get_nowait()  # Remove the oldest frame to make space
            except Empty:
                pass
        
        while output_frames.qsize() < 21:
            output_frames.put(processed_frame)





###----------------------------------------------------------------------
#               train()
#       Inputs    :     eventQueue: A multiprocessing queue for handling events.
#                       labelQueue: A multiprocessing queue for handling labels.
#
#       Output    :     None
#   Description   :     Starts the processes for capturing video frames and processing 
#                       them using the YOLO model in training mode. It initializes the 
#                       system for object detection with a focus on training.
###----------------------------------------------------------------------
def train(eventQueue, labelQueue):

    print_log('Entering CHOICE 2')
    # Detect Object
    scsize = 640, 480
    WIDTH, HEIGHT = scsize

    capture_process = mp.Process(target=capture_frames, args=(0, input_frames))
    worker_process = mp.Process(target=load_model_worker, args=(input_frames, output_frames, "train", eventQueue, labelQueue))

    capture_process.start() 
    print_log('Started CAPTURE')
    worker_process.start()
    print_log('Started WORKER')





###----------------------------------------------------------------------
#              detect()
#       Inputs    :     eventQueue: A multiprocessing queue for handling events.
#                       labelQueue: A multiprocessing queue for handling labels.
#
#       Output    :     None
#   Description   :     Similar to train, but configures the system to start object 
#                       detection in detection mode.
###----------------------------------------------------------------------
def detect(eventQueue, labelQueue):

    print_log('Entering CHOICE 1')
    # Detect Object
    scsize = 640, 480
    WIDTH, HEIGHT = scsize

    capture_process = mp.Process(target=capture_frames, args=(0, input_frames))
    worker_process = mp.Process(target=load_model_worker, args=(input_frames, output_frames, "detect", eventQueue, labelQueue))

    capture_process.start() 
    print_log('Started CAPTURE')
    worker_process.start()
    print_log('Started WORKER')





###----------------------------------------------------------------------
#               play()
#       Inputs    :     None (uses global variables and queues)
#
#       Output    :     None
#   Description   :     Sets up a Pyglet window for displaying processed video frames and UI elements. 
#                       Manages rendering, user interactions, and integrates with the event and label 
#                       queues to update the display based on processed data. Handles various user inputs 
#                       to control the detection or training process.
###----------------------------------------------------------------------
def play():

    display = pyglet.canvas.Display()
    screen = display.get_default_screen()

    # Initialize Pyglet window
    # WINDOW_WIDTH = screen.width
    # WINDOW_HEIGHT = screen.height 

    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 720 

    log_std("WIDTH  -> " + str(WINDOW_WIDTH))
    log_std("HEIGHT -> " + str(WINDOW_HEIGHT))

    rect_width = WINDOW_WIDTH / 5
    rect_height = WINDOW_HEIGHT / 3

    # Calculate the starting (x, y) point
    
    config = pyglet.gl.Config(double_buffer=True)
    batch = pyglet.graphics.Batch()
    window = pyglet.window.Window(config=config, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, caption='YOLOv5 Detection')

    DETECT_LABEL = 'DETECT'
    TRAIN_LABEL = 'TRAIN'
    # Flag to indicate if video is playing
    video_playing = False
    text_entry_active = False

    background = pyglet.graphics.Group(order=0)
    foreground = pyglet.graphics.Group(order=1)

    textbox = pyglet.image.load('textBox.png')

    sprite = displayer.create_scaled_sprite('textBox.png', window, batch, background)

    texture_id = GLuint()
    glGenTextures(1, ctypes.byref(texture_id))
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)


    # Calculate button positions
    button_y = WINDOW_HEIGHT / 9  # Y position for the buttons, close to the bottom
    button_spacing = 20  # Spacing between buttons

    detectPositionX = (WINDOW_WIDTH - (2 * rect_width + button_spacing)) / 2  # Centering the buttons horizontally
    trainPositionX = detectPositionX + rect_width + button_spacing
    tBoxX = (detectPositionX + trainPositionX) / 2 + (rect_width / 5) 
    tBoxY = button_y - 30

    rect_x = (WINDOW_WIDTH - (2 * rect_width)) / 2 - button_spacing
    rect_y = WINDOW_HEIGHT / 4

    previousLabels = []
    allLabels = []



    ###----------------------------------------------------------------------
    #           update_labels()
    #       Inputs    :     texts: The new label text(s) to display.
    #                       allLabels: A list holding the current Pyglet label objects.
    #                       previousLabels: A list holding the previous label texts for comparison.
    #                       batch: A Pyglet graphics batch for efficient rendering.
    #
    #       Output    :     None
    #   Description   :     Updates the displayed labels based on new text inputs. Compares new text 
    #                       with previous labels, clears old labels if necessary, and draws new labels.
    ###----------------------------------------------------------------------
    def update_labels(texts, allLabels, previousLabels, batch):
        # Convert new_texts to string if needed
        new_texts = str(texts)
        
        if new_texts != ''.join(previousLabels) or new_texts == "Object not Inside. Move it.":  # Check if it's different
            print("New texts received, updating labels")
            # Clear previous labels
            while allLabels:
                delete_last_label(allLabels)

            # Update previous texts
            previousLabels.clear()
            previousLabels.append(new_texts)  # Store new text

            # Draw new label
            draw_prompt(texts, (WINDOW_WIDTH/2, 700), batch, allLabels)



    ###----------------------------------------------------------------------
    #         delete_last_label()
    #       Inputs    :     allLabels: A list holding the current Pyglet label objects.
    #
    #       Output    :     None
    #   Description   :     Deletes the last label from the allLabels list and removes it from display.
    ###----------------------------------------------------------------------
    def delete_last_label(allLabels):
        if allLabels:
            print("Deleting last label")
            last_label = allLabels.pop()
            last_label.delete()




    ###----------------------------------------------------------------------
    #             draw_prompt()
    #       Inputs    :     text: The text to display as a label.
    #                       position: A tuple representing the x and y coordinates for label placement.
    #                       batch: A Pyglet graphics batch for efficient rendering.
    #                       allLabels: A list to store created label objects.
    #
    #       Output    :     None
    #   Description   :     Creates and displays a new label with the specified text at the given position. 
    #                       The label is added to the allLabels list for management.
    ###----------------------------------------------------------------------
    def draw_prompt(text, position, batch, allLabels):
        print(f"Creating label with text: {text}")
        for i, line in enumerate(text):
            label = pyglet.text.Label(
                str(line),
                font_name='Arial',
                font_size=12,
                x=position[0],
                y=position[1],
                anchor_x='center',
                anchor_y='center',
                color=(255, 0, 0, 255),  # Red color
                batch=batch
            )
            allLabels.append(label)




    ###----------------------------------------------------------------------
    #             on_dt_toggle()
    #       Inputs    :     state: A boolean indicating whether detection mode is active.
    #       Output    :     None
    #   Description   :     Toggles detection mode based on the state. If true, sets video_playing to True and calls the detect() function.
    ###----------------------------------------------------------------------
    def on_dt_toggle(state):
        nonlocal video_playing
        if state:
            video_playing = True
            detect(eventQueue, labelQueue)




    ###----------------------------------------------------------------------
    #            on_tr_toggle()
    #       Inputs    :     state: A boolean indicating whether training mode is active.
    #       Output    :     None
    #   Description   :     Toggles training mode based on the state. If true, sets video_playing to True and calls the train() function.
    ###----------------------------------------------------------------------
    def on_tr_toggle(state):
        nonlocal video_playing
        if state:
            video_playing = True
            train(eventQueue, labelQueue)



    ###----------------------------------------------------------------------
    #          text_entry_handler()
    #       Inputs    :     text: The text input from the user.
    #       Output    :     None
    #   Description   :     Handles text input by placing it in the eventQueue.
    ###----------------------------------------------------------------------
    def text_entry_handler(text):
        eventQueue.put(str(text))



    ###----------------------------------------------------------------------
    #         toggle_text_entry()
    #       Inputs    :     text_entry_active: A boolean indicating if text entry is currently active.
    #       Output    :     None
    #   Description   :     Toggles text entry mode. If active, removes handlers from the window; if not, adds handlers.
    ###----------------------------------------------------------------------
    def toggle_text_entry(text_entry_active):
        if text_entry_active:
            window.remove_handlers(te_b)
            text_entry_active = False
        else:
            window.push_handlers(te_b)
            text_entry_active = True

    # Render buttons
    dt_b = displayer.render_button(DETECT_LABEL, detectPositionX, button_y, 'detect_press.png', 'detect_unpress.png', 'detect_hover.png', batch)
    tr_b = displayer.render_button(TRAIN_LABEL, trainPositionX, button_y, 'train_press.png', 'train_unpress.png', 'train_hover.png', batch)
    te_b = displayer.render_TextEntry(tBoxX, tBoxY, batch)

    window.push_handlers(dt_b)
    window.push_handlers(tr_b)


    dt_b.set_handler('on_toggle', on_dt_toggle)
    tr_b.set_handler('on_toggle', on_tr_toggle)
    te_b.set_handler('on_commit', text_entry_handler)
    

    @window.event
    ###----------------------------------------------------------------------
    #               on_draw()
    #       Inputs    :     None (uses global variables and states)
    #
    #       Output    :     None
    #   Description   :     Handles the drawing of the Pyglet window content. Displays video frames if 
    #                       video_playing is True, updates labels, and handles exceptions. Renders the UI 
    #                       components and handles transitions between different states.
    ###----------------------------------------------------------------------
    def on_draw():
        window.clear()
        batch.draw()
        # Set up the top viewport (640x640)
        glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        try:
            if video_playing:
                try:
                    op = output_frames.get_nowait()  # Use get_nowait to avoid blocking
                    pyg = displayer.cv2_to_pyglet_image(op)
                    pyg.blit(rect_x, rect_y, 0)
                    
                    try:
                        new_texts = labelQueue.get_nowait()
                        update_labels([new_texts], allLabels, previousLabels, batch)
                        batch.draw()
                    except Empty:
                        pass

                except Empty:
                    # Handle the case where the queue is empty, possibly render a placeholder or wait
                    displayer.render_splash_screen()
            else:
                displayer.render_splash_screen()

        except Exception as e:
            # Log or handle the exception appropriately
            print(f"Error occurred: {e}")
            displayer.render_splash_screen()

        # Render buttons
        batch.draw()


    @window.event
    ###----------------------------------------------------------------------
    #             on_key_press()
    #       Inputs    :     symbol: The key symbol pressed by the user.
    #                       modifiers: Any modifier keys pressed. 
    #
    #       Output    :     None
    #   Description   :     Processes key press events, placing corresponding commands into the 
    #                       eventQueue and toggling text entry mode if necessary.
    ###----------------------------------------------------------------------
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.Q:
            eventQueue.put("Q")
            pyglet.app.exit()

        elif symbol == pyglet.window.key.S:
            eventQueue.put("S")

        elif symbol == pyglet.window.key.C:
            eventQueue.put("C")

        elif symbol == pyglet.window.key.T:
            eventQueue.put("T")

        elif symbol == pyglet.window.key.N:
            eventQueue.put("N")

        elif symbol == pyglet.window.key.Y:
            eventQueue.put("Y")

        elif symbol == pyglet.window.key.F:
            eventQueue.put("F")

        elif symbol == pyglet.window.key.A:
            eventQueue.put("A")

        elif symbol == pyglet.window.key.R:
            eventQueue.put("R")

        elif symbol == pyglet.window.key.X:
            eventQueue.put("X")

        elif symbol == pyglet.window.key.ENTER:
            eventQueue.put("ENTER")

        elif symbol == pyglet.window.key.BACKSPACE:
            eventQueue.put("BACKSPACE")

        elif symbol == pyglet.window.key.W:
            eventQueue.put("W")
            toggle_text_entry(text_entry_active)

        

    @window.event
    ###----------------------------------------------------------------------
    #               on_close()
    #       Inputs    :     None (uses global state)
    #
    #       Output    :     None
    #   Description   :     Handles the closing of the Pyglet window, exiting the application gracefully.
    ###----------------------------------------------------------------------
    def on_close():
        pyglet.app.exit()

    pyglet.app.run()





###----------------------------------------------------------------------
#              __main__()
#       Inputs    :     None (initializes global variables and queues)
#
#       Output    :     None
#   Description   :     The entry point of the program. Sets the multiprocessing start method 
#                       to 'spawn' and initializes queues for frame input, frame output, events, 
#                       and labels. Calls the play() function to start the application.
###----------------------------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method('spawn')
    input_frames = mp.Queue(maxsize=10)
    output_frames = mp.Queue()
    eventQueue = mp.Queue()
    labelQueue = mp.Queue(maxsize=1)

    play()