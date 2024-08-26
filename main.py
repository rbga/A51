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
import uxElements as ux
from logprinter import print_log, print_simple_log, log_std
from PIL import Image



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
#------------------------------------------------------------------
#               LAST MODIFICATION
#            Developer  :   Rishi Balasubramanian
#            Call Sign  :   RBGA
# Date of Modification  :   26th JULY 2024
#
#          Description  :   Replaced Static Image Button with Video
#                           Button Class. Modified sub functions
#                           related to button and processes.
#                           InferenceVideo is now a Sprite instead
#                           of Blit.
#                           on_draw()
#                           on_mouse_motion()
#                           on_mouse_press()
#                           on_mouse_release()
#                           on_resize()
#------------------------------------------------------------------
#
#------------------------------------------------------------------
#               LAST MODIFICATION
#            Developer  :   Rishi Balasubramanian
#            Call Sign  :   RBGA
# Date of Modification  :   11th AUG 2024
#
#          Description  :   videoButton and Ux Elements are now in
#                           their own file. Changed classes of video
#                           button to new and implemented initial 
#                           UX design with Main Window, Inference Window
#                           and text window.
#------------------------------------------------------------------
#
#------------------------------------------------------------------
#               LAST MODIFICATION
#            Developer  :   Rishi Balasubramanian
#            Call Sign  :   RBGA
# Date of Modification  :   13th AUG 2024
#
#          Description  :   Calculated Variables for every UI element
#                           or display element was constructed and 
#                           applied. Everything is now a factor of
#                           WINDOW_WIDTH and WINDOW_HEIGHT variables
#                           which determine size and scale of the elements.
#                           Code Cleanup.
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

    allobj = list(range(0, 600))
    obj = allobj

    if model_type == "detect":
        print_simple_log('DETECT Loaded')
        model = YOLO('yolov8x-oiv7.pt').to(device)
        scsize = (640, 480)
    elif model_type == "train":
        print_simple_log('SEGMENT Loaded')
        model = YOLO('yolov8x-seg.pt').to(device)
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
                results = model.predict(frame_processed, imgsz=640, conf=0.3, iou=0.5, classes=obj, verbose=False, device="cuda")
                processed_frame = object_detection.process_frame(frame_processed, frame, False, results, eventQueue, labelQueue)
            
            elif model_type == "train":
                #print_simple_log('TRAIN Predicting')
                results = model(frame_processed, max_det=1, vid_stride=20, imgsz=640, classes=obj, conf=0.3, iou=0.5, verbose=False, device="cuda")
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


def get_png_dimensions(png_file_path):
    with Image.open(png_file_path) as img:
        width, height = img.size
    return width, height


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

    # Calculate the starting (x, y) point
    
    config = pyglet.gl.Config(double_buffer=True, alpha_size=8)
    batch = pyglet.graphics.Batch()
    window = pyglet.window.Window(config=config, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, caption='YOLOv5 Detection', resizable=True)
    
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Flag to indicate if video is playing
    video_playing = False
    text_entry_active = False

    background = pyglet.graphics.Group(order=0)
    foreground = pyglet.graphics.Group(order=1)

    texture_id = GLuint()
    glGenTextures(1, ctypes.byref(texture_id))
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    detectButtonStateVisuals = {
    'idle'                  : 'detectIdleState_1_frames',
    'hover_transition'      : 'detHoverTransition_1_frames',
    'hover_idle'            : 'detHoverIdle_1_frames',
    'dehover_transition'    : 'detUnhoverTransition_1_frames',
    'press_transition'      : 'detPressedTransition_1_frames',
    'press_idle'            : 'detPressedIdle_1_frames',
    'unpress_transition'    : 'detUnpressedTransition_1_frames'
    }
    
    trainButtonStateVisuals = {
    'idle'                  : 'trainIdleState_1_frames',
    'hover_transition'      : 'trainHoverTransition_1_frames',
    'hover_idle'            : 'trainHoverIdle_1_frames',
    'dehover_transition'    : 'trainUnhoverTransition_1_frames',
    'press_transition'      : 'trainPressedTransition_1_frames',
    'press_idle'            : 'trainPressedIdle_1_frames',
    'unpress_transition'    : 'trainUnpressedTransition_1_frames'
    }

    mainWindowEntrance      = 'mainWindowFrame_frames'  # Folder containing entrance animation PNGs
    mainWindowLoop          = 'mainWindowFrameLoop_frames'  # Folder containing loop animation PNGs
    inferenceEntrance       = 'inferenceWindowFrame_frames'  # Folder containing entrance animation PNGs
    inferenceLoop           = 'infLoop_frames'  
    twEntrance              = 'textWindow_frames'  # Folder containing entrance animation PNGs
    twLoop                  = 'textWindowLoop_frames' 
    textAnimation           = 'glow'

    labelX = 0.49739583
    labelY = 0.22407407
    
    textEntryX = 0.46
    textEntryY = 0.087

    previousLabels = []
    allLabels = []
    textAnimX           = 0.2890625
    textAnimY           = 0.125
   # animation_instance = ux.uxAnimation(textAnimX, textAnimY, 800, 249, WINDOW_WIDTH, WINDOW_HEIGHT, textAnimation, batch)

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
            draw_prompt(texts, (WINDOW_WIDTH * labelX, WINDOW_HEIGHT * labelY), batch, allLabels)



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
                font_name='Play',
                font_size=16,
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

    # # Render buttons
    te_b = displayer.render_TextEntry(WINDOW_WIDTH * textEntryX, WINDOW_HEIGHT * textEntryY, batch)


    #Elements Position Calculations
    detectX         = 0.0598958333333333
    detectY         = 0.0944444444444444
    trainX          = 0.7942708333333333
    trainY          = detectY
    buttonsScaleX   = 0.140625
    buttonsScaleY   = 0.25
    
    inferenceX      = 0.1875
    inferenceY      = 0.32407
    inferenceScaleX = 0.625
    inferenceScaleY = 0.625
    
    textX           = 0.2890625
    textY           = 0.125
    textScaleX      = 0.4166666666666667
    textScaleY      = 0.1972222222222222
    
    inferenceSpriteX = 0.2864583333333333
    inferenceSpriteY = 0.3981481481481481


    detectButton = ux.uxLiveButton(
        x                   =   WINDOW_WIDTH    * detectX, 
        y                   =   WINDOW_HEIGHT   * detectY,
        width               =   WINDOW_WIDTH    * buttonsScaleX, 
        height              =   WINDOW_HEIGHT   * buttonsScaleY, 
        WINDOW_WIDTH        =   WINDOW_WIDTH, 
        WINDOW_HEIGHT       =   WINDOW_HEIGHT, 
        videos              =   detectButtonStateVisuals, 
        batch               =   batch, 
        group               =   foreground, 
        on_toggle_callback  =   on_dt_toggle, 
        verbose             =   False
        )
    
    trainButton = ux.uxLiveButton(
        x                   =   WINDOW_WIDTH    * trainX, 
        y                   =   WINDOW_HEIGHT   * trainY,
        width               =   WINDOW_WIDTH    * buttonsScaleX, 
        height              =   WINDOW_HEIGHT   * buttonsScaleY, 
        WINDOW_WIDTH        =   WINDOW_WIDTH, 
        WINDOW_HEIGHT       =   WINDOW_HEIGHT, 
        videos              =   trainButtonStateVisuals, 
        batch               =   batch, 
        group               =   foreground, 
        on_toggle_callback  =   on_tr_toggle, 
        verbose             =   False
        )
    
    main_window_frame = ux.uxWindowElements(
        x               =   0, 
        y               =   0, 
        width           =   WINDOW_WIDTH, 
        height          =   WINDOW_HEIGHT, 
        WINDOW_WIDTH    =   WINDOW_WIDTH, 
        WINDOW_HEIGHT   =   WINDOW_HEIGHT, 
        entrance_videos =   mainWindowEntrance, 
        loop_videos     =   mainWindowLoop, 
        batch           =   batch, 
        group           =   background,
        verbose         =   False
    )
    
    inference_window_frame = ux.uxWindowElements(
        x               =   WINDOW_WIDTH    *   inferenceX, 
        y               =   WINDOW_HEIGHT   *   inferenceY, 
        width           =   WINDOW_WIDTH    *   inferenceScaleX, 
        height          =   WINDOW_HEIGHT   *   inferenceScaleY, 
        WINDOW_WIDTH    =   WINDOW_WIDTH, 
        WINDOW_HEIGHT   =   WINDOW_HEIGHT, 
        entrance_videos =   inferenceEntrance, 
        loop_videos     =   inferenceLoop, 
        batch           =   batch, 
        group           =   foreground,
        verbose         =   False
    )
    
    text_window_frame = ux.uxWindowElements(
        x               =   WINDOW_WIDTH    *   textX, 
        y               =   WINDOW_HEIGHT   *   textY, 
        width           =   WINDOW_WIDTH    *   textScaleX, 
        height          =   WINDOW_HEIGHT   *   textScaleY, 
        WINDOW_WIDTH    =   WINDOW_WIDTH, 
        WINDOW_HEIGHT   =   WINDOW_HEIGHT, 
        entrance_videos =   twEntrance, 
        loop_videos     =   twLoop, 
        batch           =   batch, 
        group           =   foreground,
        verbose         =   False
    )
    
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

        # Draw the batch and buttons
        batch.draw()

        # Set up the top viewport (640x640)
        glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        try:
            if video_playing:
                try:
                    op = output_frames.get_nowait()  # Use get_nowait to avoid blocking
                    pyg = displayer.cv2_to_pyglet_image(op, WINDOW_WIDTH, WINDOW_HEIGHT)
                    inferenceSprite = pyglet.sprite.Sprite(img=pyg, x=WINDOW_WIDTH * inferenceSpriteX, y=WINDOW_HEIGHT * inferenceSpriteY, batch=batch, group=foreground)
                    
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
        
        # Draw the batch and buttons again in case they need to be redrawn
        batch.draw()


    @window.event
    ###----------------------------------------------------------------------
    #           <on_mouse_motion>()
    #       Inputs    :     x, y: The current mouse coordinates.
    #                       dx, dy: The change in mouse position along the x 
    #                       and y axes, respectively.
    #
    #       Output    :     None
    #   Description   :     This function is called whenever the mouse is 
    #                       moved. It updates the state of detectButton and 
    #                       trainButton based on the new mouse coordinates, 
    #                       allowing them to respond to hover events or other 
    #                       motion-based interactions.
    ###----------------------------------------------------------------------
    def on_mouse_motion(x, y, dx, dy):
        detectButton.on_mouse_motion(x, y, dx, dy)
        trainButton.on_mouse_motion(x, y, dx, dy)



    @window.event
    ###----------------------------------------------------------------------
    #           <on_mouse_press>()
    #       Inputs    :     x, y: The coordinates of the mouse when the button is pressed.
    #                       button: The mouse button that was pressed (e.g., left, right).
    #                       modifiers: Any modifier keys pressed (e.g., Shift, Ctrl).
    #
    #       Output    :     None
    #   Description   :     This function is triggered when a mouse button is pressed. 
    #                       It handles mouse press events for detectButton and trainButton, 
    #                       updating their states to reflect the press action. This can 
    #                       include visual feedback or triggering specific button functions.
    ###----------------------------------------------------------------------
    def on_mouse_press(x, y, button, modifiers):
        detectButton.on_mouse_press(x, y, button, modifiers)
        trainButton.on_mouse_press(x, y, button, modifiers)




    @window.event
    ###----------------------------------------------------------------------
    #           <on_mouse_release>()
    #       Inputs    :     x, y: The coordinates of the mouse when the button is pressed.
    #                       button: The mouse button that was pressed (e.g., left, right).
    #                       modifiers: Any modifier keys pressed (e.g., Shift, Ctrl).
    #
    #       Output    :     None
    #   Description   :     This function handles mouse release events, updating 
    #                       the states of detectButton and trainButton accordingly. 
    #                       It is used to detect the end of a button press and may 
    #                       trigger actions associated with a full click cycle.
    ###----------------------------------------------------------------------
    def on_mouse_release(x, y, button, modifiers):
        detectButton.on_mouse_release(x, y, button, modifiers)
        trainButton.on_mouse_release(x, y, button, modifiers)




    @window.event
    ###----------------------------------------------------------------------
    #              <on_resize>()
    #       Inputs    :     width: The new width of the window.
    #                       height: The new height of the window.
    #
    #       Output    :     None
    #   Description   :     This function is called when the window is resized. 
    #                       It adjusts the size and positioning of detectButton 
    #                       and trainButton to ensure they scale appropriately 
    #                       with the new window dimensions. This helps maintain 
    #                       a consistent layout and user experience regardless 
    #                       of the window size.
    ###----------------------------------------------------------------------
    def on_resize(width, height):
        detectButton.resize(width, height)
        trainButton.resize(width, height)
        inference_window_frame.resize(width, height)
        text_window_frame.resize(width, height)

    
    def update(dt):
        detectButton.update(dt)
        trainButton.update(dt)
        main_window_frame.update(dt)
        inference_window_frame.update(dt)
        text_window_frame.update(dt)
        
    pyglet.clock.schedule_interval(update, 1/60)  # 60 Hz update rate



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