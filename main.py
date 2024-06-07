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
from logprinter import print_log, print_simple_log, log_std


def capture_frames(video_source, input_frames):
    cap = cv2.VideoCapture(video_source)
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

def load_model_worker(input_frames, output_frames, model_type, eque):
    log_std('Entered LOAD MODEL WORKER')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    allobj = list(range(1, 80))
    obj = allobj[1:]
    obj.remove(63)
    #obj = [1, 40, 41, 42, 43, 44, 45]

    #model = YOLO('yolov8n.pt')
    
    # Start training from scratch
    #model.train(data='coco8.yaml', epochs=20, imgsz=256, resume=True, save=True).to(device)
    #print("Starting training from scratch...")
    #model.train()
    #torch.save(model.state_dict(), weights_file)
    #print("Training completed and weights saved.")

    if model_type == "detect":
        print_simple_log('DETECT Loaded')
        model = YOLO('yolov8n.pt').to(device)
        scsize = (640, 640)
    elif model_type == "train":
        print_simple_log('SEGMENT Loaded')
        model = YOLO('yolov8n-seg.pt').to(device)
        scsize = (256, 256)

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
                results = model.predict(frame_processed, max_det=7, vid_stride=20, imgsz=320, classes=obj, conf=0.3, iou=0.5, verbose=False, device="cuda")
                processed_frame = object_detection.process_frame(frame_processed, frame, False, results, eque)
            
            elif model_type == "train":
                #print_simple_log('TRAIN Predicting')
                results = model(frame_processed, max_det=1, vid_stride=20, imgsz=256, classes=obj, conf=0.3, iou=0.5, verbose=False, device="cuda")
                processed_frame = object_detection.process_frame(frame_processed, frame, True, results, eque)
        
        
        while output_frames.full():
            try:
                output_frames.get_nowait()  # Remove the oldest frame to make space
            except Empty:
                pass
        
        while output_frames.qsize() < 21:
            output_frames.put(processed_frame)

def train(eque):

    print_log('Entering CHOICE 2')
    # Detect Object
    scsize = 640, 640
    WIDTH, HEIGHT = scsize

    capture_process = mp.Process(target=capture_frames, args=(0, input_frames))
    worker_process = mp.Process(target=load_model_worker, args=(input_frames, output_frames, "train", eque))

    capture_process.start() 
    print_log('Started CAPTURE')
    worker_process.start()
    print_log('Started WORKER')

def detect(eque):

    print_log('Entering CHOICE 1')
    # Detect Object
    scsize = 640, 640
    WIDTH, HEIGHT = scsize

    capture_process = mp.Process(target=capture_frames, args=(0, input_frames))
    worker_process = mp.Process(target=load_model_worker, args=(input_frames, output_frames, "detect", eque))

    capture_process.start() 
    print_log('Started CAPTURE')
    worker_process.start()
    print_log('Started WORKER')

def play():

    # Initialize Pyglet window
    WINDOW_WIDTH = 640
    WINDOW_HEIGHT = 768  # 640 (video) + 128 (buttons)

    config = pyglet.gl.Config(double_buffer=True)
    batch = pyglet.graphics.Batch()
    window = pyglet.window.Window(config=config, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, caption='YOLOv5 Detection')

    DETECT_LABEL = 'DETECT'
    TRAIN_LABEL = 'TRAIN'
    # Flag to indicate if video is playing
    video_playing = False
    text_entry_active = False

    texture_id = GLuint()
    glGenTextures(1, ctypes.byref(texture_id))
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)


    # Create OpenGL texture

    dt = 110  # Adjust the x-coordinate to your preference
    tr = 330  # Adjust the x-coordinate to your preference
    tb = 275
    tby = 7
    button_y = 25


    def on_dt_toggle(state):
        nonlocal video_playing
        if state:
            video_playing = True
            detect(eque)


    def on_tr_toggle(state):
        nonlocal video_playing
        if state:
            video_playing = True
            train(eque)

    def text_entry_handler(text):
        eque.put(str(text))

    def toggle_text_entry(text_entry_active):
        if text_entry_active:
            window.remove_handlers(te_b)
            text_entry_active = False
        else:
            window.push_handlers(te_b)
            text_entry_active = True

    # Render buttons
    dt_b = displayer.render_button(DETECT_LABEL, dt, button_y, 'detect_press.png', 'detect_unpress.png', 'detect_hover.png', batch)
    tr_b = displayer.render_button(TRAIN_LABEL, tr, button_y, 'train_press.png', 'train_unpress.png', 'train_hover.png', batch)
    te_b = displayer.render_TextEntry(tb, tby, batch)

    window.push_handlers(dt_b)
    window.push_handlers(tr_b)


    dt_b.set_handler('on_toggle', on_dt_toggle)
    tr_b.set_handler('on_toggle', on_tr_toggle)
    te_b.set_handler('on_commit', text_entry_handler)
    

    @window.event
    def on_draw():
        window.clear()
        batch.draw()
        # Set up the top viewport (640x640)
        glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        try:
            if video_playing:
                try:
                    op = output_frames.get_nowait()  # Use get_nowait to avoid blocking
                    displayer.render_frame(op, texture_id)
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
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.Q:
            eque.put("Q")
            pyglet.app.exit()
        elif symbol == pyglet.window.key.S:
            eque.put("S")
        elif symbol == pyglet.window.key.C:
            eque.put("C")
        elif symbol == pyglet.window.key.T:
            eque.put("T")
        elif symbol == pyglet.window.key.N:
            eque.put("N")
        elif symbol == pyglet.window.key.ENTER:
            eque.put("ENTER")
        elif symbol == pyglet.window.key.BACKSPACE:
            eque.put("BACKSPACE")
        elif symbol == pyglet.window.key.W:
            toggle_text_entry(text_entry_active)

        

    @window.event
    def on_close():
        pyglet.app.exit()

    pyglet.app.run()


if __name__ == "__main__":
    mp.set_start_method('spawn')
    input_frames = mp.Queue(maxsize=30)
    output_frames = mp.Queue()
    eque = mp.Queue()

    play()



# Your other functions and code here...
"""
def display_frames(output_frames, scsize, event_queue):
    # Initialize Pygame
    WIDTH, HEIGHT = scsize
    PYGAME_FLAGS = DOUBLEBUF | OPENGL
    pygame.display.set_mode((WIDTH, HEIGHT), PYGAME_FLAGS)        
    gluPerspective(25, (WIDTH / HEIGHT), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)
    # OpenGL setup
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    while True:
        # Capture frame from output_frames queue
        frame = output_frames.get()

        if frame is None:
            break

        # Convert OpenCV frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = np.flip(frame_rgb, axis=0)

        # Update the texture with new frame data
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WIDTH, HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, frame_rgb)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        verts = [
            (-1, -1), (1, -1),
            (1, 1), (-1, 1)
        ]
        texts = [(0, 0), (1, 0), (1, 1), (0, 1)]
        #texts = ((1, 0), (1, 1), (0, 1), (0, 0))
        surf = (0, 1, 2, 3)

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture)

        glBegin(GL_QUADS)
        for i in surf:
            glTexCoord2f(texts[i][0], texts[i][1])
            glVertex2f(verts[i][0], verts[i][1])
        glEnd()
        
        glDisable(GL_TEXTURE_2D)

        pygame.display.flip()

    #pygame.quit()  
"""
"""
def display_frames(output_frames, key_queue):
    while True:
        frame = output_frames.get()
        
        if frame is None:
            break
        
        cv2.imshow('YOLOv5 Detection', frame)
        
        if not key_queue.empty():
            key = key_queue.get()
            if key == ord('q'):
                break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    return
"""
"""

# Create the main window
def main():
    
    event_queue = mp.Queue()

    

    print("Choose an option:")
    print("1. Live Camera Image Detection")
    print("2. Train New Things")
    print("3. Exit")
    #choice = input("Enter your choice (1/2/3): ")

    choice = '1'

    if choice == '1':
        print_log('Entering CHOICE 1')
        # Detect Object
        scsize = 640, 640
        WIDTH, HEIGHT = scsize
        
        #base_font = pygame.font.Font(None, 24) 

        
        capture_process = mp.Process(target=capture_frames, args=(0, input_frames))
        worker_process = mp.Process(target=load_model_worker, args=(input_frames, output_frames, "detect"))
        display_process = mp.Process(target=display_video, args=(output_frames, scsize, texture))
        #event_process = mp.Process(target=event_handler, args=(event_queue,))
        
        capture_process.start() 
        print_log('Started CAPTURE')
        worker_process.start()
        print_log('Started WORKER')
        display_process.start()
        print_log('Started DISPLAY')
        event_process.start()
        print_log('Started EVENT HANDLER')

    elif choice == '2':
        print_log('Entering CHOICE 2')
        scsize = 256, 256
        WIDTH, HEIGHT = scsize
        
        capture_process = mp.Process(target=capture_frames, args=(0, input_frames))
        training_process = mp.Process(target=load_model_worker, args=(input_frames,output_frames, "train", event_queue))
        display_process = mp.Process(target=display_frames, args=(output_frames, scsize, event_queue))
        event_process = mp.Process(target=event_handler, args=(event_queue,))
        
        capture_process.start()
        print_log('Started CAPTURE')
        training_process.start()
        print_log('Started TRAINER')
        display_process.start()
        print_log('Started DISPLAY')
        event_process.start()
        print_log('Started EVENT HANDLER')


    else:
        print("Invalid choice. Please try again.")


    # Wait for processes to finish
    capture_process.join()
    print_log('Joined CAPTURE')
    if choice == '1':
        worker_process.join()
        print_log('Joined WORKER')
    elif choice == '2':
        training_process.join()
        print_log('Joined TRAINER')
    display_process.join()
    print_log('Joined DISPLAY')
    event_process.join()
    print_log('Joined EVENT HANDLER')

    # Close queues
    input_frames.close()
    output_frames.close()
    event_queue.close()
    print_log('Closed QUEUES')

    # Quit Pygame
    pygame.quit()


"""
"""
def display_frames(output_frames, key_queue, scsize):

    WIDTH, HEIGHT = scsize
    pygame.init()
    pygame.display.set_caption('YOLOv5 Detection')
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    window.fill([0,0,0])
    
    while True:
        # Capture frame from output_frames queue
        framez = output_frames.get()

        if framez is None:
            break

        framez = cv2.cvtColor(framez, cv2.COLOR_BGR2RGB)

        surf = pygame.surfarray.make_surface(np.flip(framez, axis=0))
        surf = pygame.transform.rotate(surf, 90)
        
        # Display frame
        window.blit(surf, (0, 0))
        pygame.display.update()

        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Check for key press events
        if not key_queue.empty():
            key = key_queue.get()
            if key == ord('q'):
                pygame.quit()
                return

    pygame.quit()
"""