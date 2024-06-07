import cv2
import os
import torch
from ultralytics.utils.plotting import Annotator
from time import sleep
from queue import Empty
from logprinter import print_log, print_simple_log, log_std
import numpy as np

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_rect_coordinates(scsize, percentage=0.7):
    screen_width, screen_height = scsize
    rect_width = int(screen_width * percentage)
    rect_height = int(screen_height * percentage)

    # Center the rectangle
    x1 = (screen_width - rect_width) // 2
    y1 = (screen_height - rect_height) // 2
    x2 = x1 + rect_width
    y2 = y1 + rect_height

    return x1, y1, x2, y2


def put_text_on_rectangle(image, text, org, scsize, text_color=(255, 255, 255), rect_color=(0, 0, 0)):
    screen_width, screen_height = scsize
    
    # Proportional scaling
    font_scale = min(screen_width, screen_height) / 1000
    thickness = max(1, int(font_scale * 2))
    
    # Get the width and height of the text box
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    
    # Calculate the coordinates of the rectangle
    x, y = org
    rect_start = (x, y - text_height - baseline)
    rect_end = (x + text_width, y + baseline)
    
    # Draw the rectangle
    cv2.rectangle(image, rect_start, rect_end, rect_color, cv2.FILLED)
    
    # Put the text on the rectangle
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)



        
num_pics_per_orientation = 10
orientations = ['front', 'left', 'back', 'right', 'top', 'bottom']
flash_duration = 200  # Duration of the flash in milliseconds
save_path = 'captured_images'

if not os.path.exists(save_path):
    os.makedirs(save_path)


def is_inside_fixed_rect(box, fixed_rect):
    x1, y1, x2, y2 = box
    fx1, fy1, fx2, fy2 = fixed_rect
    return x1 >= fx1 and y1 >= fy1 and x2 <= fx2 and y2 <= fy2

def draw_dotted_box_with_crosshair(image, x1,y1,x2,y2, label):
    """
    Draws a dotted line box with crosshair-like pointers at the midpoints and a label on the top left.

    Parameters:
    - image: The image on which to draw the box.
    - box: A tuple/list of four integers (x1, y1, x2, y2) representing the coordinates of the box.
    - label: A string label to display on the top left corner of the box.

    Returns:
    - The image with the detection box and label drawn.
    """

    # Define box color and label background color
    box_color = (0, 255, 255)  # Yellow
    label_text_color = (0, 0, 0)  # Black

    # Draw the label text
    cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)

    # Draw the dotted box
    line_length = 5
    gap_length = 5
    
    # Top and bottom horizontal lines
    for x in range(x1, x2, line_length + gap_length):
        cv2.line(image, (x, y1), (min(x + line_length, x2), y1), box_color, 1)
        cv2.line(image, (x, y2), (min(x + line_length, x2), y2), box_color, 1)
    
    # Left and right vertical lines
    for y in range(y1, y2, line_length + gap_length):
        cv2.line(image, (x1, y), (x1, min(y + line_length, y2)), box_color, 1)
        cv2.line(image, (x2, y), (x2, min(y + line_length, y2)), box_color, 1)

    # Draw the crosshairs at the midpoints
    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2

    # Top center
    cv2.line(image, (mid_x, y1), (mid_x, y1 + 5), box_color, 1)
    #cv2.line(image, (mid_x, y1), (mid_x + 5, y1), box_color, 1)
    
    # Bottom center
    cv2.line(image, (mid_x, y2), (mid_x, y2 - 5), box_color, 1)
    #cv2.line(image, (mid_x, y2), (mid_x + 5, y2), box_color, 1)
    
    # Left center
    cv2.line(image, (x1, mid_y), (x1 + 5, mid_y), box_color, 1)
    #cv2.line(image, (x1, mid_y), (x1, mid_y + 5), box_color, 1)
    
    # Right center
    cv2.line(image, (x2, mid_y), (x2 - 5, mid_y), box_color, 1)
    #cv2.line(image, (x2, mid_y), (x2, mid_y + 5), box_color, 1)

    return image


def draw_crosshair(image, x1, y1, x2, y2, label):

    line_length=5
    gap_length=5
    # Define box color and label background color
   # Define box color and label background color
    box_color = (0, 255, 255)  # Yellow
    label_text_color = (0, 0, 0)  # Black

    # Draw the label text
    cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)

    # Pre-calculate the limited line length
    max_line_length = x2 - x1 - gap_length

    # Lines points for top, bottom, left, right sides
    # Lines points for top, bottom, left, right sides
    line_points = [
        [(x, y1), (min(x + line_length, x1 + max_line_length), y1)] for x in range(x1, x1 + max_line_length, line_length + gap_length)
    ] + [
        [(x, y2), (min(x + line_length, x1 + max_line_length), y2)] for x in range(x1, x1 + max_line_length, line_length + gap_length)
    ] + [
        [(x1, y), (x1, min(y + line_length, y2))] for y in range(y1, y2, line_length + gap_length)
    ] + [
        [(x2, y), (x2, min(y + line_length, y2))] for y in range(y1, y2, line_length + gap_length)
    ]


    # Flatten the list of line points
    line_points = [point for sublist in line_points for point in sublist]

    # Convert the line points to NumPy array
    line_points = np.array(line_points)

    # Draw all lines at once
    for i in range(0, len(line_points), 2):
        cv2.line(image, tuple(line_points[i]), tuple(line_points[i+1]), box_color, 1)

    # Draw the crosshairs at the midpoints
    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2

    # Crosshair points
    crosshair_points = [
        ((mid_x, y1), (mid_x, y1 + 5)),
        ((mid_x, y2), (mid_x, y2 - 5)),
        ((x1, mid_y), (x1 + 5, mid_y)),
        ((x2, mid_y), (x2 - 5, mid_y))
    ]

    # Draw all crosshairs at once
    for point_pair in crosshair_points:
        cv2.line(image, point_pair[0], point_pair[1], box_color, 1)

    return image


def detect_objects(frame_processed, original_frame, results):
    # Convert frame tensor to a numpy array
    # frame_processed = frame_processed.squeeze().permute(1, 2, 0).cpu().numpy()
    #log_std('Entering DETECT OBJECTS SUB FUNC')
    # Check if any detections were made
    for r in results:
        # Extract bounding boxes and labels for the first frame
        boxes = r.boxes
        labels = r.names
        # Draw bounding boxes and labels on the frame
        for i, box in enumerate(boxes):
            # Assuming box.xyxy provides the coordinates as a tuple or list
            x1, y1, x2, y2 = box.xyxy[0]  # Adjust this line according to the actual attribute name
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id = box.cls.item()  # Get class ID for each box
            class_name = labels[class_id]
            draw_crosshair(original_frame, x1, y1, x2, y2, class_name)
    
    return original_frame



    
def detect_an_object(frame_processed, original_frame, scsize, results, event_queue):
    # Convert frame tensor to a numpy array.
    frame_processed = frame_processed.squeeze().permute(1, 2, 0).cpu().numpy().copy()

    fixed_rect_x1, fixed_rect_y1, fixed_rect_x2, fixed_rect_y2 = calculate_rect_coordinates(scsize)

    # Draw fixed rectangle
    cv2.rectangle(original_frame, (fixed_rect_x1, fixed_rect_y1), (fixed_rect_x2, fixed_rect_y2), (0, 255, 0), 2)
    cv2.putText(original_frame, 'Place Object exactly within box', (0, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

    annotator = Annotator(original_frame, line_width=0.1)

    state = {
        "prompted_for_name": False,
        "event_S": False,
        "event_W": False,
        "ievent_C": False,
        "got_name": False,
        "frontA": False,
        "frontB": False,
        "frontC": False,
        "frontD": False,
        "Top": False,
        "Bottom": False,
        "termi": False
    }

    def update_state(event, state):
        if event == 'Q':
            state["termi"] = True
        elif event == 'S':
            state["event_S"] = True
        elif event == 'W':
            state["event_W"] = True
        elif len(event) > 1:
            state["obj_name"] = event
            state["got_name"] = True
        elif event == 'C':
            state["ievent_C"] = True

    def draw_prompt(text, area):
        cv2.putText(original_frame, text, area, cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        return original_frame

    if results:

        for r in results:

            if r.masks:
                for i, mask in enumerate(r.masks.xy):
                    class_id = r.boxes[i].cls.item()  # Get class ID for each mask

                    if not state["prompted_for_name"]:
                        class_name = r.names[class_id]  # Convert class ID to class name
                    else:
                        class_name = state["obj_name"]

                    annotator.seg_bbox(mask=mask, mask_color=(0, 255, 255), det_label=str(class_name))
                    frameT1 = original_frame.copy()

                    #if not state["event_S"]:
                    if is_inside_fixed_rect(r.boxes[i].xyxy[0].cpu().tolist(), (fixed_rect_x1, fixed_rect_y1, fixed_rect_x2, fixed_rect_y2)):
                        original_frame = draw_prompt('Aligned. Press -s- to Start.', (5, 25))
                        try:
                            event = event_queue.get(timeout=1)  # Add timeout to avoid blocking indefinitely
                            print_simple_log(event)
                            update_state(event, state)
                            if state["termi"]:
                                break
                        except Empty:
                            pass
                
                        if state["event_S"]:
                            original_frame = frameT1.copy()
                            original_frame = draw_prompt("Object INSIDE", (5, 200))
                                

                    if not is_inside_fixed_rect(r.boxes[i].xyxy[0].cpu().tolist(), (fixed_rect_x1, fixed_rect_y1, fixed_rect_x2, fixed_rect_y2)):
                        original_frame = frameT1.copy()
                        original_frame = draw_prompt("Object not Inside. Move it.", (5, 200))
                        
    return original_frame

    
def process_frame(frame_processed, original_frame, isTrainer, results, eque):
    if isTrainer:
        scsize = (256, 256)
        detected_frame = detect_an_object(frame_processed, original_frame, scsize, results, eque)
    else:
        detected_frame = detect_objects(frame_processed, original_frame, results)
    return detected_frame



"""
def detect_an_object(frame, scsize, results, key_queue):
    # Convert frame tensor to a numpy array
    frame_np = frame.squeeze().permute(1, 2, 0).cpu().numpy()
    termi = False

    fixed_rect_x1, fixed_rect_y1, fixed_rect_x2, fixed_rect_y2 = calculate_rect_coordinates(scsize)
    # Draw fixed rectangle
    cv2.rectangle(frame_np, (fixed_rect_x1, fixed_rect_y1), (fixed_rect_x2, fixed_rect_y2), (0, 255, 0), 2)
    cv2.putText(img = frame_np,
            text = 'Place Object exactly within box',
            org = (0, 5),
            fontFace = cv2.FONT_HERSHEY_DUPLEX,
            fontScale = 0.5,
            color = (0, 0, 0),
            thickness = 1
            )
    prompted_for_name = False
    
        #cv2.imshow('Frame', frame_np)
    if results:
        annotator = Annotator(frame_np, line_width=0.1)
        for r in results:
            if r.masks:
                for i, mask in enumerate(r.masks.xy):
                    x1, y1, x2, y2 = r.boxes[i].xyxy[0]  # Adjust this line according to the actual attribute name
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    class_id = r.boxes[i].cls.item()  # Get class ID for each mask
                    
                    if not prompted_for_name:
                        class_name = r.names[class_id]  # Convert class ID to class name
                    else:
                        class_name = obj_name

                    annotator.seg_bbox(mask=mask, mask_color=(0, 255, 255), det_label=str(class_name))
                    #draw_dotted_box_with_crosshair(frame_np, x1, y1, x2, y2, class_name)
                    #cv2.imshow('Frame', frame_np)
                    
        
                    if is_inside_fixed_rect(r.boxes[i].xyxy[0].cpu().tolist(), (fixed_rect_x1, fixed_rect_y1, fixed_rect_x2, fixed_rect_y2)):
                        put_text_on_rectangle(frame_np, 'Aligned. Press -s- to Start.', (5, 20), scsize, text_color=(255, 255, 255), rect_color=(0, 0, 255))
                        
                        if not key_queue.empty():
                            key = key_queue.get()
                            if key == ord('s'):
                                if not prompted_for_name:
                                    obj_name = input("Enter Obj name: ")
                                    prompted_for_name = True

                                elif prompted_for_name and obj_name:
                                #if cv2.waitKey(1) & 0xFF == ord('s'):  
                                    for orientation in orientations:
                                        put_text_on_rectangle(frame_np, f'{orientation.capitalize()} Press -c- to start.', (5, 35), scsize, text_color=(255, 255, 255), rect_color=(0, 0, 255))
                                        key = key_queue.get()
                                        if key == ord('s'):
                                            for idx in range(num_pics_per_orientation):
                                                # Capture and save the cropped images using YOLO's built-in crop function
                                                r.save_crop(save_path, f"{class_name}_{orientation}.jpygame")
                                                put_text_on_rectangle(frame_np, str(idx), (128, 225), scsize, text_color=(255, 255, 255), rect_color=(0, 0, 255))
                                                #cv2.waitKey(1)
                                    termi = True
                                    break
                                
                            if termi:
                                print ("Capture complete. Trainer Quitting")
                                return True
            
                    else:
                        #print ("Object not Inside. Move it.")
                        put_text_on_rectangle(frame_np, "Object not Inside. Move it.", (128, 200), scsize, text_color=(255, 255, 255), rect_color=(0, 0, 255))
                    #auto_trainer.process_frame_with_box_and_mask(frame_np, i, box.xyxy[0].cpu().tolist(), r, scsize)
            
            if not key_queue.empty():
                key = key_queue.get()
                if key == ord('s') | termi:
                    pass
                break              
    #else:
        #auto_trainer.process_frames_unknown(frame_np, scsize)
        return frame_np

"""