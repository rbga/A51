import cv2
import os
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import time

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

num_pics_per_orientation = 10
orientations = ['front', 'left', 'back', 'right', 'top', 'bottom']
flash_duration = 200  # Duration of the flash in milliseconds
save_path = 'captured_images'

if not os.path.exists(save_path):
    os.makedirs(save_path)


def capture_images_known(results, count, frame, scsize):
    for idx in range(count):
        # Flash effect
       # Capture and save the cropped images using YOLO's built-in crop function
        results.save_crop(save_path, "a.jpg")
        a, b = scsize
        cv2.putText(img = frame,
            text = str(idx+1),
            org = (b//2, b//4),
            fontFace = cv2.FONT_HERSHEY_DUPLEX,
            fontScale = 1,
            color = (202, 104, 0),
            thickness = 1
            )


def display_message(frame, message):
    display_frame = frame.copy()
    cv2.putText(img = display_frame,
                text = message,
                org = (0, 35),
                fontFace = cv2.FONT_HERSHEY_DUPLEX,
                fontScale = 0.4,
                color = (202, 104, 0),
                thickness = 1
                )
    #cv2.imshow('Frame', display_frame)
    #cv2.waitKey(1000)


def is_inside_fixed_rect(box, fixed_rect):
    x1, y1, x2, y2 = box
    fx1, fy1, fx2, fy2 = fixed_rect
    return x1 >= fx1 and y1 >= fy1 and x2 <= fx2 and y2 <= fy2


def process_frame_with_box_and_mask(frame, i, box, results, scsize):
    fixed_rect_x1, fixed_rect_y1, fixed_rect_x2, fixed_rect_y2 = calculate_rect_coordinates(scsize)
    # Draw fixed rectangle
    cv2.rectangle(frame, (fixed_rect_x1, fixed_rect_y1), (fixed_rect_x2, fixed_rect_y2), (0, 255, 0), 2)
    cv2.putText(img = frame,
            text = 'Place Object exactly within box',
            org = (0, 5),
            fontFace = cv2.FONT_HERSHEY_DUPLEX,
            fontScale = 0.4,
            color = (153, 76, 0),
            thickness = 1
            )

    # Check if the detected object is inside the fixed rectangle
    if is_inside_fixed_rect(box, (fixed_rect_x1, fixed_rect_y1, fixed_rect_x2, fixed_rect_y2)):
        display_message(frame, "Press r when ready")
        # Prompt user to place the object and get ready
        while True:
            if cv2.waitKey(1) & 0xFF == ord('r'):
                break

        # Capture images for each orientation
        for orientation in orientations:
            display_message(frame, f'{orientation.capitalize()} (Press r when ready)')
            while True:
                if cv2.waitKey(1) & 0xFF == ord('r'):
                    break
            capture_images_known(results, num_pics_per_orientation, frame, scsize)
            cv2.waitKey(3000)

        display_message(frame, 'Capture complete. Trainer Quitting')
        cv2.waitKey(3000)
    #    return frame
    
    else:
        display_message(frame, 'No object detected inside the box')

   # return frame  # Indicate that we should continue processing



def capture_images_unknown(frame, orientation, count, scsize):
    for i in range(count):

        fixed_rect_x1, fixed_rect_y1, fixed_rect_x2, fixed_rect_y2 = calculate_rect_coordinates(scsize)
        # Capture and save the image
        crop_img = frame[fixed_rect_y1:fixed_rect_y2, fixed_rect_x1:fixed_rect_x2]
        cv2.imwrite(f'{save_path}/{orientation}_{i + 1}.jpg', crop_img)
        a, b = scsize
        cv2.putText(img = frame,
                    text = str(i+1),
                    org = (b//2, b//4),
                    fontFace = cv2.FONT_HERSHEY_DUPLEX,
                    fontScale = 1,
                    color = (202, 104, 0),
                    thickness = 1
                    )


def process_frames_unknown(frame, scsize):
    # Draw fixed rectangle
    fixed_rect_x1, fixed_rect_y1, fixed_rect_x2, fixed_rect_y2 = calculate_rect_coordinates(scsize)
    cv2.rectangle(frame, (fixed_rect_x1, fixed_rect_y1), (fixed_rect_x2, fixed_rect_y2), (0, 255, 0), 2)
    cv2.putText(img = frame,
            text = 'Place Object exactly within box',
            org = (0, 5),
            fontFace = cv2.FONT_HERSHEY_DUPLEX,
            fontScale = 0.4,
            color = (76, 153, 0),
            thickness = 1
            )
    display_message(frame, "Press r when ready")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('r'):
            break

    # Capture images for each orientation
    for orientation in orientations:
        display_message(frame, f'{orientation.capitalize()} (Press r when ready)')
        while True:
            if cv2.waitKey(1) & 0xFF == ord('r'):
                break
        capture_images_unknown(frame, orientation, num_pics_per_orientation, scsize)
        time.sleep(3)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    display_message(frame, 'Capture complete. Quitting Trainer')
    time.sleep(3)
    #return frame