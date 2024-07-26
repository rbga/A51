import cv2
import os
import time
import numpy as np



################################################################################################
#                  INFORMATION
#
#            File Name  :   vision_support.py
#            Developer  :   Rishi Balasubramanian
#            Call Sign  :   RBGA
#   First Stable Build  :   18th MAY 2024
#             Use Case  :   Custom Python Support Library involving most things related to 
#                           OpenCV manipulation
#
#                 Type  :   Function(s)
#               Inputs  :   Many
#
#               Output  :   Many
#          Description  :   The code provides utilities for drawing graphical elements on 
#                           images using OpenCV, including calculating rectangle coordinates, 
#                           drawing crosshairs, adding text prompts, and checking if a bounding 
#                           box is within a fixed rectangle. This is often used in computer vision 
#                           applications for object detection, annotation, and UI overlays.
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
#     calculate_rect_coordinates()
#       Inputs    :     scsize (tuple): The size of the screen or image (width, height).
#                       percentage (float, optional): The percentage of the screen size 
#                                                     to use for the rectangle dimensions (default is 0.7).
#
#       Output    :     (tuple): Coordinates of the rectangle (x1, y1, x2, y2).
#   Description   :     Calculates the coordinates of a centered rectangle within a screen or image of a given
#                       size. The size of the rectangle is determined by the percentage parameter relative to 
#                       the screen dimensions.
###----------------------------------------------------------------------
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




###----------------------------------------------------------------------
#             draw_crosshair()
#       Inputs    :     image (numpy.ndarray): The image on which to draw.
#                       x1, y1, x2, y2 (int): The coordinates defining the rectangle.
#                       label (str): The label text to display above the rectangle.
#
#       Output    :     image (numpy.ndarray): The modified image with the crosshair and label.
#   Description   :     Draws a crosshair and label on the image at the specified rectangle coordinates. 
#                       The crosshair consists of lines at the midpoint and corners of the rectangle, 
#                       and the label is placed just above the top-left corner of the rectangle.
###----------------------------------------------------------------------
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




###----------------------------------------------------------------------
#             draw_prompt()
#       Inputs    :     original_frame (numpy.ndarray): The image on which to draw the prompt.
#                       text (str): The text to display.
#                       area (tuple): The coordinates where the text should be placed (x, y).
#
#       Output    :     None
#   Description   :     Draws a text prompt on the provided image at the specified location. 
#                       This is useful for adding annotations or prompts directly onto video 
#                       frames or images.
###----------------------------------------------------------------------
def draw_prompt(original_frame, text, area):
    cv2.putText(original_frame, text, area, cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 0, 255), 1)




###----------------------------------------------------------------------
#        is_inside_fixed_rect()
#       Inputs    :     box (tuple): Coordinates of a box (x1, y1, x2, y2).
#                       fixed_rect (tuple): Coordinates of the fixed rectangle (fx1, fy1, fx2, fy2).
#
#       Output    :     bool: True if the box is completely inside the fixed rectangle, False otherwise.
#   Description   :     Checks whether a given box is entirely within a specified fixed rectangle. This 
#                       function is useful for validating object positions or ensuring elements are 
#                       within a designated area on the screen.
###----------------------------------------------------------------------
def is_inside_fixed_rect(box, fixed_rect):
    x1, y1, x2, y2 = box
    fx1, fy1, fx2, fy2 = fixed_rect
    return x1 >= fx1 and y1 >= fy1 and x2 <= fx2 and y2 <= fy2
















"""
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

def draw_dotted_box_with_crosshair(image, x1,y1,x2,y2, label):
    Draws a dotted line box with crosshair-like pointers at the midpoints and a label on the top left.

    Parameters:
    - image: The image on which to draw the box.
    - box: A tuple/list of four integers (x1, y1, x2, y2) representing the coordinates of the box.
    - label: A string label to display on the top left corner of the box.

    Returns:
    - The image with the detection box and label drawn.


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
"""