import cv2



################################################################################################
#                  INFORMATION
#
#            File Name  :   cameraTester.py
#            Developer  :   Rishi Balasubramanian
#            Call Sign  :   RBGA
#   First Stable Build  :   15th JULY 2024
#             Use Case  :   Custom Python Program to test various Camera Modes of OpenCV
#
#                 Type  :   Executbale
#               Inputs  :   None
#
#               Output  :   Video Display and Log Text
#          Description  :   Custom Python Program to test various Camera Modes of OpenCV
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



# # Open the default camera (0)
# cap = cv2.VideoCapture(0, cv2.CAP_ANY)

# if not cap.isOpened():
#     print("Error: Could not open camera.")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to grab frame.")
#         break

#     cv2.imshow('Camera', frame)

#     # Exit on pressing 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the camera and close windows
# cap.release()
# cv2.destroyAllWindows()


import cv2

def has_alpha_channel(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Error opening video file {video_path}")

    has_alpha = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Print dtype and shape of the frame
        print(f"Frame dtype: {frame.dtype}")
        print(f"Frame shape: {frame.shape}")

        if len(frame.shape) == 3 and frame.shape[2] == 4:
            has_alpha = True
            break

    cap.release()
    return has_alpha

if __name__ == '__main__':
    video_path = 'UiElements/detectIdleState_1.avi'
    if has_alpha_channel(video_path):
        print("The video has an alpha channel.")
    else:
        print("The video does not have an alpha channel.")

