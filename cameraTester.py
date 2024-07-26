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



# Open the default camera (0)
cap = cv2.VideoCapture(0, cv2.CAP_ANY)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    cv2.imshow('Camera', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
