from ultralytics import YOLO
import torch

################################################################################################
#                  INFORMATION
#
#            File Name  :   trainer.py
#            Developer  :   Rishi Balasubramanian
#            Call Sign  :   RBGA
#   First Stable Build  :   8th JULY 2024
#             Use Case  :   YOLO Model Trainer Executable
#
#                 Type  :   Executbale
#               Inputs  :   None
#
#               Output  :   YOLO Model Training
#          Description  :   YOLO Model Trainer Executable
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

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #model = YOLO('C:/Users/ganes/Desktop/AI/Vision/runs/detect/train14/weights/best.pt').to(device)
    model = YOLO('yolov8m.pt').to(device)
    model.train(data='C:/Users/ganes/Desktop/AI/Vision/Cattle/cowabunga.yaml', epochs=50, imgsz=416, batch=-1, scale=0.5, save=True)

    print("DONE")