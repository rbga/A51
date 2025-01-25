# A51
An attempt...

# YOLOv5 Detection and Training Application

- This application is designed to perform object detection using the YOLOv5 model, train custom models, and visualize the results in real-time using Pyglet and OpenGL.
- The User Interface is custom designed from scratch where the concept was developed and rendered into GIF series on Adobe After Effects CC, and rendered into seamless live UI components by the Pyglet Game Engine's batch rendering abilities.
- Look into class uxElements.py for usage instructions. It is a unified UI class, that can take in any GIF series to convert it into a functioning UI element.
- Employs multiple cores to execute. A core is dedicated to capturing frames, another to process the frames where GPU is involved in AI Object Detection and rendering by OpenGL, another to display the frames where all of these cores communicate through inter-process Pipes including one for user inputs, to ensure consistent and reliable, real-time streaming with negligible delay.
- The video demo shown in this readme uses Open Images v7 dataset by Google, consisting of 600 object classes across 9 million images.
- Detect Mode uses Ultralytics YOLO v8, X sized OIV7 pretrained model to detect every day objects in real-time.
- Train Mode (under development) is envisioned to help user create datasets of common objects, through user input, capturing cropped images of object shown in camera in real-time.

https://github.com/user-attachments/assets/356d191e-1521-4bda-880c-2116d55c3889



## Features
- **Object Detection:** Capture frames from a video source and perform object detection using a pre-trained YOLOv5 model.
- **Model Training:** Train a custom YOLOv5 model on your dataset.
- **Real-time Display:** Display the processed frames in real-time using OpenGL for rendering and Pyglet for window management.
- **Multiprocessing:** Use Python's multiprocessing library for efficient frame processing and model inference.

## Requirements
- Python 3.x
- OpenCV
- PyTorch
- Ultralytics YOLO
- Pyglet
- OpenGL
- NumPy

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/yolov5-detection-app.git
    ```

2. Install the required dependencies:

    ```bash
    pip install opencv-python torch pyglet ultralytics numpy
    ```

## Usage

### Step 1: Choose Detection or Training Mode

The application offers two modes: 
1. **Detect**: Perform object detection on a live video stream.
2. **Train**: Train the YOLOv5 model on a custom dataset.

You can toggle between these modes using the provided buttons in the GUI.

### Step 2: Start the Application

Run the following command to start the application:

```bash
python main.py
```
The window will open, showing live video or training progress based on the selected mode.

## Key Controls
- **Q**: Exit the application.
- **S**: Pause detection or training.
- **C**: Clear the text entry field.
- **T**: Toggle training mode.
- **N**: Toggle detection mode.
- **ENTER**: Commit text entry.
- **BACKSPACE**: Delete last character in text entry.

## Step 3: Monitor the Output
Once the video stream is running, the processed frames will be displayed in real-time. You can visualize detected objects or training progress in the OpenGL window.

## Application Workflow
1. **Capture Frames:** A separate process continuously captures frames from the video source (e.g., webcam).
2. **Model Inference:** Another process performs inference using the YOLOv5 model (either for detection or training) and processes the frames accordingly.
3. **Display:** The processed frames are then rendered to an OpenGL window using Pyglet for visualization.

## Multiprocessing Design
- **Capture Process:** Captures frames from the video source.
- **Worker Process:** Performs inference using the YOLOv5 model.
- **Display Process:** Handles rendering the frames to the OpenGL window.

## Example
Here is an example of running the detection mode with the camera input:

```python
if __name__ == "__main__":
    mp.set_start_method('spawn')
    input_frames = mp.Queue(maxsize=30)
    output_frames = mp.Queue()
    eque = mp.Queue()

    play()  # Starts the application
```

## Detect Application

https://github.com/user-attachments/assets/0b757c3f-9f1a-47b2-9525-6130363cb53d


## Train Application (under Development)

https://github.com/user-attachments/assets/1631210b-7983-4653-aaf7-2195f6e8f3d1



Notes:

MEETING NOTES
https://docs.google.com/presentation/d/1i8qZjWOP19JCkIbSv-rHuwqp2b03QUt2VAocm_ukREA/edit?usp=sharing

KNOWLEDGE BANK
https://docs.google.com/presentation/d/1yhLDQupQcLxlD4wfGEHZpi9GepGllEBeM1k0BonbAtw/edit?usp=sharing

PROGRESS AND RESULTS
https://docs.google.com/presentation/d/15LEjfh_IZqzSwHz8GNqzD6ooeeCmD4QvO4HoV1rBC6c/edit?usp=sharing

PROJECT RESOURCES
https://docs.google.com/presentation/d/1rJHgJzzFisi86PKm76uFG0RQFeCqf45I5FcTA2uSv3g/edit?usp=sharing


Notes:

MEETING NOTES
https://docs.google.com/presentation/d/1i8qZjWOP19JCkIbSv-rHuwqp2b03QUt2VAocm_ukREA/edit?usp=sharing

KNOWLEDGE BANK
https://docs.google.com/presentation/d/1yhLDQupQcLxlD4wfGEHZpi9GepGllEBeM1k0BonbAtw/edit?usp=sharing

PROGRESS AND RESULTS
https://docs.google.com/presentation/d/15LEjfh_IZqzSwHz8GNqzD6ooeeCmD4QvO4HoV1rBC6c/edit?usp=sharing
