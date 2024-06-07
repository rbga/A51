import cv2
import multiprocessing as mp
import numpy as np
from multiprocessing import shared_memory
import time

def camera_shm_capture(queue, shm_name, shape, dtype):
    shm = shared_memory.SharedMemory(name=shm_name)
    frame_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    cap = cv2.VideoCapture(0)  # Open the default camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set to highest resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        np.copyto(frame_array, frame)
        queue.put(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

def frame_shm_display(queue, shm_name, shape, dtype):
    shm = shared_memory.SharedMemory(name=shm_name)
    frame_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    while True:
        queue.get()
        frame = frame_array.copy()
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def draw__shm_rectangle(queue, shm_name, shape, dtype):
    shm = shared_memory.SharedMemory(name=shm_name)
    frame_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    while True:
        queue.get()
        frame = frame_array.copy()
        height, width, _ = frame.shape
        top_left = (width // 4, height // 4)
        bottom_right = (3 * width // 4, 3 * height // 4)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 3)
        np.copyto(frame_array, frame)
        queue.put(1)

if __name__ == '__main__':
    frame_shape = (720,1280,3)
    frame_dtype = np.uint8

    # Convert the size to a standard Python int
    shm_size = int(np.prod(frame_shape) * np.dtype(frame_dtype).itemsize)
    
    shm = shared_memory.SharedMemory(create=True, size=shm_size)
    queue = mp.Queue(maxsize=5)
    
    p_capture = mp.Process(target=camera_shm_capture, args=(queue, shm.name, frame_shape, frame_dtype))
    p_display = mp.Process(target=frame_shm_display, args=(queue, shm.name, frame_shape, frame_dtype))
    p_rectangle = mp.Process(target=draw__shm_rectangle, args=(queue, shm.name, frame_shape, frame_dtype))

    start_time = time.time()

    p_capture.start()
    p_rectangle.start()
    p_display.start()

    p_capture.join()
    p_rectangle.join()
    p_display.join()

    end_time = time.time()
    print(f"Shared memory method time taken: {end_time - start_time} seconds")

    shm.close()
    shm.unlink()




"""

def camera_capture(queue):
    cap = cv2.VideoCapture(0)  # Open the default camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)  # Set to highest resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        queue.put(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

def frame_display(queue):
    while True:
        frame = queue.get()
        if frame is None:
            break
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def draw_rectangle(queue):
    while True:
        frame = queue.get()
        if frame is None:
            break
        height, width, _ = frame.shape
        top_left = (width // 4, height // 4)
        bottom_right = (3 * width // 4, 3 * height // 4)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 3)
        queue.put(frame)

if __name__ == '__main__':
    queue = mp.Queue(maxsize=5)
    
    p_capture = mp.Process(target=camera_capture, args=(queue,))
    p_display = mp.Process(target=frame_display, args=(queue,))
    p_rectangle = mp.Process(target=draw_rectangle, args=(queue,))

    start_time = time.time()

    p_capture.start()
    p_rectangle.start()
    p_display.start()

    p_capture.join()
    p_rectangle.join()
    p_display.join()

    end_time = time.time()
    print(f"Queue method time taken: {end_time - start_time} seconds")

"""




"""

def process_webcam_and_crop_objects(crop_name, frame, model, results):
    names = model.names

    if not os.path.exists(crop_name):
        os.mkdir(crop_name)

    idx = 0

    box = results[0].boxes.xyxy.cpu().tolist()
    cls = results[0].boxes.cls.cpu().tolist()
    annotator = Annotator(frame, line_width=2, example=names)

    
    if box is not None:
        for box_item, cls_item in zip(box, cls):
            idx += 1

            x1, y1, x2, y2 = box_item
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, str(frame), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            annotator.box_label(box_item, color=colors(int(cls_item), True), label=names[int(cls_item)])

            crop_obj = frame[int(box_item[1]):int(box_item[3]), int(box_item[0]):int(box_item[2])]

            cv2.imwrite(os.path.join(crop_name, str(idx) + ".png"), crop_obj)

"""