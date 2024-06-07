import cv2
import os
import torch
from ultralytics.utils.plotting import Annotator
import vision_support as vsp
import event_support as esp

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      
num_pics_per_orientation = 10
flash_duration = 200  # Duration of the flash in milliseconds
save_path = 'captured_images'

if not os.path.exists(save_path):
    os.makedirs(save_path)





def detect_objects(frame_processed, original_frame, results):
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
            vsp.draw_crosshair(original_frame, x1, y1, x2, y2, class_name)
    
    return original_frame
    
def detect_an_object(frame_processed, original_frame, scsize, results, event_queue):
    # Convert frame tensor to a numpy array.
    frame_processed = frame_processed.squeeze().permute(1, 2, 0).cpu().numpy().copy()

    fixed_rect_x1, fixed_rect_y1, fixed_rect_x2, fixed_rect_y2 = vsp.calculate_rect_coordinates(scsize)

    # Draw fixed rectangle
    cv2.rectangle(original_frame, (fixed_rect_x1, fixed_rect_y1), (fixed_rect_x2, fixed_rect_y2), (0, 255, 0), 2)
    cv2.putText(original_frame, 'Place Object exactly within box', (0, 10), cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 0, 0), 1)

    annotator = Annotator(original_frame, line_width=0.1)

    frameT1 = original_frame.copy()

    if results:

        for r in results:

            if r.masks:
                for i, mask in enumerate(r.masks.xy):
                    class_id = r.boxes[i].cls.item()  # Get class ID for each mask

                    if not esp.state["prompted_for_name"]:
                        class_name = r.names[class_id]  # Convert class ID to class name
                    else:
                        class_name = esp.state["obj_name"]

                    annotator.seg_bbox(mask=mask, mask_color=(0, 255, 255), det_label=str(class_name))
                    frameT1 = original_frame.copy()

                    #Initial esp.state. Just Aligned and Nothing Happened Yet. Awaiting Start Press
                    if vsp.is_inside_fixed_rect(r.boxes[i].xyxy[0].cpu().tolist(), (fixed_rect_x1, fixed_rect_y1, fixed_rect_x2, fixed_rect_y2)) and not esp.state["event_S"]:
                        vsp.draw_prompt(original_frame, 'Aligned. Press -s- to Start.', (5, 25))
                        esp.handle_event(event_queue, esp.state)

                    #Start Button Was Pressed. Looking for Name.
                    if vsp.is_inside_fixed_rect(r.boxes[i].xyxy[0].cpu().tolist(), (fixed_rect_x1, fixed_rect_y1, fixed_rect_x2, fixed_rect_y2)) and esp.state["event_S"]:
                        original_frame = frameT1.copy()
                        vsp.draw_prompt(original_frame, 'Start typing unique name. Press w to type. Press ENTER to finish', (5, 30))
                        esp.handle_event(event_queue, esp.state)

                    #W Pressed, Name Stored.
                    if vsp.is_inside_fixed_rect(r.boxes[i].xyxy[0].cpu().tolist(), (fixed_rect_x1, fixed_rect_y1, fixed_rect_x2, fixed_rect_y2)) and esp.state["event_W"]:
                        esp.handle_event(event_queue, esp.state)

                    #esp.state and Obj Name Updation
                    if vsp.is_inside_fixed_rect(r.boxes[i].xyxy[0].cpu().tolist(), (fixed_rect_x1, fixed_rect_y1, fixed_rect_x2, fixed_rect_y2)) and esp.state["got_name"]:
                        original_frame = frameT1.copy()
                        vsp.draw_prompt(original_frame, f'Name: {esp.state["obj_name"]} Stored', (5, 30))
                        esp.state["prompted_for_name"] = True
                        class_name = esp.state["obj_name"]


                    #All Pre-Setting Ready. Awaiting C Press to Continue
                    if vsp.is_inside_fixed_rect(r.boxes[i].xyxy[0].cpu().tolist(), (fixed_rect_x1, fixed_rect_y1, fixed_rect_x2, fixed_rect_y2)) and esp.state["prompted_for_name"]:
                        original_frame = frameT1.copy()
                        vsp.draw_prompt(original_frame, 'Press -c- to start Training.', (5, 30))
                        esp.handle_event(event_queue, esp.state)
           

                    #BEGIN CAPTURE. STARTING WITH FRONT
                    if vsp.is_inside_fixed_rect(r.boxes[i].xyxy[0].cpu().tolist(), (fixed_rect_x1, fixed_rect_y1, fixed_rect_x2, fixed_rect_y2)) and esp.state["ievent_C"]:
                        original_frame = frameT1.copy()
                        vsp.draw_prompt(original_frame, 'Capturing Data. Hold Steady! Press T to Start', (5, 25))
                        esp.handle_event(event_queue, esp.state)

                        
                    #FRONT
                    if vsp.is_inside_fixed_rect(r.boxes[i].xyxy[0].cpu().tolist(), (fixed_rect_x1, fixed_rect_y1, fixed_rect_x2, fixed_rect_y2)) and esp.state["tempState"]:
                        original_frame = frameT1.copy()
                        vsp.draw_prompt(original_frame, f'{esp.state["obj_name"]} - FRONT. Press N to Capture', (5, 33))
                        esp.handle_event(event_queue, esp.state)

                    if esp.state["inner"] and esp.state["tempState"]:
                        for idx in range(num_pics_per_orientation):
                            r.save_crop(save_path, f"{class_name}_FRONT_{idx}.jpg")
                        esp.state["inner"] = False
                        original_frame = frameT1.copy()
                        esp.state["tempState"] = False
                        vsp.draw_prompt(original_frame, 'FRONT DONE.', (5, 33))
                        esp.orientations["front"] = True
                        
                                                            
                    #LEFT
                    if vsp.is_inside_fixed_rect(r.boxes[i].xyxy[0].cpu().tolist(), (fixed_rect_x1, fixed_rect_y1, fixed_rect_x2, fixed_rect_y2)) and esp.orientations["front"]:
                        original_frame = frameT1.copy()
                        vsp.draw_prompt(original_frame, f'{esp.state["obj_name"]} - LEFT. Press N to Capture', (5, 33))
                        esp.handle_event(event_queue, esp.state)

                    if esp.state["inner"] and esp.orientations["front"]:
                        for idx in range(num_pics_per_orientation):
                            r.save_crop(save_path, f"{class_name}_LEFT_{idx}.jpg")
                        esp.state["inner"] = False
                        original_frame = frameT1.copy()
                        esp.state["tempState"] = False
                        vsp.draw_prompt(original_frame, 'LEFT DONE.', (5, 33))
                        esp.orientations["left"] = True
                        esp.orientations["front"] = False
                        

                    #BACK
                    if vsp.is_inside_fixed_rect(r.boxes[i].xyxy[0].cpu().tolist(), (fixed_rect_x1, fixed_rect_y1, fixed_rect_x2, fixed_rect_y2)) and esp.orientations["left"]:
                        original_frame = frameT1.copy()
                        vsp.draw_prompt(original_frame, f'{esp.state["obj_name"]} - BACK. Press N to Capture', (5, 33))
                        esp.handle_event(event_queue, esp.state)

                    if esp.state["inner"] and esp.orientations["left"]:
                        for idx in range(num_pics_per_orientation):
                            r.save_crop(save_path, f"{class_name}_BACK_{idx}.jpg")
                        esp.state["inner"] = False
                        original_frame = frameT1.copy()
                        esp.state["tempState"] = False
                        vsp.draw_prompt(original_frame, 'BACK DONE.', (5, 33))
                        esp.orientations["back"] = True
                        esp.orientations["left"] = False
                        


                    #RIGHT
                    if vsp.is_inside_fixed_rect(r.boxes[i].xyxy[0].cpu().tolist(), (fixed_rect_x1, fixed_rect_y1, fixed_rect_x2, fixed_rect_y2)) and esp.orientations["back"]:
                        original_frame = frameT1.copy()
                        vsp.draw_prompt(original_frame, f'{esp.state["obj_name"]} - RIGHT. Press N to Capture', (5, 33))
                        esp.handle_event(event_queue, esp.state)                        

                    if esp.state["inner"] and esp.orientations["back"]:
                        for idx in range(num_pics_per_orientation):
                            r.save_crop(save_path, f"{class_name}_RIGHT_{idx}.jpg")
                        esp.state["inner"] = False
                        original_frame = frameT1.copy()
                        esp.state["tempState"] = False
                        vsp.draw_prompt(original_frame, 'RIGHT DONE.', (5, 33))
                        esp.orientations["right"] = True
                        esp.orientations["back"] = False


                    #TOP
                    if vsp.is_inside_fixed_rect(r.boxes[i].xyxy[0].cpu().tolist(), (fixed_rect_x1, fixed_rect_y1, fixed_rect_x2, fixed_rect_y2)) and esp.orientations["right"]:
                        original_frame = frameT1.copy()
                        vsp.draw_prompt(original_frame, f'{esp.state["obj_name"]} - TOP. Press N to Capture', (5, 33))
                        esp.handle_event(event_queue, esp.state) 

                    if esp.state["inner"] and esp.orientations["right"]:
                        for idx in range(num_pics_per_orientation):
                            r.save_crop(save_path, f"{class_name}_TOP_{idx}.jpg")
                        esp.state["inner"] = False
                        original_frame = frameT1.copy()
                        esp.state["tempState"] = False
                        vsp.draw_prompt(original_frame, 'TOP DONE.', (5, 33))
                        esp.orientations["top"] = True
                        esp.orientations["right"] = False
                            


                    #BOTTOM
                    if vsp.is_inside_fixed_rect(r.boxes[i].xyxy[0].cpu().tolist(), (fixed_rect_x1, fixed_rect_y1, fixed_rect_x2, fixed_rect_y2)) and esp.orientations["top"]:
                        original_frame = frameT1.copy()
                        vsp.draw_prompt(original_frame, f'{esp.state["obj_name"]} - BOTTOM. Press N to Capture', (5, 33))
                        esp.handle_event(event_queue, esp.state) 

                    if esp.state["inner"] and esp.orientations["top"]:
                        for idx in range(num_pics_per_orientation):
                                r.save_crop(save_path, f"{class_name}_BOTTOM_{idx}.jpg")
                        esp.state["inner"] = False
                        original_frame = frameT1.copy()
                        esp.state["tempState"] = False
                        vsp.draw_prompt(original_frame, 'BOTTOM DONE. Press Q to QUIT', (5, 33))
                        esp.orientations["bottom"] = True
                        esp.orientations["top"] = False

                    if esp.state["termi"] or esp.orientations["bottom"]:
                        esp.orientations["bottom"] = False
                        break

                    if not vsp.is_inside_fixed_rect(r.boxes[i].xyxy[0].cpu().tolist(), (fixed_rect_x1, fixed_rect_y1, fixed_rect_x2, fixed_rect_y2)):
                        original_frame = frameT1.copy()
                        vsp.draw_prompt(original_frame, "Object not Inside. Move it.", (5, 200))
                        
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
                    
        
                    if vsp.is_inside_fixed_rect(r.boxes[i].xyxy[0].cpu().tolist(), (fixed_rect_x1, fixed_rect_y1, fixed_rect_x2, fixed_rect_y2)):
                        put_text_on_rectangle(frame_np, 'Aligned. Press -s- to Start.', (5, 20), scsize, text_color=(255, 255, 255), rect_color=(0, 0, 255))
                        
                        if not key_queue.empty():
                            key = key_queue.get()
                            if key == ord('s'):
                                if not prompted_for_name:
                                    obj_name = input("Enter Obj name: ")
                                    prompted_for_name = True

                                elif prompted_for_name and obj_name:
                                #if cv2.waitKey(1) & 0xFF == ord('s'):  
                                    for orientation in esp.orientations:
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









if state["ievent_C"] and vsp.is_inside_fixed_rect(r.boxes[i].xyxy[0].cpu().tolist(), (fixed_rect_x1, fixed_rect_y1, fixed_rect_x2, fixed_rect_y2)):
                        original_frame = frameT1.copy()
                        vsp.draw_prompt(original_frame, f'Place {state["obj_name"]} -{orientations[:1]}- facing inside.', (5, 30))
               
                        for idx in range(num_pics_per_orientation):
                            r.save_crop(save_path, f"{class_name}_{orientations[:1]}_{idx}.jpg")
                        state["frontA"] = True
                        
                    if state["frontA"] and vsp.is_inside_fixed_rect(r.boxes[i].xyxy[0].cpu().tolist(), (fixed_rect_x1, fixed_rect_y1, fixed_rect_x2, fixed_rect_y2)):
                        original_frame = frameT1.copy()
                        vsp.draw_prompt(original_frame, f'Place {state["obj_name"]} -{orientations[:2]}- facing inside.', (5, 30))
                    
                        for idx in range(num_pics_per_orientation):
                            r.save_crop(save_path, f"{class_name}_{orientations[:2]}_{idx}.jpg")
                        state["frontB"] = True    

                    if state["frontB"] and vsp.is_inside_fixed_rect(r.boxes[i].xyxy[0].cpu().tolist(), (fixed_rect_x1, fixed_rect_y1, fixed_rect_x2, fixed_rect_y2)):
                        original_frame = frameT1.copy()
                        vsp.draw_prompt(original_frame, f'Place {state["obj_name"]} -{orientations[:3]}- facing inside.', (5, 30))
                  
                        for idx in range(num_pics_per_orientation):
                            r.save_crop(save_path, f"{class_name}_{orientations[:3]}_{idx}.jpg")
                        state["frontC"] = True 

                    if state["frontC"] and vsp.is_inside_fixed_rect(r.boxes[i].xyxy[0].cpu().tolist(), (fixed_rect_x1, fixed_rect_y1, fixed_rect_x2, fixed_rect_y2)):
                        original_frame = frameT1.copy()
                        vsp.draw_prompt(original_frame, f'Place {state["obj_name"]} -{orientations[:4]}- facing inside.', (5, 30))
                    
                        for idx in range(num_pics_per_orientation):
                            r.save_crop(save_path, f"{class_name}_{orientations[:4]}_{idx}.jpg")
                        state["frontD"] = True 
                        
                    if state["frontD"] and vsp.is_inside_fixed_rect(r.boxes[i].xyxy[0].cpu().tolist(), (fixed_rect_x1, fixed_rect_y1, fixed_rect_x2, fixed_rect_y2)):
                        original_frame = frameT1.copy()
                        vsp.draw_prompt(original_frame, f'Place {state["obj_name"]} -{orientations[:5]}- facing inside.', (5, 30))
            
                        for idx in range(num_pics_per_orientation):
                            r.save_crop(save_path, f"{class_name}_{orientations[:5]}_{idx}.jpg")
                        state["Top"] = True 

                    if state["Top"] and vsp.is_inside_fixed_rect(r.boxes[i].xyxy[0].cpu().tolist(), (fixed_rect_x1, fixed_rect_y1, fixed_rect_x2, fixed_rect_y2)):
                        original_frame = frameT1.copy()
                        vsp.draw_prompt(original_frame, f'Place {state["obj_name"]} -{orientations[:6]}- facing inside.', (5, 30))
                
                        for idx in range(num_pics_per_orientation):
                            r.save_crop(save_path, f"{class_name}_{orientations[:6]}_{idx}.jpg")
                        state["Bottom"] = True 

"""