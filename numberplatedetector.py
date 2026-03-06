import cv2
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO

model = YOLO('license_plate_detector.pt') # loading our download pt file-- model.

#now the working for the video 

root = tk.Tk()
root.title('Number plate detector')
root.minsize(800,700)
root.resizable(False, False)

video_player = tk.Label(root)
video_player.pack()

cap = cv2.VideoCapture('datasets/Numberplate.mp4')

def update_frame():
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame,(800,700))
        results = model(frame)
        
        # Convert colors for Tkinter before we start drawing on it
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for r in results:
            boxes = r.boxes

            for box in boxes:
                # Extract the 4 coordinates (x1, y1, x2, y2) and convert them to standard integers
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw the rectangle on your main frame using OpenCV
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # --- STEP 1: THE CROP ---
                cropped_plate = frame[y1:y2, x1:x2]
                
                # --- STEP 2: THE ZOOM ---
                plate_w = x2 - x1
                plate_h = y2 - y1

                zoom_w = plate_w * 3
                zoom_h = plate_h * 3

                # Safety check: ensure the crop is valid before resizing
                if plate_w > 0 and plate_h > 0:
                    zoomed_Plate = cv2.resize(cropped_plate, (zoom_w, zoom_h))
                    
                    # --- STEP 3: THE FLOATING OVERLAY & LINE ---
                    # Find the exact horizontal center of the license plate box
                    center_x = x1 + (plate_w // 2)
                    
                    # Calculate X so the zoomed image is centered directly above the plate
                    start_x = center_x - (zoom_w // 2)
                    end_x = start_x + zoom_w
                    
                    # Calculate Y so the zoomed image hovers  pixels above the plate
                    end_y = y1 - 350
                    start_y = end_y - zoom_h
                    
                    # Safety check: Only draw if the zoomed image fits inside the window completely
                    if start_y > 0 and start_x > 0 and end_x < 800:
                        
                        # 1. Draw the green connecting line (from the box center up to the zoomed image)
                        cv2.line(frame, (center_x, y1), (center_x, end_y), (0, 255, 0), 2)
                        
                        # 2. Paste the zoomed pixels
                        frame[start_y:end_y, start_x:end_x] = zoomed_Plate
                        
                        # 3. Draw a green border around the zoomed image to make it pop
                        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        # Display the final frame on Tkinter
        img = Image.fromarray(frame)
        img_tk = ImageTk.PhotoImage(image=img)
        video_player.config(image=img_tk)
        video_player.img_tk = img_tk #type:ignore
    
    video_player.after(10, update_frame)

update_frame()
root.mainloop()