import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime

def get_motion_area(frame1, frame2):
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    area = 0
    for contour in contours:
        area += cv2.contourArea(contour)
    return area

# Initialize YOLO model
model = YOLO('yolov8n.pt')

# Open video stream
cap = cv2.VideoCapture(0)  # Change to your video source if needed

# Get the video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize video writer (we'll create it when motion is detected)
out = None

# Read two initial frames for motion detection
ret, frame1 = cap.read()
ret, frame2 = cap.read()

motion_detected = False
motion_frames = 0
MOTION_THRESHOLD = 10000  # Adjust this value based on your needs

while True:
    try:
        # Perform detection
        results = model(frame2)

        # Check for motion
        motion_area = get_motion_area(frame1, frame2)
        
        if motion_area > MOTION_THRESHOLD:
            if not motion_detected:
                print("Motion detected!")
                # Create a new video writer
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out = cv2.VideoWriter(f'motion_{timestamp}.mp4', 
                                      cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
            motion_detected = True
            motion_frames = 30  # Continue recording for 30 more frames
        elif motion_frames > 0:
            motion_frames -= 1
        else:
            motion_detected = False
            if out is not None:
                out.release()
                out = None

        # Process and visualize results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Class name and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = f'{model.names[cls]} {conf:.2f}'
                cv2.putText(frame2, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Add motion indicator
        if motion_detected:
            cv2.putText(frame2, "Motion Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show frame
        cv2.imshow('YOLOv8 Detection with Motion', frame2)

        # Save frame if motion is detected
        if motion_detected and out is not None:
            out.write(frame2)

        # Update frames for next iteration
        frame1 = frame2
        ret, frame2 = cap.read()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"An error occurred: {e}")

# Clean up
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()