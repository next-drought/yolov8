import warnings
import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime, timedelta
import glob

# Suppress urllib3 warning
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

# Redirect stderr
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

# Restore stderr
sys.stderr = stderr

# Number of days to keep files
DAYS_TO_KEEP = 10

def delete_old_files(days=DAYS_TO_KEEP):
    current_time = datetime.now()
    for file in glob.glob("motion_*.mp4"):
        try:
            file_datetime = datetime.strptime(file, "motion_%Y%m%d.mp4")
            if (current_time - file_datetime).days >= days:
                os.remove(file)
                print(f"Deleted old file: {file}")
        except ValueError:
            print(f"Couldn't parse datetime from filename: {file}")

def get_object_classes(results):
    return {int(box.cls[0]): model.names[int(box.cls[0])] for r in results for box in r.boxes}

# Function to apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

# Function to adjust exposure automatically
def adjust_exposure(frame, target_brightness=125):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    current_brightness = np.mean(gray)
    
    exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    if current_brightness < target_brightness:
        exposure = min(exposure + 0.1, 0)  # Increase exposure, max 0
    elif current_brightness > target_brightness:
        exposure = max(exposure - 0.1, -10)  # Decrease exposure, min -10
    
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    return frame

def reduce_brightness(image, percentage=10):
    """Reduce the brightness of the image to the specified percentage."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    v = cv2.multiply(v, percentage / 100.0)
    v = np.clip(v, 0, 255).astype(np.uint8)
    
    final_hsv = cv2.merge((h, s, v))
    image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return image

# Initialize YOLO model
model = YOLO('yolov8n.pt')

# Open video stream
cap = cv2.VideoCapture(0)  # Change to your video source if needed

# Adjust camera settings to handle overexposure
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Disable auto-exposure
cap.set(cv2.CAP_PROP_EXPOSURE, -13)  # Set exposure to a very low value
cap.set(cv2.CAP_PROP_GAIN, 0)       # Minimum gain
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)  # Minimum brightness
cap.set(cv2.CAP_PROP_CONTRAST, 32)   # Lower contrast

# Get the video properties
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
original_fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define new, reduced resolution and frame rate
new_width = 640
new_height = int(new_width * original_height / original_width)
new_fps = 10

# Initialize video writer
out = None
current_day = datetime.now().date()

# Read initial frame
ret, frame = cap.read()
frame = adjust_exposure(frame)
frame_clahe = apply_clahe(frame)
frame_dark = reduce_brightness(frame_clahe, percentage=10)  # Reduce brightness to 10%
frame_resized = cv2.resize(frame_dark, (new_width, new_height))

prev_results = model(frame_resized)
prev_classes = get_object_classes(prev_results)

new_object_detected = False
new_object_frames = 0
NEW_OBJECT_THRESHOLD = 5  # Number of consecutive frames to confirm a new object

# Initialize last_cleanup_time
last_cleanup_time = datetime.now()

while True:
    try:
        # Check if it's time to clean up old files (once per day)
        if (datetime.now() - last_cleanup_time).days >= 1:
            delete_old_files()
            last_cleanup_time = datetime.now()

        # Check if it's a new day
        if datetime.now().date() != current_day:
            if out is not None:
                out.release()
            current_day = datetime.now().date()
            out = None

        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply exposure adjustment and CLAHE
        frame = adjust_exposure(frame)
        frame_clahe = apply_clahe(frame)
        frame_resized = cv2.resize(frame_clahe, (new_width, new_height))

        # Perform detection on resized frame
        results = model(frame_resized)

        # Get current object classes
        current_classes = get_object_classes(results)

        # Check for new objects
        new_objects = set(current_classes.keys()) - set(prev_classes.keys())
        if new_objects:
            if not new_object_detected:
                new_object_frames = NEW_OBJECT_THRESHOLD
                new_detected_objects = [current_classes[cls] for cls in new_objects]
            else:
                new_object_frames = max(new_object_frames - 1, 0)
        else:
            new_object_frames = max(new_object_frames - 1, 0)

        new_object_detected = new_object_frames > 0

        if new_object_detected and new_object_frames == NEW_OBJECT_THRESHOLD:
            print(f"New object(s) detected: {', '.join(new_detected_objects)}")
            if out is None:
                # Create a new video writer with H.264 codec
                timestamp = datetime.now().strftime("%Y%m%d")
                out = cv2.VideoWriter(f'motion_{timestamp}.mp4', 
                                      cv2.VideoWriter_fourcc(*'avc1'), new_fps, (new_width, new_height))

        # Process and visualize results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add new object indicator
        if new_object_detected:
            cv2.putText(frame_resized, f"New Object(s): {', '.join(new_detected_objects)}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show frame
        cv2.imshow('YOLOv8 Detection with New Object', frame_resized)

        # Save frame if video writer is initialized
        if out is not None:
            out.write(frame_resized)

        # Update previous classes
        prev_classes = current_classes

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"An error occurred: {e}")

# Clean up
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()