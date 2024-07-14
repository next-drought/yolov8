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
DAYS_TO_KEEP = 7

# New constants for recording optimization
RECORD_DURATION_AFTER_DETECTION = 10  # seconds to continue recording after new object disappears
SIGNIFICANT_CHANGE_THRESHOLD = 0.1  # fraction of pixels that need to change to be considered significant

def delete_old_files(days=DAYS_TO_KEEP):
    current_time = datetime.now()
    for file in glob.glob("motion_*.mp4"):
        try:
            file_datetime = datetime.strptime(file, "motion_%Y%m%d_%H%M%S.mp4")
            if (current_time - file_datetime).days >= days:
                os.remove(file)
                print(f"Deleted old file: {file}")
        except ValueError:
            print(f"Couldn't parse datetime from filename: {file}")

def get_object_classes(results):
    return set(int(box.cls[0]) for r in results for box in r.boxes)

def is_significant_change(prev_frame, current_frame, threshold):
    diff = cv2.absdiff(prev_frame, current_frame)
    changed_pixels = np.sum(diff > 30)  # Consider pixels with difference > 30 as changed
    total_pixels = prev_frame.shape[0] * prev_frame.shape[1]
    return changed_pixels / total_pixels > threshold

# Initialize YOLO model
model = YOLO('yolov8n.pt')

# Open video stream
cap = cv2.VideoCapture(0)  # Change to your video source if needed

# Get the video properties
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
original_fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define new, reduced resolution and frame rate
new_width = 640
new_height = int(new_width * original_height / original_width)
new_fps = 10

# Initialize variables
out = None
current_day = datetime.now().date()
prev_frame = None
recording_start_time = None
last_new_object_time = None
prev_classes = set()

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

        # Read and resize frame
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (new_width, new_height))

        # Perform detection on resized frame
        results = model(frame_resized)

        # Get current object classes
        current_classes = get_object_classes(results)

        # Check for new objects
        new_object_detected = bool(current_classes - prev_classes)

        # Determine if we should be recording
        current_time = datetime.now()
        should_record = new_object_detected or (
            last_new_object_time and 
            (current_time - last_new_object_time).total_seconds() < RECORD_DURATION_AFTER_DETECTION
        )

        if should_record:
            if new_object_detected:
                print("New object detected!")
                last_new_object_time = current_time

            if out is None:
                # Start a new recording
                timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                out = cv2.VideoWriter(f'motion_{timestamp}.mp4', 
                                      cv2.VideoWriter_fourcc(*'avc1'), new_fps, (new_width, new_height))
                recording_start_time = current_time

            # Write frame to video
            out.write(frame_resized)

        elif out is not None:
            # Check if we should stop recording due to lack of significant changes
            if prev_frame is not None and not is_significant_change(prev_frame, frame_resized, SIGNIFICANT_CHANGE_THRESHOLD):
                out.release()
                out = None
                print(f"Stopped recording due to lack of significant changes. Duration: {(current_time - recording_start_time).total_seconds():.2f} seconds")

        # Process and visualize results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Class name and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = f'{model.names[cls]} {conf:.2f}'
                cv2.putText(frame_resized, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Add recording indicator
        if out is not None:
            cv2.putText(frame_resized, "Recording", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show frame
        cv2.imshow('YOLOv8 Detection with Optimized Recording', frame_resized)

        # Update previous classes and frame
        prev_classes = current_classes
        prev_frame = frame_resized.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"An error occurred: {e}")

# Clean up
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()