
import cv2
import numpy as np
from ultralytics import YOLO
import cvzone

# Initialize YOLO model
model = YOLO("yolo.pt")
names = model.model.names

# Define input file (change this to your desired input)
input_file = 'classes.jpg'  # Change this to 'classroom.mp4' for video input

# Check if the input is an image or a video
is_image = input_file.lower().endswith(('.png', '.jpg', '.jpeg'))

if is_image:
    frame = cv2.imread(input_file)
    if frame is None:
        print("Error: Image file not loaded properly!")
        exit()
    frames = [frame]  # Store image in a list to process like a video
else:
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print("Error: Video file not loaded properly!")
        exit()
    frames = None  # Will read frames from video

count = 0
active_people = set()  # Store currently detected person IDs

while True:
    if is_image:
        frame = frames[0].copy()  # Process the image once
    else:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 3 != 0:
            continue

    frame = cv2.resize(frame, (1020, 600))
# risizing so that all size will be on same size
    # Run YOLO tracking
    results = model.track(frame, persist=True, classes=0)
    #track students across frames, ensuring an accurate count even if they move.

    current_frame_ids = set()  # Stores the IDs of people currently visible in the frame.

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, class_id, track_id in zip(boxes, class_ids, track_ids):
            x1, y1, x2, y2 = box
            c = names[class_id]

            # Track only people currently in the frame
            current_frame_ids.add(track_id)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'ID: {track_id}', (x1, y2), 1, 1)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

    # This ensures only currently present people are counted, avoiding duplicates.
    active_people = current_frame_ids

    # Display total number of people present
    total_people = len(active_people)
    cvzone.putTextRect(frame, f'Total People: {total_people}', (50, 50), 2, 2)

    # Show the frame
    cv2.imshow("RGB", frame)

    # Wait for 'q' key press to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release resources
if not is_image:
    cap.release()
cv2.destroyAllWindows()
