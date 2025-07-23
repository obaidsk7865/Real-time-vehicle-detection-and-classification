# import cv2
# from ultralytics import YOLO

# # Load the trained YOLOv8 model
# model = YOLO("best.pt")  # Replace with your trained model path if different

# # Define class names (ensure they match your training labels)
# class_names = {0: "Bus", 1: "Car", 2: "motorcycle", 3: "Truck"}

# # Start video capture
# cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with video file path

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Run YOLOv8 inference on the frame
#     results = model(frame)
    
#     # Loop through detections
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
#             conf = box.conf[0].item()  # Confidence score
#             cls = int(box.cls[0].item())  # Class index
            
#             # Draw bounding box and label
#             label = f"{class_names.get(cls, 'Unknown')} {conf:.2f}"
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
#     # Display the frame
#     cv2.imshow("Vehicle Detection", frame)
    
#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()





import cv2
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO("best.pt")  # Replace with your trained model path

# Define class names (ensure they match your training labels)
class_names = {0: "Bus", 1: "Car", 2: "motorcycle", 3: "Truck"}

# Start video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with video file path

# Initialize overall vehicle counters
vehicle_counts = {"Bus": 0, "Car": 0, "motorcycle": 0, "Truck": 0}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Frame-level counts
    current_frame_counts = {"Bus": 0, "Car": 0, "motorcycle": 0, "Truck": 0}

    # Loop through detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            label_name = class_names.get(cls, "Unknown")
            label = f"{label_name} {conf:.2f}"

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Update both current and overall counts
            if label_name in current_frame_counts:
                current_frame_counts[label_name] += 1
                vehicle_counts[label_name] += 1

    # Display current frame count on the frame
    y_offset = 30
    for vehicle, count in current_frame_counts.items():
        count_text = f"{vehicle}: {count}"
        cv2.putText(frame, count_text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        y_offset += 25

    # Show the frame
    cv2.imshow("Vehicle Detection and Counting", frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Print final total counts after video ends
print("\n--- Overall Vehicle Counts ---")
for vehicle, total_count in vehicle_counts.items():
    print(f"{vehicle}: {total_count}")
