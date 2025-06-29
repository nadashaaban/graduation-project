
import numpy as np
import cv2
import face_recognition
import os

# Directory for known faces
KNOWN_FACES_DIR = 'D:/nada/grad_project/Face Recognition - Final/known_faces'
TOLERANCE = 0.5  # Lower tolerance to reduce false positives
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'hog'  # 'hog' is faster, 'cnn' is more accurate

# Function to generate colors for names
def name_to_color(name):
    return [(ord(c.lower()) - 97) * 8 for c in name[:3]]

print('Loading known faces...')
known_face_encodings = []
known_face_names = []

# Load known faces
for name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    if not os.path.isdir(person_dir):
        continue

    for filename in os.listdir(person_dir):
        img_path = os.path.join(person_dir, filename)
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        try:
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

print(f'Loaded {len(known_face_encodings)} known faces.')

# Open webcam
cameraWindow = cv2.VideoCapture(0)
if not cameraWindow.isOpened():
    print("Error: Could not open camera.")
    exit()

frame_counter = 0  # Skip some frames for better speed

while True:
    flag, frame = cameraWindow.read()
    if not flag:
        print("Failed to grab frame")
        break

    # Only process every 3rd frame to improve speed
    if frame_counter % 3 == 0:
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces and get encodings
        face_locations = face_recognition.face_locations(rgb_frame, model=MODEL)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        recognized_names = []
        recognized_locations = []

        for face_encoding, face_location in zip(face_encodings, face_locations):
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            min_distance_index = np.argmin(distances)  # Get best match
            min_distance = distances[min_distance_index]

            if min_distance < TOLERANCE:
                match = known_face_names[min_distance_index]
            else:
                match = "Unknown"  # Label explicitly as "Unknown"

            recognized_names.append(match)
            recognized_locations.append(face_location)

    # Draw rectangles and names
    for match, face_location in zip(recognized_names, recognized_locations):
        color = (0, 0, 255) if match == "Unknown" else name_to_color(match)
        top, right, bottom, left = [val * 2 for val in face_location]

        # Draw rectangle around face
        cv2.rectangle(frame, (left, top), (right, bottom), color, FRAME_THICKNESS)

        # Draw label background
        cv2.rectangle(frame, (left, bottom - 20), (right, bottom), color, cv2.FILLED)

        # Write name or "Unknown"
        cv2.putText(frame, match, (left + 10, bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), FONT_THICKNESS)

    # Show frame
    cv2.imshow("Face Recognition", frame)

    frame_counter += 1  # Increase frame counter

    # Exit on 'q' or 'Esc' key
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        print("Exiting...")
        break

# Release camera and close windows
cameraWindow.release()
cv2.destroyAllWindows()
