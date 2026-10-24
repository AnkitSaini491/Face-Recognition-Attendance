

import cv2
import face_recognition
import os
import csv
from datetime import datetime

FACES_FOLDER = "known_faces"
ATTENDANCE_FILE = "attendance.csv"

os.makedirs(FACES_FOLDER, exist_ok=True)


def load_faces():
    encodings = []
    names = []

    for file in os.listdir(FACES_FOLDER):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(FACES_FOLDER, file)
            image = face_recognition.load_image_file(path)
            faces = face_recognition.face_encodings(image)

            if faces:
                encodings.append(faces[0])
                names.append(os.path.splitext(file)[0])

    return encodings, names


def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    already_marked = set()

    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)

            for row in reader:
                if len(row) >= 2 and row[0] == name and row[1] == today:
                    already_marked.add(name)

    if name not in already_marked:
        with open(ATTENDANCE_FILE, "a", newline="") as f:
            writer = csv.writer(f)

            if os.path.getsize(ATTENDANCE_FILE) == 0:
                writer.writerow(["Name", "Date", "Time"])

            writer.writerow([name, today, current_time])

        print(f"Attendance marked for {name}")


known_encodings, known_names = load_faces()

if not known_encodings:
    print("No registered faces found!")
    print("First run register_face.py to register a face.")
    raise SystemExit

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Webcam not available.")
    raise SystemExit

print("Attendance system started. Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(
        frame, (0, 0), fx=0.25, fy=0.25
    )
    rgb_frame = cv2.cvtColor(
        small_frame, cv2.COLOR_BGR2RGB
    )

    locations = face_recognition.face_locations(rgb_frame)
    encodings = face_recognition.face_encodings(
        rgb_frame, locations
    )

    for encoding, location in zip(encodings, locations):
        matches = face_recognition.compare_faces(
            known_encodings, encoding, tolerance=0.5
        )

        name = "Unknown"

        if known_encodings:
            distances = face_recognition.face_distance(
                known_encodings, encoding
            )
            best_match = distances.argmin()

            if matches[best_match]:
                name = known_names[best_match]
                mark_attendance(name)

        top, right, bottom, left = location

        # Scale coordinates back to original frame
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        cv2.rectangle(
            frame, (left, top), (right, bottom), color, 2
        )
        cv2.putText(
            frame, name, (left, max(top - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
