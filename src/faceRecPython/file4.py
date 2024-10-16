import cv2
import face_recognition
import pickle
import sys

# Load encodings from file
with open("d:/My Projects/Python/faceRecognition/encodings.pkl", "rb") as f:
    encodings = pickle.load(f)
    
num_encodings = len(encodings)
# print(f"Number of encodings: {num_encodings}")

frame = cv2.imread(sys.argv[1])

rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
face_locations = face_recognition.face_locations(rgb_frame)
face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

for face_encoding in face_encodings:
        for name, stored_encoding in encodings.items():
            result = face_recognition.compare_faces([stored_encoding], face_encoding)
            if result[0] : 
                print(name)

