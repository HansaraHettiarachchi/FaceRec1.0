import cv2
import face_recognition
import os
import pickle

# Directory containing images
image_dir = "images"

# Dictionary to hold encodings
encodings = {}

# Loop through each image file in the directory
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_enc = face_recognition.face_encodings(rgb_img)
        if face_enc:
            encodings[filename] = face_enc[0]
            print("Hair")

# Save encodings to a file
with open("encodings.pkl", "wb") as f:
    pickle.dump(encodings, f)
