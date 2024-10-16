import cv2
import face_recognition
import json

# import face_recognition_models
print("1ddsfsf")
img = cv2.imread("d:/My Projects/Python/faceRecognition/images/Ronaldo.jpeg")
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_img)[0]

print("1")
img1 = cv2.imread("d:/My Projects/Python/faceRecognition/search/ronaldo.jpg")
rgb_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img_encoding1 = face_recognition.face_encodings(rgb_img1)[0]

print("2")
result = face_recognition.compare_faces([img_encoding], img_encoding1)

print(result)

if result[0]:
    cv2.imshow("Img",img1)
    cv2.waitKey(0)