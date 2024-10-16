import cv2
import face_recognition

img = cv2.imread("images/Elon.jpg")
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
face_rec = face_recognition.face_encodings(rgb_img)[0]

capture = cv2.VideoCapture(0)

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

 
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    
    for face_encoding in face_encodings:
        result = face_recognition.compare_faces([face_rec], face_encoding)
        print(result)

   
    cv2.imshow("Img", frame)

  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
