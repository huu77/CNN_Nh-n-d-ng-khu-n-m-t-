import cv2
import numpy as np
from tensorflow.keras.models import load_model

label =[ 'huu' ,'mask','none','none']
# Load the face recognition model
model = load_model('trained_model.keras')

# Load the face detection classifier
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open the video capture
cam = cv2.VideoCapture(0) # this is using for camera 
# cam = cv2.VideoCapture('video.mp4')

while True:
    # Read a frame from the video
    ok, frame = cam.read()

    # Perform face detection
    faces = face_detector.detectMultiScale(frame, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = cv2.resize(frame[y:y+h, x:x+w], (32, 32))
        result = np.argmax(model.predict(roi.reshape((-1, 32, 32, 3))))

        cv2.rectangle(frame, (x, y), (x+w, y+h), (125, 255, 144), 1)
        cv2.putText(frame,label[result],(x,y),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,40,255),2)

    # Display the frame
    cv2.imshow("frame", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cam.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()
