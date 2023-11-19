import cv2
import os

# Initialize webcam (using the default camera with dimensions 640x480)
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Set video width
cam.set(4, 480)  # Set video height

# Load the Haar Cascade classifier for face detection
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_user_name = input('\nEnter user name and press <Enter>: ')

print("\n[INFO] Start capturing faces. Look at the camera and wait...")

# Initialize a counter to keep track of the number of face images captured
count = 0
numberFace = 100
user_directory = os.path.join('dataset', face_user_name)
os.makedirs(user_directory, exist_ok=True)

while True:
    
    ret, img = cam.read()

    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    img = cv2.flip(img, 1)  # Flip the video horizontally
    
    faces = face_detector.detectMultiScale(img, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # Save the captured face image in the user's directory with a unique filename
        face_image_path = os.path.join(user_directory, f"{count}.jpg")
        cv2.imwrite(face_image_path, img[y:y + h, x:x + w])

        cv2.imshow('Image', img)

    k = cv2.waitKey(1) & 0xff  # Reduce delay to 1 millisecond
    if k == 27 or count >= numberFace:
        break

# Clean up resources
print("\n[INFO] Program completed. Cleaning up resources.")
cam.release()
cv2.destroyAllWindows()
