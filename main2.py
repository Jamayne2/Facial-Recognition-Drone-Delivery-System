import os
import cv2
import numpy as np
from djitellopy import Tello
import face_recognition
import KeyPressModule as kp
import time

# Set up the Tello
me = Tello()
kp.init()

me.connect()
me.streamon()
print(f"Battery: {me.get_battery()}%")

# Path to the folder containing images for recognition
path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print("Images in the directory:", myList)

# Load the images and their corresponding class names
for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])
print("Class Names:", classNames)

# Function to encode the images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Encode the known images
encodeListKnown = findEncodings(images)
print('Encoding Complete')


def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50
    if kp.getKey("LEFT"):
        lr = -speed
    elif kp.getKey("RIGHT"):
        lr = speed
    if kp.getKey("UP"):
        fb = speed
    elif kp.getKey("DOWN"):
        fb = -speed
    if kp.getKey("w"):
        ud = speed
    elif kp.getKey("s"):
        ud = -speed
    if kp.getKey("a"):
        yv = -speed
    elif kp.getKey("d"):
        yv = speed
    if kp.getKey("q"):
        me.land()
        time.sleep(3)
    if kp.getKey("e"):
        me.takeoff()
    return [lr, fb, ud, yv]


# Main loop
while True:
    vals = getKeyboardInput()
    me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    img = me.get_frame_read().frame
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Detect faces in the current frame
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurrFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # Compare faces with known encodings
    for encodeFace, faceLoc in zip(encodesCurrFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(f"Face recognized: {name}")
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            vals = [0, 0, 0, 0]
            me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "Face Recognized : " + name, (x1 + 6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Drone Camera', img)
            cv2.waitKey(2)
            time.sleep(1)

            vals = [0, 0, 0, 0]
            me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
            me.land()

    # Display the camera feed
    cv2.imshow('Drone Camera', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        me.land()
        break

# Clean up
cv2.destroyAllWindows()

