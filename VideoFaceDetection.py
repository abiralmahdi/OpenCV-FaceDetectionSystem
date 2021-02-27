# Import the required libraries
import  cv2
from random import randrange

# Train the data
faceTrained = cv2.CascadeClassifier('trains.xml')

# Initiate the webcam
webcam = cv2.VideoCapture(0)

# Iterate over the frames forever
while True:
    successfulFramRead, frame = webcam.read()

    # Convert the frame to greyscale
    grayScaledFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the face in the frames
    faceCoordinates = faceTrained.detectMultiScale(grayScaledFrame)

    # Draw rectangels around the faces for multiple faces
    for (x, y, w, h) in faceCoordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Show the frame
    cv2.imshow('Abirs Face Detector Application', frame)
    key = cv2.waitKey(1)

    # Break out of this program using a key
    if key == 81 or key == 113:
        break
    
