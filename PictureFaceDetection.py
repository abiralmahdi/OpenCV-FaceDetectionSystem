import cv2
from random import randrange
# Import Data
face_trained = cv2.CascadeClassifier('trains.xml')

# Read the image
img = cv2.imread('srk2.jpg')

# Convert the image to grayScale
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect Faces
faceCoordinates = face_trained.detectMultiScale(grayscale_img)

# Draw rectangles around the faces
for (x, y, w, h) in faceCoordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256),  randrange(256)), 2)
    # cv2.rectangle(img, (x-5, y-5), (x+w+5, y+h+5), (randrange(256), randrange(256),  randrange(256)), 1)

#Show the resultant Image
cv2.imshow('Abirs Face Detector App', img)
cv2.waitKey()
