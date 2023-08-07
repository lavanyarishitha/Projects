#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import numpy as np
from PIL import Image
import os

# Preprocess an image by resizing, converting to grayscale, and applying histogram equalization
def preprocess_image(image_path, target_size):
    img = Image.open(image_path).convert('L')
    img = img.resize(target_size, Image.LANCZOS)

    # Apply histogram equalization
    img = np.array(img)
    img = cv2.equalizeHist(img)

    preprocessed_img = np.array(img, 'uint8')
    return preprocessed_img

# Train the face classifier using Fisherfaces algorithm
def train_classifier(data_dir, target_size=(100, 100)):
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []

    for image in path:
        preprocessed_img = preprocess_image(image, target_size)
        id = int(os.path.split(image)[1].split(".")[1])
        faces.append(preprocessed_img)
        ids.append(id)

    ids = np.array(ids)
    clf = cv2.face.FisherFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.yml")

# Detect faces in an image and draw rectangles around them
def detect(img, faceCascade, img_id, classifier, confidence_threshold=4000):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0)}
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = faceCascade.detectMultiScale(gray_img, 1.1, 10)

    for (x, y, w, h) in features:
        roi_img = img[y:y + h, x:x + w]
        gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.resize(gray_roi, (100, 100))  # Resize ROI to match training size

        label, confidence = classifier.predict(gray_roi)
        if confidence < confidence_threshold:
            if label == 1:
                cv2.rectangle(img, (x, y), (x + w, y + h), color['green'], 2)
                cv2.putText(img, f"Matched ({confidence:.2f},{label})", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            color['green'], 1, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), color['red'], 2)
            cv2.putText(img, f"Not Matched ({confidence:.2f},{label})", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        color['red'], 1, cv2.LINE_AA)

    return img

# Load the pre-trained face cascade classifier
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
img_id = 0
video_capture = cv2.VideoCapture(0)

# Set the desired output frame size
output_width = 1200
output_height = 800

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, output_width)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, output_height)

# Load or train the classifier
if os.path.isfile("classifier.yml"):
    classifier = cv2.face.FisherFaceRecognizer_create()
    classifier.read("classifier.yml")
else:
    train_classifier("user_data")
    classifier = cv2.face.FisherFaceRecognizer_create()
    classifier.read("classifier.yml")

while True:
    _, img = video_capture.read()
    img = cv2.flip(img, 1)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detect(img, faceCascade, img_id, classifier)
    cv2.imshow("face_detection", img)
    img_id += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()


# In[ ]:




