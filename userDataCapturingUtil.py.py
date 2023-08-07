#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This utility used to capture user data from a videoCapture which will be used to train the face detector algorithm
import cv2

# storing dataset with unique filenames
def generate_dataset(img, id, img_id):
    cv2.imwrite("user."+str(id)+"."+str(img_id)+".jpg",img)
    
# Draw square boundary around detected faces and returns the coordinates     
def draw_boundary(img , classifier , scale_factor , minNeighbors , color , text):
    gray_img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scale_factor, minNeighbors)
    coords = []
    for (x,y,w,h) in features:
        cv2.rectangle(img , (x,y),(x+w,y+h),color,2)
        cv2.putText(img , text, (x,y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x,y,w,h]
    return coords, img

# detecting faces and saving the detected faces
def detect(img , faceCascade, img_id):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0)}
    coords , img = draw_boundary(img , faceCascade , 1.1 , 10, color['green'], "face")
    
    if len(coords)==4:
        x,y,w,h = coords
        roi_img = img[ y:y+h , x:x+w]
        user_id = 1
        generate_dataset(roi_img, user_id, img_id)
        
    return img

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

img_id = 0
video_capture = cv2.VideoCapture(0)   
while True:
    _,img = video_capture.read()
     # Flip the frame horizontally
    img = cv2.flip(img, 1)
    img = detect(img , faceCascade ,img_id)
    cv2.imshow("face_detection",img)
    img_id+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()


# In[ ]:




