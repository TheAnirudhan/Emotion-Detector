import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')  # Loading Haarcascade
model = tf.keras.models.load_model("trained weights 1.h5") # Loading Weights of pretrained model


def predict_emotion(img): # Proccess the image to feed into the Nueral network and get prediction
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray,1.3,5)
            
    for (x,y,w,h) in faces:
        roi = np.expand_dims(cv.resize(frame[y:y+h, x:x+w], (150,150)),0)
        pred  = model.predict(roi)
        text_idx=np.argmax(pred)
        text_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        if text_idx == 0:
            text= text_list[0]
        if text_idx == 1:
            text= text_list[1]
        elif text_idx == 2:
            text= text_list[2]
        elif text_idx == 3:
            text= text_list[3]
        elif text_idx == 4:
            text= text_list[4]
        elif text_idx == 5:
            text= text_list[5]
        elif text_idx == 6:
            text= text_list[6]
        
        return(text)
    

cap = cv.VideoCapture(0) # Start Video Capture

while True:
    ret,frame = cap.read()
    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # Convert to gray
        faces = face_cascade.detectMultiScale(gray,1.3,5) # Find Faces in the frame
        for (x,y,w,h) in faces:
            text = predict_emotion(frame) # Predict Emotion of Each Face

            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),5)
            cv.putText(frame, text, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        cv.imshow("Face",frame)

    if cv.waitKey(10) == 27: break
cap.release()
cv.destroyAllWindows()
