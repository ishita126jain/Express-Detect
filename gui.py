import tkinter as tk
from tkinter import filedialog
from tkinter import *
import threading
from sklearn import metrics

from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

# donwload haarcascade_frontalface_default from here "https://github.com/opencv/opencv/tree/master/data/haarcascades"
# Here are kaggel links of the dataset which I have used them in my project
# Eye dataset "https://www.kaggle.com/code/nandalald/eyes-dataset-prediction" 
# Mouth dataset "https://www.kaggle.com/datasets/davidvazquezcic/yawn-dataset"
# Wrinkles dataset "https://www.kaggle.com/datasets/rishantrokaha/skin-wrinkles-vs-nonwrinkles?select=10.Junaid-Ahmed-selfies-750x750.jpg" and "https://www.kaggle.com/datasets/mohit335448/ageing-dataset"

def FacialExpressionModel(json_file, weights_file):
    with open(json_file,"r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer ='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

    return model

def EyeDetectionModel(json_file, weights_file):
    with open(json_file,"r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer ='adam', loss='binary_crossentropy', metrics = ['accuracy'])

    return model

def MouthDetectionModel(json_file, weights_file):
    with open(json_file,"r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer ='adam', loss='binary_crossentropy', metrics = ['accuracy'])

    return model

def WrinkleDetectionModel(json_file, weights_file):
    with open(json_file,"r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer ='adam', loss='binary_crossentropy', metrics = ['accuracy'])

    return model

top =tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
model1 = FacialExpressionModel("model_a.json","model_weights.h5")
model2 = EyeDetectionModel("model_eye.json","model_eye.h5")
model3 = MouthDetectionModel("model_mouth.json","model_mouth.h5")
model4 = WrinkleDetectionModel("model_wrinkle.json","model_wrinkle.h5")


EMOTIONS_LIST = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]

def Detect(file_path):
    global Label_packed

    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_image,1.3,5)
    try:
        for (x,y,w,h) in faces:
            fc = gray_image[y:y+h,x:x+w]
            roi = cv2.resize(fc,(48,48))
            pred = EMOTIONS_LIST[np.argmax(model1.predict(roi[np.newaxis,:,:,np.newaxis]))]
        print("Predicted Emotion is " + pred)
        label1.configure(foreground="#011638",text = pred)
    except:
        label1.configure(foreground="#011638",text = "Unable to detect")

def Detect_eye(file_path):
    global Label_packed

    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_image,1.3,5)
    if faces==():
        label1.configure(foreground="#011638",text = "Unable to detect")
        return
    try:
        pred = None
        for (x,y,w,h) in faces:
            fc = gray_image[y:y+h,x:x+w]
            eye = eyes.detectMultiScale(fc)
            for ex,ey,ew,eh in eye:
                eye_x = x+ex
                eye_y = y+ey
                eye_w = ew
                eye_h = eh
                eye_roi = image[eye_y:eye_y+eye_h, eye_x:eye_x+eye_w]
                resized_eye = cv2.resize(eye_roi, (24, 24))
                resized_eye = resized_eye / 255.0
                input_data = np.reshape(resized_eye, (1, 24, 24, 3))
                pred = (model2.predict(input_data))       
        if pred:
            eye_state = 'Open'
        else:
            eye_state = 'Close'
        print("Predicted Eye state is" + eye_state)
        label1.configure(foreground="#011638",text = eye_state)
    except:
        label1.configure(foreground="#011638",text = "Unable to detect")

def Detect_mouth(file_path):
    global Label_packed

    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_image,1.3,5)
    if faces==():
        label1.configure(foreground="#011638",text = "Unable to detect")
        return
    try:
        pred = None
        for (x,y,w,h) in faces:
            fc = gray_image[y:y+h,x:x+w]
            mouth_cascade = cv2.CascadeClassifier('haarcascade_mouth.xml')
            mouths = mouth_cascade.detectMultiScale(fc, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            global res
            res=0
		
        if mouths == ():
            label1.configure(foreground="#011638",text = "Unable to detect")
            return
		
        if len(mouths) > 0:
            (mx, my, mw, mh) = mouths[0]
		    
            mouth_x = x + mx
            mouth_y = y + my
            mouth_width = mw
            mouth_height = mh
            		
            mouth_image = image[mouth_y:mouth_y+mouth_height, mouth_x:mouth_x+mouth_width]
            resized_mouth = cv2.resize(mouth_image, (64, 64))
            resized_mouth = resized_mouth / 255.0
            input_data = np.reshape(resized_mouth, (1, 64, 64, 3))
            pred = model3.predict(input_data)
            res = pred.item()
            res = round(res,2)   
        print(res) 
        if res <= 0.88:
            mouth_state = 'Open'
        else:
            mouth_state = 'Close'
        print("Predicted Mouth state is " + mouth_state)
        label1.configure(foreground="#011638",text = mouth_state)
    except:
        label1.configure(foreground="#011638",text = "Unable to detect")

def Detect_wrinkle(file_path):
    global Label_packed

    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_image,1.3,5)
    if faces==():
        label1.configure(foreground="#011638",text = "Unable to detect")
        return
    try:
        prediction = None
        for x,y,w,h in faces:
            face_roi = image[y:y+h, x:x+w]
            
            global res
            res=0
            resized_img = cv2.resize(face_roi, (64, 64))
            resized_img = resized_img / 255.0
            
        prediction = model4.predict(np.expand_dims(resized_img,axis=0))
        res = prediction.item()
        res = round(res,2)  
        print(res) 
        if res >= 0.6:
            wrinkle_state = 'Non Wrinkle'
        else:
            wrinkle_state = 'Wrinkle'
        print("Predicted Wrinkle state is " + wrinkle_state)
        label1.configure(foreground="#011638",text = wrinkle_state)
    except:
        label1.configure(foreground="#011638",text = "Unable to detect")

def show_Detect_button(file_path):
    detect_b = Button(top,text="Detect Emotion", command= lambda: Detect(file_path),padx=10,pady=5)
    detect_b.configure(background="#364156",foreground='white',font=('arial',10,'bold'))
    detect_b.place(relx =0.79,rely=0.26)

def show_Detect_eye_button(file_path):
    detect_b = Button(top,text="Detect Eye State", command= lambda: Detect_eye(file_path),padx=10,pady=5)
    detect_b.configure(background="#364156",foreground='white',font=('arial',10,'bold'))
    detect_b.place(relx =0.79,rely=0.36)

def show_Detect_mouth_button(file_path):
    detect_b = Button(top,text="Detect Mouth State", command= lambda: Detect_mouth(file_path),padx=10,pady=5)
    detect_b.configure(background="#364156",foreground='white',font=('arial',10,'bold'))
    detect_b.place(relx =0.79,rely=0.46)

def show_Detect_Wrinkle_button(file_path):
    detect_b = Button(top,text="Detect Wrinkle State", command= lambda: Detect_wrinkle(file_path),padx=10,pady=5)
    detect_b.configure(background="#364156",foreground='white',font=('arial',10,'bold'))
    detect_b.place(relx =0.79,rely=0.56)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        detect_button_thread = threading.Thread(target=lambda: show_Detect_button(file_path))
        detect_eye_button_thread = threading.Thread(target=lambda: show_Detect_eye_button(file_path))
        detect_mouth_button_thread = threading.Thread(target=lambda: show_Detect_mouth_button(file_path))
        detect_wrinkle_button_thread = threading.Thread(target=lambda: show_Detect_Wrinkle_button(file_path))

        # Start both threads
        detect_button_thread.start()
        detect_eye_button_thread.start()
        detect_mouth_button_thread.start()
        detect_wrinkle_button_thread.start()
    except:
        pass

upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156",foreground='white',font=('arial',20,'bold'))
upload.pack(side='bottom',pady=50)
sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')
heading = Label(top,text='Emotion, Eye, Mouth and Wrinkle Detector ',pady=20,font=('arial',25,'bold'))
heading.configure(background='#CDCDCD',foreground="#364156")
heading.pack()
top.mainloop()
