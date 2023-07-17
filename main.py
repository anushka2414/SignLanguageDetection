import os 
import tensorflow as tf 
import cv2 
import mediapipe as mp 
from keras.models import load_model
import numpy as np 
import time
import pandas as pd 

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

savedModel = load_model('signLanguage.h5')
mpHands = mp.solutions.hands 

hands = mpHands.Hands()

mp_drawing = mp.solutions.drawing_utils 

cap = cv2.VideoCapture(0) 

_, frame = cap.read() 
h,w,c = frame.shape 

analysisFrame = '' 
letterPred =  ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

while True: 
    _, frame = cap.read() 

    k = cv2.waitKey(1)

    if k%256 == 27:
        break

    elif k%256 == 32: 



        analysisFrame = frame 
        showFrame = analysisFrame 
        cv2.imshow("Frame", showFrame) 
        framerAnalysis = cv2.cvtColor(analysisFrame, cv2.COLOR_BGR2RGB)
        resultAnalysis = hands.process(framerAnalysis)
        handLandmarkAnalysis  = resultAnalysis.multi_hand_landmarks

        if handLandmarkAnalysis: 
            for handLMAnalysis in handLandmarkAnalysis:
        
                x_max = 0 
                y_max = 0 
                x_min = w 
                y_min = h 

                for lm in handLMAnalysis.landmark: 
                    x,y = int(lm.x * w), int(lm.y * h)

                    if x > x_max: 
                        x_max = x 
                    if x < x_min: 
                        x_min = x 

                    if y> y_max: 
                        y_max = y 

                    if y<y_min: 
                        y_min = y 

                y_min -= 20 
                y_max += 20 
                x_min -= 20 
                x_max += 20  

            analysisFrame = cv2.cvtColor(analysisFrame, cv2.COLOR_BGR2GRAY)
            analysisFrame = analysisFrame[y_min : y_max, x_min : x_max]
            analysisFrame = cv2.resize(analysisFrame, (28,28))

            nlist = [] 

            rows, cols = analysisFrame.shape 

            for i in range(rows):
                for j in range(cols):
                    k = analysisFrame[i,j]
                    nlist.append(k)

            data = pd.DataFrame(nlist).T
            colName = [] 

            for val in range(784):
                colName.append(val)

            data.columns = colName 

            pixelData = data.values 
            pixelData = pixelData / 255 
            pixelData = pixelData.reshape(-1, 28, 28 , 1)
            prediction = savedModel.predict(pixelData)
            predArray = np.array(prediction[0])
            letterPredictDict = {letterPred[i]: predArray[i] for i in range(len(letterPred))}

            predArraySorted = sorted(predArray, reverse = True ) 

            maxConfidence1 = predArraySorted[0]
            maxConfidence2 = predArraySorted[1]
            maxConfidence3 = predArraySorted[2]

            for key,value in letterPredictDict.items(): 
                if value == maxConfidence1 or value == maxConfidence2 or value == maxConfidence3:
                    print("Predicted Character!: ", key)
                    print("Confidence: ", 100*value)

            time.sleep(5)
                
    
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frameRGB)

    handLandmarks = result.multi_hand_landmarks 

    if handLandmarks: 
        for handLMs in handLandmarks: 
            x_max = 0 
            y_max = 0 
            x_min = w 
            y_min = h 

            for lm in handLMs.landmark: 
                x,y = int(lm.x * w), int(lm.y * h)

                if x > x_max: 
                    x_max = x 
                if x < x_min: 
                    x_min = x 

                if y> y_max: 
                    y_max = y 

                if y<y_min: 
                    y_min = y 

            y_min -= 20 
            y_max += 20 
            x_min -= 20 
            x_max += 20 

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            mp_drawing.draw_landmarks(frame, handLMs, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Sign Language Detector", frame) 

cap.release()

cv2.destroyAllWindows() 



