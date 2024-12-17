import os
import numpy as np
import cv2
import joblib
import time
import matplotlib.pyplot as plt
from keras_facenet import FaceNet
from dnn_face_detection import detect_face
from inference import *

'''
    Get models description
'''
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--type',required=True, type=str, choices= ['img','emb'])
parser.add_argument('--model',required=True, type=str, choices= ['knn','svm'])

args = parser.parse_args()


'''
    Load Model
'''

model = joblib.load(f"../face_attendance_model/{args.type}_{args.model}_model.pkl")
embedder = FaceNet()


# Main function to capture video and detect faces
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, img = cap.read()
            img = cv2.flip(img,1)    

            if not ret:
                break
            
            boxes = detect_face(img)
            for box in boxes:
                (x, y, x1, y1) = box.astype("int")
                cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
                
                if args.type == 'img':
                    prediction,proba = predict_face_img(img, box, model)
                else:
                    prediction,proba = predict_face_emb(img, box, embedder, model)

                predicted_class = prediction
                cv2.putText(img, f'{predicted_class}, {proba}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow("Webcam Face Detection and Classification", img)

            if cv2.waitKey(1) == ord("q"):
                break
    except Exception as e:
        print(e)
    finally:
        cap.release()
        cv2.destroyAllWindows()
