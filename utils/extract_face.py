import cv2
import os
import tqdm
from dnn_face_detection import detect_face 
import numpy as np

video_path = "../dataset/video"
destination =  "../dataset/images"


parent_directory = "../dataset/images"

if not os.path.exists(parent_directory):
    os.makedirs(parent_directory)


def rand():
    return np.random.binomial(n=1,p=0.2,size=[1])

def random_num(i):
    return int(np.random.randint(0,i))


for root, files, videos in tqdm.tqdm(os.walk(video_path)):
    for video in videos:
        video_id = root.split("\\")[-1]
        destination_path = os.path.join(destination, video_id)
        if not os.path.isdir(destination_path):
            os.mkdir(destination_path)
        cap = cv2.VideoCapture(os.path.join(root, video))
       
        #For balancing images
        balance=[]
        balance_count=0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)
            faces = detect_face(frame)
            for bbounding_box in faces:
                x = int(bbounding_box[0])
                y = int(bbounding_box[1])
                x2 = int(bbounding_box[2])
                y2 = int(bbounding_box[3])

                crop_face = frame[y:y2, x:x2]
                crop_face = cv2.resize(crop_face, (160,160))
                balance.append(crop_face)
                cv2.imwrite(f"{destination_path}/{balance_count}.png",crop_face)
                
            balance_count += 1
       
        
        #For balancing our data set
        while balance_count<1000:
                if rand():
                    index=random_num(len(balance))
                    balance_count+=1
                    cv2.imwrite(f"{destination_path}/{balance_count}.png",balance[index])
        
                