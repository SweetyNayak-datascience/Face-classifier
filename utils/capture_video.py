import cv2
import os
import time
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')

# pass person's name
parser.add_argument('--name',  type=str,
                    help='Name')
args = parser.parse_args()


# capture person's video using face detection

cap = cv2.VideoCapture(0)
_,frame = cap.read()
fh, fw,_ = frame.shape
fps = cap.get(cv2.CAP_PROP_FPS)

# create directory
parent_directory = "../dataset/video"

if not os.path.exists(parent_directory):
    os.makedirs(parent_directory)

# save in a path
destination_path = f"../dataset/video/{args.name}"

if not os.path.isdir(destination_path): 
    os.mkdir(destination_path)

out = cv2.VideoWriter(f"{destination_path}/{args.name}.avi",cv2.VideoWriter_fourcc(*'mp4v'),fps,(fw,fh))
start = time.time()

while cap.isOpened() :
    ret, frame = cap.read()   
    frame = cv2.flip(frame,1)    
    count_time = int((time.time() - start))

    cv2.putText(frame, str(count_time),(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,25,24))
    out.write(frame)
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q") or count_time >= 10:
        break



