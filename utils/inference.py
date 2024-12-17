import cv2
import numpy as np
'''
    Image height width
'''
IMG_HEIGHT, IMG_WIDTH = 64, 64

def predict_face_img(img, box, model):
    (x, y, x1, y1) = box.astype("int")
    face = img[y:y1, x:x1]
    face_resized = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))
    face_flattened = face_resized.reshape(1, -1)

    prediction = model.predict(face_flattened)
    proba = int(np.max(model.predict_proba(face_flattened))*100)

    return prediction[0], proba

def predict_face_emb(img, box, embedder, model):
    
    (x, y, x1, y1) = box.astype("int")
    face = img[y:y1, x:x1]
    emb=  embedder.embeddings(np.expand_dims(face,axis=0))
    
    prediction = model.predict(emb)
    proba = int(np.max(model.predict_proba(emb))*100)
    
    return prediction[0],proba