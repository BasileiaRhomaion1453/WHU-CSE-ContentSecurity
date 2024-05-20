import os
import cv2
import dlib
import numpy as np
import csv
facepath = './output'
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('./model/shape_predictor_68_face_landmarks.dat')
r=0
l=[]
print(os.listdir(facepath))
for i in os.listdir(facepath):

    img = cv2.imdecode(np.fromfile(os.path.join(facepath, i), dtype=np.uint8), cv2.IMREAD_COLOR)
    dets = detector(img,0)
    for k, d in enumerate(dets):
        shape = sp(img,d)
        '''
        for q in range(shape.num_parts):
            cv2.circle(img, (shape.part(q).x, shape.part(q).y), 2, (0,255,0), -1,1)
            cv2.putText(img, str(q), (shape.part(q).x, shape.part(q).y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        '''
        face_chip = dlib.get_face_chip(img, shape,size=750)      
    #cv2.imshow("Output", img)
    #cv2.waitKey(0)
    cv2.imwrite(os.path.join('./output',str(r)+'.png'), face_chip)
    r+=1
'''
    l.append(str(i[:-4]))
print(l)
p=range(len(l))
with open('name.csv','w',encoding='utf8',newline='') as f:
    cw = csv.writer(f)
    for k,t in zip(l,p) :
        cw.writerow([t,k]) 
        '''
