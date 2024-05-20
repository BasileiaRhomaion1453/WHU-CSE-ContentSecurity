import threading
import cv2
import os
import dlib
import numpy as np
from imutils import face_utils
from mtcnn import MTCNN
from keras.models import load_model

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("./model/shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1('./model/dlib_face_recognition_resnet_model_v1.dat')
facepath = './output'
recognizer = cv2.FaceRecognizerSF.create('./model/face_recognizer_fast.onnx','')
model = load_model('./model/fas.h5')
def load_mtcnn_model(model_path):
    mtcnn = MTCNN(model_path)
    return mtcnn
def test_one(X):
    TEMP = X.copy()
    X = (cv2.resize(X,(224,224))-127.5)/127.5
    t = model.predict(np.array([X]),verbose=0)[0]
    return t

#win = dlib.image_window()
EYE_AR_THRESH = 0.3# EAR阈值
EYE_AR_CONSEC_FRAMES = 1# 当EAR小于阈值时，接连多少帧一定发生眨眼动作


RIGHT_EYE_START = 37 - 1
RIGHT_EYE_END = 42 - 1
LEFT_EYE_START = 43 - 1
LEFT_EYE_END = 48 - 1

class OpcvCapture(threading.Thread):
    def __init__(self, win_name, cam_name):
        super().__init__()
        self.cam_name = cam_name
        self.win_name = win_name
        self.frame_counter = 0# 连续帧计数 
        self.blink_counter = 0# 眨眼计数
        self.rect = None
        x = input('请选择方法,1=静默,2=眨眼\n')
        if x == '1':
            self.rect = self.rect1
        elif x == '2':
            self.rect = self.rect2

        else:
            print('输入错误')
            exit(1)
        # self.frame = np.zeros([400, 400, 3], np.uint8)

    def photo(self):
        name=[]
        if os.path.isfile('./csvdata/figures.csv'):
            print('csv file exist, start to read csv')
            with open('./csvdata/figures.csv', newline='',encoding='utf-8') as csvfile:
                d = np.loadtxt(csvfile , delimiter=",", dtype=float)
                #print(d)
            with open('./csvdata/name.csv', newline='', encoding='utf-8') as f:
                for row in f:
                    name.append(row[:-2])
                print('successfully read csv')
            return d,name
        else:
            print('csv file not exist, start to write csv')
            print('accurating......')
            d=[]
            path = os.listdir(facepath)
            path.sort(key=lambda x:int(x[:-4]))
            for i in path:
                img = cv2.imdecode(np.fromfile(os.path.join(facepath, i), dtype=np.uint8), cv2.IMREAD_COLOR)
                faces = detector(img,0)
                for face in faces:
                    face_descriptor = None
                    shape = predictor(img,face)
                    face_descriptor = facerec.compute_face_descriptor(img, shape)
                    d.append(face_descriptor)
            print('Start to write csv, please wait...')
            d=np.array(d)
            d.reshape(len(d),128)
            #print(d)
            with open('./csvdata/figures.csv', 'w', newline='',encoding='utf-8') as csvfile:
                np.savetxt(csvfile, d, delimiter=',', fmt='%f')
            with open('./csvdata/name.csv', newline='', encoding='utf-8') as f:
                for row in f:
                    name.append(row[:-2])
            print('successfully write csv')
            return d,name


    def method(self,gray,face,list,name):
        landmarks = predictor(gray,face)
        self.eyejudge(landmarks)
        video = np.array(facerec.compute_face_descriptor(gray, landmarks))
        q=[]
        for i in list:
            q.append(np.linalg.norm(np.array(i) - np.array(video)))
        min_value = min(q) # 求列表最小值
        min_idx = q.index(min_value)
        return name[min_idx]
    
    def EAR(self,eye):
    # 默认二范数：求特征值，然后求最大特征值得算术平方根
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)
    
    def eyejudge(self,landsmark):
    # 提取左眼和右眼坐标，然后使用该坐标计算两只眼睛的眼睛纵横比
        points = face_utils.shape_to_np(landsmark)
        leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]# 取出左眼对应的特征点
        rightEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]# 取出右眼对应的特征点
        ear = self.EAR(leftEye) + self.EAR(rightEye) / 2.0
        # 判断眼睛纵横比是否低于眨眼阈值
        if ear < EYE_AR_THRESH:
            self.frame_counter += 1
        else:
    # 检测到一次闭眼
            if self.frame_counter >= EYE_AR_CONSEC_FRAMES:
                self.blink_counter += 1
            self.frame_counter = 0

    def rect1(self,frame,face,score):
        cv2.putText(frame, str(score), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)#(5,50)
        if score > 0.74:
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)#bgr
        else:
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)
    
    def rect2(self,frame,face,score):
        cv2.putText(frame, "Blinks:{0}".format(self.blink_counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)#(10, 30)
        if self.frame_counter == 1:
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        elif self.blink_counter >=1:
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)

    def run(self):
        p,name=self.photo()
        capture = cv2.VideoCapture(self.cam_name)
        capture.set(3, 1280)
        capture.set(4, 720)
        while (True):
            # 获取一帧
            ret, frame = capture.read()
            faces = detector(frame,0)
            score = test_one(frame)
            if len(faces) != 0:
                for face in faces:
                    cv2.putText(frame, self.method(frame,face,p,name), (face.left(), face.bottom() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    self.rect(frame,face,score)
            else:
                self.blink_counter = 0
            cv2.imshow(self.win_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()

    def start(self):
        super().start()

if __name__ == "__main__":
    camera1 = OpcvCapture("camera1", 0)
    camera1.start()
    