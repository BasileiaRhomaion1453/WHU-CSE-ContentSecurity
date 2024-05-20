import threading
import cv2
import os
import dlib
import numpy as np
from imutils import face_utils
from mtcnn import MTCNN
from keras.models import load_model
from flask import Flask, request, jsonify , render_template, Response
import csv

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("./model/shape_predictor_68_face_landmarks.dat")

# Load the face recognition model 68特征->128特征
facerec = dlib.face_recognition_model_v1('./model/dlib_face_recognition_resnet_model_v1.dat')

#照片路径，data是原始照片，output是处理过后的人脸照片，精度更好
facepath = './output'
#opencv的人脸检测，效果不好
#recognizer = cv2.FaceRecognizerSF.create('./model/face_recognizer_fast.onnx','')
#静默检测模型
model = load_model('./model/fas.h5')

#静默算法
'''
def load_mtcnn_model(model_path):
    mtcnn = MTCNN(model_path)
    return mtcnn
'''
def test_one(X):#计算分数
    TEMP = X.copy()
    X = (cv2.resize(X,(224,224))-127.5)/127.5
    t = model.predict(np.array([X]),verbose=0)[0]
    return t

#眨眼算法的值
EYE_AR_THRESH = 0.3# EAR阈值
EYE_AR_CONSEC_FRAMES = 2# 当EAR小于阈值时，接连多少帧一定发生眨眼动作，越小感觉检测的越准，

#眼部特征点
RIGHT_EYE_START = 37 - 1
RIGHT_EYE_END = 42 - 1
LEFT_EYE_START = 43 - 1
LEFT_EYE_END = 48 - 1

qiandaobiao=[0]*64
namelist = []
#创建flask
app = Flask(__name__)

class OpcvCapture(threading.Thread):
    def __init__(self, win_name, cam_name):
        super().__init__()
        self.cam_name = cam_name
        self.win_name = win_name
        self.frame_counter = 0# 连续帧计数 
        self.blink_counter = 0# 眨眼计数
        self.rect = self.rect1
        self.qiandao = '等待中'
        self.namemax = ''#检测到的人脸的名字
        """
        x = input('请选择方法,1=静默,2=眨眼\n')
        if x == '1':
            self.rect = self.rect1
        elif x == '2':
            self.rect = self.rect2

        else:
            print('输入错误')
            exit(1)
        # self.frame = np.zeros([400, 400, 3], np.uint8)
        """

    def photo(self):#对人脸特征预处理，只有第一次运行需要计算，加速后续启动
        if os.path.isfile('./csvdata/figures.csv'):#如果已经计算过了就直接读取特征
            print('csv file exist, start to read csv')
            with open('./csvdata/figures.csv', newline='',encoding='utf-8') as csvfile:
                d = np.loadtxt(csvfile , delimiter=",", dtype=float)
                #print(d)
            with open('./csvdata/name.csv', newline='', encoding='utf-8') as f:#读取对应名字
                for row in f:
                    namelist.append(row[:-2])
                print('successfully read csv')
            return d
        else:
            print('csv file not exist, start to write csv')
            print('accurating......')
            d=[]
            path = os.listdir(facepath)
            path.sort(key=lambda x:int(x[:-4]))
            for i in path:
                img = cv2.imdecode(np.fromfile(os.path.join(facepath, i), dtype=np.uint8), cv2.IMREAD_COLOR)
                faces = detector(img,0)
                for face in faces:                      #没与处理过，先把人脸数据计算一次并保存
                    face_descriptor = None
                    shape = predictor(img,face)
                    face_descriptor = facerec.compute_face_descriptor(img, shape)
                    d.append(face_descriptor)
            print('Start to write csv, please wait...')
            d=np.array(d)
            d.reshape(len(d),128)#形状要保持一致
            #print(d)E
            with open('./csvdata/figures.csv', 'w', newline='',encoding='utf-8') as csvfile:
                np.savetxt(csvfile, d, delimiter=',', fmt='%f')
            with open('./csvdata/name.csv', newline='', encoding='utf-8') as f:
                for row in f:
                    (namelist).append(row[:-2])
            print('successfully write csv')
            return d


    def method(self,gray,face,list):#摄像头人脸特征检测
        landmarks = predictor(gray,face)
        self.eyejudge(landmarks)
        video = np.array(facerec.compute_face_descriptor(gray, landmarks))#68->128
        q=[]
        for i in list:
            q.append(np.linalg.norm(np.array(i) - np.array(video)))#计算欧式距离
        min_value = min(q) # 求距离最小值，对应的就是那个人
        min_idx = q.index(min_value)
        self.namemax = namelist[min_idx]
        qiandaobiao[min_idx]=1
        return namelist[min_idx]
    
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

    def rect1(self,frame,face):#检测方法一，静默检测，分数超过0.74则签到成功，框从红色变为蓝色
        score = test_one(frame)#计算静默检测分数
        cv2.putText(frame, '静默活体检测'+str(score), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)#(5,50)
        if score > 0.74:
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)#bgr
            self.qiandao = '签到成功!'
        else:
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)
    
    def rect2(self,frame,face):#检测方法二，眨眼检测，眨眼次数超过1次则签到成功，框从红色变为蓝色，眨眼时框变绿色
        cv2.putText(frame, '眨眼检测'+"Blinks:{0}".format(self.blink_counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)#(10, 30)
        if self.frame_counter == 1:
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        elif self.blink_counter >=1:
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)
            self.qiandao = '签到成功!'
        else:
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)

    def run(self):
        p=self.photo()
        capture = cv2.VideoCapture(self.cam_name)
        capture.set(3, 854)#设置框大小
        capture.set(4, 480)
        #capture.set(3, 1280)#设置框大小
        #capture.set(4, 720)
        while (True):
            # 获取一帧
            ret, frame = capture.read()
            faces = detector(frame,0)
            if len(faces) != 0:
                for face in faces:
                    #向图片添加检测到的人员信息
                    self.rect(frame,face)
                    cv2.putText(frame, self.method(frame,face,p), (face.left(), face.bottom() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                self.blink_counter = 0
                self.qiandao = '等待中'
            #cv2.imshow(self.win_name, frame) #后端展示
            ret1,buffer = cv2.imencode('.jpg',frame)#前端需编码传送
            frame = buffer.tobytes()
            yield  (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        '''
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        capture.release()
        cv2.destroyAllWindows()
        '''

#运行前先创建一个类，方便后面路由器函数操作
camera1 = OpcvCapture("camera1", 0)

#从模板创建网页
@app.route('/')
def index():
    return render_template('index.html')

#传送处理过的摄像头信息
@app.route('/video_feed0')
def video_feed0():
    return Response(camera1.run(),mimetype='multipart/x-mixed-replace; boundary=frame')

#按钮1切换成模式一
@app.route('/change_mode1')
def change_mode1():
    camera1.rect=camera1.rect1

#按钮2切换成模式二
@app.route('/change_mode2')
def change_mode2():
    camera1.rect=camera1.rect2

#传送签到信息
@app.route('/get_text')
def get_text():
    if camera1.qiandao == '等待中':
        text = camera1.qiandao
    else:
        text = camera1.qiandao + camera1.namemax
    return jsonify({'text': text}) 

@app.route('/writeqiandaobiao')
def writeqiandao():
    with open('./csvdata/qiandaobiao.csv', 'w', newline='',encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入数据到多列中
        for item1, item2 in zip(namelist, qiandaobiao):
            writer.writerow([item1, item2])
    camera1.qiandao = '写入完成！'
    print("写入完成！")
    os._exit(0)


if __name__ == '__main__':
    app.run()