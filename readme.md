1.data文件夹内放人脸照片
2.output文件夹是人脸切片，运行preparation.py后会生成
3.model内存放模型文件，分别是：dlib_face_recognition_resnet_model_v1.dat，face_recognizer_fast.onnx，fas.h5，shape_predictor_68_face_landmarks.dat
模型使用的是dlib68特征，68转128特征，人脸检测，静默活体检测，这个算法在github上有
4.csvdata内放人脸特征、人名和签到信息三个csv文件，运行主程序后自然生成
5.mtcnn.py是静默活体算法文件
6.templates文件夹存放html文件
5.app.py是主程序