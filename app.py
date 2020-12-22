#Import necessary libraries
from flask import Flask, render_template, Response
import cv2
import dlib
import numpy as np

#Initialize the Flask app
app = Flask(__name__)

detector = dlib.get_frontal_face_detector()
 
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

camera = cv2.VideoCapture(0)

f = 0;

def gen_frames():  

    ALL = list(range(0, 68)) 
    RIGHT_EYEBROW = list(range(17, 22))     
    LEFT_EYEBROW = list(range(22, 27))  
    RIGHT_EYE = list(range(36, 42))  
    LEFT_EYE = list(range(42, 48))  
    NOSE = list(range(27, 36))
    MOUTH_OUTLINE = list(range(48, 61))  
    MOUTH_INNER = list(range(61, 68)) 
    JAWLINE = list(range(0, 17)) 

    index = ALL

    while True:
        
        success, img_frame = camera.read()
        
        img_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)

        dets = detector(img_gray, 1)

        for face in dets:
    
            shape = predictor(img_frame, face) #얼굴에서 68개 점 찾기

            list_points = []
            for p in shape.parts():
                list_points.append([p.x, p.y])

            list_points = np.array(list_points)


            for i,pt in enumerate(list_points[index]):

                pt_pos = (pt[0], pt[1])
                cv2.circle(img_frame, pt_pos, 2, (0, 255, 0), -1)

            
            cv2.rectangle(img_frame, (face.left(), face.top()), (face.right(), face.bottom()),
                (0, 0, 255), 3)
            

        cv2.imshow('frame2', img_frame)

        key = cv2.waitKey(1)

        if key == 27:
            break
    
        elif key == ord('1'):
            index = ALL
        elif key == ord('2'):
            index = LEFT_EYEBROW + RIGHT_EYEBROW
        elif key == ord('3'):
            index = LEFT_EYE + RIGHT_EYE
        elif key == ord('4'):
            index = NOSE
        elif key == ord('5'):
            index = MOUTH_OUTLINE+MOUTH_INNER
        elif key == ord('6'):
            index = JAWLINE


        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', img_frame)
            img_frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_frame + b'\r\n')



@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/')
def index():
    return render_template('page1.html')

@app.route('/beta')
def beta():
    return render_template('beta.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)