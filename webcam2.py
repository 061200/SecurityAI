import dlib
import cv2 as cv
import numpy as np

# addition
import face_recognition
import pickle

encoding_file = './ai_cv/encodings3.pickle'
unknown_name = 'Unknown'
model_method = 'hog'
output_name = './ai_cv/video/output_' + model_method + '.avi'

# ----------------------

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./ai_cv/model/shape_predictor_68_face_landmarks.dat')
cap = cv.VideoCapture(0)

if not cap.isOpened:
    print('### Error opening video ###')
    exit(0)



# range는 끝값이 포함안됨   
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

def detectFaceAndDisplay(frame):
    img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    dets = detector(img_gray, 1)

    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model=model_method)
    encodings = face_recognition.face_encodings(rgb, boxes)

    names = []
    count = 0
    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"],encoding)
        name = 'Unknown'

        

        # check to see if we have found a match

        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            count = len(matchedIdxs)
            print(matchedIdxs)

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number of
                # votes (note: in the event of an unlikely tie Python will
                # select first entry in the dictionary)
                name = max(counts, key=counts.get)
                if(counts.get(name) < 5):
                    name = unknown_name
    
        names.append(name)
    
    
  
    for name, face in zip(names, dets):

        shape = predictor(frame, face) #얼굴에서 68개 점 찾기

        list_points = []
        for p in shape.parts():
            list_points.append([p.x, p.y])

        list_points = np.array(list_points)

        for i,pt in enumerate(list_points[index]):
            pt_pos = (pt[0], pt[1])
            cv.circle(frame, pt_pos, 2, (0, 255, 0), -1)

        color = (0, 255, 0)
        line = 2
        if(name == unknown_name):
            color = (0, 0, 255)
            line = 2
            name = unknown_name
        cv.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), color, line)
        y = face.top() - 15 if face.top() - 15 > 15 else face.top() + 15
        cv.putText(frame, name, (face.left(), y), cv.FONT_HERSHEY_SIMPLEX, 0.75, color, line)
        
                
    cv.imshow('result', frame)

    


# load the known faces and embeddings
data = pickle.loads(open(encoding_file, "rb").read())

while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        # close the video file pointers
        cap.release()
        # close the writer point
        writer.release()
        break
    
    detectFaceAndDisplay(frame)

    key = cv.waitKey(1)

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


cv.destroyAllWindows()
