import cv2
import mediapipe as mp

# camera
camera = cv2.VideoCapture(0)

# drawing utility
mp_drawing = mp.solutions.drawing_utils

# hand recognition
mp_hands = mp.solutions.hands
hand = mp_hands.Hands()

# face detection
mp_face = mp.solutions.face_detection
face = mp_face.FaceDetection()

# face mesh
mp_face_mesh = mp.solutions.face_mesh
faceMesh = mp_face_mesh.FaceMesh()


def handDetection(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h,w,_ = img.shape
    results = hand.process(img)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # this is the basic connection and drawing functionality provided by default
            #mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # detecting and tracking only the tip of the thumb -> check their docs
            lm = hand_landmarks.landmark[4] # landmark[4] => tip of the thumb
            cv2.circle(img, (int(w * lm.x),int(lm.y * h)), 10, (0,255,0), cv2.FILLED)

            #for iterating through all the 20 landmarks
            #for id, lm in enumerate(hand_landmarks.landmark):
            #    print(id, lm)

    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def faceDetection(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h,w,_ = img.shape
    results = face.process(img)

    if results.detections:
        for detection in results.detections:

            # have a look at all the information that is available in detection
            #print(detection)

            # this is the basic drawing functionality provided by default
            #mp_drawing.draw_detection(img, detection)

            # drawing bounding box of each detection
            bounding_box = detection.location_data.relative_bounding_box
            lx, ly = int(bounding_box.xmin * w), int(bounding_box.ymin * h)
            ux, uy = lx + int(bounding_box.width * w), ly + int(bounding_box.height * h)
            point1 = (lx, ly)
            point2 = (ux, uy)

            cv2.rectangle(img,point1,point2,(0,0,255),2)

    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def faceMeshDetection(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(img)
    h,w,_ = img.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # default drawing functionality provided
            #mp_drawing.draw_landmarks(img, face_landmarks, mp_face_mesh.FACE_CONNECTIONS)

            # placing a circle on each landmark point using opencv
            landmarks = face_landmarks.landmark
            for point in landmarks: cv2.circle(img, (int(point.x * w), int(point.y * h)),2,(0,255,0),1)

    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


while True:
    ret, img = camera.read()

    img = handDetection(img)
    #img = faceDetection(img)
    #img = faceMeshDetection(img)

    cv2.imshow('frame', img)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

camera.release()
cv2.destroyAllWindows()