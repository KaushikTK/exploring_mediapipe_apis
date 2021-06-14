import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands()

camera = cv2.VideoCapture(0)


def recogniseHands(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hand.process(img)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # this is the basic connection and drawing functionality provided by default
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # finding only the position of a specific point -> check their doc for finding what index represents what
            #lm = hand_landmarks.landmark[4]
            #h,w,_ = img.shape
            #cv2.circle(img, (int(w * lm.x),int(lm.y * h)), 10, (0,255,0), cv2.FILLED)

            #for iterating through all the 20 landmarks
            #for id, lm in enumerate(hand_landmarks.landmark):
            #    print(id, lm)

    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


while True:
    ret, img = camera.read()
    img = recogniseHands(img)
    cv2.imshow('frame', img)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

camera.release()
cv2.destroyAllWindows()