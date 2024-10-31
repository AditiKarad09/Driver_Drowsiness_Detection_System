
import cv2
import os
import numpy as np
from keras.models import load_model
from pygame import mixer
import time

def init_sound():
    mixer.init()
    return mixer.Sound('alarm.wav')

def load_cascades():
    base_path = '/Users/aditikarad/Desktop/RITAssignments/SEM4/Adv_CV/Project/Drowsiness detection [OPEN CV]/haar cascade files/'
    face_cascade = cv2.CascadeClassifier(os.path.join(base_path, 'haarcascade_frontalface_alt.xml'))
    leye_cascade = cv2.CascadeClassifier(os.path.join(base_path, 'haarcascade_lefteye_2splits.xml'))
    reye_cascade = cv2.CascadeClassifier(os.path.join(base_path, 'haarcascade_righteye_2splits.xml'))
    return face_cascade, leye_cascade, reye_cascade

def detect_eyes(eye_cascade, gray_frame):
    eyes = eye_cascade.detectMultiScale(gray_frame)
    return eyes

def predict_eye_status(eye, model):
    eye_gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    eye_resized = cv2.resize(eye_gray, (24, 24))
    eye_normalized = eye_resized / 255.0
    eye_reshaped = eye_normalized.reshape(24, 24, -1)
    eye_expanded = np.expand_dims(eye_reshaped, axis=0)
    prediction = np.argmax(model.predict(eye_expanded), axis=-1)
    return prediction

def main():
    sound = init_sound()
    face_cascade, leye_cascade, reye_cascade = load_cascades()
    model = load_model('models/cnncat2.h5')
    cap = cv2.VideoCapture(0)
    score = 0
    thicc = 2
    path = os.getcwd()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detect_eyes(face_cascade, gray)
        left_eye = detect_eyes(leye_cascade, gray)
        right_eye = detect_eyes(reye_cascade, gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)

        eye_status = 'Open'  # default status
        for eye_cascade in [right_eye, left_eye]:
            for (x, y, w, h) in eye_cascade:
                eye = frame[y:y+h, x:x+w]
                prediction = predict_eye_status(eye, model)
                eye_status = 'Closed' if prediction == 0 else 'Open'
                break

        display_text = f'{eye_status} Score:{score}'
        cv2.putText(frame, display_text, (10, height - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)
        if eye_status == 'Closed':
            score += 1
        else:
            score = max(score - 1, 0)

        if score > 15:
            cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
            try:
                sound.play()
            except:
                pass
            thicc = min(16, thicc + 2)
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
        else:
            thicc = max(2, thicc - 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
