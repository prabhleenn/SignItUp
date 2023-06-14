import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

engine = pyttsx3.init()

f = open(r"C:\Users\user\PycharmProjects\new\model.p", "rb")
model_dict = pickle.load(f)
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',  9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19 : 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Hello', 27: 'thank you', 28: 'name', 29: 'where', 30: 'how', 31: 'what', 32: 'I', 33: 'my', 34: "goodbye",35:'fine',36:'help',37:'no' , 38:'need' ,39:'You ',40:'whatsupp',41:'live',42:'Good'}

number_of_classes = 42


# Number of features expected by the model
num_expected_features = 126

word = ""  # Initialize an empty word
sentence = ""

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Pad the data_aux list with zeros to match the expected number of features
        padded_data_aux = data_aux[:num_expected_features] + [0] * (num_expected_features - len(data_aux))

        prediction = model.predict([np.asarray(padded_data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        # Add predicted character to the word when 'y' key is pressed
        key = cv2.waitKey(1)
        if key == ord('y'):
            word += predicted_character

        if key == ord("s"):
            sentence += word + ' '
            word = ''
        elif key == ord("d"):
            word = word[:-1]

        if key == ord("r"):
            print(sentence)
            engine.say(sentence)
            engine.runAndWait()

        if key == ord("x"):
            sentence = ""

        # Display the predicted word and complete sentence on the video screen
        cv2.putText(frame, "Predicted Word: " + word, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (21, 44, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Complete Sentence: " + sentence, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (45, 121, 255), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27:  # Press Esc key to exit
        break

cap.release()
cv2.destroyAllWindows()
