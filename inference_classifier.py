# '''# import pickle
# # import cv2
# # import mediapipe as mp
# # import numpy as np

# # model_dict = pickle.load(open('./model.p', 'rb'))
# # model = model_dict['model']

# # cap = cv2.VideoCapture(0)

# # mp_hands = mp.solutions.hands
# # mp_drawing = mp.solutions.drawing_utils
# # mp_drawing_styles = mp.solutions.drawing_styles

# # hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# # labels_dict = {0: 'A', 1: 'B', 2: 'L'}
# # while True:

# #     data_aux = []
# #     x_ = []
# #     y_ = []

# #     ret, frame = cap.read()

# #     H, W, _ = frame.shape

# #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# #     results = hands.process(frame_rgb)
# #     if results.multi_hand_landmarks:
# #         for hand_landmarks in results.multi_hand_landmarks:
# #             mp_drawing.draw_landmarks(
# #                 frame,  # image to draw
# #                 hand_landmarks,  # model output
# #                 mp_hands.HAND_CONNECTIONS,  # hand connections
# #                 mp_drawing_styles.get_default_hand_landmarks_style(),
# #                 mp_drawing_styles.get_default_hand_connections_style())

# #         for hand_landmarks in results.multi_hand_landmarks:
# #             for i in range(len(hand_landmarks.landmark)):
# #                 x = hand_landmarks.landmark[i].x
# #                 y = hand_landmarks.landmark[i].y

# #                 x_.append(x)
# #                 y_.append(y)

# #             for i in range(len(hand_landmarks.landmark)):
# #                 x = hand_landmarks.landmark[i].x
# #                 y = hand_landmarks.landmark[i].y
# #                 data_aux.append(x - min(x_))
# #                 data_aux.append(y - min(y_))

# #         x1 = int(min(x_) * W) - 10
# #         y1 = int(min(y_) * H) - 10

# #         x2 = int(max(x_) * W) - 10
# #         y2 = int(max(y_) * H) - 10

# #         prediction = model.predict([np.asarray(data_aux)])

# #         predicted_character = labels_dict[int(prediction[0])]

# #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
# #         cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
# #                     cv2.LINE_AA)

# #     cv2.imshow('frame', frame)
# #     cv2.waitKey(1)


# # cap.release()
# # cv2.destroyAllWindows()


# '''
# # #-----------------------------------
# # import time
# # import pickle
# # import cv2
# # import mediapipe as mp
# # import numpy as np
# # import pyttsx3
# # import threading

# # # Load the model
# # model_dict = pickle.load(open('./model.p', 'rb'))
# # model = model_dict['model']

# # # Initialize MediaPipe
# # mp_hands = mp.solutions.hands # type: ignore
# # mp_drawing = mp.solutions.drawing_utils # type: ignore
# # mp_drawing_styles = mp.solutions.drawing_styles # type: ignore
# # hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# # # Define labels for recognition
# # labels_dict = {
# #     0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G',
# #     7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N',
# #     14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
# #     21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
# #     26: '0', 27: '1', 28: '2', 29: '3', 30: '4',
# #     31: '5', 32: '6', 33: '7', 34: '8', 35: '9',
# #     36: ' '  # Add a label for a space character
# # }

# # # Initialize the camera
# # cap = cv2.VideoCapture(0)

# # # Create a text-to-speech engine
# # engine = pyttsx3.init()

# # # Initialize variables for character recognition
# # recognized_characters = []
# # current_character = ''
# # last_recognition_time = time.time()
# # max_character_duration = 2.0  # 2 seconds for each character
# # word = ''

# # local_dictionary = {
# #     'hello': 'A common greeting',
# #     'world': 'The earth and all people and things on it',
# #     'example': 'A representative form or pattern',
# #     'ac': 'this is ac',
# #     'a': 'this is a'
# #     # Add more words and meanings as needed
# # }

# # time_difference = 0  # Initialize time_difference outside the loop

# # # Define a function to speak the word and its meaning
# # def speak_word(word, meaning):
# #     engine.say(word)
# #     # engine.say(meaning)
# #     engine.runAndWait()

# # while True:
# #     ret, frame = cap.read()

# #     if not ret:
# #         print("Error: Camera not found or could not be opened.")
# #         break

# #     H, W, _ = frame.shape

# #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# #     results = hands.process(frame_rgb)

# #     if results.multi_hand_landmarks:
# #         for hand_landmarks in results.multi_hand_landmarks:
# #             data_aux = []
# #             x_ = []
# #             y_ = []

# #             for i in range(len(hand_landmarks.landmark)):
# #                 x = hand_landmarks.landmark[i].x
# #                 y = hand_landmarks.landmark[i].y

# #                 x_.append(x)
# #                 y_.append(y)

# #             for i in range(len(hand_landmarks.landmark)):
# #                 x = hand_landmarks.landmark[i].x
# #                 y = hand_landmarks.landmark[i].y
# #                 data_aux.append(x - min(x_))
# #                 data_aux.append(y - min(y_))

# #             x1 = int(min(x_) * W) - 10
# #             y1 = int(min(y_) * H) - 10

# #             x2 = int(max(x_) * W) - 10
# #             y2 = int(max(y_) * H) - 10

# #             prediction = model.predict([np.asarray(data_aux)])
# #             predicted_character = labels_dict[int(prediction[0])]

# #             current_time = time.time()
# #             time_difference = current_time - last_recognition_time

# #             if time_difference >= max_character_duration:
# #                 if current_character:
# #                     recognized_characters.append(current_character)
# #                     word += current_character
# #                     current_character = ''

# #             if predicted_character != ' ':
# #                 current_character = predicted_character
# #                 last_recognition_time = current_time

# #             cv2.putText(frame, current_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

# #     key = cv2.waitKey(2)
# #     if key == 27:  # Press Esc key to exit
# #         break

# #     if time_difference > max_character_duration:
# #         if recognized_characters:
# #             full_word = ''.join(recognized_characters)
# #             recognized_characters = []
# #             if word:
# #                 meaning = local_dictionary.get(word.lower())
# #                 if meaning:
# #                     # Use threading to say the word and meaning without blocking the frame capture
# #                     threading.Thread(target=speak_word, args=(word, meaning)).start()
# #                 else:
# #                     engine.say(f"No meaning found for '{word}'")a
# #                     engine.runAndWait()
# #             word = ''

# #     cv2.imshow('Sign Language Recognition', frame)

# # cap.release()
# # cv2.destroyAllWindows()



# import time
# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np
# import pyttsx3
# import threading

# # Load the model
# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']

# # Initialize MediaPipe
# mp_hands = mp.solutions.hands # type: ignore
# mp_drawing = mp.solutions.drawing_utils # type: ignore
# mp_drawing_styles = mp.solutions.drawing_styles # type: ignore
# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# # Define labels for recognition
# labels_dict = {
#     0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G',
#     7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N',
#     14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
#     21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
#     26: '0', 27: '1', 28: '2', 29: '3', 30: '4',
#     31: '5', 32: '6', 33: '7', 34: '8', 35: '9',
#     36: ' i love you'  # Add a label for a space character
# }

# # Initialize the camera
# cap = cv2.VideoCapture(0)

# # Create a text-to-speech engine
# engine = pyttsx3.init()

# # Initialize variables for character recognition
# recognized_characters = []  # Initialize a list to store recognized characters
# current_character = ''
# last_recognition_time = time.time()
# max_character_duration = 1.0  # 2 seconds for each character
# word = ''

# local_dictionary = {'a': 'a', 'b': 'b', 'c': 'c', 'd': 'd', 'e': 'e', 'f': 'f', 'g': 'g', 'h': 'h', 'i': 'i', 'j': 'j', 'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'o': 'o', 'p': 'p', 'q': 'q', 'r': 'r', 's': 's', 't': 't', 'u': 'u', 'v': 'v', 'w': 'w', 'x': 'x', 'y': 'y', 'z': 'z', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9', 'i love you':'i love you'}
#     # Add more words and meanings as needed


# def speak_recognized_word(word):
#     meaning = local_dictionary.get(word.lower())
#     if meaning:
#         engine.say(word)
#         #engine.say(meaning)  #---------------------------------
        
#         engine.runAndWait()
#     else:
#         engine.say(f"No word found for '{word}'")
#         engine.runAndWait()

# while True:
    
#     #---------------------------------
#     current_time = time.time()  # Update the current time inside the loop
#     time_difference = current_time - last_recognition_time
    
#     #----------------------------
    
#     ret, frame = cap.read()

#     if not ret:
#         print("Error: Camera not found or could not be opened.")
#         break

#     H, W, _ = frame.shape

#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     results = hands.process(frame_rgb)   #-----------------------
#     if results.multi_hand_landmarks: 
#         for hand_landmarks in results.multi_hand_landmarks:
#             data_aux = []
#             x_ = []
#             y_ = []

#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y

#                 x_.append(x)
#                 y_.append(y)

#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 data_aux.append(x - min(x_))
#                 data_aux.append(y - min(y_))

#             x1 = int(min(x_) * W) - 10
#             y1 = int(min(y_) * H) - 10

#             x2 = int(max(x_) * W) - 10
#             y2 = int(max(y_) * H) - 10

#             # prediction = model.predict([np.array(data_aux)])
#             # prediction = model. predict(list(data_aux))
#             # prediction = model.predict([data_aux.reshape(1, -1)]) # type: ignore
#             prediction = model.predict(np.asarray(data_aux).reshape(1, -1))
            


#             # prediction = model.predict([np.asarray(data_aux)])
#             # prediction = model.predict([data_aux])  # Make sure the dimensionality of data_aux matches the model's requirements
#             # prediction = model.predict(data_aux)  
#             predicted_character = labels_dict[int(prediction[0])] ;print(f" predicetd char {predicted_character}")#----------------------

#             current_time = time.time()
#             time_difference = current_time - last_recognition_time

#             if time_difference >= max_character_duration:
#                 if current_character:
#                     recognized_characters.append(current_character)
#                     current_character = ''; print(f" current char {current_character}") #------------------------

#             if predicted_character != ' ':
#                 current_character = predicted_character
#                 last_recognition_time = current_time

#             cv2.putText(frame, current_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

#             if time_difference >= 2.0:  # Check for a gap of 4 seconds or more
#                 if recognized_characters: 
#                     print(f" list  {recognized_characters}") #-------------------------------------------------
#                     full_word = ''.join(recognized_characters)   ; print(f" full  Word {full_word}") #----------------
#                     recognized_characters = []  # Clear the list
#                     if full_word:
#                         threading.Thread(target=speak_recognized_word, args=(full_word,)).start()

#     key = cv2.waitKey(2)
#     if key == 27:  # Press Esc key to exit
#         break

#     cv2.imshow('Sign Language Recognition', frame)

# cap.release()
# cv2.destroyAllWindows()


import pickle

import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# labels_dict = {0: 'A', 1: 'B', 2: 'L'}
labels_dict = {
     0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G',
     7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N',
     14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
     21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
     26: '0', 27: '1', 28: '2', 29: '3', 30: '4',
     31: '5', 32: '6', 33: '7', 34: '8', 35: '9',
     36: ' i love you'  # Add a label for a space character
 }
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
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
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

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()