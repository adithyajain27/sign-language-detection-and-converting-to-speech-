
import os
import cv2
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
number_of_classes = 60
dataset_size = 100
class_count = 0  # Initialize a variable to track the class count
# Initialize the camera
cap = cv2.VideoCapture(0)  # Change the camera index to  work on cameras

if not cap.isOpened():
    print("Error: Camera not opened. Check the camera index.")
    exit(1)

# Set the frame size (you can adjust these values)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Create a named window with specific size and title

cv2.namedWindow('Capturing the image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Capturing the image', 800, 600)  # Adjust the size as needed

while class_count < number_of_classes:  # Limit collection to a specific number of classes
    if not os.path.exists(os.path.join(DATA_DIR, str(class_count))):
        os.makedirs(os.path.join(DATA_DIR, str(class_count)))

    # print('Collecting data for class {}'.format(class_count))
    print(f'Collecting data for class {class_count}')

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? \n Press "y"-> start \n "Esc"-> end ', (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3, cv2.LINE_AA) #(255, 0, 0) is bgr 0.8 is font size
        cv2.imshow('Capturing the image', frame)
        key = cv2.waitKey(250)  # this is 25milisec for taking a single photo vary it to making speed or slow now is 25mili / 1000 mili i.e 1 sec
        if key == ord('y'):
            break
        elif key == 27:  # 27 is the ASCII code for the 'esc' key
            cv2.destroyAllWindows()
            exit(0)  # Exit the program when 'Esc' is pressed

    counter = 0
    # capture_interval = 2000  # Set a delay of 2000 milliseconds (2 seconds) per capture
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('Capturing the image', frame)
        key = cv2.waitKey(250)
        if key == ord('y'):
            break
        elif key == 27:  # 27 is the ASCII code for the 'esc' key
            cv2.destroyAllWindows()
            exit(0)  # Exit the program when 'Esc' is pressed
        cv2.imwrite(os.path.join(DATA_DIR, str(class_count), '{}.jpg'.format(counter)), frame)
        counter += 1

    class_count += 1

# Release the camera and destroy the window when done
cap.release()
cv2.destroyAllWindows()
