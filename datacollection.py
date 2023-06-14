import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

#number_of_classes =  # 0:'A' , 1:'B' , 2:'C' ,3:'D' ,4:'E' ,5:'F'
                      # 6:'G' ,  7:'H' , 8:'I' ,  9:'J' ,  10 :'K', 11:'L'
                      # 12: 'M' , 13:'N' ,14:'O' ,15:'P' ,16:'Q' ,17:'R'
                      # 18:'S' , 19:'T' ,20:'U' ,21:'V' ,22:'W' ,23:'X'
                      #24:'Y',25:'Z'


#26:'Hello' ,  27:'me' , 28:'You', 29:'My' ,30:'Name' ,31:'How' , 32:'nice' ,33:'meet' ,34:'You',35:'Goodbye',36:'help'


# # 0:'A' , 1:'B' , 2:'C' ,3:'D' ,4:'E' ,5:'F' ,6:'G' ,  7:'H' , 8:'I' ,  9:'J' ,  10 :'K', 11:'L' ,26:'Hello' , 27:'thank you' , 28:'name' ,29:'where' , 30:'how' ,31:'what' , 32:'I' , 33:'my',34:"goodbye',35:'fine',36:'help',37:'You' , 38:'need' ,39:'no' ,40:'whatsupp',41:'Take care}
number_of_classes = 6  # 'what' ,'help' ,'name'
dataset_size = 400

cap = cv2.VideoCapture(0)
for j in range(42,43):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "s" to start capturing or "q" to quit.', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(25)
        if key == ord('s'):
            break
        elif key == ord('q'):
            done = True
            break

    if done:
        break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        key = cv2.waitKey(25)
        if key == ord('p'):
            cv2.putText(frame, 'Paused. Press any key to resume.', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            cv2.waitKey(0)  # Wait until any key is pressed to resume capturing
        elif key == ord('q'):
            done = True
            break

        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
        counter += 1
        print(counter)

    if done:
        break

cap.release()
cv2.destroyAllWindows()
