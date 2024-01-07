import cv2
import numpy as np
import os
# from matplotlib import pyplot as plt
import mediapipe as mp
import tensorflow as tf
# print("done")

def mediapipe_detection(image, model):
    if(image is not None):
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        return
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    print("a")
    return image, results


    
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    print(len(pose),len(face),len(lh),len(rh))
    return np.concatenate([pose, face, lh, rh])



def translatee(videoo):
    model=tf.keras.saving.load_model("C:\\Users\\sumen\\Downloads\\sign_translation\\sign_translation\\utils\\singlemodel_withvalloss_doubled_28.keras", custom_objects=None, compile=True, safe_mode=True)
    sequence = []
    predictions = []
    threshold = 0.5
    fc=0
    prr={}
    actions=np.load("sign_translation\\utils\\actions.npy")
    os.chdir('sign_translation\\utils')
    cap = cv2.VideoCapture(videoo) 
    mp_holistic = mp.solutions.holistic # Holistic model
# Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=threshold) as holistic:
        while cap.isOpened():
            # print("enter")

            # Read feed
            ret, frame = cap.read()
            if frame is None :
                break
            fc+=1
            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30 :
                if fc% 30==0 and fc%60 !=0:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])
                    # if fc % 15 ==0:

                    predictions.append(actions[np.argmax(res)])
                    prr[fc]=actions[np.argmax(res)]            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        # print(predictions)
    
    return(predictions)
