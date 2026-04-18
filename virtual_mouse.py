import cv2
import mediapipe as mp
import utili
import pyautogui
import random
from  pynput.mouse import Button , Controller


mouse = Controller()
screen_width , screen_height = pyautogui.size()


mphands = mp.solutions.hands
hands = mphands.Hands(
    static_image_mode = False , 
    model_complexity = 1 , 
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.7,
    max_num_hands = 1
)


def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmark= processed.multi_hand_landmarks[0]
        return hand_landmark.landmark[mphands.HandLandmark.INDEX_FINGER_TIP]
    

def move_mouse(index_finger_tip):
    if index_finger_tip is not None:
        x = int(index_finger_tip.x * screen_width)
        y = int(index_finger_tip.y * screen_height)
        pyautogui.moveTo(x, y , duration=0.1)


def is_left_click(landmark_list , thumb_index_dist):
    return(
        utili.get_angle(landmark_list[5] , landmark_list[6] , landmark_list[8]) < 50 and
        utili.get_angle(landmark_list[9] , landmark_list[10] , landmark_list[12]) > 90 and
        thumb_index_dist > 50
    )


def is_right_click(landmark_list , thumb_index_dist):
    return(
        utili.get_angle(landmark_list[9] , landmark_list[10] , landmark_list[12]) < 50 and
        utili.get_angle(landmark_list[5] , landmark_list[6] , landmark_list[8])>90 and
        thumb_index_dist > 50
    )


def is_double_click(landmark_list , thumb_index_dist):
    return(
        utili.get_angle(landmark_list[5], landmark_list[6] , landmark_list[8]) <50 and 
        utili.get_angle(landmark_list[9] , landmark_list[10] , landmark_list[12]) < 50 and 
        thumb_index_dist > 50
    )

def is_screenshot(landmark_list):
    return(
        utili.get_angle(landmark_list[1] , landmark_list[2] , landmark_list[4]) > 140 and
        utili.get_angle(landmark_list[5] , landmark_list[6] , landmark_list[8]) < 70 and
        utili.get_angle(landmark_list[9] , landmark_list[10] , landmark_list[12]) < 70 and
        utili.get_angle(landmark_list[13] , landmark_list[14] , landmark_list[16]) < 70 and
        utili.get_angle(landmark_list[17] , landmark_list[18] , landmark_list[20]) > 140 
    )

def is_scroll(landmark_list):
    return(
        utili.get_angle(landmark_list[5] , landmark_list[6] , landmark_list[8]) > 160 and
        utili.get_angle(landmark_list[9] , landmark_list[10] , landmark_list[12]) > 160 and
        utili.get_angle(landmark_list[13] , landmark_list[14] , landmark_list[16]) > 160 and
        utili.get_angle(landmark_list[17] , landmark_list[18] , landmark_list[20]) > 160  
    )

def detect_gesture(frame , landmarks_list , processed):
    if len(landmarks_list) >=21 :
        index_finger_tip = find_finger_tip(processed)
        thumb_index_dist = utili.get_distance([landmarks_list[4] , landmarks_list[5]])

        if is_scroll(landmarks_list):
            if index_finger_tip.y < 0.4:
                pyautogui.scroll(100)
                cv2.putText(frame, "scroll up" , (50 ,50) , cv2.FONT_HERSHEY_SIMPLEX, 1 , (0,255,255) , 2)

            elif index_finger_tip.y > 0.6:
                pyautogui.scroll(-100)
                cv2.putText(frame, "scroll down" , (50 ,50) , cv2.FONT_HERSHEY_SIMPLEX, 1 , (0,255,255) , 2)
        
        elif thumb_index_dist < 50 and utili.get_angle(landmarks_list[5] , landmarks_list[6] , landmarks_list[8]) > 90:
            move_mouse(index_finger_tip)
        
        elif is_left_click(landmarks_list , thumb_index_dist):
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "left click" , (50 ,50) , cv2.FONT_HERSHEY_SIMPLEX, 1 , (0,255,0) , 2)

        elif is_right_click(landmarks_list , thumb_index_dist):
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "right click" , (50 ,50) , cv2.FONT_HERSHEY_SIMPLEX, 1 , (0,0,255) ,2)

        elif is_double_click(landmarks_list , thumb_index_dist):
           pyautogui.doubleClick()
           cv2.putText(frame, "double click" , (50 ,50) , cv2.FONT_HERSHEY_SIMPLEX, 1 , (0,255,255) , 2)

        elif is_screenshot(landmarks_list ):
            iml = pyautogui.screenshot()
            label = random.randint(1,1000)
            iml.save(f"ss_{label}.png")
            cv2.putText(frame, "Screenshot taken" , (50 ,50) , cv2.FONT_HERSHEY_SIMPLEX, 1 , (0,255,255) , 2)
            print(f"SS saved as : ss_{label}.png")


def main():
    cap = cv2.VideoCapture(0)
    draw = mp.solutions.drawing_utils

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame , 1)
            frameRGB  = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)
            landmarks_list = []


            if processed.multi_hand_landmarks:
                hands_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame , hands_landmarks , mphands.HAND_CONNECTIONS)
                for lm in hands_landmarks.landmark:
                    landmarks_list.append((lm.x , lm.y))

            
            detect_gesture(frame, landmarks_list ,processed)
            cv2.imshow("virtual_mouse" , frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()



