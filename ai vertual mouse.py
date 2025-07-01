import streamlit as st
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import tempfile
import time
from PIL import Image
import os

# ------------------ Utility Functions ------------------
def get_angle(a, b, c):
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle

def get_distance(landmark_list):
    if len(landmark_list) < 2:
        return
    (x1, y1), (x2, y2) = landmark_list[0], landmark_list[1]
    L = np.hypot(x2 - x1, y2 - y1)
    return np.interp(L, [0, 1], [0, 1000])

# ------------------ Virtual Mouse Class ------------------
class VirtualMouse:
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=1
        )
        self.current_mode = "Normal"
        self.last_gesture_time = 0
        self.gesture_cooldown = 1.0
        self.draw = mp.solutions.drawing_utils

    def find_finger_tip(self, processed):
        if processed.multi_hand_landmarks:
            hand_landmarks = processed.multi_hand_landmarks[0]
            return hand_landmarks.landmark[self.mpHands.HandLandmark.INDEX_FINGER_TIP]
        return None

    def move_mouse(self, index_finger_tip):
        if index_finger_tip is not None:
            x = int(index_finger_tip.x * self.screen_width)
            y = int(index_finger_tip.y / 2 * self.screen_height)
            pyautogui.moveTo(x, y)

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = self.hands.process(frameRGB)

        landmark_list = []
        if processed.multi_hand_landmarks:
            hand_landmarks = processed.multi_hand_landmarks[0]
            self.draw.draw_landmarks(frame, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                landmark_list.append((lm.x, lm.y))

        self.detect_gesture(frame, landmark_list, processed)
        return frame

    def detect_gesture(self, frame, landmark_list, processed):
        if len(landmark_list) >= 21:
            index_finger_tip = self.find_finger_tip(processed)
            thumb_index_dist = get_distance([landmark_list[4], landmark_list[5]]) if len(landmark_list) > 5 else 100
            current_time = time.time()
            if current_time - self.last_gesture_time < self.gesture_cooldown:
                return

            if self.current_mode == "Normal":
                if get_distance([landmark_list[4], landmark_list[5]]) < 50 and \
                   get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90:
                    self.move_mouse(index_finger_tip)

                if get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and \
                   get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90 and \
                   thumb_index_dist > 50:
                    pyautogui.click()
                    self.last_gesture_time = current_time
                    cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                elif get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and \
                     get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90 and \
                     thumb_index_dist > 50:
                    pyautogui.rightClick()
                    self.last_gesture_time = current_time
                    cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                elif get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and \
                     get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and \
                     thumb_index_dist > 50:
                    pyautogui.doubleClick()
                    self.last_gesture_time = current_time
                    cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                elif get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and \
                     get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and \
                     thumb_index_dist < 50:
                    screenshot = pyautogui.screenshot()
                    screenshot_path = os.path.join(tempfile.gettempdir(), f'screenshot_{int(time.time())}.png')
                    screenshot.save(screenshot_path)
                    self.last_gesture_time = current_time
                    cv2.putText(frame, "Screenshot Taken", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    st.session_state.last_screenshot = screenshot_path

            elif self.current_mode == "Presentation":
                index_tip_x = landmark_list[8][0] if len(landmark_list) > 8 else 0.5
                if index_tip_x > 0.8:
                    pyautogui.press('right')
                    self.last_gesture_time = current_time
                    cv2.putText(frame, "Next Slide", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255), 2)
                elif index_tip_x < 0.2:
                    pyautogui.press('left')
                    self.last_gesture_time = current_time
                    cv2.putText(frame, "Previous Slide", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255), 2)

            elif self.current_mode == "Gaming":
                if get_angle(landmark_list[2], landmark_list[3], landmark_list[4]) < 30:
                    pyautogui.press('space')
                    self.last_gesture_time = current_time
                    cv2.putText(frame, "Jump", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

# ------------------ Streamlit App ------------------
def main():
    st.set_page_config(page_title="AI Virtual Mouse", page_icon="🖱️", layout="wide")
    st.title("🖱️ AI Virtual Mouse Control")
    st.markdown("Control your computer with hand gestures using AI")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Configuration")
        mode = st.radio("Select Mode", ["Normal", "Presentation", "Gaming"], index=0)
        sensitivity = st.slider("Mouse Sensitivity", 0.5, 2.0, 1.0)
        gesture_cooldown = st.slider("Gesture Cooldown (seconds)", 0.1, 2.0, 1.0, step=0.1)

        st.subheader("Instructions")
        st.markdown("""
        - **Normal Mode**: 
          - 👆 Point to move cursor
          - 🤟 Left click
          - 🤘 Right click
          - ✌️ Double click
          - 🤏 Screenshot
        - **Presentation Mode**:
          - 👉 Swipe right for next slide
          - 👈 Swipe left for previous slide
        - **Gaming Mode**:
          - 👍 Thumbs up to jump (space)
        """)

        if 'last_screenshot' in st.session_state:
            try:
                st.image(st.session_state.last_screenshot, caption="Last Screenshot", use_container_width=True)
            except:
                pass

    with col2:
        st.subheader("Live Camera Feed")
        run = st.checkbox("Enable Virtual Mouse", value=False)

        if run:
            virtual_mouse = VirtualMouse()
            virtual_mouse.current_mode = mode
            virtual_mouse.gesture_cooldown = gesture_cooldown

            FRAME_WINDOW = st.image([])

            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            while run:
                ret, frame = camera.read()
                if not ret:
                    st.error("Failed to capture video")
                    break

                frame = virtual_mouse.process_frame(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (1280, 720))  # Resize to fit wider window
                FRAME_WINDOW.image(frame, use_container_width=True)

            camera.release()
        else:
            st.info("Enable the checkbox to start the virtual mouse")

if __name__ == "__main__":
    main()
