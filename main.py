import cv2
import mediapipe as mp
import math
import numpy as np
import subprocess
import platform

# Windows-specific imports
if platform.system() == 'Windows':
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize Mediapipe drawing and hands utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Webcam Setup
wCam, hCam = 640, 480
cam = cv2.VideoCapture(0)
cam.set(3, wCam)
cam.set(4, hCam)

# Volume control via osascript
def change_volume(vol_per):
    """Change system volume based on the OS"""
    vol = int(np.interp(vol_per, [0, 100], [0, 100]))
    if platform.system() == 'Darwin':  # macOS
        subprocess.call(["osascript", "-e", f"set volume output volume {vol} --100%"])
    elif platform.system() == 'Windows':  # Windows
        # Windows volume control using PyCaw
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volume_range = volume.GetVolumeRange()
        min_vol = volume_range[0]
        max_vol = volume_range[1]
        volume.SetMasterVolumeLevel(np.interp(vol_per, [0, 100], [min_vol, max_vol]), None)

# Pause/Play media using osascript
def pause_audio():
    """Pause or play audio on macOS using osascript"""
    subprocess.call(["osascript", "-e", "tell application \"System Events\" to key code 16"])


# Mediapipe Hand Landmark Model
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cam.isOpened():
        success, image = cam.read()

        # Check if the frame was properly captured
        if not success:
            print("Failed to capture image from webcam.")
            continue  # Skip the rest of the loop if no image is captured

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # Get hand landmarks position
        lmList = []
        if results.multi_hand_landmarks:
            myHand = results.multi_hand_landmarks[0]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

        # Check for thumb and index finger positions for volume control
        if len(lmList) != 0:
            x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tip
            x2, y2 = lmList[8][1], lmList[8][2]  # Index finger tip

            # Mark Thumb and Index finger
            cv2.circle(image, (x1, y1), 15, (255, 255, 255), -1)
            cv2.circle(image, (x2, y2), 15, (255, 255, 255), -1)
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Calculate the distance between the thumb and index finger
            length = math.hypot(x2 - x1, y2 - y1)

            # Set volume based on the length (distance) between thumb and index finger
            volPer = np.interp(length, [50, 220], [0, 100])
            change_volume(volPer)

            # Volume Bar
            volBar = np.interp(length, [50, 220], [400, 150])
            cv2.rectangle(image, (50, 150), (85, 400), (0, 0, 0), 3)
            cv2.rectangle(image, (50, int(volBar)), (85, 400), (0, 0, 0), cv2.FILLED)
            cv2.putText(image, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)

            # Detect if open palm (all fingers extended)s
            fingers = []
            # Thumb
            if lmList[4][1] > lmList[3][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            # Other fingers
            for id in range(8, 21, 4):
                if lmList[id][2] < lmList[id - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            
            # If all fingers are up, it's an open palm gesture
            if fingers.count(1) == 5:
                pause_audio()  # Call the function to pause/play audio
                cv2.putText(image, 'Audio Paused', (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # Display the image
        cv2.imshow('Gesture Volume Control', image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close windows
cam.release()
cv2.destroyAllWindows()
