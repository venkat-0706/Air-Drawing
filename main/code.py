import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands with better detection/tracking confidence
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Create drawing canvas
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

# Colors and drawing settings
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Red, Green, Blue
color_names = ['Red', 'Green', 'Blue']
color_index = 0
brush_thickness = 20
eraser_thickness = 70

# Palette box positions
palette_positions = [(50, 10), (150, 10), (250, 10)]
box_size = 50

# Helper to detect fingers up
def fingers_up(lm_list):
    tips = [8, 12, 16, 20]
    fingers = []
    fingers.append(1 if lm_list[4][0] < lm_list[3][0] else 0)  # Thumb
    for tip in tips:
        fingers.append(1 if lm_list[tip][1] < lm_list[tip - 2][1] else 0)
    return fingers

# Start capturing
cap = cv2.VideoCapture(0)
prev_x, prev_y = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (1280, 720))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    index_x, index_y = 0, 0
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if lm_list:
                finger_status = fingers_up(lm_list)
                index_x, index_y = lm_list[8]  # Index fingertip

                # Color selection by hovering
                for i, (px, py) in enumerate(palette_positions):
                    if px < index_x < px + box_size and py < index_y < py + box_size:
                        color_index = i
                        cv2.putText(frame, f"Color: {color_names[i]}", (10, 160),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, colors[color_index], 2)

                # ✍️ Draw if only index finger is up
                if finger_status == [0, 1, 0, 0, 0]:
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = index_x, index_y
                    cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), colors[color_index], brush_thickness)
                    prev_x, prev_y = index_x, index_y

                # 🧽 Erase if all fingers are down
                elif sum(finger_status) == 0:
                    cv2.circle(canvas, (index_x, index_y), eraser_thickness, (0, 0, 0), -1)
                    cv2.putText(frame, "Erasing", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    prev_x, prev_y = 0, 0
                else:
                    prev_x, prev_y = 0, 0

    # Draw color palette
    for i, (x, y) in enumerate(palette_positions):
        cv2.rectangle(frame, (x, y), (x + box_size, y + box_size), colors[i], -1)
        if i == color_index:
            cv2.rectangle(frame, (x, y), (x + box_size, y + box_size), (255, 255, 255), 2)

    # Overlay canvas on frame
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv_mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
    inv_mask = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, inv_mask)
    frame = cv2.bitwise_or(frame, canvas)

    # Instructions
    cv2.putText(frame, "Press Q to Quit", (1000, 700),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Display
    cv2.imshow("Hand Gesture Drawing", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
