import mediapipe as mp
import cv2
import torch
import torch.nn as nn
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Define a simple CNN for gesture classification
class SimpleGestureCNN(nn.Module):
    def __init__(self):
        super(SimpleGestureCNN, self).__init__()
        self.fc1 = nn.Linear(21 * 3, 64)  # 21 keypoints * 3 coordinates (x, y, z)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 5)  # Assuming 5 gesture classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = SimpleGestureCNN()

# Capture video and process hand landmarks
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.append([lm.x, lm.y, lm.z])
            keypoints = np.array(keypoints).flatten()  # Flatten to 1D

            # Convert to tensor and classify gesture
            with torch.no_grad():
                input_tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0)
                gesture_prediction = model(input_tensor)
                predicted_class = torch.argmax(gesture_prediction, dim=1).item()
                
                # Display gesture on screen
                cv2.putText(frame, f'Gesture: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
