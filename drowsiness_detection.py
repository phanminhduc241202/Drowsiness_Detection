'''import cv2
import mediapipe as mp
import numpy as np
import time
def calculate_ear(eye_landmarks):
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear


def calculate_mar(mouth_landmarks):
    A = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[10])
    B = np.linalg.norm(mouth_landmarks[4] - mouth_landmarks[8])
    C = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[6])
    mar = (A + B) / (2.0 * C)
    return mar

# Thêm hàm tính toán góc giữa các điểm để phân tích chuyển động đầu
def calculate_head_movement(landmarks):
    nose_tip = np.array([landmarks[1].x, landmarks[1].y])
    chin = np.array([landmarks[152].x, landmarks[152].y])
    left_ear = np.array([landmarks[234].x, landmarks[234].y])
    right_ear = np.array([landmarks[454].x, landmarks[454].y])

    # Góc pitch (gật đầu)
    pitch = np.arctan2(nose_tip[1] - chin[1], nose_tip[0] - chin[0]) * 180 / np.pi

    # Góc yaw (quay đầu trái/phải)
    yaw = np.arctan2(left_ear[1] - right_ear[1], left_ear[0] - right_ear[0]) * 180 / np.pi

    return pitch, yaw

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

EAR_THRESHOLD = 0.2
MAR_THRESHOLD = 1.8
BLINK_TIME_THRESHOLD = 5  # Giới hạn thời gian nhắm mắt 2 giây
HEAD_PITCH_THRESHOLD = 80  # Ngưỡng gật đầu

blink_count = 0
eye_closed = False
mouth_open = False
mouth_count = 0
blink_start_time = None
sleepy_warning = False

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)
                )

                landmarks = face_landmarks.landmark

                left_eye_landmarks = np.array([
                    [landmarks[33].x, landmarks[33].y],
                    [landmarks[160].x, landmarks[160].y],
                    [landmarks[158].x, landmarks[158].y],
                    [landmarks[133].x, landmarks[133].y],
                    [landmarks[153].x, landmarks[153].y],
                    [landmarks[144].x, landmarks[144].y]
                ])

                right_eye_landmarks = np.array([
                    [landmarks[362].x, landmarks[362].y],
                    [landmarks[385].x, landmarks[385].y],
                    [landmarks[387].x, landmarks[387].y],
                    [landmarks[263].x, landmarks[263].y],
                    [landmarks[373].x, landmarks[373].y],
                    [landmarks[380].x, landmarks[380].y]
                ])

                mouth_landmarks = np.array([
                    [landmarks[61].x, landmarks[61].y],
                    [landmarks[146].x, landmarks[146].y],
                    [landmarks[91].x, landmarks[91].y],
                    [landmarks[181].x, landmarks[181].y],
                    [landmarks[84].x, landmarks[84].y],
                    [landmarks[17].x, landmarks[17].y],
                    [landmarks[314].x, landmarks[314].y],
                    [landmarks[405].x, landmarks[405].y],
                    [landmarks[318].x, landmarks[318].y],
                    [landmarks[402].x, landmarks[402].y],
                    [landmarks[323].x, landmarks[323].y],
                    [landmarks[293].x, landmarks[293].y]
                ])

                left_ear = calculate_ear(left_eye_landmarks)
                right_ear = calculate_ear(right_eye_landmarks)
                avg_ear = (left_ear + right_ear) / 2.0
                mar = calculate_mar(mouth_landmarks)

                cv2.putText(frame, f'EAR: {avg_ear:.2f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f'MAR: {mar:.2f}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if avg_ear < EAR_THRESHOLD:
                    if not eye_closed:
                        blink_start_time = time.time()  # Bắt đầu đếm thời gian nhắm mắt
                        blink_count += 1
                        eye_closed = True
                    cv2.putText(frame, "Mat nham!", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    if time.time() - blink_start_time > BLINK_TIME_THRESHOLD:
                        sleepy_warning = True  # Cảnh báo buồn ngủ
                else:
                    eye_closed = False
                    cv2.putText(frame, "Mat mo!", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if mar > MAR_THRESHOLD:
                    cv2.putText(frame, "Mieng dong!", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    mouth_open = False
                else:
                    if not mouth_open:
                        mouth_count += 1
                        mouth_open = True
                    cv2.putText(frame, "Mieng mo!", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Tính toán chuyển động đầu
                pitch, yaw = calculate_head_movement(landmarks)
                cv2.putText(frame, f'Pitch: {pitch:.2f}', (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'Yaw: {yaw:.2f}', (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if abs(pitch) < HEAD_PITCH_THRESHOLD:
                    sleepy_warning = True  # Cảnh báo buồn ngủ khi gật đầu

                # Cảnh báo buồn ngủ
                if sleepy_warning:
                    cv2.putText(frame, "Canh bao: Tai xe dang buon ngu!", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    cv2.putText(frame, "Dang trong trang thai theo doi!!!", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                cv2.putText(frame, f'Number of Blinks: {blink_count}', (30, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f'Number of open_mouth: {mouth_count}',(30, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Face Mesh", frame)        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()'''

import cv2
import mediapipe as mp
import numpy as np
import time

def calculate_ear(eye_landmarks):
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_mar(mouth_landmarks):
    A = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[10])
    B = np.linalg.norm(mouth_landmarks[4] - mouth_landmarks[8])
    C = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[6])
    mar = (A + B) / (2.0 * C)
    return mar

# Thêm hàm tính toán góc giữa các điểm để phân tích chuyển động đầu
def calculate_head_movement(landmarks):
    nose_tip = np.array([landmarks[1].x, landmarks[1].y])
    chin = np.array([landmarks[152].x, landmarks[152].y])
    left_ear = np.array([landmarks[234].x, landmarks[234].y])
    right_ear = np.array([landmarks[454].x, landmarks[454].y])

    # Góc pitch (gật đầu)
    pitch = np.arctan2(nose_tip[1] - chin[1], nose_tip[0] - chin[0]) * 180 / np.pi

    # Góc yaw (quay đầu trái/phải)
    yaw = np.arctan2(left_ear[1] - right_ear[1], left_ear[0] - right_ear[0]) * 180 / np.pi

    return pitch, yaw

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

EAR_THRESHOLD = 0.2
MAR_THRESHOLD = 1.8
BLINK_TIME_THRESHOLD = 5  # Giới hạn thời gian nhắm mắt 2 giây
HEAD_PITCH_THRESHOLD = 80  # Ngưỡng gật đầu

blink_count = 0
eye_closed = False
mouth_open = False
mouth_count = 0
blink_start_time = None
sleepy_warning = False
max_blink_time = 0  # Thời gian tối đa mắt nhắm

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)
                )

                landmarks = face_landmarks.landmark

                left_eye_landmarks = np.array([
                    [landmarks[33].x, landmarks[33].y],
                    [landmarks[160].x, landmarks[160].y],
                    [landmarks[158].x, landmarks[158].y],
                    [landmarks[133].x, landmarks[133].y],
                    [landmarks[153].x, landmarks[153].y],
                    [landmarks[144].x, landmarks[144].y]
                ])

                right_eye_landmarks = np.array([
                    [landmarks[362].x, landmarks[362].y],
                    [landmarks[385].x, landmarks[385].y],
                    [landmarks[387].x, landmarks[387].y],
                    [landmarks[263].x, landmarks[263].y],
                    [landmarks[373].x, landmarks[373].y],
                    [landmarks[380].x, landmarks[380].y]
                ])

                mouth_landmarks = np.array([
                    [landmarks[61].x, landmarks[61].y],
                    [landmarks[146].x, landmarks[146].y],
                    [landmarks[91].x, landmarks[91].y],
                    [landmarks[181].x, landmarks[181].y],
                    [landmarks[84].x, landmarks[84].y],
                    [landmarks[17].x, landmarks[17].y],
                    [landmarks[314].x, landmarks[314].y],
                    [landmarks[405].x, landmarks[405].y],
                    [landmarks[318].x, landmarks[318].y],
                    [landmarks[402].x, landmarks[402].y],
                    [landmarks[323].x, landmarks[323].y],
                    [landmarks[293].x, landmarks[293].y]
                ])

                left_ear = calculate_ear(left_eye_landmarks)
                right_ear = calculate_ear(right_eye_landmarks)
                avg_ear = (left_ear + right_ear) / 2.0
                mar = calculate_mar(mouth_landmarks)

                cv2.putText(frame, f'EAR: {avg_ear:.2f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f'MAR: {mar:.2f}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if avg_ear < EAR_THRESHOLD:
                    if not eye_closed:
                        blink_start_time = time.time()  # Bắt đầu đếm thời gian nhắm mắt
                        blink_count += 1
                        eye_closed = True
                    cv2.putText(frame, "Mat nham!", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    blink_time = time.time() - blink_start_time
                    if blink_time > max_blink_time:
                        max_blink_time = blink_time  # Cập nhật thời gian nhắm mắt tối đa

                    if blink_time > BLINK_TIME_THRESHOLD:
                        sleepy_warning = True  # Cảnh báo buồn ngủ
                else:
                    eye_closed = False
                    cv2.putText(frame, "Mat mo!", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if mar > MAR_THRESHOLD:
                    cv2.putText(frame, "Mieng dong!", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    mouth_open = False
                else:
                    if not mouth_open:
                        mouth_count += 1
                        mouth_open = True
                    cv2.putText(frame, "Mieng mo!", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Tính toán chuyển động đầu
                pitch, yaw = calculate_head_movement(landmarks)
                cv2.putText(frame, f'Pitch: {pitch:.2f}', (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'Yaw: {yaw:.2f}', (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if abs(pitch) < HEAD_PITCH_THRESHOLD:
                    sleepy_warning = True  # Cảnh báo buồn ngủ khi gật đầu

                # Cảnh báo buồn ngủ
                if sleepy_warning:
                    cv2.putText(frame, "Canh bao: Tai xe dang buon ngu!", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    cv2.putText(frame, "Dang trong trang thai theo doi!!!", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                cv2.putText(frame, f'Number of Blinks: {blink_count}', (30, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f'Max blink time: {max_blink_time:.2f}s', (30, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f'Number of Blinks: {mouth_count}', (30, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Face Mesh", frame)        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
