# import mediapipe as mp
# import cv2
# import sys
# import numpy as np
# import serial

# counter = 0
# arm_down = True
# arduino = serial.Serial("COM3", 9600, timeout=0.1) 

# cap1 = cv2.VideoCapture(0)
# cap1.set(3, 1020) #가로
# cap1.set(4, 800) #세로

# fps = cv2.CAP_PROP_FPS
# delay_time = 1/fps

# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils
# mp = mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)

# def calculate_angle(a, b, c):
#     a = np.array(a)  # First
#     b = np.array(b)  # Mid
#     c = np.array(c)  # End
    
#     radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
#     angle = np.abs(radians * 180.0 / np.pi)
    
#     if angle > 180.0:
#         angle = 360 - angle
        
#     return angle

# def map_value(value, in_min, in_max, out_min, out_max):
#     return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
# min_angle = 0  # 최소 각도
# max_angle = 180  # 최대 각도
# min_mapped_value = 0  # 최소 매핑 값
# max_mapped_value = 180  # 최대 매핑 값

# if not cap1.isOpened():
#     print("Could not open a Camera!")
#     sys.exit()

# # print(img1.shape) (height, width, channel)

# while True:
    
#     ret1, img1 = cap1.read()
#     image = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
#     image.flags.writeable = False
#     result = mp.process(image)
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
#     try:
#         landmarks = result.pose_landmarks.landmark
#         # mp_drawing.draw_landmarks(img1, landmark, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec((255,0,0), 2, 2))
#         # elbow_x = (landmarks[13].x * img1.shape[1])
#         # elbow_y = (landmarks[13].y * img1.shape[0])
#         # print("hello")
#         # print(shoulder_x)
#         # print(shoulder_y)
#         # shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        
#         # cv2.circle(img1, (int(elbow_x), int(elbow_y)), 10, (0, 0, 255), 5)
        
#         shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
#         elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
#         wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        
#         shoulder_pos = tuple(np.multiply(shoulder, [640, 480]).astype(int))
#         elbow_pos = tuple(np.multiply(elbow, [640, 480]).astype(int))
#         wrist_pos = tuple(np.multiply(wrist, [640, 480]).astype(int))

#         cv2.circle(image, shoulder_pos, 10, (255,0,0), -1)
#         cv2.circle(image, elbow_pos, 10, (0,255,0), -1)
#         cv2.circle(image, wrist_pos, 10, (0,0,255), -1)             
#         cv2.line(image, shoulder_pos, elbow_pos,   (255,255,0), 3)
#         cv2.line(image, elbow_pos, wrist_pos, (0,255,255), 3)
        
#         # 어깨와 팔 사이의 각도 계산하기
#         angle = calculate_angle(shoulder, elbow, wrist)
#         # 어깨 각도 구하기
#         shoulder_angle = calculate_angle(elbow, shoulder, [shoulder[0], shoulder[1] - 0.1])
#         # 손목 각도 구하기
#         wrist_angle = calculate_angle(elbow, wrist, [wrist[0], wrist[1] - 0.1])
#         # 각도 표시하기
#         cv2.putText(image, f'Elbow-Shoulder Angle: {round(angle, 2)}',
#                     tuple(np.multiply(elbow, [640, 480]).astype(int)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
#                     cv2.LINE_AA)
#         cv2.putText(image, f'Shoulder Angle: {round(shoulder_angle, 2)}',
#                     tuple(np.multiply(shoulder, [640, 480]).astype(int)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
#                     cv2.LINE_AA)
#         cv2.putText(image, f'Wrist Angle: {round(wrist_angle, 2)}',
#                     tuple(np.multiply(wrist, [640, 480]).astype(int)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
#                     cv2.LINE_AA)
#         # 팔이 일정 각도 이하로 내려갔을 때 카운터를 증가시킵니다.
#         if angle <= 45 and not arm_down:
#             counter += 1
#             arm_down = True
            
#         if angle > 45 :
#             arm_down = False 
#         # 카운터를 화면에 표시합니다.
#         cv2.putText(image, f'Count: {counter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
#         # 아두이노에 각도 정보를 전송합니다.
#         arduino.write(f'{angle}\n'.encode())
        
#     except:
#         pass
    
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break
#     cv2.imshow("final",img1)
    
# cap1.release()
# cv2.destroyAllWindows()





import mediapipe as mp  # Mediapipe 라이브러리를 불러옵니다.
import cv2  # OpenCV 라이브러리를 불러옵니다.
import numpy as np  # NumPy 라이브러리를 불러옵니다. 이미지 처리를 위해 사용됩니다.
import serial  # 시리얼 통신 라이브러리를 불러옵니다.
import time  # 시간 관련 라이브러리를 불러옵니다.
import sys

# 카운터 및 팔의 상태 변수를 초기화합니다.
counter = 0
arm_down = True

# Mediapipe에서 포즈 관련 모듈 및 도구를 가져옵니다.
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 포즈 추정을 위한 객체를 생성하며, 최소 감지 신뢰도 및 최소 추적 신뢰도를 설정합니다.
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 아두이노와의 시리얼 통신을 설정합니다. (COM 포트 및 통신 속도)
arduino = serial.Serial('COM3', 9600, timeout=0.1)

# 카메라를 통해 영상을 캡처하기 위한 VideoCapture 객체를 생성합니다.
cap = cv2.VideoCapture(0)

# 카메라의 해상도를 설정합니다.
cap.set(3, 1020)
cap.set(4, 800)

def calculate_angle(a, b, c):
    # 모든 점이 제대로 존재하는지 확인합니다.
    if a is None or b is None or c is None:
        return None
    
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# 카메라가 정상적으로 열리지 않았을 경우 에러 메시지를 출력하고 프로그램을 종료합니다.
if not cap.isOpened():
    print("Camera is not detected")
    sys.exit()

# 영상을 처리하는 루프를 시작합니다.
while cap.isOpened():
    # 카메라에서 프레임을 읽어옵니다.
    ret, frame = cap.read()
    
    # RGB 이미지로 변환합니다.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Mediapipe를 사용하여 포즈를 추정합니다.
    results = pose.process(image)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    try:
        # 추정된 포즈의 랜드마크들을 가져옵니다.
        landmarks = results.pose_landmarks.landmark
        
        # 각 관절의 위치를 추출합니다.
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        # 각 관절의 화면 상의 위치를 픽셀로 변환합니다.
        shoulder_pos = tuple(np.multiply(shoulder, [640, 480]).astype(int))
        elbow_pos = tuple(np.multiply(elbow, [640, 480]).astype(int))
        wrist_pos = tuple(np.multiply(wrist, [640, 480]).astype(int))

        # 관절 위치를 시각화합니다.
        cv2.circle(image, shoulder_pos, 10, (255,0,0), -1)
        cv2.circle(image, elbow_pos, 10, (0,255,0), -1)
        cv2.circle(image, wrist_pos, 10, (0,0,255), -1)             
        cv2.line(image, shoulder_pos, elbow_pos,   (255,255,0), 3)
        cv2.line(image, elbow_pos, wrist_pos, (0,255,255), 3)
        
        # 관절 각도를 계산합니다.
        angle = calculate_angle(shoulder, elbow, wrist)
        
        # 화면에 각도를 표시합니다.
        cv2.putText(image, f'Angle: {round(angle, 2)}', tuple(np.multiply(elbow, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 팔이 일정 각도 이하로 내려갔을 때 카운터를 증가시킵니다.
        if angle <= 45 and not arm_down:
            counter += 1
            arm_down = True
            
        if angle > 45 :
            arm_down = False 

        # 카운터를 화면에 표시합니다.
        cv2.putText(image, f'Count: {counter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # 아두이노에 각도 정보를 전송합니다.
        arduino.write(f'{angle}\n'.encode())
    except:
        pass
    
    # 프레임에 추정된 포즈를 시각화하여 화면에 표시합니다.
    cv2.imshow('Mediapipe Feed', image)

    # 'q' 키를 누르면 루프를 종료합니다.
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# 루프를 빠져나오면 카메라를 해제합니다.
cap.release()

# 모든 창을 닫습니다.
cv2.destroyAllWindows()


# 참고 ( 같은 내용인데 이건 안됨 왜일까?)
# import cv2
# import mediapipe as mp
# import numpy as np
# import serial
# import time
# counter = 0
# arm_down = True
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose
# # 아두이노와의 시리얼 통신을 설정합니다.
# arduino = serial.Serial('COM4', 115200, timeout=0.1)
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)
# def calculate_angle(a, b, c):
#     a = np.array(a)  # First
#     b = np.array(b)  # Mid
#     c = np.array(c)  # End
#     radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
#     angle = np.abs(radians * 180.0 / np.pi)
#     if angle > 180.0:
#         angle = 360 - angle
#     return angle
# def map_value(value, in_min, in_max, out_min, out_max):
#     return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
# min_angle = 0  # 최소 각도
# max_angle = 180  # 최대 각도
# min_mapped_value = 0  # 최소 매핑 값
# max_mapped_value = 180  # 최대 매핑 값
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         # RGB 이미지로 변환하기
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False
#         results = pose.process(image)
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#         try:
#             landmarks = results.pose_landmarks.landmark
#             # 각 관절의 위치점 찾기
#             shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
#                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
#             elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
#                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
#             wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
#                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
#             shoulder_pos = tuple(np.multiply(shoulder, [640, 480]).astype(int))
#             elbow_pos = tuple(np.multiply(elbow, [640, 480]).astype(int))
#             wrist_pos = tuple(np.multiply(wrist, [640, 480]).astype(int))
#             # 포인트 비주얼 처리
#             cv2.circle(image, shoulder_pos, 10, (255,0,0), -1)
#             cv2.circle(image, elbow_pos, 10, (0,255,0), -1)
#             cv2.circle(image, wrist_pos, 10, (0,0,255), -1)
#             cv2.line(image, shoulder_pos, elbow_pos,   (255,255,0), 3)
#             cv2.line(image, elbow_pos, wrist_pos, (0,255,255), 3)
#             # 어깨와 팔 사이의 각도 계산하기
#             shoulder_angle = calculate_angle(elbow, shoulder, [shoulder[0], shoulder[1] - 0.1])
#             # 팔꿈치 각도 구하기
#             elbow_angle = calculate_angle(shoulder, elbow, wrist)
#             # 손목 각도 구하기
#             wrist_angle = calculate_angle(elbow, wrist, [wrist[0], wrist[1] - 0.1])
#             # 각도 표시하기
#             cv2.putText(image, f'Shoulder Angle: {round(shoulder_angle, 2)}',
#                         tuple(np.multiply(shoulder, [640, 480]).astype(int)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
#                         cv2.LINE_AA)
#             cv2.putText(image, f'Elbow Angle: {round(elbow_angle, 2)}',
#                         tuple(np.multiply(elbow, [640, 480]).astype(int)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
#                         cv2.LINE_AA)
#             cv2.putText(image, f'Wrist Angle: {round(wrist_angle, 2)}',
#                         tuple(np.multiply(wrist, [640, 480]).astype(int)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
#                         cv2.LINE_AA)
#             if elbow_angle <= 45 and not arm_down:
#                 counter += 1
#                 arm_down = True
#             if elbow_angle > 45:
#                 arm_down = False
#             cv2.putText(image, f'Count: {counter}',
#                         (10, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
#             # 아두이노에 각도 송신하기
#             mapped_shoulder_angle = int(map_value(shoulder_angle, min_angle, max_angle, min_mapped_value, max_mapped_value))
#             mapped_elbow_angle = int(map_value(elbow_angle, min_angle, max_angle, min_mapped_value, max_mapped_value))
#             mapped_wrist_angle = int(map_value(wrist_angle, min_angle, max_angle, min_mapped_value, max_mapped_value))
#             arduino.write(f'{mapped_shoulder_angle},{mapped_elbow_angle},{mapped_wrist_angle}\n'.encode())
#         except:
#             pass
#         cv2.imshow('Mediapipe Feed', image)
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()