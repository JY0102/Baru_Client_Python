import cv2
import numpy as np
import mediapipe as mp
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from PoseFSM import PoseFSM
import threading

class PoseAnalyzer:
    # 운동 이름 , npy 확장파일
    # 각종 초기화
    
    def __init__(self, exercise_type, reference_npy):
        # MediaPipe 초기화
        self.pose = mp.solutions.pose.Pose(static_image_mode=False)

        # self.exercise_type = "" # 전달 받은 동작 넣기
        self.fsm = PoseFSM(exercise_type) # FSM 초기화

        # 기준 동작 시퀀스 로드
        # self.reference_pose_sequence = np.load(f"{exercise_type}_reference.npy")  # 예: squat_reference.npy 
        print(reference_npy)
        # self.reference_pose_sequence = np.load(reference_npy ,allow_pickle=True)
        raw_seq = np.load(reference_npy, allow_pickle=True)  # (T, 33, 3) 라면
        self.reference_pose_sequence = [frame.flatten() for frame in raw_seq]


        # 실시간 사용자 포즈 시퀀스 저장
        self.live_pose_sequence = []
        self.lock = threading.Lock()

    # 실시간 프레임 분석 함수 (JPEG 바이트 입력 → 정확도 및 카운트 반환)
    def process_frame(self, jpeg_bytes):
        frame = self._decode_jpeg_bytes(jpeg_bytes) # JPEG 바이트 -> 이미지 변환 함수 호출
        pose_vec_flat, landmarks = self._extract_pose_vector(frame) # 포즈 벡터 및 랜드마크 추출 함수 호출
        if pose_vec_flat is not None and landmarks is not None:
            with self.lock:
                # self.live_pose_sequence.append(np.array(pose_vec_flat).flatten()) # 프레임 누적시키기
                self.live_pose_sequence.append(pose_vec_flat)
                if len(self.live_pose_sequence) >= 60: # 30프레임당 1초 ( 추후 수정 )
                    similarity = self._compare_with_dtw(self.live_pose_sequence)
                    count = self.fsm.update(landmarks, similarity) # FSM 업데이트로 카운트 확인
                    self.live_pose_sequence.clear() # 다음 분석을 위해서 초기화
                    
                    return round(similarity, 1) # 정확도                        
                    
        return None # 유효하지 않은 프레임은 None 반환

    #region 각종 유틸함수
    
    # JPEG 바이트 데이터를 OpenCV 이미지로 디코딩
    def _decode_jpeg_bytes(self, jpeg_bytes):
        np_arr = np.frombuffer(jpeg_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # 이미지에서 포즈 벡터와 3D 랜드마크 추출
    def _extract_pose_vector(self, image):
        result = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not result.pose_landmarks:
            return None, None
        keypoints = [(lm.x, lm.y, lm.z) for lm in result.pose_landmarks.landmark]
        return np.array(keypoints).flatten(), keypoints

    # DTW를 이용해 정확도 계산
    def _compare_with_dtw(self, live_sequence):
        distance, _ = fastdtw(live_sequence, self.reference_pose_sequence, dist=euclidean)
        similarity = np.exp(-distance / 100)
        return similarity * 100

    #endregion