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
        print("기준 NPY shape:", reference_npy.shape, reference_npy[0].shape)

        # 정상적인 npy 받았는지 확인
        if isinstance(reference_npy[0], (list, np.ndarray)) and np.array(reference_npy[0]).ndim == 2:
            self.reference_pose_sequence = [
                np.array(self._normalize_keypoints(np.array(frame))).flatten()
                # np.array(frame).flatten()
                for frame in reference_npy
            ]
        else:
            raise ValueError("기준 포즈 데이터는 (33, 3) 형식이어야 합니다.")

        # 실시간 사용자 포즈 시퀀스 저장
        self.live_pose_sequence = []
        self.lock = threading.Lock()

    # 실시간 프레임 분석 함수 (JPEG 바이트 입력 → 정확도 및 카운트 반환)
    def process_frame(self, jpeg_bytes):
        frame = self._decode_jpeg_bytes(jpeg_bytes) # JPEG 바이트 -> 이미지 변환 함수 호출
        if frame is None:
            print("JPEG 디코딩 실패")
            return "프레임 디코딩 실패"
        
        pose_vec_flat, landmarks = self._extract_pose_vector(frame) # 포즈 벡터 및 랜드마크 추출 함수 호출
        if pose_vec_flat is not None and landmarks is not None:
            with self.lock:
                # self.live_pose_sequence.append(np.array(pose_vec_flat).flatten()) # 프레임 누적시키기
                self.live_pose_sequence.append(pose_vec_flat)
                print("프레임 누적 개수:", len(self.live_pose_sequence))
                if len(self.live_pose_sequence) >= 5: # 30프레임당 1초 ( 추후 수정 )
                    similarity = self._compare_with_dtw(self.live_pose_sequence)
                    count = self.fsm.update(landmarks, similarity) # FSM 업데이트로 카운트 확인
                    print(f"Count: {count}")
                    self.live_pose_sequence.clear() # 다음 분석을 위해서 초기화   
                    print(round(similarity, 1))
                    return {"정확도":round(similarity, 1), "카운트":count} # 정확도 
        else:
            print("포즈 인식 실패")           
        return None

    #region 각종 유틸함수
    
    # JPEG 바이트 데이터를 OpenCV 이미지로 디코딩
    def _decode_jpeg_bytes(self, jpeg_bytes):
        np_arr = np.frombuffer(jpeg_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # 이미지에서 포즈 벡터와 랜드마크 추출
    def _extract_pose_vector(self, image):
        result = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # print("landmarks:", result.pose_landmarks)
        if not result.pose_landmarks:
            return None, None
        keypoints = [(lm.x, lm.y, lm.z) for lm in result.pose_landmarks.landmark]
        keypoints_norm = self._normalize_keypoints(keypoints)
        return np.array(keypoints_norm).flatten(), keypoints

    # DTW 정확도 계산을 위한 정규화(어깨 기준 체형 보정)
    def _normalize_keypoints(self, keypoints):
        ls = np.array(keypoints[11])
        rs = np.array(keypoints[12])
        shoulder_width = np.linalg.norm(ls - rs) + 1e-6
        keypoints_norm = [(x / shoulder_width, y / shoulder_width, z / shoulder_width) for (x, y, z) in keypoints]
        # print("정규화된 키포인트 (첫 5개):", keypoints_norm[:5])
        return keypoints_norm

    # DTW를 이용해 정확도 계산
    def _compare_with_dtw(self, live_sequence):
        distance, _ = fastdtw(live_sequence, self.reference_pose_sequence, dist=euclidean)
        max_dist = 30000
        similarity = max(0, 100 - (distance / max_dist) * 100)
        print("DTW distance:", distance)
        return similarity

    #endregion
