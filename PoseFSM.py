import numpy as np
# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/index?hl=ko 에서 포즈 랜드마크 번호 이미지 다운함
class PoseFSM:
    # 운동 유형별 FSM 초기화
    def __init__(self, exercise_type): 
        self.exercise_type = exercise_type
        self.state = 0
        self.count = 0
        self.prev_similarity = 0
        self.baseline = None

    # 두 keypoint 간 거리 계산 (정규화된 좌표 기준)
    def _calculate_ratio(self, landmarks, p1, p2): 
        a = np.array(landmarks[p1])
        b = np.array(landmarks[p2])
        return np.linalg.norm(a - b)

    # 더 많이 접힌 다리를 찾아서 해당 다리의 굽힘 비율 반환 (스쿼트, 런지에 사용)
    def _dominant_leg_fold_ratio(self, landmarks):
        upper_leg_l = self._calculate_ratio(landmarks, 23, 25) # 왼쪽 골반 마커부터 무릎 마커 사이의 거리 계산
        lower_leg_l = self._calculate_ratio(landmarks, 25, 27) # 왼쪽 무릎 마커부터 발목 마커 사이의 거리 계산
        upper_leg_r = self._calculate_ratio(landmarks, 24, 26) # 오른쪽 골반 마커부터 무릎 마커 사이의 거리 계산
        lower_leg_r = self._calculate_ratio(landmarks, 26, 28) # 오른쪾 무릎 마커부터 발목 마커 사이의 거리 계산

        # 왼쪽, 오른쪽중 어느 쪽이 더 접힌건지 비율 계산, 결과가 0이 되는걸 막으면서 연산 결과에 영향을 최소화 하기 위해서 1e-6(0.000001) 사용
        fold_ratio_l = upper_leg_l / (lower_leg_l + 1e-6) # 계산된 왼쪽 허벅지 / 종아리
        fold_ratio_r = upper_leg_r / (lower_leg_r + 1e-6) # 계산된 오른쪽 허벅지 / 종아리

        return fold_ratio_l if abs(fold_ratio_l - 1) > abs(fold_ratio_r - 1) else fold_ratio_r

    # 현재 운동 유형에 따른 FSM 로직 호출
    def update(self, landmarks, similarity):
        print(f"[FSM] 호출 | 운동 유형: {self.exercise_type}", flush=True)
        if self.exercise_type == "Squat":
            return self._squat_logic(landmarks, similarity)
        elif self.exercise_type == "lunge":
            return self._lunge_logic(landmarks, similarity)
        elif self.exercise_type == "sidestretch":
            return self._sidestretch_logic(landmarks, similarity)
        else:
            return self.count

    # squat 로직
    def _squat_logic(self, landmarks, similarity):
        leg_fold_ratio = self._dominant_leg_fold_ratio(landmarks)
        print(f"[로그]similarity: {similarity:.2f}", flush=True)
        print(f"[로그]fold_ratio: {leg_fold_ratio:.2f}", flush=True)

        if self.state == 0:
            if leg_fold_ratio < 0.95 and similarity > 50:
                self.state = 1
        elif self.state == 1:
            if leg_fold_ratio > 1.05 and similarity > 50:
                self.state = 0
                self.count += 1
        return self.count

    # lunge 로직
    def _lunge_logic(self, landmarks, similarity):
        left_fold = self._calculate_ratio(landmarks, 23, 25)  # 왼쪽 골반-무릎
        right_fold = self._calculate_ratio(landmarks, 24, 26)  # 오른쪽 골반-무릎

        # 더 접힌 쪽이 현재 동작 중인 다리
        threshold_down = 0.6
        threshold_up = 0.9

        if self.state == 0:
            if left_fold < threshold_down and similarity > 50:
                self.state = 1  # 왼쪽 런지 내려감
        elif self.state == 1:
            if left_fold > threshold_up and similarity > 50:
                self.state = 2  # 왼쪽 런지 올라옴
        elif self.state == 2:
            if right_fold < threshold_down and similarity > 50:
                self.state = 3  # 오른쪽 런지 내려감
        elif self.state == 3:
            if right_fold > threshold_up and similarity > 50:
                self.state = 0  # 오른쪽 런지 올라옴 → 1회 완료
                self.count += 1
        return self.count

    # sidestretch 로직
    def _sidestretch_logic(self, landmarks, similarity):
        left_torso = self._calculate_ratio(landmarks, 11, 23)   # 왼쪽 어깨-골반
        right_torso = self._calculate_ratio(landmarks, 12, 24)  # 오른쪽 어깨-골반

        if self.baseline is None: # 좌우로 움직일 때 기준이 될 최초 정지상태에서의 기준 길이
            self.baseline = (left_torso + right_torso) / 2
        
        if self.state == 0:
            if left_torso < self.baseline * 0.85 and similarity > 50:
                self.state = 1
            elif right_torso < self.baseline * 0.85 and similarity > 50:
                self.state = 2
        elif self.state == 1:
            if left_torso >= self.baseline * 0.95 and similarity > 50:
                self.state = 0
                self.count += 1
        elif self.state == 2:
            if right_torso >= self.baseline * 0.95 and similarity > 50:
                self.state = 0
                self.count += 1
        return self.count