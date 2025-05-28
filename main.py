# Client 에 붙은 Python

from fastapi import FastAPI, HTTPException , Request
import numpy as np
import httpx
import io

app = FastAPI()

# 전역 이미지 저장 변수
latest_image = None

# ** 디버깅용 화면 출력기 ** 
import threading
import cv2
lock = threading.Lock()

def show_window():
    global latest_image
    cv2.namedWindow("Live View", cv2.WINDOW_NORMAL)

    try:
        while True:
            with lock:
                img_bytes = latest_image

            if img_bytes is not None:
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img is not None:
                    cv2.imshow("Live View", img)

            if cv2.waitKey(30) == 27:
                break
    finally:
        cv2.destroyAllWindows()
threading.Thread(target=show_window, daemon=True).start()


# uvicorn main:app --reload

# 포트 6천 
# 서버 주소
BASE_URL = "http://127.0.0.1:6000"

# 현재 운동하고있는 종류
CurrentExercise = None
CurrentNpy = None

# 정확도 값
Accuracies = []

# 운동 종류 -> DB -> npy 파일
@app.post("/type/")
async def GetNpy(exercise : str):
    global CurrentNpy
    global CurrentExercise
    
    if CurrentExercise != exercise and exercise:
        
        try:            
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{BASE_URL}/get/type/?exercise={exercise}")
            
            if response.status_code == 200:
                bytes_io = io.BytesIO(response.content)
                CurrentNpy = np.load(bytes_io)
                CurrentExercise = exercise                
            else:
                raise HTTPException(status_code=401 , detail="GetNpy Error")
            
        except HTTPException:
            raise
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")

# !! 백그라운드 스레드에서 운동 비교하는 함수 제작 예정 !!
# 운동시작
@app.post("/byte/")
async def Post_Check(request: Request):           

    # jpeg Byte
    # type = bytes
    global latest_image
    latest_image  = await request.body()

    # 반환값 저장
    # Accuracies.append()


# 정확도 값 반환 및 DB 저장
@app.get("/get/accuracy")
async def Get_Accuracy(id : str):      
    global Accuracies
    reuslt = Accuracies.copy()
    Accuracies.clear()    
    
    # DB 저장
    async with httpx.AsyncClient() as client:
        await client.post(f"{BASE_URL}/user/beforeinfo/", json={    "id" : id , "Accuracies": reuslt})                
    
    return reuslt

