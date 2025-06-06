#region 참조 모음

# fastAPI 에 필요
from fastapi import FastAPI, HTTPException , Request 
import httpx
import io
import json
# 서버에서 npy 로드
import numpy as np
from pydantic import BaseModel

import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

# 백그라운드 스레드 활성화에 필요함
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 정확도 체크
from RealTime_PoseCompare import PoseAnalyzer
# 현재 시간
from datetime import datetime


#endregion

app = FastAPI()



    
# 전역 이미지 저장 변수
latest_image = None

#region   ** 디버깅용 화면 출력기 ** 

# import threading
# import cv2
# lock = threading.Lock()

# def show_window():
#     global latest_image
#     cv2.namedWindow("Live View", cv2.WINDOW_NORMAL)

#     try:
#         while True:
#             with lock:
#                 img_bytes = latest_image

#             if img_bytes is not None:
#                 np_arr = np.frombuffer(img_bytes, np.uint8)
#                 img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#                 if img is not None:
#                     cv2.imshow("Live View", img)

#             if cv2.waitKey(30) == 27:
#                 break
#     finally:
#         cv2.destroyAllWindows()
# threading.Thread(target=show_window, daemon=True).start()

#endregion

# 실행 명령어
# uvicorn main:app --reload
# uvicorn main:app --host 220.90.180.114 --port 8080 --reload

# 포트 6천 
# 서버 주소
# BASE_URL = "http://127.0.0.1:6000"
BASE_URL = "http://220.90.180.114:8000"

# 현재 운동하고있는 종류
CurrentExercise = None
CurrentNpy = None

# 정확도 값
Accuracies = []
# 총 개수
Count  = None


# 정확도 비교하는 클래스
Compare = None

# 정확도 비교후 저장
def process_frame_in_thread(image , uid):
    global Compare
    global Count
    
    result = Compare.process_frame(latest_image)
    print(result)
    
    if result:
        Accuracies.append(result['정확도'])
        Count = result['카운트']
        
        data = {
        "Count": Count,
        "Accuracies": Accuracies.copy()
        }
            
        
        app.state.redis.set(f"result:{uid}", json.dumps(data), ex=10)
    else:
        return 
    

# 운동 종류 -> DB -> npy 파일
@app.post("/start/")
async def GetNpy(exercise : str):
    global CurrentNpy
    global CurrentExercise
    global Compare
        
    print(exercise)
    
    if CurrentExercise != exercise and exercise:
        
        try:            
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{BASE_URL}/get/type/?exercise={exercise}")
            if response.status_code == 200:
                try:
                    bytes_io = io.BytesIO(response.content)
                    CurrentNpy = np.load(bytes_io, allow_pickle=True)
                    CurrentExercise = exercise                
                    Compare = PoseAnalyzer(exercise_type=CurrentExercise , reference_npy=CurrentNpy) 
                    print('Compare 생성 성공')  
                    
                    import uuid
                    uid = str(uuid.uuid4())  # 고유 id 생성
    
                    return {"uid": uid}
                except Exception as e:
                    print("기준 NPY 로딩 실패:", str(e))   
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

    executor = ThreadPoolExecutor(max_workers=20)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, process_frame_in_thread, latest_image)
    
# 목표량 저장
@app.post("/goal/")
async def Post_Goal(request : Request):
    
    body = await request.body()
    
    async with httpx.AsyncClient() as client:
        await client.post(f"{BASE_URL}/data/insert/beforeinfo", json=json.loads(body))         

# 실시간 정확도 , 개수 출력
@app.get("/get/current/")
async def Get_Accuracy(uid : str):
    key = f"result:{uid}"
    data = await app.state.redis.get(key)
    
    if data:
        await app.state.redis.delete(key)
        return json.loads(data)
    else :
        raise HTTPException(status_code=400)

# 정확도 값 반환 및 DB 저장
@app.get("/get/accuracy/")
async def Get_Accuracy(id : str):      
    
    global Accuracies    
    global Count
    
    result = (Count , Accuracies.copy() ) 
        
    # DB 저장
    async with httpx.AsyncClient() as client:
        json_str ={
            "baru_id" : id , 
            "date" : datetime.now().strftime("%Y-%m-%d"),
            "name" : CurrentExercise ,
            "count" : Count ,
            "accuracies": Accuracies
            }
        # post 요청하고 따로 기다리지는 않음
        _= client.post(f"{BASE_URL}/data/insert/play/", json=json_str)                
    
    Count = None
    Accuracies.clear()    
    # C#에 전송
    return result


