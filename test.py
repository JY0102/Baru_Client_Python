#region 참조 모음

# fastAPI 에 필요
from fastapi import FastAPI, HTTPException , Request
import httpx
import io
# mediapipe pose 확장파일
import numpy as np

# 백그라운드 스레드 활성화에 필요함
import asyncio
from concurrent.futures import ThreadPoolExecutor

from RealTime_PoseCompare import PoseAnalyzer
#endregion

BASE_URL = "http://127.0.0.1:6000"

CurrentNpy = None
CurrentExercise = None
Compare = None

async def GetNpy(exercise : str):
    global CurrentNpy
    global CurrentExercise
    global Compare
    
    if CurrentExercise != exercise and exercise:
        
        try:            
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{BASE_URL}/get/type/?exercise={exercise}")

            if response.status_code == 200:
                bytes_io = io.BytesIO(response.content)
                # CurrentNpy = np.load(bytes_io)
                CurrentExercise = exercise                
                Compare = PoseAnalyzer(exercise_type= CurrentExercise , reference_npy= bytes_io)       
                print('성공')         
            else:
                raise HTTPException(status_code=401 , detail="GetNpy Error")
            
        except HTTPException:
            raise
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")
        

asyncio.run(GetNpy('squat'))