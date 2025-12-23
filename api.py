# api.py
from fastapi import FastAPI
from ai_service import ZJHAIService

app = FastAPI()
ai = ZJHAIService("models/a30_psro3_best.pth")

@app.post("/zjh/ai/decide")
def ai_decide(payload: dict):
    print(payload)
    return ai.decide(payload)
