# api.py
from fastapi import FastAPI
from ai_service import ZJHAIService

app = FastAPI()
ai = ZJHAIService("models/a30_psro5_final_step_500000.pth")

@app.post("/zjh/ai/decide")
def ai_decide(payload: dict):
    print(payload)
    return ai.decide(payload)


