import os
import asyncio
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Dict

from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

# Import your functions here (predict_approval, predict_agent, runner, etc.)
# Assuming your previous code is inside predictor_module.py
from predictor_module import predict_agent, session_service, runner, APP_NAME, USER_ID, SESSION_ID

# ----------------------------
# FastAPI Setup
# ----------------------------
app = FastAPI(title="Drug Approval Predictor API")

class PredictRequest(BaseModel):
    indication: str
    molecule_type: str
    agencies: str
    start_date: str

async def call_agent_async(query: str, runner, user_id, session_id):
    content = types.Content(role="user", parts=[types.Part(text=query)])
    final_response_text = "Agent did not produce a final response."

    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
            break
    return final_response_text


@app.on_event("startup")
async def startup_event():
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)


@app.post("/predict")
async def predict(req: PredictRequest) -> Dict:
    query = (
        f"Indication={req.indication}, "
        f"MoleculeType={req.molecule_type}, "
        f"Agencies={req.agencies}, "
        f"StartDate={req.start_date}"
    )
    response_text = await call_agent_async(query, runner, USER_ID, SESSION_ID)
    return {"response": response_text}


@app.get("/")
async def root():
    return {"message": "Welcome to the Drug Approval Predictor API"}
