import os
import asyncio
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

# Import your functions here
from predictor_module import predict_agent, session_service, runner, APP_NAME, USER_ID, SESSION_ID

# ----------------------------
# FastAPI Setup
# ----------------------------
app = FastAPI(title="Drug Approval Predictor API")

# ----------------------------
# CORS Configuration
# ----------------------------
origins = [
    "*",  # Allow all origins, or specify frontend URLs like "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Request Model
# ----------------------------
class PredictRequest(BaseModel):
    indication: str
    molecule_type: str
    agencies: str
    start_date: str

# ----------------------------
# Agent call function
# ----------------------------
async def call_agent_async(query: str, runner, user_id, session_id):
    content = types.Content(role="user", parts=[types.Part(text=query)])
    final_response_text = "Agent did not produce a final response."

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
            break
    return final_response_text

# ----------------------------
# Robust Response Parser
# ----------------------------
def parse_agent_response(response_text: str) -> Dict:
    result = {"approvals": []}

    # Split response into chunks per agency
    agency_blocks = re.split(r'(?="Agency":")', response_text)

    for block in agency_blocks:
        if not block.strip():
            continue

        # Extract Agency
        agency_match = re.search(r'"Agency":"([^"]+)"', block)
        if not agency_match:
            continue
        agency_name = agency_match.group(1)

        # Extract Tentative Approval
        approval_match = re.search(r'"Tentative approval":"([^"]+)"', block)
        tentative_approval = approval_match.group(1) if approval_match else ""

        # Extract Reason
        reason_match = re.search(r'"Reason":\s*"([^"]+)"', block)
        reason = reason_match.group(1) if reason_match else ""

        # Extract reimbursement dates
        reimb_dict = {}
        reimb_matches = re.findall(r'(\w+)\s*[:→]\s*(\d+)\s*months', block)
        for item in reimb_matches:
            reimb_dict[item[0]] = f"{item[1]} months"

        reimb_sorted = dict(
            sorted(reimb_dict.items(), key=lambda x: int(x[1].split()[0]))
        )

        result["approvals"].append({
            "agency": agency_name,
            "tentative_approval": tentative_approval,
            "reason": reason,
            "reimbursement_approval_dates": reimb_sorted
        })

    # Sort approvals by tentative approval date if possible
    result["approvals"] = sorted(
        result["approvals"],
        key=lambda x: x["tentative_approval"]
    )

    return result

# ----------------------------
# Startup event
# ----------------------------
@app.on_event("startup")
async def startup_event():
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )

# ----------------------------
# API endpoints
# ----------------------------
@app.post("/predict")
async def predict(req: PredictRequest) -> Dict:
    # Force agent to return response for each agency separately
    query = (
        f"Indication={req.indication}, "
        f"MoleculeType={req.molecule_type}, "
        f"Agencies={req.agencies} "
        f"(return response for EACH agency separately in structured format), "
        f"StartDate={req.start_date}"
    )

    response_text = await call_agent_async(query, runner, USER_ID, SESSION_ID)

    # DEBUG: Print raw agent response in logs
    print("\n====== RAW AGENT RESPONSE ======\n", response_text, "\n===============================\n")

    # Parse agent response into structured JSON with rankings
    structured_response = parse_agent_response(response_text)
    return structured_response

@app.get("/")
async def root():
    return {"message": "Welcome to the Drug Approval Predictor API"}
