import os
import asyncio
import pandas as pd
import dateutil.parser
from typing import Dict
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

import warnings
warnings.filterwarnings("ignore")

# ----------------------------
# 1. Model constants
# ----------------------------
MODEL_GEMINI_2_0_FLASH = "gemini-2.0-flash"

# ----------------------------
# 2. API Keys (replace with yours)
# ----------------------------


os.environ["GOOGLE_API_KEY"] = "AIzaSyDAOcP7uNLrb2F_0w8MpXIK8OeXoi-pwfo"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"
print("✅ API Keys configured.")

# os.getenv("GOOGLE_API_KEY")
# ----------------------------
# 3. Load Excel Dataset
# ----------------------------

EXCEL_FILE = "model_input_final.xlsx"  # must contain required columns
REASON_FILE = "reason_final.txt"       # text file with reasons

def load_data() -> pd.DataFrame:
    if not os.path.exists(EXCEL_FILE):
        print(f"❌ File not found: {EXCEL_FILE}")
        return pd.DataFrame()
    df = pd.read_excel(EXCEL_FILE)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Normalize key columns
    for col in ["indication", "molecule_type", "agency", "hta"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    print(f"📊 Loaded {len(df)} rows from {EXCEL_FILE}")
    return df

# ----------------------------
# 3a. Load reason text file
# ----------------------------
def load_reason_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        print(f"❌ Reason file not found: {file_path}")
        return ""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text.lower()  # lowercase for easier matching

REASON_TEXT = load_reason_file(REASON_FILE)

def get_reason_from_text(agency: str, molecule_type: str, indication: str, prediction_output: str = "", user_input: dict = None) -> str:
    """
    Synthesizes a multi-sentence reason for the prediction, using both prediction output/user input and all relevant text file reasons.
    """
    agency = agency.lower()
    molecule_type = molecule_type.lower()
    indication = indication.lower()
    fallback_reason = "No specific reason available for this combination in the text file."

    # Collect all relevant sentences
    sentences = [s.strip().capitalize() for s in REASON_TEXT.split(".") if agency in s and molecule_type in s and indication in s]
    if not sentences:
        # fallback: try with just agency and molecule_type
        sentences = [s.strip().capitalize() for s in REASON_TEXT.split(".") if agency in s and molecule_type in s]
    if not sentences:
        return fallback_reason

    # Synthesize reasoning
    # 25% from prediction/user input
    user_part = ""
    if user_input:
        user_part = (
            f"Based on the provided indication ({user_input.get('indication','')}), molecule type ({user_input.get('molecule_type','')}), and agency ({user_input.get('agency','').upper()}), "
            f"the predicted approval timeline is {prediction_output}. "
        )
    else:
        user_part = "Based on the prediction output and user input, "

    # 75% from text file, rephrased
    text_part = " ".join(sentences)
    text_part = text_part.replace("according to the text file", "based on historical studies")
    text_part = text_part.replace("as per the text file", "based on recent trends")

    # Combine and ensure at least 5 sentences
    combined = user_part + text_part
    if combined.count(".") < 5:
        combined += " " + " ".join(sentences[:5 - combined.count(".")])

    return combined.strip()

# ----------------------------
# 4. Predictor Tool
# ----------------------------
def normalize_molecule_type(mol_type: str) -> str:
    mol_type = mol_type.strip().lower()
    if mol_type in ["b", "biologic", "biologics"]:
        return "biologic"
    elif mol_type in ["s", "small", "small molecule", "small molecules"]:
        return "small molecule"
    return mol_type

def expand_molecule_type(mol_type: str) -> str:
    return "Biologics" if mol_type == "biologic" else "Small Molecule" if mol_type == "small molecule" else mol_type

def predict_approval(indication: str, molecule_type: str, agencies: str, start_date: str) -> Dict:
    """
    Predict approval based on agency+molecule+indication (completion→approval lag).
    Predict reimbursement based on agency+molecule+indication+HTA (approval→reimb lag).
    """
    df = load_data()
    if df.empty:
        return {"error": "No data available."}

    mol_type_norm = normalize_molecule_type(molecule_type)
    results = {}
    ranking = []
    start_dt = dateutil.parser.parse(start_date)

    # global fallback lags
    if "reimbursement_date" in df.columns and "approval_date" in df.columns:
        all_lags = (pd.to_datetime(df["reimbursement_date"]) - pd.to_datetime(df["approval_date"])).dt.days.dropna()
        all_lags = all_lags[(all_lags > 0) & (all_lags < 1095)]
        global_avg_reimb_lag = int(all_lags.median()) if not all_lags.empty else 180
    else:
        global_avg_reimb_lag = 180

    # process each agency
    for agency in [a.strip().lower() for a in agencies.split(",") if a.strip()]:
        subset = df[
            df["indication"].str.contains(indication.strip().lower(), na=False) &
            df["molecule_type"].str.contains(mol_type_norm, na=False) &
            df["agency"].str.contains(agency, na=False)
        ]

        if subset.empty:
            results[agency.upper()] = (
                f"{agency.upper()} approval: No data available for {indication}, {expand_molecule_type(mol_type_norm)}, {agency.upper()}\n"
                f"Reason: No data available for this combination of parameters."
            )
            continue

        # --- Approval lag (study completion → approval)
        lags = (
            pd.to_datetime(subset["approval_date"], errors="coerce") -
            pd.to_datetime(subset["actual_study_completion_date"], errors="coerce")
        ).dt.days.dropna()
        lags = lags[(lags > 0) & (lags < 2000)]

        avg_lag_days = int(lags.mean()) if not lags.empty else 365
        predicted_approval_date = start_dt + pd.Timedelta(days=avg_lag_days)

        # --- HTA reimbursement lags (approval → reimb)
        reimbursement_summary = []
        if "hta" in subset.columns:
            unique_countries = subset["hta"].dropna().str.lower().unique()
            for country in unique_countries:
                hta_subset = subset[subset["hta"].str.lower() == country]

                hta_lags = (
                    pd.to_datetime(hta_subset["reimbursement_date"], errors="coerce") -
                    pd.to_datetime(hta_subset["approval_date"], errors="coerce")
                ).dt.days.dropna()

                hta_lags = hta_lags[(hta_lags > 0) & (hta_lags < 1095)]
                avg_hta_lag = int(hta_lags.median()) if not hta_lags.empty else global_avg_reimb_lag

                hta_months = round(avg_hta_lag / 30)
                reimbursement_summary.append(f"{country.upper()} → {hta_months} months")

        reimbursement_output = "Prediction of reimbursement post-approval (" + agency.upper() + ")\n"
        reimbursement_output += ", ".join(reimbursement_summary) if reimbursement_summary else "No HTA data available"

        # --- Reason
        reason_text = get_reason_from_text(agency, mol_type_norm, indication)

        # --- Quarter formatting
        q_approval = (predicted_approval_date.month - 1) // 3 + 1

        results[agency.upper()] = (
            f"{agency.upper()} approval: Q{q_approval} {predicted_approval_date.year}\n"
            f"Reason: {reason_text}\n\n"
            f"{reimbursement_output}"
        )

        # Ranking info
        months = round((predicted_approval_date - start_dt).days / 30)
        ranking.append((agency.upper(), months, predicted_approval_date))

    # --- Ranking
    ranking.sort(key=lambda x: x[1])
    ranking_lines = []
    for i, (agency, months, predicted_approval_date) in enumerate(ranking, start=1):
        approval_text = results[agency].split("\n")[0]
        suffix = "th"
        if i == 1: suffix = "st"
        elif i == 2: suffix = "nd"
        elif i == 3: suffix = "rd"
        ranking_lines.append(f"{i}{suffix} approval {approval_text}")

    ranking_str = "\n".join(ranking_lines)

    return {
        "predictions": results,
        "ranking": ranking_str,
    }

# ----------------------------
# 5. Define ADK Agent
# ----------------------------
predict_agent = Agent(
    name="drug_approval_predictor",
    model=MODEL_GEMINI_2_0_FLASH,
    description="Predicts drug approval timelines based on indication, molecule type, agencies, and start date.",
    instruction=(
        """
        You are a Drug Approval and Reimbursement Predictor Agent. Your task is to predict the likelihood of a drug's approval and reimbursement based on the user's input.

        1. Input Requirements:
        When a user provides the following information, proceed with the prediction:
        Indication
        MoleculeType
        Agencies
        start_date

        2. Prediction Process:
        Use the provided data to call the predict_approval tool.

        3. Justification and Reasoning:
        -while providing reasons dont use According to the text file or As per the text file,use based on historical studies or based on recent trends.
        - For each agency, you must provide a detailed, multi-sentence reason for the prediction.
        - The reason must be at least five sentences.
        -please search the text file for relevant reasons that match the agency, molecule type, and indication.
        - 25% of the reasoning should be based on the prediction output and user input (e.g., timelines, data availability, molecule type, agency, indication) you need to search the text file for relevant reasons that match the agency, molecule type, and indication.
        - 75% of the reasoning must be synthesized from all relevant information found in the uploaded text file, especially sentences that mention the selected indication, molecule type, and agency. Like what factors influence the approval timelines, any recent changes in regulations, historical approval trends, or specific challenges associated with the indication or molecule type.
        - If multiple relevant reasons are found in the text file, combine them into a comprehensive justification.
        - If a strong, contradictory reason is found in the text file, you may override the prediction and explain why.
        - Always use all matching reasons from the text file for the selected parameters.

        4. Response Format:
        For each agency, return the response in this format:

        approval predict fastest to slowest ranking:
        "Agency":"<agency name>",
        "Tentative approval":"QX YYYY",
        "Reason": "<Comprehensive, multi-sentence reason for prediction, using both prediction output/user input and all relevant text file reasons>",
        "reimbursement approval dates":<HTA Country> → <months> months.......

        Example:
        approval predict fastest to slowest ranking:
        "Agency":"FDA",
        "Tentative approval":"Q2 2025",
        "Reason": "Based on the provided indication and molecule type, the predicted approval timeline is Q2 2025. The dataset shows similar cases with a median lag of 14 months. According to the uploaded text file, FDA approvals for this indication and molecule type are often influenced by recent clinical trial outcomes and regulatory priorities. The text file also notes that expedited pathways may apply for this agency. Therefore, the prediction reflects both the data-driven estimate and the contextual factors described in the text file.",
        "reimbursement approval dates":US → 8 months, EU → 12 months

        Always follow this structure for each agency in the ranking.
        """
    ),
    tools=[predict_approval],
)
print(f"✅ Agent '{predict_agent.name}' created.")

# ----------------------------
# 6. Session + Runner
# ----------------------------
session_service = InMemorySessionService()
APP_NAME = "drug_predictor_app"
USER_ID = "user_1"
SESSION_ID = "session_001"

runner = Runner(
    agent=predict_agent,
    app_name=APP_NAME,
    session_service=session_service
)
print("✅ Runner created.")

# ----------------------------
# 7. Interaction Function
# ----------------------------
async def call_agent_async(query: str, runner, user_id, session_id):
    print(f"\n>>> User Query: {query}")
    content = types.Content(role="user", parts=[types.Part(text=query)])
    final_response_text = "Agent did not produce a final response."

    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
            break

    print(f"<<< Agent Response:\n{final_response_text}\n")
    return final_response_text

# ----------------------------
# 8. Run Example Conversation
# ----------------------------

async def run_conversation():
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    indication = input("Enter Indication: ")
    molecule_type = input("Enter Molecule Type (Biologic / Small Molecule): ")
    agencies = input("Enter Agencies (comma separated, e.g., FDA, EMA, PMDA): ")
    start_date = input("Enter Study Completion Date (e.g., Jan 2020): ")

    query = f"Indication={indication}, MoleculeType={molecule_type}, Agencies={agencies}, StartDate={start_date}"
    await call_agent_async(query, runner, USER_ID, SESSION_ID)

if __name__ == "__main__":
    try:
        asyncio.run(run_conversation())
    except Exception as e:
        print(f"❌ Error: {e}")
