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
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")#"AIzaSyDAOcP7uNLrb2F_0w8MpXIK8OeXoi-pwfo"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"
print("✅ API Keys configured.")

# ----------------------------
# 3. Load Excel Dataset
# ----------------------------

EXCEL_FILE = "model_input_final.xlsx"  # must contain required columns including `hta`
REASON_FILE = "reason_final.txt"  # text file with reasons

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

def get_reason_from_text(agency: str, molecule_type: str, indication: str) -> str:
    """
    Search the text file for a reason that matches agency, molecule_type, and indication keywords.
    Returns first matching sentence or fallback message.
    """
    agency = agency.lower()
    molecule_type = molecule_type.lower()
    indication = indication.lower()
    fallback_reason = "No specific reason available for this combination in the text file."

    # Split text into sentences
    sentences = REASON_TEXT.split(".")
    for sent in sentences:
        if agency in sent and molecule_type in sent:
            if indication in sent or True:  # optional: partial match
                return sent.strip().capitalize() + "."
    return fallback_reason

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
    Predict approval based on indication, molecule_type, agencies, and start date.
    If an exact start_date (month+year) match exists → use approval_date from Excel.
    Else → calculate average approval lag.
    """
    df = load_data()
    if df.empty:
        return {"error": "No data available."}

    mol_type_norm = normalize_molecule_type(molecule_type)
    results = {}
    ranking = []
    start_dt = dateutil.parser.parse(start_date)

    # Extract month-year from user start_date
    user_month = start_dt.month
    user_year = start_dt.year

    # Compute global average reimbursement lag for fallback
    if "reimbursement_date" in df.columns and "approval_date" in df.columns:
        all_lags = (pd.to_datetime(df["reimbursement_date"]) - pd.to_datetime(df["approval_date"])).dt.days.dropna()
        all_lags = all_lags[(all_lags > 0) & (all_lags < 1095)]
        global_avg_reimb_lag = int(all_lags.median()) if not all_lags.empty else 180
    else:
        global_avg_reimb_lag = 180

    # Check each agency
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

        # --- Check exact start_date month+year match ---
        subset["start_date"] = pd.to_datetime(subset["start_date"], errors="coerce")
        match = subset[
            (subset["start_date"].dt.month == user_month) &
            (subset["start_date"].dt.year == user_year)
        ]

        if not match.empty and "approval_date" in match.columns:
            # Use actual approval_date from Excel
            approval_date = pd.to_datetime(match.iloc[0]["approval_date"], errors="coerce")
            predicted_approval_date = approval_date if not pd.isna(approval_date) else start_dt
            source_type = "Exact match from dataset"
        else:
            # --- Fallback to average lag ---
            lags = (pd.to_datetime(subset["approval_date"]) - pd.to_datetime(subset["start_date"])).dt.days.dropna()
            avg_lag_days = int(lags.mean()) if not lags.empty else 1000
            predicted_approval_date = start_dt + pd.Timedelta(days=avg_lag_days)
            if predicted_approval_date < start_dt:
                predicted_approval_date = start_dt
            source_type = "Predicted using average lag"

        # --- Reimbursement lag ---
        if "reimbursement_date" in subset.columns:
            reimb_lags = (pd.to_datetime(subset["reimbursement_date"]) - pd.to_datetime(subset["approval_date"])).dt.days.dropna()
            reimb_lags = reimb_lags[(reimb_lags > 0) & (reimb_lags < 1095)]
            avg_reimb_lag_days = int(reimb_lags.median()) if not reimb_lags.empty else global_avg_reimb_lag
        else:
            avg_reimb_lag_days = global_avg_reimb_lag

        predicted_reimb_date = predicted_approval_date + pd.Timedelta(days=avg_reimb_lag_days)

        # --- HTA-specific reimbursement for this agency ---
        reimbursement_summary = []
        if "hta" in subset.columns:
            unique_countries = subset["hta"].dropna().str.lower().unique()
            for country in unique_countries:
                hta_subset = subset[subset["hta"].str.lower() == country]

                # Check if exact approval date exists for this HTA
                exact_reimb_match = hta_subset[
                    (pd.to_datetime(hta_subset["approval_date"], errors="coerce") == predicted_approval_date)
                ]
                if not exact_reimb_match.empty and "reimbursement_date" in exact_reimb_match.columns:
                    reimb_date = pd.to_datetime(exact_reimb_match.iloc[0]["reimbursement_date"], errors="coerce")
                    if pd.isna(reimb_date):
                        avg_hta_lag = global_avg_reimb_lag
                    else:
                        avg_hta_lag = (reimb_date - predicted_approval_date).days
                else:
                    hta_lags = (pd.to_datetime(hta_subset["reimbursement_date"]) - pd.to_datetime(hta_subset["approval_date"])).dt.days.dropna()
                    hta_lags = hta_lags[(hta_lags > 0) & (hta_lags < 1095)]
                    avg_hta_lag = int(hta_lags.median()) if not hta_lags.empty else global_avg_reimb_lag

                hta_months = round(avg_hta_lag / 30)
                reimbursement_summary.append(f"{country.upper()} → {hta_months} months")

        reimbursement_output = "Prediction of reimbursement post-approval (" + agency.upper() + ")\n"
        reimbursement_output += ", ".join(reimbursement_summary) if reimbursement_summary else "No HTA data available"

        # # --- Reason from uploaded text file ---
        reason_text = get_reason_from_text(agency, mol_type_norm, indication)
        
        

        # --- Quarter format ---
        q_approval = (predicted_approval_date.month - 1) // 3 + 1
        q_reimb = (predicted_reimb_date.month - 1) // 3 + 1

        results[agency.upper()] = (
            f"{agency.upper()} approval: Q{q_approval} {predicted_approval_date.year}\n"
            f"{agency.upper()} reimbursement_date: Q{q_reimb} {predicted_reimb_date.year}\n"
            f"Reason: {reason_text}\n\n"
            f"{reimbursement_output}"
        )

        # Ranking info
        months = round((predicted_approval_date - start_dt).days / 30)
        ranking.append((agency.upper(), months, predicted_approval_date))

    # --- Ranking ---
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
        You must provide a clear and meaningful reason for the prediction.
        The reason is based on why it predicted that value, what are the things affecting the prediction of approval for selected indication, molecule type, and agency.
        This reason should be a synthesis of the prediction output, the user's input, and information from the uploaded text file.
        The justification must be between four and five sentences in length.
        The 25% reasoning should be based on the prediction output and user input, while up to 75% can be derived from the uploaded text file.
        
        4. Conditional Override:
        You have the authority to change the prediction if you find a strong, contradictory reason within the uploaded text file. The final decision should reflect the most compelling evidence available.

        5. Output Format:
        Always return your response as a JSON array, where each element contains:
        - agency: <agency name>
        - tentative_approval: <QX YYYY>
        - reason: <reason for prediction>
        - reimbursement_approval_dates: <HTA Country> → <months> months, ...

        Example:
        [
          {
            "agency": "FDA",
            "tentative_approval": "Q2 2025",
            "reason": "Your detailed reason here.",
            "reimbursement_approval_dates": "NICE → 8 months, HAS → 10 months"
          },
          {
            "agency": "EMA",
            "tentative_approval": "Q3 2025",
            "reason": "Your detailed reason here.",
            "reimbursement_approval_dates": "G-BA → 12 months"
          }
        ]
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
    start_date = input("Enter Start Date (e.g., Jan 2020): ")

    query = f"Indication={indication}, MoleculeType={molecule_type}, Agencies={agencies}, StartDate={start_date}"
    await call_agent_async(query, runner, USER_ID, SESSION_ID)

if __name__ == "__main__":
    try:
        asyncio.run(run_conversation())
    except Exception as e:
        print(f"❌ Error: {e}")
