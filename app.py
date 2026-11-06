import streamlit as st
import os
import json
import io
import random
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple, Optional

# --- SYSTEM PROMPTS ---
DESIGNER_SYSTEM_PROMPT = """You are an expert Survey Methodologist and Data Scientist.
Your task is to design a high-quality survey based on the user's research objective and target audience.

Strictly follow these design rules:
1. Avoid leading, double-barrelled, confusing, or ambiguous questions.
2. Order questions logically: from easy/general inquiries to sensitive/specific questions.
3. Every single question must map precisely to the target research objective.
4. Ensure appropriate type configurations matching requested formats.

You must return your output strictly as a valid JSON object matching the defined schema below. Do not include any conversational preamble or markdown formatting blocks.

REQUIRED JSON SCHEMA:
{
  "survey_title": "string",
  "introduction": "string",
  "questions": [
    {
      "id": "Q1",
      "text": "string",
      "type": "multiple_choice | likert | rating | yes_no | open_ended",
      "options": ["string"] or null,
      "purpose": "Clear sentence describing what this specific question measures."
    }
  ],
  "design_notes": ["bias mitigation or best-practice tip"]
}"""

QUALITATIVE_SYSTEM_PROMPT = """You are an expert qualitative data analyst.
Your task is to analyze open-ended text survey responses for a single question column.
Examine the provided list of responses carefully. Identify primary themes, compute an approximate count for each theme, extract an illustrative quote, classify sentiment counts, and synthesize recommendations.

You must return your output strictly as a valid JSON object matching the defined schema below. Do not include any conversational preamble or markdown formatting blocks.

REQUIRED JSON SCHEMA:
{
  "summary": "2-3 sentence high-level overview of the textual feedback.",
  "themes": [
    {
      "theme": "string name of theme",
      "count_estimate": 10,
      "example_quote": "verbatim text example from responses"
    }
  ],
  "sentiment": {
    "positive": 5,
    "neutral": 3,
    "negative": 2
  },
  "recommendations": ["clear actionable strategic recommendation"]
}"""

SUMMARY_SYSTEM_PROMPT = """You are a senior executive research consultant. 
Your task is to synthesize quantitative survey statistics and qualitative theme evaluations into a comprehensive Executive Summary report.
Analyze the provided analytical metadata summaries and draft an institutional-grade, insights-focused executive overview.
Provide a clean, readable Markdown layout with clear headers, major bullet points, and key actionable metrics. No preamble or JSON wrapping required."""

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Survey Designer and Analyzer", layout="wide", page_icon="📝")

# --- SESSION STATE INITIALIZATION ---
if "survey_data" not in st.session_state:
    st.session_state.survey_data = None
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "qual_analysis" not in st.session_state:
    st.session_state.qual_analysis = {}
if "executive_summary" not in st.session_state:
    st.session_state.executive_summary = None

# --- CORE INTEGRATION UTILITIES ---
def get_groq_client() -> Optional[Groq]:
    """Retrieves API client parameters securely from secrets or environment variables."""
    from groq import Groq
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)

def call_groq_json(client: Groq, model: str, system: str, user: str, temperature: float = 0.2, retry: bool = True) -> Dict[str, Any]:
    """Queries Groq endpoint with a strict JSON format declaration guarantee."""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices.message.content)
    except Exception as e:
        if retry:
            return call_groq_json(client, model, system, user, temperature, retry=False)
        raise RuntimeError(f"API Processing Error: {str(e)}")

# --- TAB 1: LOGICAL ENGINE ---
def generate_survey_markdown(survey: Dict[str, Any]) -> str:
    """Transforms structural survey JSON mapping profiles into a clean Markdown document string."""
    md = f"# {survey.get('survey_title', 'Survey')}\n\n{survey.get('introduction', '')}\n\n"
    for q in survey.get("questions", []):
        md += f"### {q.get('id')}: {q.get('text')}\n"
        md += f"*Type: {q.get('type')}*  \n"
        if q.get("options"):
            md += "Options:\n"
            for opt in q["options"]:
                md += f"- [ ] {opt}\n"
        md += f"*Purpose: {q.get('purpose')}*\n\n"
    if survey.get("design_notes"):
        md += "## Methodology & Design Notes\n"
        for note in survey["design_notes"]:
            md += f"- {note}\n"
    return md

def create_csv_template(survey: Dict[str, Any]) -> str:
    """Generates an empty template layout matching question columns to standardize collection streams."""
    columns = [q.get("id", "Unknown") for q in survey.get("questions", [])]
    df = pd.DataFrame(columns=columns)
    return df.to_csv(index=False)

# --- TAB 2: LOGICAL ENGINE ---
@st.cache_data
def load_survey_dataframe(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """Reads structural spreadsheet data streams securely via cached file input blocks."""
    if file_name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    return pd.read_excel(io.BytesIO(file_bytes))

def detect_column_types(df: pd.DataFrame, survey_json: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Classifies column data attributes dynamically using layout profiles or automated row sampling models."""
    types_map = {}
    
    # Map questions directly if template matches existing JSON properties
    if survey_json:
        for q in survey_json.get("questions", []):
            q_id = q.get("id")
            if q_id in df.columns:
                types_map[q_id] = q.get("type")
                
    for col in df.columns:
        if col in types_map:
            continue
            
        non_null = df[col].dropna()
        if non_null.empty:
            types_map[col] = "open_ended"
            continue
            
        # Inspect numerical distributions
        if pd.api.types.is_numeric_dtype(df[col]):
            unique_vals = sorted(non_null.unique())
            if set(unique_vals).issubset({0, 1}) or (len(unique_vals) == 2 and set(str(v).lower() for v in unique_vals).issubset({"yes", "no", "true", "false"})):
                types_map[col] = "yes_no"
            elif all(float(v).is_integer() and 1 <= v <= 5 for v in unique_vals):
                types_map[col] = "likert"
            elif all(float(v).is_integer() and 1 <= v <= 10 for v in unique_vals):
                types_map[col] = "rating"
            else:
                types_map[col] = "rating"
        else:
            # Check unique values density threshold ratios
            if len(non_null.unique()) / len(non_null) < 0.2 or len(non_null.unique()) <= 5:
                unique_lower = [str(v).lower().strip() for v in non_null.unique()]
                if set(unique_lower).issubset({"yes", "no", "true", "false", "y", "n"}):
                    types_map[col] = "yes_no"
                else:
                    types_map[col] = "multiple_choice"
            else:
                types_map[col] = "open_ended"
                
    return types_map

def generate_sample_dataset() -> pd.DataFrame:
    """Constructs a production-grade mock survey response dataset containing 50 diverse row models."""
    mc_opts = ["Product Features", "Customer Support", "Pricing Model", "User Interface"]
    text_opts = [
        "The interface felt a bit sluggish but the feature parameters are robust.",
        "Excellent prompt execution pathways. Customer support was immediate and outstanding.",
        "Pricing strategies are slightly expensive for small startups like ours.",
        "Clean workspace layouts. Found it highly intuitive and extremely easy to navigate.",
        "Missing automated visualization dashboards. Hope they launch this component soon."
    ]
    
    rows = []
    for i in range(1, 51):
        rows.append({
            "Q1": random.choice(["Yes", "No"]),
            "Q2": random.randint(3, 5) if i % 4 != 0 else random.randint(1, 2),
            "Q3": random.randint(7, 10) if i % 5 != 0 else random.randint(2, 5),
            "Q4": random.choice(mc_opts),
            "Q5": random.choice(text_opts)
        })
    return pd.DataFrame(rows)

# --- APPLICATION INTERFACE ENVIRONMENT ---
groq_client = get_groq_client()

# --- SIDEBAR REGION ---
with st.sidebar:
    st.title("⚙️ Operations Panel")
    if not groq_client:
        st.error("❌ GROQ_API_KEY Secret Token Missing. Configuration required to boot operational processes.")
        st.stop()
        
    st.subheader("Model Selection Parameters")
    selected_model = st.selectbox("Inference Model Engine", ["llama-3.3-70b-versatile"], index=0)
    inference_temp = st.slider("Temperature Range", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

# --- MAIN LAYOUT CONTEXT ---
st.title("📊 AI Survey Designer and Analyzer")
st.write("Generate optimized qualitative survey matrices and analyze mixed quantitative outputs effortlessly.")

tab_designer, tab_analyzer = st.tabs(["📋 Tab 1: Survey Designer", "📈 Tab 2: Response Analyzer"])

# --- TAB 1: SURVEY DESIGNER UI ---
with tab_designer:
    st.header("Create Structured Surveys")
    
    with st.form("survey_design_form"):
