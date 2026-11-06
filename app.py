
import os
import io
import json
import random
import pandas as pd
import streamlit as st
import plotly.express as px
from groq import Groq


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI Survey Designer & Analyzer",
    page_icon="📝",
    layout="wide"
)

st.title("📊 AI Survey Designer & Analyzer")
st.markdown(
    "Design intelligent surveys, analyze responses, visualize "
    "research findings, and generate executive reports."
)


# =====================================================
# SESSION STATE
# =====================================================

defaults = {
    "survey_data": None,
    "uploaded_df": None,
    "qual_analysis": None,
    "executive_summary": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =====================================================
# AI SYSTEM PROMPTS
# =====================================================

DESIGNER_SYSTEM_PROMPT = """
You are an expert survey methodologist and data scientist.

Create professional surveys based on the user's research objective.

Rules:
1. Avoid leading, confusing, or double-barrelled questions.
2. Arrange questions logically.
3. Every question must support the research objective.
4. Use appropriate question types.
5. Multiple choice questions must have relevant options.
6. Likert questions should use a 1-5 scale.
7. Rating questions should use a 1-10 scale.

Return ONLY valid JSON using this schema:

{
  "survey_title": "string",
  "introduction": "string",
  "questions": [
    {
      "id": "Q1",
      "text": "string",
      "type": "multiple_choice",
      "options": ["Option 1", "Option 2"],
      "purpose": "string"
    }
  ],
  "design_notes": ["string"]
}

Allowed types:
multiple_choice, likert, rating, yes_no, open_ended

For open_ended questions, options must be null.
"""

QUALITATIVE_SYSTEM_PROMPT = """
You are an expert qualitative data analyst.

Analyze the provided survey responses.

Identify:
1. Main feedback themes.
2. Estimated number of responses for each theme.
3. Verbatim example quotes.
4. Positive, neutral, and negative sentiment.
5. Practical recommendations.

Do not invent example quotes.
Sentiment counts must add up to the number of analyzed responses.

Return ONLY valid JSON:

{
  "summary": "Overview of feedback",
  "themes": [
    {
      "theme": "Theme name",
      "count_estimate": 5,
      "example_quote": "Actual response"
    }
  ],
  "sentiment": {
    "positive": 5,
    "neutral": 3,
    "negative": 2
  },
  "recommendations": [
    "Actionable recommendation"
  ]
}
"""

SUMMARY_SYSTEM_PROMPT = """
You are a senior executive research consultant.

Create a professional executive summary using the supplied
survey statistics and qualitative analysis.

Include:
- Executive overview
- Important findings
- Quantitative insights
- Qualitative insights
- Research limitations
- Actionable recommendations

Use clear Markdown headings and bullet points.
Never invent statistics.

Return Markdown only.
"""


# =====================================================
# GROQ API CONNECTION
# =====================================================

def get_groq_client():

    api_key = None

    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass

    api_key = api_key or os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


def call_groq_json(client, model, system_prompt, user_prompt, temperature):

    try:

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=temperature,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content

        return json.loads(content)

    except Exception as e:
        raise RuntimeError(f"Groq API Error: {str(e)}")


def call_groq_text(client, model, system_prompt, user_prompt, temperature):

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=temperature
    )

    return response.choices[0].message.content or ""


# =====================================================
# SURVEY UTILITIES
# =====================================================

def generate_survey_markdown(survey):

    md = f"# {survey.get('survey_title', 'Survey')}\n\n"

    md += survey.get("introduction", "") + "\n\n"

    for question in survey.get("questions", []):

        md += f"## {question.get('id')}: {question.get('text')}\n\n"

        md += f"**Type:** {question.get('type')}\n\n"

        options = question.get("options") or []

        if options:
            md += "**Options:**\n\n"

            for option in options:
                md += f"- [ ] {option}\n"

        md += f"\n**Purpose:** {question.get('purpose')}\n\n"

    notes = survey.get("design_notes", [])

    if notes:

        md += "## Methodology and Design Notes\n\n"

        for note in notes:
            md += f"- {note}\n"

    return md


def create_csv_template(survey):

    columns = [
        q.get("id", "Question")
        for q in survey.get("questions", [])
    ]

    df = pd.DataFrame(columns=columns)

    return df.to_csv(index=False)


# =====================================================
# DATASET UTILITIES
# =====================================================

def load_dataframe(uploaded_file):

    file_bytes = uploaded_file.getvalue()

    if uploaded_file.name.lower().endswith(".csv"):

        return pd.read_csv(io.BytesIO(file_bytes))

    return pd.read_excel(io.BytesIO(file_bytes))


def detect_column_types(df, survey=None):

    types_map = {}

    if survey:

        for question in survey.get("questions", []):

            q_id = question.get("id")

            if q_id in df.columns:
                types_map[q_id] = question.get("type")

    for column in df.columns:

        if column in types_map:
            continue

        series = df[column].dropna()

        if series.empty:
            types_map[column] = "open_ended"

        elif pd.api.types.is_numeric_dtype(series):

            values = series.unique()

            if set(values).issubset({0, 1}):
                types_map[column] = "yes_no"

            elif all(
                float(v).is_integer() and 1 <= float(v) <= 5
                for v in values
            ):
                types_map[column] = "likert"

            elif all(
                float(v).is_integer() and 1 <= float(v) <= 10
                for v in values
            ):
                types_map[column] = "rating"

            else:
                types_map[column] = "rating"

        else:

            values = series.astype(str).str.strip().str.lower().unique()

            if set(values).issubset(
                {"yes", "no", "true", "false", "y", "n"}
            ):
                types_map[column] = "yes_no"

            elif len(values) <= 8:
                types_map[column] = "multiple_choice"

            else:
                types_map[column] = "open_ended"

    return types_map


# =====================================================
# SAMPLE DATA
# =====================================================

def generate_sample_dataset():

    categories = [
        "Product Features",
        "Customer Support",
        "Pricing",
        "User Interface"
    ]

    comments = [
        "The interface is simple and easy to use.",
        "Customer support was excellent and very helpful.",
        "The pricing is too expensive for small businesses.",
        "Some important features are missing.",
        "The product works well but sometimes loads slowly.",
        "Overall, I am satisfied with the experience.",
        "The dashboard should include more visualizations.",
        "The product is easy to understand and navigate."
    ]

    rows = []

    for i in range(50):

        rows.append({
            "Q1": random.choice(["Yes", "No"]),
            "Q2": random.randint(1, 5),
            "Q3": random.randint(1, 10),
            "Q4": random.choice(categories),
            "Q5": random.choice(comments)
        })

    return pd.DataFrame(rows)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("⚙️ Settings")

    selected_model = st.selectbox(
        "AI Model",
        ["llama-3.3-70b-versatile"]
    )

    temperature = st.slider(
        "AI Creativity",
        0.0,
        1.0,
        0.2,
        0.05
    )

    st.divider()

    st.subheader("API Configuration")

    st.caption(
        "Add GROQ_API_KEY to Streamlit secrets or "
        "your environment variables."
    )

    if st.button("Check API Connection"):

        if get_groq_client():
            st.success("Groq API key detected.")
        else:
            st.error("Groq API key not found.")

groq_client = get_groq_client()


# =====================================================
# MAIN TABS
# =====================================================

tab1, tab2 = st.tabs([
    "📋 Survey Designer",
    "📈 Response Analyzer"
])


# =====================================================
# TAB 1: SURVEY DESIGNER
# =====================================================

with tab1:

    st.header("Create Your AI Survey")

    st.write(
        "Enter your research objective and target audience "
        "to generate a professional survey."
    )

    with st.form("survey_design_form"):

        objective = st.text_area(
            "Research Objective",
            placeholder="Example: Understand student satisfaction with online education."
        )

        audience = st.text_input(
            "Target Audience",
            placeholder="Example: University students"
        )

        question_count = st.slider(
            "Number of Questions",
            5,
            20,
            10
        )

        additional_context = st.text_area(
            "Additional Instructions",
            placeholder="Optional instructions..."
        )

        submitted = st.form_submit_button(
            "✨ Generate Survey",
            use_container_width=True
        )

    if submitted:

        if not objective.strip() or not audience.strip():

            st.error("Please enter both research objective and target audience.")

        elif not groq_client:

            st.error("Please configure your Groq API key first.")

        else:

            prompt = f"""
            Research Objective: {objective}

            Target Audience: {audience}

            Number of Questions: {question_count}

            Additional Instructions: {additional_context}

            Generate exactly {question_count} questions.
            Use sequential question IDs beginning with Q1.
            """

            with st.spinner("AI is designing your survey..."):

                try:

                    survey = call_groq_json(
                        groq_client,
                        selected_model,
                        DESIGNER_SYSTEM_PROMPT,
                        prompt,
                        temperature
                    )

                    if not survey.get("questions"):
                        raise ValueError("No questions were generated.")

                    st.session_state.survey_data = survey

                    st.success("Survey generated successfully!")

                except Exception as e:

                    st.error(str(e))

    if st.session_state.survey_data:

        survey = st.session_state.survey_data

        st.divider()

        st.subheader(survey.get("survey_title", "Generated Survey"))

        st.info(survey.get("introduction", ""))

        for question in survey.get("questions", []):

            with st.expander(
                f"{question.get('id')}: {question.get('text')}"
            ):

                st.write("**Question Type:**", question.get("type"))

                options = question.get("options") or []

                if options:

                    st.write("**Options:**")

                    for option in options:
                        st.write("•", option)

                st.write("**Purpose:**", question.get("purpose"))

        st.subheader("Survey Design Notes")

        for note in survey.get("design_notes", []):
            st.write("•", note)

        col1, col2 = st.columns(2)

        with col1:

            st.download_button(
                "⬇️ Download Survey",
                data=generate_survey_markdown(survey),
                file_name="survey.md",
                mime="text/markdown",
                use_container_width=True
            )

        with col2:

            st.download_button(
                "⬇️ Download CSV Template",
                data=create_csv_template(survey),
                file_name="survey_template.csv",
                mime="text/csv",
                use_container_width=True
            )


# =====================================================
# TAB 2: RESPONSE ANALYZER
# =====================================================

with tab2:

    st.header("Survey Response Analysis")

    st.write(
        "Upload survey responses in CSV or Excel format "
        "to explore your research data."
    )

    col1, col2 = st.columns([3, 1])

    with col1:

        uploaded_file = st.file_uploader(
            "Upload CSV or Excel File",
            type=["csv", "xlsx", "xls"]
        )

    with col2:

        st.write("")

        st.write("")

        if st.button("Load Sample Data", use_container_width=True):

            st.session_state.uploaded_df = generate_sample_dataset()

            st.session_state.qual_analysis = None

            st.session_state.executive_summary = None

    if uploaded_file is not None:

        try:

            st.session_state.uploaded_df = load_dataframe(uploaded_file)

            st.session_state.qual_analysis = None

            st.session_state.executive_summary = None

        except Exception as e:

            st.error(f"File loading error: {e}")

    df = st.session_state.uploaded_df

    if df is not None:

        if df.empty:

            st.warning("Your dataset is empty.")

        else:

            st.success(
                f"Successfully loaded {len(df)} responses."
            )

            st.subheader("Dataset Preview")

            st.dataframe(
                df.head(20),
                use_container_width=True
            )

            # ---------------- METRICS ----------------

            st.subheader("Dataset Overview")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Total Responses", len(df))

            c2.metric("Total Questions", len(df.columns))

            c3.metric(
                "Missing Values",
                int(df.isna().sum().sum())
            )

            c4.metric(
                "Duplicate Rows",
                int(df.duplicated().sum())
            )

            # ---------------- MISSING VALUES ----------------

            st.subheader("Missing Value Analysis")

            missing_df = pd.DataFrame({
                "Column": df.columns,
                "Missing Values": [
                    int(df[c].isna().sum())
                    for c in df.columns
                ],
                "Missing Percentage": [
                    round(df[c].isna().mean() * 100, 2)
                    for c in df.columns
                ]
            })

            st.dataframe(
                missing_df,
                use_container_width=True
            )

            # ---------------- CHARTS ----------------

            st.subheader("Interactive Data Visualization")

            types_map = detect_column_types(
                df,
                st.session_state.survey_data
            )

            selected_column = st.selectbox(
                "Select Survey Question",
                list(df.columns)
            )

            series = df[selected_column].dropna()

            if not series.empty:

                if pd.api.types.is_numeric_dtype(series):

                    chart_df = (
                        series.value_counts()
                        .sort_index()
                        .rename_axis("Response")
                        .reset_index(name="Count")
                    )

                    fig = px.bar(
                        chart_df,
                        x="Response",
                        y="Count",
                        text="Count",
                        title=f"Response Distribution: {selected_column}"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    a, b, c = st.columns(3)

                    a.metric("Average", round(series.mean(), 2))

                    b.metric("Minimum", series.min())

                    c.metric("Maximum", series.max())

                else:

                    counts = (
                        series.astype(str)
                        .value_counts()
                        .rename_axis("Response")
                        .reset_index(name="Count")
                    )

                    chart_type = st.radio(
                        "Choose Chart Type",
                        ["Bar Chart", "Pie Chart"],
                        horizontal=True
                    )

                    if chart_type == "Bar Chart":

                        fig = px.bar(
                            counts,
                            x="Response",
                            y="Count",
                            text="Count",
                            title="Response Distribution"
                        )

                    else:

                        fig = px.pie(
                            counts,
                            names="Response",
                            values="Count",
                            title="Response Distribution"
                        )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

            # ---------------- QUALITATIVE AI ANALYSIS ----------------

            st.divider()

            st.subheader("🧠 AI Qualitative Analysis")

            open_columns = [
                c for c in df.columns
                if types_map.get(c) == "open_ended"
                and df[c].dropna().astype(str).str.strip().ne("").any()
            ]

            if open_columns:

                text_column = st.selectbox(
                    "Select Open-Ended Question",
                    open_columns
                )

                max_responses = st.slider(
                    "Maximum Responses for AI",
                    10,
                    200,
                    min(100, max(10, len(df))),
                    10
                )

                if st.button(
                    "Analyze Written Feedback",
                    type="primary",
                    disabled=groq_client is None
                ):

                    responses = (
                        df[text_column]
                        .dropna()
                        .astype(str)
                        .str.strip()
                    )

                    responses = [
                        r for r in responses.tolist() if r
                    ][:max_responses]

                    if not responses:

                        st.warning("No written responses found.")

                    else:

                        prompt = f"""
                        Analyze these survey responses.

                        Question Column: {text_column}

                        Responses:
                        {json.dumps(responses, ensure_ascii=False)}
                        """

                        with st.spinner("AI is analyzing feedback..."):

                            try:

                                result = call_groq_json(
                                    groq_client,
                                    selected_model,
                                    QUALITATIVE_SYSTEM_PROMPT,
                                    prompt,
                                    temperature
                                )

                                st.session_state.qual_analysis = result

                                st.success("Qualitative analysis completed!")

                            except Exception as e:

                                st.error(str(e))

            else:

                st.info(
                    "No open-ended text column was detected. "
                    "You can still use the quantitative analysis."
                )

            # ---------------- DISPLAY QUALITATIVE RESULTS ----------------

            analysis = st.session_state.qual_analysis

            if analysis:

                st.subheader("Feedback Summary")

                st.write(analysis.get("summary", ""))

                themes = analysis.get("themes", [])

                if themes:

                    st.subheader("Main Feedback Themes")

                    st.dataframe(
                        pd.DataFrame(themes),
                        use_container_width=True
                    )

                sentiment = analysis.get("sentiment", {})

                if sentiment:

                    sentiment_df = pd.DataFrame({
                        "Sentiment": list(sentiment.keys()),
                        "Count": list(sentiment.values())
                    })

                    fig = px.pie(
                        sentiment_df,
                        names="Sentiment",
                        values="Count",
                        title="Customer Sentiment Analysis"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                st.subheader("AI Recommendations")

                for recommendation in analysis.get("recommendations", []):
                    st.write("💡", recommendation)

            # ---------------- EXECUTIVE SUMMARY ----------------

            st.divider()

            st.subheader("📑 Executive Summary Generator")

            if st.button(
                "Generate Executive Summary",
                disabled=groq_client is None
            ):

                summary_data = {
                    "total_responses": len(df),
                    "columns": list(df.columns),
                    "descriptive_statistics": df.describe(
                        include="all"
                    ).fillna("").to_dict(),
                    "missing_values": missing_df.to_dict(
                        orient="records"
                    ),
                    "qualitative_analysis": analysis or {}
                }

                with st.spinner("Generating executive report..."):

                    try:

                        summary = call_groq_text(
                            groq_client,
                            selected_model,
                            SUMMARY_SYSTEM_PROMPT,
                            json.dumps(
                                summary_data,
                                default=str,
                                ensure_ascii=False
                            ),
                            temperature
                        )

                        st.session_state.executive_summary = summary

                    except Exception as e:

                        st.error(str(e))

            if st.session_state.executive_summary:

                st.markdown(
                    st.session_state.executive_summary
                )

                st.download_button(
                    "⬇️ Download Executive Report",
                    data=st.session_state.executive_summary,
                    file_name="executive_summary.md",
                    mime="text/markdown"
                )

    else:

        st.info(
            "Upload your survey response file or click "
            "'Load Sample Data' to get started."
        )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "AI Survey Designer & Analyzer | Powered by Python, "
    "Streamlit, Plotly and Groq"
)
