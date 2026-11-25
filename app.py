import json
import os
import time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Load environment variables from .env
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Flashcard Generator",
    page_icon="🗂️",
    layout="wide",
)

st.title("🗂️ AI Flashcard Generator")
st.write(
    "Paste your study notes to generate question-and-answer flashcards for quick revision and export them as a CSV."
)


# Pydantic models to enforce structured JSON output from the LLM
class Flashcard(BaseModel):
    question: str = Field(
        description="Clear, concise revision question based directly on the study notes."
    )
    answer: str = Field(
        description="Accurate, direct answer based on the study notes."
    )


class FlashcardSet(BaseModel):
    flashcards: list[Flashcard]


def strip_markdown_fences(text: str) -> str:
    """Remove optional ```json ... ``` fences if present."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def generate_flashcards_df(
    api_key: str,
    model_name: str,
    notes: str,
    num_cards: int,
    status_container,
) -> pd.DataFrame:
    """Call Gemini to produce Q&A flashcards as JSON and load into a pandas DataFrame."""
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=30000),
    )

    prompt = f"""
You are a study assistant helping students revise quickly.
Convert the study notes below into {num_cards} question-and-answer flashcards.

Rules:
- Focus on key definitions, concepts, formulas, and important facts.
- Keep questions clear and answers concise.
- Use only facts supported by the provided study notes.

Study Notes:
{notes}
""".strip()

    response = None
    max_attempts = 4

    for attempt in range(max_attempts):
        try:
            status_container.caption(
                f"Requesting flashcards from `{model_name}` (attempt {attempt + 1}/{max_attempts})..."
            )
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=FlashcardSet,
                    temperature=0.3,
                ),
            )
            break
        except Exception as err:
            err_msg = str(err)
            is_transient = (
                "503" in err_msg
                or "UNAVAILABLE" in err_msg
                or "504" in err_msg
                or "DEADLINE_EXCEEDED" in err_msg
            )
            if is_transient and attempt < max_attempts - 1:
                wait_seconds = 3 * (attempt + 1)
                status_container.caption(
                    f"Model busy or slow ({err_msg[:60]}...). Retrying in {wait_seconds}s..."
                )
                time.sleep(wait_seconds)
                continue
            raise

    raw_text = strip_markdown_fences(response.text or "")
    parsed_json = json.loads(raw_text)
    cards = parsed_json.get("flashcards", [])

    if not cards:
        raise ValueError("No flashcards were returned by the model.")

    df = pd.DataFrame(cards)
    df = df[["question", "answer"]].rename(
        columns={"question": "Question", "answer": "Answer"}
    )
    return df


# Sidebar controls
with st.sidebar:
    st.header("Settings")
    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    sidebar_key = st.text_input(
        "Gemini API Key",
        value=env_key,
        type="password",
        help="Loaded automatically from .env if present.",
    ).strip()
    api_key = sidebar_key or env_key

    model_choice = st.selectbox(
        "Gemini Model",
        options=[
            "gemini-flash-latest",
            "gemini-flash-lite-latest",
            "gemini-3.5-flash",
            "gemini-3.8-flash",
            "Custom",
        ],
        index=0,
    )

    if model_choice == "Custom":
        model_name = st.text_input(
            "Enter Custom Model ID",
            value="gemini-flash-latest",
        ).strip()
    else:
        model_name = model_choice

    num_cards = st.slider(
        "Number of Flashcards",
        min_value=3,
        max_value=15,
        value=5,
    )

# Main input area
notes_text = st.text_area(
    "Paste your study notes here:",
    height=220,
    placeholder="Paste lecture notes, textbook paragraphs, or topic summaries...",
)

if st.button("Generate Flashcards", type="primary"):
    if not api_key:
        st.error("Please enter your GEMINI_API_KEY in .env or in the sidebar.")
    elif not model_name:
        st.error("Please select or enter a valid model name.")
    elif not notes_text.strip():
        st.warning("Please paste some study notes before generating flashcards.")
    else:
        status_box = st.empty()
        try:
            with st.spinner("Generating flashcards..."):
                df = generate_flashcards_df(
                    api_key=api_key,
                    model_name=model_name,
                    notes=notes_text,
                    num_cards=num_cards,
                    status_container=status_box,
                )
            status_box.empty()
            st.session_state["flashcards_df"] = df
            st.success(f"Generated {len(df)} flashcards!")
        except Exception as exc:
            status_box.empty()
            st.error(f"Error: {exc}")

# Render flashcards and CSV export button
if (
    "flashcards_df" in st.session_state
    and isinstance(st.session_state["flashcards_df"], pd.DataFrame)
    and not st.session_state["flashcards_df"].empty
):
    df = st.session_state["flashcards_df"]

    st.divider()
    top_left, top_right = st.columns([3, 1])
    with top_left:
        st.subheader("Revision Flashcards")
    with top_right:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv_bytes,
            file_name="flashcards.csv",
            mime="text/csv",
            use_container_width=True,
        )

    grid_cols = st.columns(2)
    for idx, row in df.iterrows():
        with grid_cols[idx % 2]:
            with st.container(border=True):
                st.markdown(f"**Card {idx + 1}**")
                st.markdown(f"**Q:** {row['Question']}")
                with st.expander("Show Answer"):
                    st.write(row["Answer"])

    with st.expander("View DataFrame Table"):
        st.dataframe(df, use_container_width=True)