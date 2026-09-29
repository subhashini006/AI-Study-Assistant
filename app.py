"""AI Study Assistant - main Streamlit app."""

import os

import streamlit as st
from dotenv import load_dotenv

from utils.pdf_processor import extract_text_from_pdf
from utils.study_tools import generate_summary, answer_question, generate_quiz
from utils.vector_store import chunk_text, build_index, search_index

# Load the API key and settings from the .env file
load_dotenv()

# ---------- Page setup ----------
st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="🎓",
    layout="wide",
)


# ---------- Session state ----------
# st.session_state remembers values while the user moves between pages.
def init_state():
    defaults = {
        "pdf_name": None,
        "pdf_text": "",
        "page_count": 0,
        "summary": "",
        "chunks": None,
        "index": None,
        "chat_history": [],
        "quiz_questions": None,
        "quiz_submitted": False,
        "quiz_answers": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# ---------- Pages ----------
def home_page():
    st.title("🎓 AI Study Assistant")
    st.write("Turn any PDF study material into a personal study tool.")

    st.subheader("What you can do")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
- 📄 **Upload** your PDF notes or textbook
- 📝 **Summary** with headings and bullet points
"""
        )
    with col2:
        st.markdown(
            """
- 💬 **Ask AI** questions about your material
- ❓ **Quiz** yourself with multiple-choice questions
"""
        )

    st.info("👈 Start by opening **Upload Material** in the sidebar.")


def upload_page():
    st.title("📄 Upload Material")
    st.write("Upload a PDF that contains selectable text (not just scanned images).")

    uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded is not None:
        # Only process the file if it is new (avoids re-reading on every click)
        if uploaded.name != st.session_state.pdf_name:
            with st.spinner("Extracting text from your PDF..."):
                try:
                    text, pages = extract_text_from_pdf(uploaded.read())
                except Exception as error:
                    st.error(f"Could not read this PDF: {error}")
                    return

            if not text:
                st.warning(
                    "No text found. This PDF may be a scanned image. "
                    "Please try a PDF with selectable text."
                )
                return

            st.session_state.pdf_name = uploaded.name
            st.session_state.pdf_text = text
            st.session_state.page_count = pages
            # Clear old data since this is a new document
            st.session_state.chunks = None
            st.session_state.index = None
            st.session_state.chat_history = []
            st.session_state.summary = ""
            st.session_state.quiz_questions = None
            st.session_state.quiz_submitted = False
            st.session_state.quiz_answers = {}
            st.success("✅ PDF uploaded and text extracted!")

    # Show details if a PDF is loaded
    if st.session_state.pdf_name:
        text = st.session_state.pdf_text

        st.subheader(f"📘 {st.session_state.pdf_name}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Pages", st.session_state.page_count)
        col2.metric("Words", f"{len(text.split()):,}")
        col3.metric("Characters", f"{len(text):,}")

        with st.expander("Preview extracted text"):
            st.text(text[:3000] + ("..." if len(text) > 3000 else ""))


def summary_page():
    st.title("📝 Summary")

    if not st.session_state.pdf_name:
        st.warning("Please upload a PDF first (📄 Upload Material).")
        return

    st.write(f"Material: **{st.session_state.pdf_name}**")

    if st.button("Generate Summary", type="primary"):
        with st.spinner("Reading your material and writing a summary..."):
            try:
                st.session_state.summary = generate_summary(st.session_state.pdf_text)
            except Exception as error:
                st.error(f"Something went wrong: {error}")
                return

    if st.session_state.summary:
        st.divider()
        st.markdown(st.session_state.summary)

        st.download_button(
            "⬇️ Download summary as text",
            data=st.session_state.summary,
            file_name="summary.md",
            mime="text/markdown",
        )
    else:
        st.info("Click **Generate Summary** to create your study summary.")


def ensure_index_built():
    """Build the semantic search index once per PDF, and cache it."""
    if st.session_state.index is None:
        with st.spinner("Preparing your material for Q&A (first time only)..."):
            chunks = chunk_text(st.session_state.pdf_text)
            index = build_index(chunks)
            st.session_state.chunks = chunks
            st.session_state.index = index


def ask_ai_page():
    st.title("💬 Ask AI")

    if not st.session_state.pdf_name:
        st.warning("Please upload a PDF first (📄 Upload Material).")
        return

    st.write(f"Ask questions about **{st.session_state.pdf_name}**")

    try:
        ensure_index_built()
    except Exception as error:
        st.error(f"Could not prepare the document for search: {error}")
        return

    # Show past questions and answers
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])

    question = st.chat_input("Type your question about the material...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching your material..."):
                try:
                    relevant_chunks = search_index(
                        question, st.session_state.index, st.session_state.chunks
                    )
                    answer = answer_question(question, relevant_chunks)
                except Exception as error:
                    answer = f"Something went wrong: {error}"
            st.write(answer)

        st.session_state.chat_history.append({"question": question, "answer": answer})

    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat"):
            st.session_state.chat_history = []
            st.rerun()


def quiz_page():
    st.title("❓ Quiz")

    if not st.session_state.pdf_name:
        st.warning("Please upload a PDF first (📄 Upload Material).")
        return

    st.write(f"Test yourself on **{st.session_state.pdf_name}**")

    col1, col2 = st.columns([1, 3])
    with col1:
        num_questions = st.selectbox("Number of questions", [5, 6, 7, 8, 9, 10], index=1)

    if st.button("🎲 Generate New Quiz", type="primary"):
        with st.spinner("Creating your quiz..."):
            try:
                st.session_state.quiz_questions = generate_quiz(
                    st.session_state.pdf_text, num_questions=num_questions
                )
                st.session_state.quiz_submitted = False
                st.session_state.quiz_answers = {}
            except Exception as error:
                st.error(f"Could not generate quiz: {error}")
                return

    questions = st.session_state.quiz_questions
    if not questions:
        st.info("Click **Generate New Quiz** to start.")
        return

    st.divider()

    # ---- Quiz form: all questions submit together ----
    with st.form("quiz_form"):
        for i, q in enumerate(questions):
            st.markdown(f"**{i + 1}. {q['question']}**")
            selected = st.radio(
                "Choose one:",
                options=list(range(4)),
                format_func=lambda idx, opts=q["options"]: opts[idx],
                key=f"quiz_q_{i}",
                index=None,
                label_visibility="collapsed",
            )
            st.session_state.quiz_answers[i] = selected
            st.write("")  # small spacing

        submitted = st.form_submit_button("✅ Submit Quiz", type="primary")

    if submitted:
        unanswered = [
            i for i, ans in st.session_state.quiz_answers.items() if ans is None
        ]
        if unanswered:
            st.warning("Please answer all questions before submitting.")
        else:
            st.session_state.quiz_submitted = True

    # ---- Results ----
    if st.session_state.quiz_submitted:
        st.divider()
        st.subheader("📊 Results")

        score = 0
        for i, q in enumerate(questions):
            user_choice = st.session_state.quiz_answers[i]
            correct_index = q["correct_index"]
            is_correct = user_choice == correct_index

            if is_correct:
                score += 1

            with st.container(border=True):
                st.markdown(f"**{i + 1}. {q['question']}**")
                if is_correct:
                    st.success(f"✅ Your answer: {q['options'][user_choice]} — Correct!")
                else:
                    st.error(f"❌ Your answer: {q['options'][user_choice]}")
                    st.info(f"Correct answer: {q['options'][correct_index]}")
                st.caption(f"💡 {q.get('explanation', '')}")

        percent = round((score / len(questions)) * 100)
        st.divider()
        st.metric("Final Score", f"{score} / {len(questions)}", f"{percent}%")

        if st.button("🔄 Retake with New Questions"):
            st.session_state.quiz_questions = None
            st.session_state.quiz_submitted = False
            st.session_state.quiz_answers = {}
            st.rerun()


# ---------- Sidebar navigation ----------
PAGES = [
    "🏠 Home",
    "📄 Upload Material",
    "📝 Summary",
    "💬 Ask AI",
    "❓ Quiz",
]

st.sidebar.title("🎓 AI Study Assistant")
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.divider()

# Status boxes so you can see what is ready
if st.session_state.pdf_name:
    st.sidebar.success(f"Loaded: {st.session_state.pdf_name}")
else:
    st.sidebar.info("No PDF loaded yet")

if os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_API_KEY") != "paste_your_key_here":
    st.sidebar.success("API key found ✅")
else:
    st.sidebar.warning("API key missing (check .env)")


# ---------- Show the selected page ----------
if page == "🏠 Home":
    home_page()
elif page == "📄 Upload Material":
    upload_page()
elif page == "📝 Summary":
    summary_page()
elif page == "💬 Ask AI":
    ask_ai_page()
elif page == "❓ Quiz":
    quiz_page()