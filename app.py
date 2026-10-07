import os
import json

import numpy as np
import faiss
import streamlit as st
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader


# Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("Gemini API key not found. Please check your .env file.")
    st.stop()


# Create Gemini client
client = genai.Client(api_key=API_KEY)
def classify_question(question, pdf_available):
    """
    Decide which feature should handle the student's question.
    """

    router_prompt = f"""
You are an AI router for a student study assistant.

Choose exactly ONE category for the student's request:

- chat: General study questions or explanations.
- pdf: Questions that ask about the uploaded document or notes.
- quiz: Requests to create, generate, or take a quiz.

PDF available: {pdf_available}

Student request:
{question}

Return ONLY one word:
chat
pdf
quiz
"""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=router_prompt
    )

    category = response.text.strip().lower()

    if category not in ["chat", "pdf", "quiz"]:
        category = "chat"

    return category

@st.cache_data
def create_chunks(text, chunk_size=1500):
    """
    Split document text into smaller chunks.
    """

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def retrieve_relevant_chunks(question, chunks, top_k=3):
    """
    Find the document chunks most relevant to the question.
    """

    question_words = set(question.lower().split())

    scored_chunks = []

    for chunk in chunks:

        chunk_words = set(chunk.lower().split())

        score = len(
            question_words.intersection(chunk_words)
        )

        scored_chunks.append((score, chunk))

    scored_chunks.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [
        chunk
        for score, chunk in scored_chunks[:top_k]
        if score > 0
    ]

@st.cache_resource
def create_faiss_index(chunks):
    """
    Create a FAISS vector index from document chunks.
    """

    if not chunks:
        return None

    # Generate embeddings for all document chunks
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunks
    )

    embeddings = np.array(
        [embedding.values for embedding in result.embeddings],
        dtype="float32"
    )

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    # Create FAISS index
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index

def search_faiss(question, chunks, index, top_k=3):
    """
    Find the most relevant document chunks using FAISS.
    """

    # Create an embedding for the student's question
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question
    )

    question_embedding = np.array(
        [result.embeddings[0].values],
        dtype="float32"
    )

    # Normalize for cosine similarity
    faiss.normalize_L2(question_embedding)

    # Search FAISS
    scores, indices = index.search(
        question_embedding,
        min(top_k, len(chunks))
    )

    relevant_chunks = []

    for index_position in indices[0]:

        if index_position >= 0:
            relevant_chunks.append(
                chunks[index_position]
            )

    return relevant_chunks

# Page configuration
st.set_page_config(
    page_title="AI Student Study Assistant",
    page_icon="🎓",
    layout="centered"
)


# Title
st.title("🎓 AI Student Study Assistant")
st.write(
    "Ask questions about programming, computer science, "
    "and other study topics."
)

# PDF upload
uploaded_file = st.file_uploader(
    "📄 Upload your study notes or PDF",
    type=["pdf"]
)

if uploaded_file:
    reader = PdfReader(uploaded_file)

    pdf_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pdf_text += text + "\n"

    st.success(
        f"PDF uploaded successfully! "
        f"{len(reader.pages)} pages found."
    )

    pdf_chunks = create_chunks(pdf_text)

    st.info(
        f"Document divided into {len(pdf_chunks)} chunks for retrieval."
    )

    pdf_index = create_faiss_index(tuple(pdf_chunks))

    st.success("FAISS vector index created successfully.")
    
# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "quiz" not in st.session_state:
    st.session_state.quiz = None

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

# Quiz Generator
st.subheader("🧠 AI Quiz Generator")

if uploaded_file and pdf_text.strip():

    if st.button("Generate Quiz from PDF"):

        with st.spinner("Generating quiz from your document..."):

            quiz_chunks = search_faiss(
                "important concepts, topics, definitions, architecture, objectives, methods, and key information from the document",
                pdf_chunks,
                pdf_index,
                top_k=5
            )

            quiz_context = "\n\n---\n\n".join(
                quiz_chunks
            )

            quiz_prompt = f"""
You are an AI quiz generator.

Create a quiz with exactly 5 multiple-choice questions
based ONLY on the retrieved sections from the uploaded document.

Each question must have:
- A question
- Exactly 4 options
- One correct answer

Make the questions test different concepts from the
provided study material.

Return ONLY valid JSON in this format:

[
    {{
        "question": "Question text",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "answer": "Option A"
    }}
]

Do not include markdown or any explanation.

RETRIEVED STUDY MATERIAL:
{quiz_context}
"""

            try:

                response = client.models.generate_content(
                    model="gemini-3.1-flash-lite",
                    contents=quiz_prompt
                )

                quiz_text = response.text.strip()

                if quiz_text.startswith("```"):
                    quiz_text = quiz_text.replace("```json", "")
                    quiz_text = quiz_text.replace("```", "")
                    quiz_text = quiz_text.strip()

                st.session_state.quiz = json.loads(quiz_text)
                st.session_state.quiz_submitted = False

            except Exception as e:

                st.error(f"Could not generate quiz: {e}")


# Display Quiz
if st.session_state.quiz:

    st.divider()
    st.subheader("📝 Your Quiz")

    answers = {}

    for i, question in enumerate(st.session_state.quiz):

        st.write(f"**{i + 1}. {question['question']}**")

        answers[i] = st.radio(
            "Choose an answer:",
            question["options"],
            key=f"question_{i}"
        )

    if st.button("Submit Quiz"):

        score = 0

        for i, question in enumerate(st.session_state.quiz):

            if answers[i] == question["answer"]:
                score += 1

        st.session_state.quiz_submitted = True

        st.success(
            f"🎉 Your score: {score}/{len(st.session_state.quiz)}"
        )

        if score == len(st.session_state.quiz):
            st.balloons()

else:

    if not uploaded_file:
        st.info("Upload a PDF to generate an AI quiz.")

# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# Chat input
question = st.chat_input("Ask your question...")


if question:

    # Display user's question
    with st.chat_message("user"):
        st.write(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Decide which AI feature should handle the request
    pdf_available = uploaded_file is not None and pdf_text.strip()

    try:

        category = classify_question(
            question,
            pdf_available
        )

        # Show which agent was selected
        st.caption(f"🤖 Agent selected: {category.upper()}")

        with st.chat_message("assistant"):

            # -----------------------------------------
            # PDF AGENT
            # -----------------------------------------

            if category == "pdf" and pdf_available:

                relevant_chunks = search_faiss(
                    question,
                    pdf_chunks,
                    pdf_index,
                    top_k=3
                )

                if relevant_chunks:

                    retrieved_context = "\n\n---\n\n".join(
                        relevant_chunks
                    )

                else:

                    retrieved_context = pdf_text

                prompt = f"""
            You are the Document Q&A Agent of an AI Student Study Assistant.

            Answer the student's question using ONLY the retrieved
            sections from the uploaded document.

            Rules:
            - Use the retrieved document sections as the primary source.
            - Do not invent information.
            - If the answer is not available in the retrieved sections,
            clearly say so.
            - Explain the answer clearly and simply.

            RETRIEVED DOCUMENT SECTIONS:
            {retrieved_context}

            STUDENT QUESTION:
            {question}
            """

            # -----------------------------------------
            # QUIZ AGENT
            # -----------------------------------------

            elif category == "quiz":

                if not pdf_available:

                    answer = (
                        "Please upload a PDF first so I can "
                        "generate a quiz from your study material."
                    )

                    st.write(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                    st.stop()

                quiz_chunks = search_faiss(
                    question,
                    pdf_chunks,
                    pdf_index,
                    top_k=5
                )

                quiz_context = "\n\n---\n\n".join(
                    quiz_chunks
                )

                prompt = f"""
                You are the Quiz Agent of an AI Student Study Assistant.

                The student wants a quiz based on their uploaded document.

                Create a short quiz based ONLY on the retrieved
                sections from the document.

                RETRIEVED STUDY MATERIAL:
                {quiz_context}

                STUDENT REQUEST:
                {question}

                Provide 5 multiple-choice questions with
                four options each and clearly indicate the correct answer.
                """

            # -----------------------------------------
            # CHAT AGENT
            # -----------------------------------------

            else:

                prompt = f"""
You are the Chat Agent of an AI Student Study Assistant.

Answer the student's question clearly and simply.

Use examples when useful.

STUDENT QUESTION:
{question}
"""

            # -----------------------------------------
            # Generate response
            # -----------------------------------------

            if category != "quiz" or pdf_available:

                response = client.models.generate_content(
                    model="gemini-3.1-flash-lite",
                    contents=prompt
                )

                answer = response.text

                st.write(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

    except Exception as e:

        st.error(
            f"Something went wrong while contacting Gemini: {e}"
        )