# app.py - FINAL VERSION WITH FAISS INSTEAD OF CHROMA

import os
import streamlit as st
from dotenv import load_dotenv
import json

# LangChain components
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Pydantic for data structure
from pydantic import BaseModel, Field
from typing import List

# --- Load Environment Variables ---
load_dotenv()

@st.cache_resource
def get_llm():
    """
    Initializes and returns the Gemini LLM.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("Google API Key not found. Please set it in your .env file.")

    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.2
    )

@st.cache_resource
def get_embedding_model():
    """
    Initializes and returns the Google Generative AI embedding model.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("Google API Key not found. Please set it in your .env file.")

    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

@st.cache_resource
def get_vector_store():
    """Initializes and returns the FAISS vector store (persistent)."""
    embedding_function = get_embedding_model()
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embedding_function, allow_dangerous_deserialization=True)
    else:
        vector_store = FAISS.from_texts(["initial placeholder"], embedding_function)
        vector_store.save_local("faiss_index")
        return vector_store

# --- Helper Function to Load Documents ---
def load_document(file):
    try:
        name, extension = os.path.splitext(file.name)
        extension = extension.lower()
        temp_file_path = f"temp_{file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(file.getbuffer())

        if extension == '.pdf':
            loader = PyPDFLoader(temp_file_path)
        elif extension == '.docx':
            loader = Docx2txtLoader(temp_file_path)
        elif extension == '.txt':
            loader = TextLoader(temp_file_path)
        else:
            st.error(f"Unsupported file format: {extension}")
            return None

        documents = loader.load()
        text = " ".join([doc.page_content for doc in documents])
        os.remove(temp_file_path)
        return text
    except Exception as e:
        st.error(f"Error loading or parsing file: {e}")
        return None

# --- Pydantic Model for Structured Output ---
class ResumeAnalysis(BaseModel):
    match_score: int = Field(description="The suitability score as a percentage (0-100).", ge=0, le=100)
    strengths: List[str] = Field(description="Key strengths of the candidate.")
    weaknesses: List[str] = Field(description="Key weaknesses or missing qualifications.")
    summary: str = Field(description="Overall summary of the candidate's fit.")

# --- Main App Logic ---
def main():
    st.set_page_config(page_title="🎯 AI Resume Screener", layout="wide")

    st.markdown("""
    <style>
        .stButton>button { background-color: #4CAF50; color: white; border-radius: 8px; padding: 10px 20px; border: none; font-size: 16px; }
        .stProgress > div > div > div > div { background-color: #4CAF50; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🎯 AI-Powered Resume Screener & Search")

    # --- Gracefully Initialize Models ---
    try:
        llm = get_llm()
        vector_store = get_vector_store()
    except ValueError as e:
        st.error(e)
        st.stop()

    # --- UI Tabs ---
    tab1, tab2 = st.tabs(["Analyze New Resume", "Search Past Candidates"])

    # === TAB 1: Analyze New Resume ===
    with tab1:
        st.header("Analyze a New Resume")
        with st.form("analysis_form"):
            col1, col2 = st.columns([2, 1])
            with col1:
                job_description = st.text_area("📋 **Job Description**", height=300, placeholder="e.g., Senior Python Developer...")
            with col2:
                uploaded_file = st.file_uploader("📄 **Upload Resume**", type=["pdf", "docx", "txt"])

            submitted = st.form_submit_button("Analyze Resume")

        if submitted and job_description and uploaded_file:
            with st.spinner("Analyzing... The AI is at work! 🧠"):
                resume_text = load_document(uploaded_file)
                if resume_text:
                    parser = PydanticOutputParser(pydantic_object=ResumeAnalysis)
                    prompt = PromptTemplate(
                        template="""You are an expert HR analyst. Analyze the provided resume against the job description.
                        Your goal is to provide a structured analysis in JSON format.\n
                        Job Description:\n{job_description}\nResume Text:\n{resume_text}\n{format_instructions}""",
                        input_variables=["job_description", "resume_text"],
                        partial_variables={"format_instructions": parser.get_format_instructions()},
                    )
                    chain = prompt | llm | parser
                    try:
                        analysis_result = chain.invoke({"job_description": job_description, "resume_text": resume_text})
                        st.success("✅ Analysis Complete!")

                        # Split and add resume to FAISS
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                        chunks = text_splitter.create_documents([resume_text])
                        for chunk in chunks:
                            chunk.metadata = {
                                "filename": uploaded_file.name,
                                "analysis_summary": analysis_result.summary,
                                "match_score": analysis_result.match_score,
                                "strengths": json.dumps(analysis_result.strengths),
                                "weaknesses": json.dumps(analysis_result.weaknesses),
                            }
                        vector_store.add_documents(chunks, ids=[f"{uploaded_file.name}_{i}" for i in range(len(chunks))])
                        vector_store.save_local("faiss_index")

                        # Show results
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("📊 Suitability Score")
                            score = analysis_result.match_score
                            st.progress(score / 100)
                            st.metric(label="Match Percentage", value=f"{score}%")
                        with col2:
                            st.subheader("📝 Overall Summary")
                            st.write(analysis_result.summary)

                        st.subheader("👍 Strengths")
                        for s in analysis_result.strengths: st.markdown(f"- {s}")
                        st.subheader("👎 Weaknesses")
                        for w in analysis_result.weaknesses: st.markdown(f"- {w}")
                    except Exception as e:
                        st.error(f"An error occurred during analysis: {e}")

    # === TAB 2: Search Past Candidates ===
    with tab2:
        st.header("Search the Candidate Knowledge Base")
        search_query = st.text_input("Enter keywords to search for in resumes", "")
        if st.button("Search Candidates"):
            if search_query:
                with st.spinner("Searching..."):
                    try:
                        results = vector_store.similarity_search_with_score(search_query, k=5)
                        st.success(f"Found {len(results)} relevant document chunks.")
                        candidates = {}
                        for doc, score in results:
                            filename = doc.metadata.get('filename')
                            if filename not in candidates:
                                candidates[filename] = {
                                    'score': score,
                                    'summary': doc.metadata.get('analysis_summary', 'N/A'),
                                    'match_score': doc.metadata.get('match_score', 'N/A'),
                                    'strengths': json.loads(doc.metadata.get('strengths', '[]')),
                                    'weaknesses': json.loads(doc.metadata.get('weaknesses', '[]')),
                                    'chunks': []
                                }
                            candidates[filename]['chunks'].append(doc.page_content)

                        for filename, data in candidates.items():
                            with st.expander(f"**📄 {filename}** (Relevance: {data['score']:.2f} | Original Match: {data['match_score']}%)"):
                                st.markdown(f"**Summary:** {data['summary']}")
                                st.markdown("**Strengths:**")
                                for s in data['strengths']: st.write(f"- {s}")
                                st.markdown("**Weaknesses:**")
                                for w in data['weaknesses']: st.write(f"- {w}")
                                st.markdown("**Relevant Excerpts from Resume:**")
                                for chunk in data['chunks']: st.info(f"...{chunk}...")
                    except Exception as e:
                        st.error(f"An error occurred during search: {e}")
            else:
                st.warning("Please enter a search query.")

if __name__ == "__main__":
    main()
