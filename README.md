# 📄 Resume Screener

An AI-powered **Resume Screening System** that helps recruiters and hiring managers automatically filter, rank, and shortlist candidates based on job descriptions.  

This project leverages **LangChain, Google Generative AI, and ChromaDB** to analyze resumes, extract key skills, and provide relevance scores for better decision-making.  

---

## 🚀 Features
- ✅ Upload **Resumes** (PDF, DOCX, TXT)  
- ✅ Parse and Extract Information (Name, Skills, Experience, Education)  
- ✅ Compare Resumes with **Job Descriptions**  
- ✅ Relevance Scoring & Ranking  
- ✅ Interactive **Streamlit UI**  
- ✅ Vector DB (**ChromaDB**) for storage and retrieval  
- ✅ Uses **LangChain + Google Generative AI** for semantic matching  

---

## 🛠️ Tech Stack
- **Python 3.10+**
- **Frameworks & Libraries**:
  - `streamlit` – UI framework  
  - `langchain`, `langchain-community`, `langchain-google-genai` – LLM integration  
  - `google-generativeai` – Google Gemini API  
  - `chromadb` – Vector database for resume embeddings  
  - `python-dotenv` – Environment variable management  
  - `docx2txt`, `pypdf` – Resume parsing  
- **Other**: Pydantic pinned for compatibility  

---

## 📂 Project Structure
```

resume\_screener/
│-- app.py                # Main Streamlit app
│-- requirements.txt      # Project dependencies
│-- .env                  # API keys and environment variables
│-- chroma\_store/         # Vector DB storage
│-- sample\_resumes/       # Example resumes
│-- sample\_jd/            # Example job descriptions

````

---

## ⚡ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/AgileBA/Resume_screener.git
cd Resume_screener
````

### 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # On Linux/Mac
.venv\Scripts\activate      # On Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Setup Environment Variables

Create a **.env** file in the project root and add your API keys:

```
GOOGLE_API_KEY=your_google_generative_ai_key
```

### 5️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 🎯 Usage

1. Upload one or more resumes.
2. Paste or upload a Job Description.
3. Click **"Screen Candidates"**.
4. Get **ranked resumes** with match scores.


## 🤝 Contributing

Contributions are welcome! Fork this repo, make changes, and submit a PR.

---



