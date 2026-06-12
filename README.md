# 🎯 AI Resume Analyzer & Skill Gap Detector

An AI-powered Resume Analyzer built using **NLP, Machine Learning, TF-IDF, Semantic Similarity, and ATS Simulation** to evaluate how well a resume matches a target job description.

The system analyzes resumes against job descriptions, computes multiple similarity metrics, identifies missing skills, simulates ATS scoring, and generates personalized improvement recommendations.

---

## 🚀 Features

### Resume Analysis

* Upload resumes in **PDF** or **TXT** format
* Extract and preprocess resume text
* Analyze resume against a target job description

### Match Scoring Engine

* TF-IDF Vectorization
* Cosine Similarity
* Sentence Transformer Semantic Similarity
* Keyword Overlap Analysis
* Skill Coverage Scoring

### Skill Gap Detection

* Extract technical skills from resumes
* Extract required skills from job descriptions
* Identify:

  * Matched skills
  * Missing skills
  * Additional skills
* Recommend top skills to learn

### ATS Resume Evaluation

Simulates Applicant Tracking System checks:

* Resume sections detection
* Action verb analysis
* Quantified achievement detection
* Keyword matching
* Resume length optimization

### Keyword Intelligence

* Important keyword extraction from job descriptions
* Missing keyword detection
* Keyword density analysis

### Visualization Dashboard

* Match Score Gauge
* ATS Score Gauge
* Skill Coverage Charts
* Missing Skill Priority Charts
* Score Breakdown Visualizations

### Multiple Resume Comparison

* Compare up to 3 resumes simultaneously
* Rank candidates against the same job description
* Side-by-side performance metrics

### Downloadable Reports

Generate detailed resume analysis reports including:

* Match scores
* ATS scores
* Skill gap analysis
* Missing keywords
* Improvement recommendations

---

## 🧠 Machine Learning Pipeline

Resume → Text Extraction → Text Cleaning → Tokenization → Stopword Removal → Lemmatization → TF-IDF Vectorization → Semantic Embeddings → Similarity Scoring → Skill Extraction → ATS Evaluation → Recommendations

---

## 🛠 Tech Stack

### Frontend

* Streamlit

### Data Processing

* Pandas
* NumPy

### Natural Language Processing

* NLTK
* TF-IDF
* Sentence Transformers

### Machine Learning

* Scikit-learn
* Cosine Similarity

### Visualization

* Matplotlib

### PDF Processing

* pdfplumber
* PyPDF2

---

## 📂 Project Structure

```text
AI-Resume-Analyzer/
│
├── app.py
│
├── data/
│   └── skills_database.csv
│
├── utils/
│   ├── text_preprocessing.py
│   ├── skill_extractor.py
│   └── similarity.py
│
├── requirements.txt
│
└── README.md
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/AI-Resume-Analyzer.git

cd AI-Resume-Analyzer
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / Mac**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will launch locally in your browser.

---

## 📊 Scoring Methodology

The final match score combines multiple signals:

| Component           | Purpose                           |
| ------------------- | --------------------------------- |
| TF-IDF Similarity   | Measures textual similarity       |
| Semantic Similarity | Captures contextual meaning       |
| Keyword Overlap     | Detects exact keyword matches     |
| Skill Coverage      | Measures required skills coverage |

These signals are combined into a final score ranging from **0–100**.

---

## 🎯 ATS Evaluation Factors

The ATS simulator evaluates:

* Resume Sections
* Action Verbs
* Quantified Achievements
* Job Description Keyword Match
* Resume Length

It then generates ATS feedback and optimization suggestions.

---

## 📈 Example Output

The system provides:

* Overall Match Score
* ATS Score
* Skills Matched
* Skills Missing
* Missing Keywords
* Skill Coverage Percentage
* Personalized Recommendations
* Downloadable Analysis Report

---

## Future Improvements

* Resume Authentication & User Accounts
* Resume History Tracking
* LLM-Based Resume Suggestions
* Resume Rewrite Assistant
* Cover Letter Generator
* Job Recommendation Engine
* Cloud Deployment
* Recruiter Dashboard

---

## Author

Built by an Electronics and Communication Engineering undergraduate interested in:

* Machine Learning
* Natural Language Processing
* Generative AI
* Applied AI Systems
* Software Development

---

## 📸 Demo

<p align="center">
  <img src="demo pics/1st.png" width="800">
</p>

<p align="center">
  <img src="demo pics/3rd.png" width="800">
</p>

<p align="center">
  <img src="demo pics/5th.png" width="800">
</p>

<p align="center">
  <img src="demo pics/6th.png" width="800">
</p>

## License

This project is intended for educational, research, and portfolio purposes.
