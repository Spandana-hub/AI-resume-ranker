"""
app.py — AI Resume Ranker & Skill Gap Analyzer
================================================
A Streamlit application that compares a candidate's resume
against a job description using classical NLP and ML techniques.

Features:
  ✅ Resume upload (PDF / TXT)
  ✅ TF-IDF + Cosine Similarity match scoring
  ✅ Skill gap detection from a curated skills database
  ✅ ATS score simulation
  ✅ Keyword analysis (found / missing)
  ✅ Personalized improvement suggestions
  ✅ Visualizations (match gauge, skill coverage, missing skills)
  ✅ Downloadable analysis report
  ✅ Multiple resume comparison
"""

import io
import os
import sys
import time
import base64
import datetime
import warnings

warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (required for Streamlit)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec

# ── Path setup so utils/ is importable ──────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from utils.text_preprocessing import preprocess_text
from utils.skill_extractor import (
    load_skills_database,
    extract_skills_from_text,
    compare_skills,
    get_top_suggested_skills,
    keyword_density_analysis,
)
from utils.similarity import (
    compute_match_score,
    compute_ats_score,
    extract_important_keywords,
    find_keywords_in_resume,
    generate_improvement_suggestions,
)


# ═══════════════════════════════════════════════════════════════════
#  PAGE CONFIG & GLOBAL STYLES
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a cohesive, premium dark-themed dashboard
st.markdown("""
<style>
            
/* Fix: Keep the header active so the sidebar can reopen, but make it match the background color */
header[data-testid="stHeader"] {
    background-color: #0f1117 !important;
    border-bottom: none !important;
}

/* Ensure the reopen arrow button is visible against the dark background */
header[data-testid="stHeader"] button {
    color: #e2e8f0 !important;
}
/* 1. Global App Base & Text Overrides */
.stApp { 
    background: #0f1117; 
}
.stApp, .stMarkdown, p, h1, h2, h3, h4, h5, h6, label {
    color: #e2e8f0 !important;
}

/* 2. Force Dark Mode Theme onto Streamlit Widgets */
div[data-baseweb="textarea"] textarea, 
div[data-baseweb="base-input"] input,
div[data-baseweb="select"] {
    background-color: #161925 !important;
    color: #e2e8f0 !important;
    border: 1px solid #2d3150 !important;
    border-radius: 10px !important;
}

/* Clean up widget focus borders */
div[data-baseweb="textarea"]:focus-within, 
div[data-baseweb="base-input"]:focus-within {
    border-color: #7c3aed !important;
}

/* Style the Drag-and-Drop File Uploader box */
div[data-testid="stFileUploader"] {
    background-color: #161925 !important;
    border: 1px dashed #4c1d95 !important;
    border-radius: 12px;
    padding: 15px;
}
div[data-testid="stFileUploader"] section {
    background-color: transparent !important;
}

/* Fix sidebar color matching */
section[data-testid="stSidebar"] {
    background-color: #12141c !important;
    border-right: 1px solid #1e2230;
}

/* 3. Original Component Layouts */
.metric-card {
    background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
    border-radius: 16px;
    padding: 20px 24px;
    border: 1px solid #2d3150;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    margin-bottom: 12px;
}
.metric-card h2 { color: #4c6ef5 !important; margin: 0 0 4px 0; font-size: 2rem; }
.metric-card p { color: #8892b0 !important; margin: 0; font-size: 0.85rem; }

.section-header {
    background: linear-gradient(90deg, #7c3aed22 0%, transparent 100%);
    border-left: 3px solid #7c3aed;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    margin: 24px 0 16px 0;
}
.section-header h3 { color: #e2e8f0 !important; margin: 0; }

/* Skill tags */
.skill-tag-green {
    display: inline-block;
    background: #064e3b; color: #6ee7b7;
    border: 1px solid #10b981;
    border-radius: 20px; padding: 3px 12px;
    margin: 3px; font-size: 0.8rem; font-weight: 500;
}
.skill-tag-red {
    display: inline-block;
    background: #450a0a; color: #fca5a5;
    border: 1px solid #ef4444;
    border-radius: 20px; padding: 3px 12px;
    margin: 3px; font-size: 0.8rem; font-weight: 500;
}
.skill-tag-blue {
    display: inline-block;
    background: #1e3a5f; color: #93c5fd;
    border: 1px solid #3b82f6;
    border-radius: 20px; padding: 3px 12px;
    margin: 3px; font-size: 0.8rem; font-weight: 500;
}

.suggestion-card {
    background: #1a1f35;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    border-left: 3px solid #7c3aed;
    color: #cbd5e1;
    font-size: 0.9rem;
    line-height: 1.6;
}

.info-pill {
    display: inline-block;
    background: #1e293b; color: #94a3b8;
    border-radius: 20px; padding: 4px 14px;
    font-size: 0.78rem; margin: 4px 2px;
    border: 1px solid #334155;
}
            
 /* Target both key action buttons natively using Streamlit selectors */
button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-secondary"] {
    background-color: #334155 !important; /* Premium slate grey */
    color: #cbd5e1 !important;            /* Clean light-grey text */
    border: 1px solid #475569 !important; /* Subtly defined border */
    transition: all 0.3s ease-in-out !important;
}

/* Hover effects for both buttons */
button[data-testid="stBaseButton-primary"]:hover,
button[data-testid="stBaseButton-secondary"]:hover {
    background-color: #1e293b !important; /* Turns a bit darker grey */
    color: #ffffff !important;            /* Text turns crisp white */
    border-color: #334155 !important;    /* Darkens border edge */
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  PDF TEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_file) -> str:
    """
    Extracts plain text from an uploaded PDF file.
    Tries pdfplumber first, then PyPDF2 as fallback.
    """
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: PyPDF2
    try:
        import PyPDF2
        pdf_file.seek(0)
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Could not extract PDF text: {e}")
        return ""


def read_uploaded_file(uploaded_file) -> str:
    """Reads text from an uploaded TXT or PDF file."""
    if uploaded_file is None:
        return ""

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".txt"):
        try:
            return uploaded_file.read().decode("utf-8", errors="replace")
        except Exception as e:
            st.error(f"Error reading TXT file: {e}")
            return ""

    elif file_name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)

    else:
        st.warning("Unsupported file format. Please upload .txt or .pdf files.")
        return ""


# ═══════════════════════════════════════════════════════════════════
#  VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def plot_score_gauge(score: float, title: str = "Match Score", color: str = "#7c3aed") -> plt.Figure:
    """
    Draws a semicircular gauge chart showing the match score.
    Uses matplotlib patches to create the arc-style gauge.
    """
    fig, ax = plt.subplots(figsize=(5, 3), facecolor="#0f1117")
    ax.set_facecolor("#0f1117")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.1, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")

    # Background arc (grey track)
    theta = np.linspace(np.pi, 0, 200)
    x_bg = np.cos(theta)
    y_bg = np.sin(theta)
    ax.plot(x_bg, y_bg, color="#2d3150", linewidth=18, solid_capstyle="round")

    # Foreground arc (colored fill based on score)
    fill_angle = np.pi * (1 - score / 100)
    theta_fill = np.linspace(np.pi, fill_angle, 200)
    x_fill = np.cos(theta_fill)
    y_fill = np.sin(theta_fill)
    ax.plot(x_fill, y_fill, color=color, linewidth=18, solid_capstyle="round", alpha=0.9)

    # Score text in center
    ax.text(0, 0.30, f"{score:.0f}%", ha="center", va="center",
            fontsize=30, fontweight="bold", color="white", fontfamily="monospace")
    ax.text(0, 0.08, title, ha="center", va="center",
            fontsize=10, color="#8892b0")

    # Scale labels
    ax.text(-1.15, -0.05, "0", ha="center", va="center", fontsize=8, color="#4a5568")
    ax.text(1.15, -0.05, "100", ha="center", va="center", fontsize=8, color="#4a5568")
    ax.text(0, 1.15, "50", ha="center", va="center", fontsize=8, color="#4a5568")

    plt.tight_layout(pad=0.2)
    return fig


def plot_skills_coverage(comparison: dict) -> plt.Figure:
    """
    Horizontal bar chart showing skill coverage per category.
    Green portion = skills present, Red portion = skills missing.
    """
    by_cat = comparison.get("by_category", {})
    if not by_cat:
        return None

    # Only show categories that have JD requirements
    categories = []
    present_counts = []
    missing_counts = []

    for cat, data in by_cat.items():
        total = len(data["present"]) + len(data["missing"])
        if total > 0:
            categories.append(cat)
            present_counts.append(len(data["present"]))
            missing_counts.append(len(data["missing"]))

    if not categories:
        return None

    fig, ax = plt.subplots(figsize=(8, max(3, len(categories) * 0.7 + 1)),
                           facecolor="#0f1117")
    ax.set_facecolor("#0f1117")

    y = np.arange(len(categories))
    bar_h = 0.5

    # Stacked horizontal bars
    bars1 = ax.barh(y, present_counts, bar_h, color="#10b981", alpha=0.85, label="Present")
    bars2 = ax.barh(y, missing_counts, bar_h, left=present_counts, color="#ef4444", alpha=0.75, label="Missing")

    # Labels
    ax.set_yticks(y)
    ax.set_yticklabels(categories, color="#cbd5e1", fontsize=9)
    ax.set_xlabel("Skill Count", color="#8892b0", fontsize=9)
    ax.tick_params(colors="#8892b0", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#2d3150")
    ax.spines["bottom"].set_color("#2d3150")

    # Value labels on bars
    for bar, val in zip(bars1, present_counts):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                    str(val), ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    for bar, val, off in zip(bars2, missing_counts, present_counts):
        if val > 0:
            ax.text(off + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                    str(val), ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    legend = ax.legend(loc="lower right", framealpha=0.2, labelcolor="#cbd5e1", fontsize=8)
    legend.get_frame().set_facecolor("#1e2130")
    legend.get_frame().set_edgecolor("#2d3150")

    ax.set_title("Skill Coverage by Category", color="#e2e8f0", fontsize=11, pad=12)
    plt.tight_layout()
    return fig


def plot_missing_skills(missing_skills: list, top_n: int = 12) -> plt.Figure:
    """
    Bar chart of missing skills ranked by importance.
    """
    if not missing_skills:
        return None

    skills_to_show = missing_skills[:top_n]

    # Importance score (higher for in-demand skills)
    PRIORITY = {
        "python": 10, "sql": 9, "docker": 8, "kubernetes": 8, "aws": 9,
        "azure": 7, "gcp": 7, "tensorflow": 8, "pytorch": 8, "pandas": 7,
        "numpy": 7, "react": 7, "fastapi": 6, "kafka": 6, "spark": 6,
        "git": 8, "mlflow": 6, "airflow": 6, "postgresql": 6, "scikit-learn": 7,
    }
    scores = [PRIORITY.get(s, 3) for s in skills_to_show]
    colors = ["#ef4444" if s >= 8 else "#f97316" if s >= 6 else "#eab308"
              for s in scores]

    fig, ax = plt.subplots(figsize=(8, max(3, len(skills_to_show) * 0.55 + 1.5)),
                           facecolor="#0f1117")
    ax.set_facecolor("#0f1117")

    y = np.arange(len(skills_to_show))
    bars = ax.barh(y, scores, 0.6, color=colors, alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(skills_to_show, color="#cbd5e1", fontsize=9)
    ax.set_xlabel("Priority Score", color="#8892b0", fontsize=9)
    ax.tick_params(colors="#8892b0", labelsize=8)
    ax.set_xlim(0, 12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#2d3150")
    ax.spines["bottom"].set_color("#2d3150")

    for bar, val in zip(bars, scores):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", color="#94a3b8", fontsize=8)

    # Priority legend
    high = mpatches.Patch(color="#ef4444", alpha=0.85, label="High Priority")
    med = mpatches.Patch(color="#f97316", alpha=0.85, label="Medium Priority")
    low = mpatches.Patch(color="#eab308", alpha=0.85, label="Low Priority")
    legend = ax.legend(handles=[high, med, low], loc="lower right",
                       framealpha=0.2, labelcolor="#cbd5e1", fontsize=8)
    legend.get_frame().set_facecolor("#1e2130")
    legend.get_frame().set_edgecolor("#2d3150")

    ax.set_title("Missing Skills — Priority Ranking", color="#e2e8f0", fontsize=11, pad=12)
    plt.tight_layout()
    return fig


def plot_score_breakdown(match_score_data: dict, ats_score: int) -> plt.Figure:
    """
    Radar/bar chart showing score component breakdown.
    """
    #labels = ["TF-IDF\nSimilarity", "Keyword\nOverlap", "Skill\nCoverage", "ATS\nScore"]
    labels = [
    "TF-IDF",
    "Semantic",
    "Keyword",
    "Skill",
    "ATS"
    ]
    values = [
        match_score_data["tfidf_similarity"],
        match_score_data["semantic_similarity"],
        match_score_data["keyword_overlap"],
        match_score_data["skill_coverage"],
        ats_score,
    ]
    #colors = ["#7c3aed", "#3b82f6", "#10b981", "#f59e0b"]
    colors = [
    "#7c3aed",
    "#06b6d4",
    "#3b82f6",
    "#10b981",
    "#f59e0b"
    ]

    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#0f1117")
    ax.set_facecolor("#0f1117")

    bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.55, edgecolor="#0f1117")

    # Value labels on top of bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.0f}%", ha="center", va="bottom", color="white",
                fontsize=10, fontweight="bold")

    ax.set_ylim(0, 115)
    ax.set_ylabel("Score (%)", color="#8892b0", fontsize=9)
    ax.tick_params(colors="#8892b0", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#2d3150")
    ax.spines["bottom"].set_color("#2d3150")
    ax.yaxis.grid(True, color="#1e2130", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Score Component Breakdown", color="#e2e8f0", fontsize=11, pad=12)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════
#  REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════

def generate_text_report(
    match_data: dict,
    ats_data: dict,
    skill_comparison: dict,
    suggestions: list,
    found_keywords: list,
    missing_keywords: list,
    top_skills: list,
    resume_name: str = "Resume",
) -> str:
    """Generates a plain-text downloadable report."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "=" * 60,
        "  AI RESUME RANKER — ANALYSIS REPORT",
        f"  Generated: {now}",
        f"  Resume: {resume_name}",
        "=" * 60,
        "",
        "── MATCH SCORES ─────────────────────────────────",
        f"  Overall Match Score : {match_data['final_score']:.1f}%  ({match_data['score_label']})",
        f"  TF-IDF Similarity   : {match_data['tfidf_similarity']:.1f}%",
        f"  Semantic Similarity : {match_data['semantic_similarity']:.1f}%",
        f"  Keyword Overlap     : {match_data['keyword_overlap']:.1f}%",
        f"  Skill Coverage      : {match_data['skill_coverage']:.1f}%",
        f"  ATS Score           : {ats_data['ats_score']}%  ({ats_data['ats_label']})",
        "",
        "── SKILLS ANALYSIS ──────────────────────────────",
        f"  Skills in Resume    : {skill_comparison['resume_total']}",
        f"  Skills in Job Desc  : {skill_comparison['jd_total']}",
        f"  Skills Matched      : {skill_comparison['matched_total']}",
        f"  Skills Missing      : {len(skill_comparison['skills_missing'])}",
        "",
        "  ✅ Skills Present:",
    ]
    for s in skill_comparison["skills_present"]:
        lines.append(f"     • {s}")

    lines += [
        "",
        "  ❌ Skills Missing:",
    ]
    for s in skill_comparison["skills_missing"]:
        lines.append(f"     • {s}")

    lines += [
        "",
        f"  🏆 Top 5 Skills to Learn: {', '.join(top_skills)}",
        "",
        "── KEYWORD ANALYSIS ─────────────────────────────",
        "  Important Keywords Found:",
    ]
    for kw, score in found_keywords[:10]:
        lines.append(f"     • {kw} (score: {score:.3f})")

    lines += [
        "",
        "  Important Keywords Missing:",
    ]
    for kw, score in missing_keywords[:10]:
        lines.append(f"     • {kw} (score: {score:.3f})")

    lines += [
        "",
        "── ATS FEEDBACK ─────────────────────────────────",
    ]
    for fb in ats_data["feedback"]:
        lines.append(f"  • {fb}")

    lines += [
        "",
        "── IMPROVEMENT SUGGESTIONS ──────────────────────",
    ]
    for i, s in enumerate(suggestions, 1):
        # Strip markdown bold markers for plain text
        s_clean = s.replace("**", "")
        lines.append(f"  {i}. {s_clean}")

    lines += [
        "",
        "=" * 60,
        "  End of Report — AI Resume Analyzer",
        "=" * 60,
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════

# def render_sidebar():
#     with st.sidebar:
#         st.markdown("## 🎯 AI Resume Ranker")
#         st.markdown(
#             "<span class='info-pill'>v1.0</span> "
#             "<span class='info-pill'>NLP + ML</span> "
#             "<span class='info-pill'>TF-IDF</span>",
#             unsafe_allow_html=True,
#         )
#         st.divider()

#         st.markdown("### 📖 How It Works")
#         st.markdown("""
# 1. **Upload** your resume (PDF/TXT)
# 2. **Paste** the job description
# 3. **Click Analyze** to run the ML pipeline
# 4. Review your **match score**, **skill gaps**, and **suggestions**
#         """)
#         st.divider()

#         st.markdown("### ⚙️ ML Pipeline")
#         steps = [
#             "📄 Text Extraction",
#             "🧹 Text Cleaning",
#             "✂️ Tokenization",
#             "🚫 Stopword Removal",
#             "📊 TF-IDF Vectorization",
#             "📐 Cosine Similarity",
#             "🔍 Skill Extraction",
#             "🤖 ATS Simulation",
#         ]
#         for step in steps:
#             st.markdown(f"<span class='info-pill'>{step}</span>", unsafe_allow_html=True)

#         # st.divider()
#         # st.markdown("### 📚 Skills Database")
#         # try:
#         #     db = load_skills_database()
#         #     total = sum(len(v) for v in db.values())
#         #     st.metric("Total Skills Tracked", total)
#         #     st.metric("Categories", len(db))
#         # except Exception:
#         #     st.warning("Skills DB not loaded")

#         st.divider()
#         st.caption("Built with Python · Streamlit · scikit-learn · NLTK · matplotlib")
def render_sidebar():
    with st.sidebar:
        
        st.markdown(
            "<h1 style='font-size: 2.1rem; font-weight: 800; color: #ffffff; margin-bottom: 5px; padding-bottom: 0px;'>"
            "🎯 AI Resume Analyzer"
            "</h1>", 
            unsafe_allow_html=True
        )
        st.markdown(
            "<span class='info-pill'>v1.0</span>"
            "<span class='info-pill'>⚡ Fast Analysis</span>",
            unsafe_allow_html=True,
        )
        st.divider()

        # How It Works Section — Using a clean Modern Sans-Serif font with high-contrast accenting
        st.markdown("### 📖 How It Works")
        st.markdown("""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 0.92rem; line-height: 1.6; color: #cbd5e1;">
            <p><strong style="color: #4c6ef5; font-weight: 600;">1. Upload</strong> your resume (PDF or TXT format).</p>
            <p><strong style="color: #4c6ef5; font-weight: 600;">2. Paste</strong> the target job description.</p>
            <p><strong style="color: #4c6ef5; font-weight: 600;">3. Click Analyze</strong> to evaluate alignment.</p>
            <p><strong style="color: #4c6ef5; font-weight: 600;">4. Review</strong> your match score, missing keywords, and tailored suggestions.</p>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        # Privacy Section — Styled with an elegant Editorial Serif font to feel trustworthy and distinct
        st.markdown("### 🔒 Privacy First")
        st.markdown("""
        <p style="font-family: 'Georgia', Cambria, 'Times New Roman', serif; font-style: italic; font-size: 0.88rem; color: #94a3b8; line-height: 1.5; padding-left: 2px;">
            Your data security is important. Resumes are processed locally, analyzed instantly, and are never saved or stored permanently on our servers.
        </p>
        """, unsafe_allow_html=True)

        st.divider()
        
        # Dual Footers at the very bottom
        st.caption("AI Resume Analyzer © 2026")
        st.caption("Built with Python · Streamlit · scikit-learn · NLTK · matplotlib")


# ═══════════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════════

def run_analysis(resume_text: str, jd_text: str, resume_name: str = "Resume") -> dict:
    """
    Orchestrates the entire analysis pipeline and returns all results.
    """
    skills_db = load_skills_database()

    # ── Skill Extraction ──
    resume_skills = extract_skills_from_text(resume_text, skills_db)
    jd_skills = extract_skills_from_text(jd_text, skills_db)
    skill_comparison = compare_skills(resume_skills, jd_skills)

    # ── Match Scoring ──
    match_data = compute_match_score(
        resume_text, jd_text,
        skill_coverage_pct=skill_comparison["coverage_pct"]
    )

    # ── ATS Scoring ──
    ats_data = compute_ats_score(resume_text, jd_text)

    # ── Keyword Analysis ──
    important_kw = extract_important_keywords(jd_text, top_n=20)
    found_kw, missing_kw = find_keywords_in_resume(important_kw, resume_text)

    # ── Top Skills to Learn ──
    top_skills = get_top_suggested_skills(skill_comparison["skills_missing"], n=5)

    # ── Improvement Suggestions ──
    suggestions = generate_improvement_suggestions(
        match_score=match_data["final_score"],
        skills_missing=skill_comparison["skills_missing"],
        missing_keywords=missing_kw,
        ats_score=ats_data["ats_score"],
        ats_feedback=ats_data["feedback"],
    )

    # ── Keyword Density ──
    all_found_skills = skill_comparison["skills_present"] + list(
        s for s in skill_comparison.get("extra_skills", [])
    )
    kw_density = keyword_density_analysis(resume_text, all_found_skills[:30])

    return {
        "match_data": match_data,
        "ats_data": ats_data,
        "skill_comparison": skill_comparison,
        "resume_skills": resume_skills,
        "jd_skills": jd_skills,
        "found_keywords": found_kw,
        "missing_keywords": missing_kw,
        "top_skills": top_skills,
        "suggestions": suggestions,
        "kw_density": kw_density,
        "resume_name": resume_name,
    }


def render_results(results: dict):
    """Renders the full analysis dashboard."""
    md = results["match_data"]
    ats = results["ats_data"]
    sc = results["skill_comparison"]

    # ── TOP METRICS ROW ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    st.write("TF-IDF Similarity:", md["tfidf_similarity"], "%")
    st.write("Semantic Similarity:", md["semantic_similarity"], "%")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{md['final_score']:.0f}%</h2>
            <p>Overall Match Score</p>
            <p>{md['score_label']}</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{ats['ats_score']}%</h2>
            <p>ATS Score</p>
            <p>{ats['ats_label']}</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{sc['matched_total']}</h2>
            <p>Skills Matched</p>
            <p>of {sc['jd_total']} required</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{len(sc['skills_missing'])}</h2>
            <p>Skills Missing</p>
            <p>from job description</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── GAUGE CHARTS ROW ─────────────────────────────────────────
    col_g1, col_g2, col_g3 = st.columns(3)

    with col_g1:
        fig = plot_score_gauge(md["final_score"], "Overall Match", md["score_color"])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_g2:
        ats_color = "#00C853" if ats["ats_score"] >= 70 else "#FFD600" if ats["ats_score"] >= 45 else "#D50000"
        fig = plot_score_gauge(ats["ats_score"], "ATS Score", ats_color)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_g3:
        cov = sc["coverage_pct"]
        cov_color = "#00C853" if cov >= 70 else "#FFD600" if cov >= 40 else "#D50000"
        fig = plot_score_gauge(cov, "Skill Coverage", cov_color)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── SCORE BREAKDOWN CHART ─────────────────────────────────────
    st.markdown("""<div class="section-header"><h3>📈 Score Component Breakdown</h3></div>""",
                unsafe_allow_html=True)
    fig = plot_score_breakdown(md, ats["ats_score"])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── SKILLS SECTION ────────────────────────────────────────────
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("""<div class="section-header"><h3>✅ Skills Found in Resume</h3></div>""",
                    unsafe_allow_html=True)
        if sc["skills_present"]:
            tags = " ".join(f'<span class="skill-tag-green">{s}</span>'
                            for s in sorted(sc["skills_present"]))
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.info("No matching skills detected.")

        if sc.get("extra_skills"):
            st.markdown("**Extra skills (not in JD):**")
            tags = " ".join(f'<span class="skill-tag-blue">{s}</span>'
                            for s in sorted(sc["extra_skills"])[:15])
            st.markdown(tags, unsafe_allow_html=True)

    with col_s2:
        st.markdown("""<div class="section-header"><h3>❌ Skills Missing from Resume</h3></div>""",
                    unsafe_allow_html=True)
        if sc["skills_missing"]:
            tags = " ".join(f'<span class="skill-tag-red">{s}</span>'
                            for s in sorted(sc["skills_missing"]))
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.success("🎉 You have all required skills!")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TOP 5 SKILLS TO LEARN ─────────────────────────────────────
    if results["top_skills"]:
        st.markdown("""<div class="section-header"><h3>🏆 Top 5 Suggested Skills to Learn</h3></div>""",
                    unsafe_allow_html=True)
        cols = st.columns(5)
        icons = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
        for i, (col, skill) in enumerate(zip(cols, results["top_skills"])):
            with col:
                st.markdown(f"""
                <div class="metric-card" style="text-align:center">
                    <h2 style="font-size:1.5rem">{icons[i]}</h2>
                    <p style="color:#c4b5fd; font-weight:600; font-size:0.95rem">{skill}</p>
                    <p>Learn Next</p>
                </div>""", unsafe_allow_html=True)

    # ── SKILL COVERAGE CHART ──────────────────────────────────────
    st.markdown("""<div class="section-header"><h3>📊 Skill Coverage by Category</h3></div>""",
                unsafe_allow_html=True)
    fig = plot_skills_coverage(sc)
    if fig:
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("No category-level skill data to display.")

    # ── MISSING SKILLS CHART ──────────────────────────────────────
    if sc["skills_missing"]:
        st.markdown("""<div class="section-header"><h3>📉 Missing Skills Priority Chart</h3></div>""",
                    unsafe_allow_html=True)
        fig = plot_missing_skills(sc["skills_missing"])
        if fig:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # ── KEYWORD ANALYSIS ──────────────────────────────────────────
    col_k1, col_k2 = st.columns(2)

    with col_k1:
        st.markdown("""<div class="section-header"><h3>🔑 Important Keywords Found</h3></div>""",
                    unsafe_allow_html=True)
        if results["found_keywords"]:
            for kw, score in results["found_keywords"][:12]:
                bar_width = int(score * 1000)
                st.markdown(
                    f'<span class="skill-tag-green">{kw}</span>',
                    unsafe_allow_html=True,
                )
        else:
            st.warning("No important keywords detected in resume.")

    with col_k2:
        st.markdown("""<div class="section-header"><h3>⚠️ Important Keywords Missing</h3></div>""",
                    unsafe_allow_html=True)
        if results["missing_keywords"]:
            for kw, score in results["missing_keywords"][:12]:
                st.markdown(
                    f'<span class="skill-tag-red">{kw}</span>',
                    unsafe_allow_html=True,
                )
        else:
            st.success("All important keywords are present!")

    # ── KEYWORD DENSITY ───────────────────────────────────────────
    if results["kw_density"]:
        st.markdown("""<div class="section-header"><h3>📈 Keyword Density Analysis</h3></div>""",
                    unsafe_allow_html=True)
        density_df = pd.DataFrame(
            list(results["kw_density"].items()),
            columns=["Skill/Keyword", "Count in Resume"]
        )
        st.dataframe(
            density_df.head(15),
            use_container_width=True,
            hide_index=True,
        )

    # ── ATS FEEDBACK ──────────────────────────────────────────────
    st.markdown("""<div class="section-header"><h3>🤖 ATS Analysis Details</h3></div>""",
                unsafe_allow_html=True)
    col_a1, col_a2 = st.columns(2)

    with col_a1:
        d = ats["details"]
        st.markdown(f"""
        | ATS Factor | Value |
        |---|---|
        | Sections Found | {', '.join(d.get('sections_found', [])) or 'None'} |
        | Action Verbs | {d.get('action_verbs_found', 0)} |
        | Quantified Metrics | {d.get('quantified_metrics', 0)} |
        | JD Keyword Match | {d.get('jd_keyword_match_pct', 0):.1f}% |
        | Word Count | {d.get('word_count', 0)} |
        """)

    with col_a2:
        st.markdown("**ATS Optimization Tips:**")
        for fb in ats["feedback"]:
            st.markdown(f"<div class='suggestion-card'>⚡ {fb}</div>", unsafe_allow_html=True)

    # ── IMPROVEMENT SUGGESTIONS ───────────────────────────────────
    st.markdown("""<div class="section-header"><h3>💡 Personalized Improvement Suggestions</h3></div>""",
                unsafe_allow_html=True)
    for suggestion in results["suggestions"]:
        st.markdown(f"<div class='suggestion-card'>{suggestion}</div>",
                    unsafe_allow_html=True)

    # ── DOWNLOAD REPORT ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📥 Download Your Report")
    report_text = generate_text_report(
        match_data=results["match_data"],
        ats_data=results["ats_data"],
        skill_comparison=results["skill_comparison"],
        suggestions=results["suggestions"],
        found_keywords=results["found_keywords"],
        missing_keywords=results["missing_keywords"],
        top_skills=results["top_skills"],
        resume_name=results["resume_name"],
    )
    st.download_button(
        label="⬇️  Download Full Analysis Report (.txt)",
        data=report_text,
        file_name=f"resume_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        use_container_width=True,
    )

# ═══════════════════════════════════════════════════════════════════
#  MULTIPLE RESUME COMPARISON
# ═══════════════════════════════════════════════════════════════════

def render_comparison_tab(jd_text: str):
    """Allows comparing up to 3 resumes side by side."""
    st.markdown("### 📋 Compare Multiple Resumes")
    st.info("Upload up to 3 resumes and compare them against the same job description.")

    num_resumes = st.selectbox("Number of resumes to compare", [2, 3], index=0)
    resume_texts = []
    resume_names = []

    cols = st.columns(num_resumes)
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"**Resume {i+1}**")
            uploaded = st.file_uploader(
                f"Upload Resume {i+1}", type=["pdf", "txt"], key=f"compare_resume_{i}"
            )
            if uploaded:
                text = read_uploaded_file(uploaded)
                resume_texts.append(text)
                resume_names.append(uploaded.name)
            else:
                resume_texts.append(None)
                resume_names.append(f"Resume {i+1}")

    # No more manual wrapper HTML blocks needed here! Natively handled by CSS now.
    if st.button("⚡ Compare Resumes", use_container_width=True, key="compare_btn"):
        if not jd_text.strip():
            st.error("Please enter a job description first.")
            return

        valid = [(t, n) for t, n in zip(resume_texts, resume_names) if t and t.strip()]
        if len(valid) < 2:
            st.error("Please upload at least 2 resumes.")
            return

        comparison_results = []
        for text, name in valid:
            r = run_analysis(text, jd_text, name)
            comparison_results.append(r)

        # Comparison Table
        st.markdown("#### 📊 Side-by-Side Comparison")
        rows = []
        for r in comparison_results:
            rows.append({
                "Resume": r["resume_name"],
                "Match Score (%)": r["match_data"]["final_score"],
                "ATS Score (%)": r["ats_data"]["ats_score"],
                "Skills Matched": r["skill_comparison"]["matched_total"],
                "Skills Missing": len(r["skill_comparison"]["skills_missing"]),
                "Verdict": r["match_data"]["score_label"],
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Bar chart comparison
        fig, ax = plt.subplots(figsize=(8, 3.5), facecolor="#0f1117")
        ax.set_facecolor("#0f1117")
        x = np.arange(len(comparison_results))
        w = 0.25

        match_scores = [r["match_data"]["final_score"] for r in comparison_results]
        ats_scores = [r["ats_data"]["ats_score"] for r in comparison_results]
        cov_scores = [r["skill_comparison"]["coverage_pct"] for r in comparison_results]

        ax.bar(x - w, match_scores, w, label="Match Score", color="#7c3aed", alpha=0.85)
        ax.bar(x, ats_scores, w, label="ATS Score", color="#10b981", alpha=0.85)
        ax.bar(x + w, cov_scores, w, label="Skill Coverage", color="#f59e0b", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([r["resume_name"][:20] for r in comparison_results], color="#cbd5e1", fontsize=8)
        ax.set_ylabel("Score (%)", color="#8892b0", fontsize=9)
        ax.tick_params(colors="#8892b0")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#2d3150")
        ax.spines["bottom"].set_color("#2d3150")
        ax.yaxis.grid(True, color="#1e2130", linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 115)
        legend = ax.legend(framealpha=0.2, labelcolor="#cbd5e1", fontsize=8)
        legend.get_frame().set_facecolor("#1e2130")
        legend.get_frame().set_edgecolor("#2d3150")
        ax.set_title("Resume Comparison", color="#e2e8f0", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Best candidate
        best = max(comparison_results, key=lambda r: r["match_data"]["final_score"])
        st.success(f"🏆 Best Match: **{best['resume_name']}** with a score of **{best['match_data']['final_score']:.1f}%**")


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main():
    render_sidebar()

    # ── Header ──────────────────────────────────────────────────────
    # Modern AI Header with Gemini-inspired Navy/Blue/White Gradient Glow
    st.markdown("""
    <div style="
        background: radial-gradient(circle at center, rgba(76, 110, 245, 0.22) 0%, rgba(15, 17, 23, 0) 65%);
        padding: 48px 20px 32px 20px;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 10px;
    ">
        <h1 style="
            font-size: 3.2rem; 
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff 30%, #8da2fb 65%, #4c6ef5 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            letter-spacing: -0.03em;
        ">
             AI Resume Analyzer
        </h1>
        <p style="
            color: #94a3b8; 
            font-size: 1.05rem; 
            margin-top: 12px; 
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            font-weight: 500;
            opacity: 0.85;
        ">
            Powered by TF-IDF · Sentence Transformers · Cosine Similarity · NLP
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["🔍 Single Resume Analysis", "📊 Compare Multiple Resumes"])

    with tab1:
        # Wrap the whole setup step inside a clean layout card
        with st.container(border=True):
            st.markdown("### 📂 Step 1 — Upload Your Resume")
            col_upload, col_info = st.columns([2, 1])

            with col_upload:
                uploaded_file = st.file_uploader(
                    "Upload Resume (PDF or TXT)",
                    type=["pdf", "txt"],
                    label_visibility="collapsed", 
                    help="Supported formats: PDF, TXT. Max size: 10MB.",
                )

            with col_info:
                st.markdown("""
                <div class="metric-card" style="margin-bottom:0; padding:15px 20px;">
                    <p style="color:#4c6ef5; font-weight:600; margin-bottom:4px;">📌 Tips for Best Results</p>
                    <p style="font-size:0.8rem; line-height:1.4;">• Use a clean, text-based PDF</p>
                    <p style="font-size:0.8rem; line-height:1.4;">• Avoid image-only or scanned PDFs</p>
                    <p style="font-size:0.8rem; line-height:1.4;">• TXT files give the most accurate results</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("---") 
            
            st.markdown("### 📋 Step 2 — Paste Job Description")
            jd_text = st.text_area(
                "Job Description",
                label_visibility="collapsed", 
                placeholder="Paste the full job description here...\n\nInclude required skills, responsibilities, and qualifications.",
                height=220,
                help="Paste the complete job description for the most accurate analysis.",
            )

            # Optional: Paste Resume as Text
            with st.expander("📝 Or paste your resume as text alternative"):
                resume_text_input = st.text_area(
                    "Resume Text",
                    placeholder="Paste your resume content here if you don't have a file layout...",
                    height=200,
                    key="manual_resume",
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Analyze Button (Centered and Shortened) ───────────────────
        col_space_left, col_btn, col_space_right = st.columns([2, 1, 2])
        
        with col_btn:
            analyze_btn = st.button(
                "🚀 Analyze Resume",
                use_container_width=True, 
                type="primary",
            )

        if analyze_btn:
            resume_text = ""
            resume_name = "Resume"

            if uploaded_file is not None:
                with st.spinner("📄 Reading your resume..."):
                    resume_text = read_uploaded_file(uploaded_file)
                    resume_name = uploaded_file.name
            elif resume_text_input.strip():
                resume_text = resume_text_input
                resume_name = "Pasted Resume"

            if not resume_text.strip():
                st.error("❗ Please upload a resume file or paste your resume text.")
                st.stop()
            if not jd_text.strip():
                st.error("❗ Please paste a job description.")
                st.stop()
            if len(resume_text.strip()) < 50:
                st.warning("⚠️ Resume text seems very short. Results may be inaccurate.")
            if len(jd_text.strip()) < 50:
                st.warning("⚠️ Job description seems very short. Results may be inaccurate.")

            progress_bar = st.progress(0)
            status = st.empty()

            stages = [
                (10, "🧹 Cleaning and tokenizing text..."),
                (30, "📊 Running TF-IDF vectorization..."),
                (50, "📐 Computing cosine similarity..."),
                (70, "🔍 Extracting and comparing skills..."),
                (85, "🤖 Running ATS simulation..."),
                (95, "💡 Generating recommendations..."),
                (100, "✅ Analysis complete!"),
            ]

            for pct, msg in stages:
                progress_bar.progress(pct)
                status.markdown(f"**{msg}**")
                time.sleep(0.15)

            results = run_analysis(resume_text, jd_text, resume_name)
            status.empty()
            progress_bar.empty()

            st.session_state["results"] = results
            st.success(f"✅ Analysis complete! Match Score: **{results['match_data']['final_score']:.1f}%**")

        if "results" in st.session_state:
            render_results(st.session_state["results"])

    with tab2:
        jd_for_compare = st.text_area(
            "Job Description for Comparison",
            placeholder="Paste the job description here to compare multiple resumes against it...",
            height=180,
            key="jd_compare",
        )
        render_comparison_tab(jd_for_compare)


if __name__ == "__main__":
    main()