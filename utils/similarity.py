"""
similarity.py
-------------
Implements the core ML pipeline for computing resume-to-job similarity.

Pipeline:
  1. Preprocess both texts
  2. TF-IDF Vectorization (converts text to numerical feature vectors)
  3. Cosine Similarity (measures angle between vectors → 0 to 1)
  4. Score generation with multiple weighting factors
  5. ATS scoring (Applicant Tracking System simulation)
  6. Keyword analysis
"""

import re
import math
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from utils.text_preprocessing import preprocess_text, clean_text

# Load model only once
semantic_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ─────────────────────────────────────────────
# CORE: TF-IDF + COSINE SIMILARITY
# ─────────────────────────────────────────────

def compute_tfidf_cosine_similarity(resume_text: str, jd_text: str) -> float:
    """
    Computes semantic similarity between resume and job description
    using TF-IDF vectors and cosine similarity.

    How it works:
      - TF-IDF assigns importance scores to words (rare words in the corpus
        but frequent in a document get high scores).
      - Cosine similarity measures the angle between two vectors.
        Score of 1.0 = identical direction, 0.0 = completely different.

    Args:
        resume_text: Preprocessed resume text
        jd_text: Preprocessed job description text

    Returns:
        Float between 0.0 and 1.0
    """
    if not resume_text.strip() or not jd_text.strip():
        return 0.0

    # TF-IDF Vectorizer configuration:
    # - ngram_range=(1,2): captures single words AND two-word phrases
    # - max_features=5000: limit vocabulary size for performance
    # - sublinear_tf=True: apply log normalization to term frequency
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
        min_df=1,
    )

    try:
        # Fit on both documents together, then transform
        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])

        # tfidf_matrix[0] = resume vector, tfidf_matrix[1] = jd vector
        score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        return float(score)
    except Exception:
        return 0.0
    
def compute_semantic_similarity(
    resume_text: str,
    jd_text: str
) -> float:
    """
    Sentence Transformer based semantic similarity.

    Returns:
        float between 0 and 1
    """

    if not resume_text.strip() or not jd_text.strip():
        return 0.0

    try:
        embeddings = semantic_model.encode(
            [resume_text, jd_text]
        )

        score = cosine_similarity(
            [embeddings[0]],
            [embeddings[1]]
        )[0][0]

        return float(score)

    except Exception:
        return 0.0


def compute_keyword_overlap_score(resume_text: str, jd_text: str) -> float:
    """
    Simple keyword overlap score as a complementary signal.
    Counts what fraction of JD keywords appear in the resume.

    This catches cases where TF-IDF might miss exact keyword matches.
    """
    if not resume_text or not jd_text:
        return 0.0

    # Extract meaningful words (length > 2, not purely numeric)
    def extract_keywords(text):
        words = re.findall(r"\b[a-z][a-z0-9\+\#\-]{1,}\b", text.lower())
        return set(words)

    resume_kw = extract_keywords(resume_text)
    jd_kw = extract_keywords(jd_text)

    if not jd_kw:
        return 0.0

    overlap = len(resume_kw.intersection(jd_kw))
    return overlap / len(jd_kw)


def compute_match_score(
    resume_raw: str,
    jd_raw: str,
    skill_coverage_pct: float = 0.0,
) -> dict:
    """
    Master scoring function that combines multiple signals into
    a final 0–100 match score.

    Weighting:
      - 50%: TF-IDF cosine similarity (semantic overlap)
      - 30%: Keyword overlap (exact term matching)
      - 20%: Skill coverage (% of required skills present)

    Args:
        resume_raw: Raw resume text
        jd_raw: Raw job description text
        skill_coverage_pct: Precomputed skill coverage percentage (0-100)

    Returns:
        Dictionary with final score and component breakdowns
    """
    # Preprocess both texts for ML pipeline
    resume_processed = preprocess_text(resume_raw, return_string=True)
    jd_processed = preprocess_text(jd_raw, return_string=True)

    # Component 1: TF-IDF Cosine Similarity (semantic meaning)
    tfidf_score = compute_tfidf_cosine_similarity(resume_processed, jd_processed)

    # Component 2: Sentence Transformer Semantic Similarity
    semantic_score = compute_semantic_similarity( clean_text(resume_raw),
    clean_text(jd_raw))
    
    # Component 2: Keyword Overlap (exact word matching)
    keyword_score = compute_keyword_overlap_score(resume_processed, jd_processed)

    # Component 3: Skill Coverage (normalized to 0–1)
    skill_score = skill_coverage_pct / 100.0

    # Weighted combination
    WEIGHTS = {"tfidf": 0.50,"semantic": 0.40, "keyword": 0.30, "skill": 0.20}
    combined = (
        WEIGHTS["tfidf"] * tfidf_score
        + WEIGHTS["semantic"] * semantic_score
        + WEIGHTS["keyword"] * keyword_score
        + WEIGHTS["skill"] * skill_score
    )

    # Scale to 0–100 and cap at 100
    final_score = min(round(combined * 100, 1), 100.0)

    return {
        "final_score": final_score,
        "tfidf_similarity": round(tfidf_score * 100, 1),
        "semantic_similarity": round(semantic_score * 100, 1),
        "keyword_overlap": round(keyword_score * 100, 1),
        "skill_coverage": round(skill_coverage_pct, 1),
        "score_label": _get_score_label(final_score),
        "score_color": _get_score_color(final_score),
    }


def _get_score_label(score: float) -> str:
    """Returns a human-readable label for the match score."""
    if score >= 80:
        return "Excellent Match 🌟"
    elif score >= 65:
        return "Good Match ✅"
    elif score >= 45:
        return "Moderate Match ⚠️"
    elif score >= 25:
        return "Weak Match 🔸"
    else:
        return "Poor Match ❌"


def _get_score_color(score: float) -> str:
    """Returns a color hex for score visualization."""
    if score >= 80:
        return "#00C853"  # Green
    elif score >= 65:
        return "#64DD17"  # Light Green
    elif score >= 45:
        return "#FFD600"  # Yellow
    elif score >= 25:
        return "#FF6D00"  # Orange
    else:
        return "#D50000"  # Red


# ─────────────────────────────────────────────
# ATS SCORE
# ─────────────────────────────────────────────

def compute_ats_score(resume_raw: str, jd_raw: str) -> dict:
    """
    Simulates an ATS (Applicant Tracking System) score.

    ATS systems typically evaluate:
    1. Keyword density match
    2. Section detection (Experience, Education, Skills, etc.)
    3. Format compatibility (no tables/images in parsed text)
    4. Measurable achievements (numbers, percentages)
    5. Action verbs presence

    Returns a 0–100 ATS score with explanations.
    """
    score = 0
    feedback = []
    details = {}

    resume_lower = resume_raw.lower()
    jd_lower = jd_raw.lower()

    # ── Check 1: Standard Resume Sections (25 points) ──
    section_keywords = {
        "experience": ["experience", "work history", "employment"],
        "education": ["education", "degree", "university", "bachelor", "master", "phd"],
        "skills": ["skills", "technical skills", "competencies"],
        "projects": ["projects", "portfolio"],
        "summary": ["summary", "objective", "profile", "about"],
    }
    sections_found = []
    for section, keywords in section_keywords.items():
        if any(kw in resume_lower for kw in keywords):
            sections_found.append(section)

    section_score = min(len(sections_found) * 5, 25)
    score += section_score
    details["sections_found"] = sections_found
    if len(sections_found) < 4:
        missing = [s for s in section_keywords if s not in sections_found]
        feedback.append(f"Add these sections: {', '.join(missing).title()}")

    # ── Check 2: Action Verbs (20 points) ──
    action_verbs = [
        "developed", "designed", "implemented", "built", "created", "managed",
        "led", "improved", "optimized", "delivered", "achieved", "increased",
        "reduced", "automated", "deployed", "architected", "mentored", "trained",
        "analyzed", "engineered", "launched", "collaborated"
    ]
    found_verbs = [v for v in action_verbs if v in resume_lower]
    verb_score = min(len(found_verbs) * 2, 20)
    score += verb_score
    details["action_verbs_found"] = len(found_verbs)
    if len(found_verbs) < 5:
        feedback.append("Use more action verbs (e.g., Developed, Implemented, Optimized)")

    # ── Check 3: Measurable Achievements (20 points) ──
    # Look for numbers, percentages, dollar amounts
    numbers_pattern = re.findall(r"\d+[%$kmb]?|\$[\d,]+", resume_lower)
    achievement_score = min(len(numbers_pattern) * 2, 20)
    score += achievement_score
    details["quantified_metrics"] = len(numbers_pattern)
    if len(numbers_pattern) < 5:
        feedback.append("Add quantifiable achievements (e.g., 'Improved performance by 40%')")

    # ── Check 4: JD Keyword Match (25 points) ──
    jd_words = set(re.findall(r"\b[a-z][a-z]{2,}\b", jd_lower))
    resume_words = set(re.findall(r"\b[a-z][a-z]{2,}\b", resume_lower))
    common = jd_words.intersection(resume_words)
    jd_match_pct = (len(common) / len(jd_words) * 100) if jd_words else 0
    jd_score = min(int(jd_match_pct * 0.25), 25)
    score += jd_score
    details["jd_keyword_match_pct"] = round(jd_match_pct, 1)
    if jd_match_pct < 50:
        feedback.append("Mirror more keywords from the job description in your resume")

    # ── Check 5: Resume Length (10 points) ──
    word_count = len(resume_raw.split())
    details["word_count"] = word_count
    if 300 <= word_count <= 800:
        score += 10
    elif word_count < 300:
        feedback.append("Resume is too short. Aim for 300–800 words for ATS optimization.")
    else:
        score += 5
        feedback.append("Resume may be too long. ATS systems prefer concise resumes.")

    return {
        "ats_score": min(score, 100),
        "feedback": feedback,
        "details": details,
        "ats_label": _get_score_label(score),
    }


# ─────────────────────────────────────────────
# KEYWORD ANALYSIS
# ─────────────────────────────────────────────

def extract_important_keywords(jd_text: str, top_n: int = 20) -> list:
    """
    Extracts the most important keywords from a job description
    using TF-IDF scores. These are the terms the JD emphasizes most.

    Args:
        jd_text: Raw job description text
        top_n: Number of top keywords to return

    Returns:
        List of (keyword, score) tuples sorted by importance
    """
    if not jd_text.strip():
        return []

    processed = preprocess_text(jd_text, return_string=True)
    if not processed.strip():
        return []

    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000, sublinear_tf=True)
        tfidf_matrix = vectorizer.fit_transform([processed])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]

        keyword_scores = list(zip(feature_names, scores))
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        return keyword_scores[:top_n]
    except Exception:
        return []


def find_keywords_in_resume(keywords: list, resume_raw: str) -> tuple:
    """
    Checks which important JD keywords are present/absent in the resume.

    Args:
        keywords: List of (keyword, score) from extract_important_keywords
        resume_raw: Raw resume text

    Returns:
        (found_keywords, missing_keywords) — both as lists of (keyword, score)
    """
    resume_lower = resume_raw.lower()
    found, missing = [], []

    for keyword, score in keywords:
        if re.search(r"\b" + re.escape(keyword) + r"\b", resume_lower):
            found.append((keyword, score))
        else:
            missing.append((keyword, score))

    return found, missing


# ─────────────────────────────────────────────
# IMPROVEMENT SUGGESTIONS
# ─────────────────────────────────────────────

def generate_improvement_suggestions(
    match_score: float,
    skills_missing: list,
    missing_keywords: list,
    ats_score: int,
    ats_feedback: list,
) -> list:
    """
    Generates personalized, actionable improvement suggestions
    based on all analysis results.

    Returns a list of suggestion strings ordered by priority.
    """
    suggestions = []

    # Score-based overall advice
    if match_score < 30:
        suggestions.append(
            "🔴 Your resume has low alignment with this job. Consider tailoring it "
            "specifically for this role by adding relevant experience and keywords."
        )
    elif match_score < 55:
        suggestions.append(
            "🟡 Your resume partially matches this job description. Focus on adding "
            "missing skills and mirroring key phrases from the job posting."
        )
    elif match_score < 75:
        suggestions.append(
            "🟢 Good alignment! Fine-tune your resume by including a few more "
            "specific technologies and quantifying your achievements."
        )
    else:
        suggestions.append(
            "🌟 Excellent match! Your profile is well-aligned. Make sure your cover "
            "letter reinforces the key themes from the job description."
        )

    # Skills-based suggestions
    if skills_missing:
        top_missing = skills_missing[:5]
        suggestions.append(
            f"📚 Priority skills to add/learn: **{', '.join(top_missing)}**. "
            "Consider adding these to your resume if you have any exposure, "
            "or take online courses to build them."
        )

    # Keyword-based suggestions
    if missing_keywords:
        top_kw = [kw for kw, _ in missing_keywords[:5]]
        suggestions.append(
            f"🔑 Include these important keywords from the JD in your resume: "
            f"**{', '.join(top_kw)}**. Use them naturally in your experience bullets."
        )

    # ATS suggestions
    if ats_score < 60:
        suggestions.append(
            "🤖 Your ATS score is low. Many companies use ATS to filter resumes before "
            "human review. Use a clean, single-column format and standard section headings."
        )
    for fb in ats_feedback[:3]:
        suggestions.append(f"📋 {fb}")

    # General best practices
    suggestions.extend([
        "📊 Quantify your impact: Replace 'improved performance' with "
        "'improved performance by 35% using caching strategies'.",
        "🎯 Customize your resume summary/objective to mirror the job title "
        "and core requirements of this specific role.",
        "🔗 Add relevant certifications (AWS, GCP, TensorFlow Developer) to "
        "strengthen your technical credibility.",
        "📝 Keep bullet points concise: Start with action verbs, include context, "
        "and end with measurable impact.",
    ])

    return suggestions