"""
skill_extractor.py
------------------
Loads a predefined skills database and extracts technical skills
from resume and job description text.

Produces:
  - Skills found in resume
  - Skills required by job description
  - Skills present (intersection)
  - Skills missing (in JD but not in resume)
  - Top suggested skills to learn
"""

import os
import re
import pandas as pd
from collections import Counter

# Path to the skills database CSV (relative to this file)
SKILLS_DB_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "skills_database.csv"
    )
)


def load_skills_database() -> dict:
    """
    Loads skills from CSV into a dictionary keyed by category.
    Returns: { "Programming Languages": ["python", "java", ...], ... }
    """
    try:
        df = pd.read_csv(SKILLS_DB_PATH)
        # Normalize all skill names to lowercase for case-insensitive matching
        df["skill"] = df["skill"].str.lower().str.strip()
        df["category"] = df["category"].str.strip()

        skills_by_category = {}
        for category, group in df.groupby("category"):
            skills_by_category[category] = sorted(group["skill"].tolist())

        return skills_by_category
    except Exception as e:
        # Return a minimal fallback if CSV is missing
        return {
            "Programming Languages": ["python", "java", "javascript", "r"],
            "ML Libraries": ["tensorflow", "pytorch", "scikit-learn", "pandas", "numpy"],
            "Cloud Technologies": ["aws", "azure", "gcp"],
            "Databases": ["sql", "mongodb", "postgresql"],
            "Web Technologies": ["flask", "django", "fastapi", "react"],
            "DevOps Tools": ["docker", "kubernetes", "git"],
        }


def get_all_skills_flat(skills_db: dict) -> list:
    """Returns a flat list of all skills across all categories."""
    all_skills = []
    for skills in skills_db.values():
        all_skills.extend(skills)
    return all_skills


def extract_skills_from_text(text: str, skills_db: dict) -> dict:
    """
    Scans raw text for known skills using exact word/phrase matching.

    Strategy:
    1. Lowercase the text
    2. For each skill in the database, check if it appears as a whole word
    3. Handle multi-word skills (e.g., "scikit-learn", "google cloud")
    4. Return matched skills grouped by category

    Args:
        text: Raw resume or job description text
        skills_db: Dictionary of {category: [skills]}

    Returns:
        Dictionary of {category: [matched_skills]}
    """
    if not text:
        return {}

    text_lower = text.lower()

    # Replace hyphens/underscores with space for flexible matching
    text_normalized = re.sub(r"[-_]", " ", text_lower)

    found_by_category = {}

    for category, skills in skills_db.items():
        found_in_category = []
        for skill in skills:
            # Normalize skill name the same way
            skill_normalized = re.sub(r"[-_]", " ", skill)

            # Use word boundary matching to avoid partial matches
            # e.g., "r" should not match "react" or "docker"
            if len(skill_normalized) <= 2:
                # Short skills (c, r, go): require word boundaries + not inside longer words
                pattern = r"\b" + re.escape(skill_normalized) + r"\b"
            else:
                pattern = r"\b" + re.escape(skill_normalized) + r"\b"

            if re.search(pattern, text_normalized):
                found_in_category.append(skill)

        if found_in_category:
            found_by_category[category] = found_in_category

    return found_by_category


def flatten_skills(skills_by_category: dict) -> set:
    """Converts a {category: [skills]} dict to a flat set of skill names."""
    flat = set()
    for skills in skills_by_category.values():
        flat.update(skills)
    return flat


def compare_skills(resume_skills: dict, jd_skills: dict) -> dict:
    """
    Compares skills found in resume vs job description.

    Returns a detailed comparison:
    {
        "skills_present":  skills in both resume AND job description,
        "skills_missing":  skills in JD but NOT in resume,
        "extra_skills":    skills in resume but NOT required by JD,
        "coverage_pct":    what % of JD skills does the resume cover,
        "by_category": {
            category: {
                "present": [...],
                "missing": [...],
            }
        }
    }
    """
    resume_flat = flatten_skills(resume_skills)
    jd_flat = flatten_skills(jd_skills)

    skills_present = resume_flat.intersection(jd_flat)
    skills_missing = jd_flat.difference(resume_flat)
    extra_skills = resume_flat.difference(jd_flat)

    # Coverage: what fraction of JD-required skills does the resume have?
    coverage_pct = (len(skills_present) / len(jd_flat) * 100) if jd_flat else 0.0

    # Per-category breakdown
    all_categories = set(resume_skills.keys()) | set(jd_skills.keys())
    by_category = {}
    for category in all_categories:
        r_set = set(resume_skills.get(category, []))
        j_set = set(jd_skills.get(category, []))
        by_category[category] = {
            "present": sorted(r_set.intersection(j_set)),
            "missing": sorted(j_set.difference(r_set)),
            "extra": sorted(r_set.difference(j_set)),
        }

    return {
        "skills_present": sorted(skills_present),
        "skills_missing": sorted(skills_missing),
        "extra_skills": sorted(extra_skills),
        "coverage_pct": round(coverage_pct, 1),
        "by_category": by_category,
        "resume_total": len(resume_flat),
        "jd_total": len(jd_flat),
        "matched_total": len(skills_present),
    }


def get_top_suggested_skills(skills_missing: list, n: int = 5) -> list:
    """
    Returns the top N most important missing skills to learn.
    Priority is based on a manually curated importance score —
    high-demand skills rank higher.

    Args:
        skills_missing: List of skills not found in the resume
        n: Number of top skills to return

    Returns:
        Top N missing skills sorted by priority
    """
    # Priority weights for high-demand skills (higher = more important)
    PRIORITY = {
        "python": 10, "sql": 9, "docker": 8, "kubernetes": 8,
        "aws": 9, "azure": 7, "gcp": 7, "tensorflow": 8, "pytorch": 8,
        "scikit-learn": 7, "pandas": 7, "numpy": 7, "react": 7,
        "fastapi": 6, "kafka": 6, "spark": 6, "git": 8, "mlflow": 6,
        "airflow": 6, "postgresql": 6, "mongodb": 6, "redis": 5,
        "transformers": 7, "langchain": 6, "nodejs": 6, "typescript": 6,
    }

    scored = [(skill, PRIORITY.get(skill, 3)) for skill in skills_missing]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [skill for skill, _ in scored[:n]]


def keyword_density_analysis(text: str, skills: list) -> dict:
    """
    Analyzes how frequently each skill keyword appears in the text.
    Useful for ATS optimization — more mentions = stronger signal.

    Returns: {skill: count} for skills that appear at least once.
    """
    if not text or not skills:
        return {}

    text_lower = text.lower()
    density = {}
    for skill in skills:
        count = len(re.findall(r"\b" + re.escape(skill) + r"\b", text_lower))
        if count > 0:
            density[skill] = count

    # Sort by frequency descending
    return dict(sorted(density.items(), key=lambda x: x[1], reverse=True))