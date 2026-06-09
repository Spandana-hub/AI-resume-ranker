"""
text_preprocessing.py
---------------------
Handles all text cleaning and normalization tasks.
Converts raw resume/job description text into clean tokens
ready for TF-IDF vectorization and skill extraction.
"""

import re
import string
import nltk
import logging

# Download required NLTK data (only runs once per environment)
def download_nltk_resources():
    """Download required NLTK resources silently."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception:
                pass

download_nltk_resources()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer (reduces words to root form, e.g., "running" → "run")
lemmatizer = WordNetLemmatizer()

# Load English stopwords (common words like "the", "is", "and" that carry little meaning)
try:
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = set()

# Extra domain-specific stopwords to remove from resumes
EXTRA_STOPWORDS = {
    "experience", "work", "years", "year", "month", "months",
    "company", "team", "project", "projects", "developed", "development",
    "implemented", "using", "used", "working", "worked", "responsible",
    "ability", "skills", "knowledge", "strong", "good", "excellent",
    "proficient", "familiar", "understanding", "including", "across",
    "various", "multiple", "well", "also", "etc", "eg", "ie"
}
STOPWORDS = STOPWORDS.union(EXTRA_STOPWORDS)


def clean_text(text: str) -> str:
    """
    Step 1: Basic text cleaning.
    - Lowercase everything
    - Remove URLs, emails, special characters
    - Normalize whitespace
    """
    if not text or not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs (http://..., www....)
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove phone numbers
    text = re.sub(r"\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b", " ", text)

    # Remove punctuation except hyphens (useful for "scikit-learn", "c++")
    text = re.sub(r"[^\w\s\-\+\#]", " ", text)

    # Normalize multiple spaces/newlines into a single space
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text: str) -> list:
    """
    Step 2: Tokenization.
    Splits the cleaned text into individual word tokens.
    Example: "machine learning engineer" → ["machine", "learning", "engineer"]
    """
    try:
        tokens = word_tokenize(text)
    except Exception:
        # Fallback to simple split if NLTK tokenizer fails
        tokens = text.split()
    return tokens


def remove_stopwords(tokens: list) -> list:
    """
    Step 3: Stopword Removal.
    Removes common, low-meaning words so TF-IDF focuses on important terms.
    Keeps tokens that are: not stopwords AND longer than 1 character.
    """
    return [
        token for token in tokens
        if token not in STOPWORDS and len(token) > 1
    ]


def lemmatize_tokens(tokens: list) -> list:
    """
    Step 4: Lemmatization.
    Reduces words to their base/root form.
    Examples: "engineers" → "engineer", "computing" → "compute"
    """
    return [lemmatizer.lemmatize(token) for token in tokens]


def preprocess_text(text: str, return_string: bool = True):
    """
    Full NLP preprocessing pipeline:
    clean → tokenize → remove stopwords → lemmatize

    Args:
        text: Raw input text (resume or job description)
        return_string: If True, returns a single joined string (for TF-IDF).
                       If False, returns list of tokens (for skill extraction).

    Returns:
        Preprocessed text as string or list of tokens.
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)

    if return_string:
        return " ".join(tokens)
    return tokens


def extract_bigrams(tokens: list) -> list:
    """
    Generates bigrams (two-word phrases) from a token list.
    Useful for detecting multi-word skills like "machine learning", "deep learning".

    Example: ["machine", "learning", "model"] → ["machine learning", "learning model"]
    """
    return [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]


def get_all_ngrams(text: str) -> list:
    """
    Returns both unigrams and bigrams from preprocessed text.
    Used by the skill extractor to match multi-word skills.
    """
    tokens = preprocess_text(text, return_string=False)
    unigrams = tokens
    bigrams = extract_bigrams(tokens)
    return unigrams + bigrams