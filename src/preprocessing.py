"""
Data preprocessing module for customer support tweets.
Implements normalization, cleaning, and tokenization as per project requirements.
"""
import re
import string
from typing import List, Optional
import pandas as pd


def normalize_text(text: str) -> str:
    """
    Normalize text: lowercasing and punctuation normalization.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    return text


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    return url_pattern.sub('', text)


def remove_mentions(text: str) -> str:
    """Remove Twitter mentions (@username)."""
    mention_pattern = re.compile(r'@\w+')
    return mention_pattern.sub('', text)


def remove_hashtags(text: str) -> str:
    """Remove hashtags."""
    hashtag_pattern = re.compile(r'#\w+')
    return hashtag_pattern.sub('', text)


def remove_emoticons(text: str) -> str:
    """
    Remove emoticons and emojis.
    """
    # Remove common emoticons
    emoticon_pattern = re.compile(
        r'[:;=][\-o]?[\)\]\(\[DpP/\\]|[\-o]?[:;=][\)\]\(\[DpP/\\]'
    )
    text = emoticon_pattern.sub('', text)
    
    # Remove emojis (basic pattern)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)
    
    return text


def remove_special_characters(text: str, keep_punctuation: bool = True) -> str:
    """
    Remove special characters. Optionally keep basic punctuation.
    """
    if keep_punctuation:
        # Keep basic punctuation, remove other special chars
        text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\-\'"]', '', text)
    else:
        # Remove all non-alphanumeric except spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def preprocess_text(
    text: str,
    normalize: bool = True,
    remove_urls_flag: bool = True,
    remove_mentions_flag: bool = True,
    remove_hashtags_flag: bool = True,
    remove_emoticons_flag: bool = True,
    remove_special_chars: bool = True,
    keep_punctuation: bool = True
) -> str:
    """
    Comprehensive text preprocessing pipeline.
    
    Args:
        text: Input text to preprocess
        normalize: Apply normalization (lowercasing)
        remove_urls_flag: Remove URLs
        remove_mentions_flag: Remove Twitter mentions
        remove_hashtags_flag: Remove hashtags
        remove_emoticons_flag: Remove emoticons/emojis
        remove_special_chars: Remove special characters
        keep_punctuation: Keep basic punctuation when removing special chars
    
    Returns:
        Preprocessed text
    """
    if not isinstance(text, str):
        text = str(text)
    
    if normalize:
        text = normalize_text(text)
    
    if remove_urls_flag:
        text = remove_urls(text)
    
    if remove_mentions_flag:
        text = remove_mentions(text)
    
    if remove_hashtags_flag:
        text = remove_hashtags(text)
    
    if remove_emoticons_flag:
        text = remove_emoticons(text)
    
    if remove_special_chars:
        text = remove_special_characters(text, keep_punctuation=keep_punctuation)
    
    return text.strip()


def preprocess_dataframe(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """
    Preprocess text column in a DataFrame.
    
    Args:
        df: DataFrame with text column
        text_column: Name of the text column to preprocess
    
    Returns:
        DataFrame with preprocessed text
    """
    df = df.copy()
    df[f"{text_column}_processed"] = df[text_column].apply(preprocess_text)
    return df

