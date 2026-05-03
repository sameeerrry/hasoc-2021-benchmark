"""
Tweet cleaning for transformer input.
Conservative: preserves case, emojis, Devanagari script.
"""
import re
import html
import unicodedata

try:
    import emoji
    HAS_EMOJI = True
except ImportError:
    HAS_EMOJI = False

URL_RE = re.compile(r'https?://\S+|www\.\S+')
MENTION_RE = re.compile(r'@\w+')
HASHTAG_RE = re.compile(r'#(\w+)')
WHITESPACE_RE = re.compile(r'\s+')
REPEAT_CHAR_RE = re.compile(r'(.)\1{2,}')
ZERO_WIDTH_RE = re.compile(r'[\u200b-\u200f\ufeff]')


def clean_tweet(text: str, keep_emoji: bool = True) -> str:
    """
    Clean a tweet for transformer input.

    Steps:
        1. NFC Unicode normalize (Devanagari consistency)
        2. HTML entity decode
        3. Remove zero-width chars
        4. URLs -> <URL>, mentions -> <USER>
        5. Hashtags -> drop # but keep word
        6. Collapse repeated chars (looool -> lool)
        7. Optionally strip emojis
        8. Collapse whitespace
    """
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize('NFC', text)
    text = html.unescape(text)
    text = ZERO_WIDTH_RE.sub('', text)
    text = URL_RE.sub('<URL>', text)
    text = MENTION_RE.sub('<USER>', text)
    text = HASHTAG_RE.sub(r'\1', text)
    text = REPEAT_CHAR_RE.sub(r'\1\1', text)

    if not keep_emoji and HAS_EMOJI:
        text = emoji.replace_emoji(text, replace='')

    text = WHITESPACE_RE.sub(' ', text).strip()
    return text


def clean_for_tfidf(text: str) -> str:
    """More aggressive cleaning for TF-IDF baseline (lowercase, no emoji)."""
    text = clean_tweet(text, keep_emoji=False)
    text = text.lower()
    return text
