"""Language detection and sentence-boundary rules for multilingual chunking."""

# Language codes supported by paraphrase-multilingual-MiniLM-L12-v2
SUPPORTED_LANGUAGES: set[str] = {
    "ar", "bg", "ca", "cs", "da", "de", "el", "en", "es", "et",
    "fa", "fi", "fr", "gl", "gu", "he", "hi", "hr", "hu", "hy",
    "id", "it", "ja", "ka", "ko", "ku", "lt", "lv", "mk", "mn",
    "mr", "ms", "my", "nb", "nl", "pl", "pt", "ro", "ru", "sk",
    "sl", "sq", "sr", "sv", "th", "tr", "uk", "ur", "vi",
}

# Basic sentence-ending punctuation — works for most Latin/Cyrillic script languages
SENTENCE_END_PATTERN = r"(?<=[.!?…])\s+(?=\p{Lu})"

# CJK sentence boundaries are more complex; for now we use a simple split
CJK_SENTENCE_END = r"(?<=[。！？])"
