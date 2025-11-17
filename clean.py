import re

def clean_text(text: str) -> str:
    text = text.replace("\xad", "")
    text = text.replace("\u2022", "")
    text = text.replace("\uf0b7", "")
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")
    text = text.replace("", "")

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\bPage\s*\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"Copyright.*?\d{4}", "", text, flags=re.IGNORECASE)

    return text.strip()
