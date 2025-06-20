
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import re
import json
from pathlib import Path
from pyvi import ViTokenizer
import unicodedata
from underthesea import word_tokenize


# Load stopwords
stopword_path = Path("/kaggle/input/stopwords-vi/stopwords-vi.json")

with stopword_path.open(encoding="utf-8") as f:
    VI_STOPWORDS = set(json.load(f))

print(f"Số stop-words nạp: {len(VI_STOPWORDS)}")


#Function preprocess data
def preprocess_text(text: str) -> str:
    """
    Hàm tiền xử lý văn bản tiếng Việt:
    - Chuẩn hoá Unicode
    - Chuyển về chữ thường
    - Loại bỏ URL, email, chữ số, dấu câu
    - Chuẩn hoá khoảng trắng
    - Tách từ (nếu có thư viện underthesea)
    - Loại bỏ stopwords
    """
    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+[\d\.]*', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text, format="text")
    filtered = [w for w in tokens.split() if w not in VI_STOPWORDS]
    return " ".join(filtered)

def read_text_with_fallback(path: Path) -> str:
    """
    Thử đọc file với nhiều encoding khác nhau.
    """
    for enc in ("utf-8", "utf-8-sig", "utf-16", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    # Cuối cùng vẫn đọc bằng latin-1, bỏ ký tự lạ
    return path.read_bytes().decode("latin-1", errors="ignore")

INPUT_ROOT  = Path("/kaggle/input/vietnamese-title-and-passage/Data/27Topics/Ver1.1/Train/new train")
OUTPUT_ROOT = Path("/kaggle/working/preprocessed_data/Train")

for in_file in INPUT_ROOT.rglob("*.txt"):
    rel     = in_file.relative_to(INPUT_ROOT)
    out_file= OUTPUT_ROOT / rel
    out_file.parent.mkdir(parents=True, exist_ok=True)

    raw = read_text_with_fallback(in_file)
    proc = preprocess_text(raw)
    out_file.write_text(proc, encoding="utf-8")