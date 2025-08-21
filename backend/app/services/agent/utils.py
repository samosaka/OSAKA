import os, json, hashlib, shutil
import pandas as pd
from typing import Dict

def load_ohlcv(path: str) -> pd.DataFrame:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        head = f.read(2048).lstrip()

    is_json = head.startswith('{') or head.startswith('[')
    ext = os.path.splitext(path)[1].lower()

    if is_json or ext in {'.json', '.ndjson'}:
        # Try NDJSON first (one JSON object per line). If it fails, parse as a JSON array.
        try:
            df = pd.read_json(path, lines=True)
        except ValueError:
            df = pd.read_json(path)
    else:
        # Not JSON → try CSV then TSV
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, sep='\t')

    # Normalize column names (lowercase, strip)
    cols_lower = {c: c.lower().strip() for c in df.columns}
    df.rename(columns=cols_lower, inplace=True)
    if "time" in cols_lower:
        df["Time"] = pd.to_datetime(df[cols_lower["time"]])
    elif "date" in cols_lower and "time" in cols_lower:
        df["Time"] = pd.to_datetime(df[cols_lower["date"]] + " " + df[cols_lower["time"]])
    else:
        df["Time"] = pd.to_datetime(df.iloc[:, 0])

    def pick(name):
        for c in df.columns:
            if c.lower().strip() == name:
                return c
        return None

    o, h, l, c, v = (pick("open"), pick("high"), pick("low"), pick("close"), pick("volume"))
    keep = [x for x in [o, h, l, c, v] if x is not None]
    out = df[["Time"] + keep].copy()
    out = out.rename(columns={o: "Open", h: "High", l: "Low", c: "Close", v: "Volume"})
    return out.set_index("Time").sort_index()

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def write_json(path: str, obj: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def copy_file(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
