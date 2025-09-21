"""
LoggerAgent: saves logs (json) and redacts PII (very simple: mask long digit sequences).
"""
import os, json, re
from pathlib import Path

PII_PATTERN = re.compile(r"\b(\d{6,})\b")  # simplistic: sequences >=6 digits

class LoggerAgent:
    def __init__(self, out_dir: str = "outputs/logs"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.logs = []

    def redact(self, text: str) -> str:
        if not text: return text
        return PII_PATTERN.sub(lambda m: f"***{m.group(1)[-4:]}", text)

    def log(self, entry: dict):
        # redact any 'text' fields
        e = json.loads(json.dumps(entry))  # shallow copy
        if "text" in e: e["text"] = self.redact(e["text"])
        if "transcript" in e: e["transcript"] = self.redact(e["transcript"])
        self.logs.append(e)

    def save(self, fname="logs.json"):
        p = self.out_dir / fname
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(self.logs, fh, ensure_ascii=False, indent=2)
        return str(p)
