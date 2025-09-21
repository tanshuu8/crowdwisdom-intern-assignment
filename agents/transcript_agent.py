# agents/transcript_agent.py
"""
TranscriptAgent: Collects conversation transcripts + saves logs.
"""

from pathlib import Path
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TranscriptAgent:
    def __init__(self, out_dir="outputs/transcripts"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.conversation = []

    def add_turn(self, speaker: str, text: str):
        ts = datetime.now().isoformat(timespec="seconds")
        entry = {"time": ts, "speaker": speaker, "text": text}
        self.conversation.append(entry)

    def save(self, fname="conversation.json"):
        path = self.out_dir / fname
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.conversation, f, ensure_ascii=False, indent=2)
        logger.info(f"Transcript saved: {path}")
        return str(path)
