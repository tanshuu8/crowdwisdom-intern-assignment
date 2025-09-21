"""
ClientAgent: simulates a caller. Can act as scripted or LLM-generated.
Produces text and optionally produces audio via TTSAgent.
"""
from typing import List, Dict

class ClientAgent:
    def __init__(self, scripts: List[str] = None):
        self.scripts = scripts or [
            "אני רוצה לבטל את המנוי לטלוויזיה שלי. אני לא משתמש בזה יותר.",
            "הבנתי — מה עליי לעשות כדי לוודא שבוטל ותשלחו לי אישור?",
        ]
        self.turn = 0

    def next_utterance(self) -> str:
        if self.turn < len(self.scripts):
            txt = self.scripts[self.turn]
            self.turn += 1
            return txt
        return ""

    def reset(self):
        self.turn = 0
