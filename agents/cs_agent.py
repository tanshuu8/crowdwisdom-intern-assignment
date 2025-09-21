# agents/cs_agent.py
"""
CustomerServiceAgent (final)
- Default: deterministic/scripted Hebrew replies for reliable demo (no API calls).
- Optional: enable OpenAI/LiteLLM by setting env var CW_CS_USE_OPENAI=1 and providing OPENAI_API_KEY.
- Always returns dict: {"reply": str, "action": "verify"|"explain"|"confirm"|"close"|"retry"}
"""

from __future__ import annotations
import os
import json
import logging
import random
import re
from typing import Dict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Try to import litellm but don't fail if not available
try:
    import litellm
    from litellm import AuthenticationError, RateLimitError
except Exception:
    litellm = None
    AuthenticationError = Exception
    RateLimitError = Exception


class CustomerServiceAgent:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.state = {}

        # Structured fallback/scripted replies
        self.scripted_flow = [
            {"reply": "אנא אשר את מספר תעודת הזהות שלך כדי שנוכל להמשיך.", "action": "verify"},
            {"reply": "המדיניות שלנו היא ביטול מיידי ללא קנסות. האם תרצה להמשיך?", "action": "explain"},
            {"reply": "הבקשה שלך בוטלה בהצלחה. נשלח לך אישור למייל תוך 24 שעות.", "action": "confirm"},
            {"reply": "תודה שפנית לשירות הלקוחות. יום נעים!", "action": "close"},
        ]

        # Extra canned replies for variety
        self.extra_replies = [
            {"reply": "להשלמת הביטול נשמח לאשר פרטי חשבון נוספים, נא לציין שם מלא.", "action": "verify"},
            {"reply": "נוכל לשלוח אישור כתוב במייל — האם תרצה שנעשה זאת?", "action": "confirm"},
            {"reply": "הבקשה התקבלה ותטופל בהקדם.", "action": "confirm"},
        ]

        # Control whether to call OpenAI/LiteLLM if available
        self.use_openai = os.getenv("CW_CS_USE_OPENAI", "0").lower() in ("1", "true", "yes")

    # Simple rule-based mapper to pick a script step
    def _scripted_reply_for(self, user_text: str) -> Dict:
        t = user_text.strip().lower()

        # Keywords-based decisions (Hebrew substrings)
        if re.search(r"בטל|לבטל|לבקש ביטול|לבטל", t):
            return self.scripted_flow[0]  # verify
        if re.search(r"אישור|אישורים|איך אדע|אישור במייל", t):
            return {"reply": "נשלח אישור למייל ברגע שהביטול יושלם. מה כתובת המייל שלך?", "action": "verify"}
        if re.search(r"מה עליי לעשות|כיצד|איך", t):
            return self.scripted_flow[1]  # explain
        if re.search(r"סיימתם|בוצע|בוצעה|סגור", t):
            return random.choice(self.extra_replies)
        # default: confirm or ask to continue
        return {"reply": "האם ברצונך שאבצע את הביטול כעת?", "action": "explain"}

    def _call_openai(self, user_text: str) -> Dict:
        """
        Attempt to call litellm/OpenAI and parse JSON response.
        Returns a dict with reply/action on success, otherwise raises.
        """
        if litellm is None:
            raise RuntimeError("litellm not installed")

        prompt = f"""
אתה נציג שירות לקוחות של חברת טלוויזיה.
הלקוח אמר: "{user_text}"
המטרה: לבצע תהליך ביטול מנוי בצורה מקצועית.
השלבים: בקשת אימות זהות -> הסבר מדיניות ביטול -> אישור סופי -> סגירה.

תן תשובה בעברית קצרה וברורה.
החזר JSON בפורמט:
{{"reply": "...", "action": "verify"|"explain"|"confirm"|"close"}}
"""
        resp = litellm.completion(model=self.model, messages=[{"role": "user", "content": prompt}])
        # Adapt to litellm response shape safely
        try:
            raw = None
            # litellm might return object-like shape; try to extract content
            if hasattr(resp, "choices"):
                # resp.choices[0].message["content"] for typical
                raw = resp.choices[0].message["content"]
            elif isinstance(resp, dict) and "choices" in resp:
                ch = resp["choices"][0]
                raw = ch.get("message", {}).get("content") if isinstance(ch, dict) else str(ch)
            else:
                raw = str(resp)
            raw = str(raw).strip()
            data = json.loads(raw)
            if not isinstance(data, dict) or "reply" not in data:
                raise ValueError("Invalid LLM format")
            return data
        except Exception as e:
            raise RuntimeError(f"LLM response parse failed: {e}")

    def reply(self, user_text: str) -> Dict:
        """
        Main entry: returns a dict with keys 'reply' and 'action'.
        Behavior:
          - If CW_CS_USE_OPENAI=1 and litellm available, try LLM with fallback to scripted replies.
          - Otherwise, use deterministic scripted replies (safe for demo).
        """
        # If OpenAI usage requested and available, attempt it
        if self.use_openai and litellm is not None:
            try:
                out = self._call_openai(user_text)
                logger.info("CS Agent: got LLM reply")
                return out
            except (AuthenticationError, RateLimitError) as e:
                logger.warning("LLM quota/auth problem: %s — falling back to scripted reply", e)
            except Exception as e:
                logger.warning("LLM call failed: %s — falling back to scripted reply", e)

        # Deterministic/scripted reply
        try:
            out = self._scripted_reply_for(user_text)
            logger.info("CS Agent: scripted reply selected: %s", out)
            return out
        except Exception as e:
            logger.exception("Scripted reply failed; returning generic fallback. %s", e)
            return {"reply": "מצטער, לא הצלחתי להבין. תוכל לחזור שוב?", "action": "retry"}


# Self-test
if __name__ == "__main__":
    cs = CustomerServiceAgent()
    for q in [
        "אני רוצה לבטל את המנוי שלי",
        "מה עליי לעשות כדי לקבל אישור?",
        "סיימתם את התהליך?"
    ]:
        print("Q:", q)
        print("A:", cs.reply(q))
