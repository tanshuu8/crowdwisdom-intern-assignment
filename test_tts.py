from agents.nikud_agent import NikudAgent
from agents.tts_agent import TTSAgent

n = NikudAgent(use_real_phonikud=True)
t = TTSAgent(backend="auto")

txt = "שלום, אבקש לבטל את המנוי שלי."
res = n.add_nikud(txt)
print("PHONEMES/VOCALIZED:", res["vocalized"])

out = t.synthesize(txt, "demo_from_text.wav", phonemes=None)
print("WAV saved:", out["path"])
