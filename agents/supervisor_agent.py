"""
SupervisorAgent: lightweight guardrails. Exposes small policy checks used by orchestration.
"""
class SupervisorAgent:
    def __init__(self, asr_threshold: float = 0.6, max_turns: int = 20):
        self.asr_threshold = asr_threshold
        self.max_turns = max_turns

    def check_asr_conf(self, conf: float) -> bool:
        return conf >= self.asr_threshold

    def allow_new_turn(self, turn:int) -> bool:
        return turn < self.max_turns
