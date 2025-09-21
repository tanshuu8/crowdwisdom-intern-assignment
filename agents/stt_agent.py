# agents/stt_agent.py
"""
Robust STTAgent for CrowdWisdom project.

- Honors CW_STT_FORCE_MOCK env var or force_mock constructor arg.
- Tries whisper_timestamped first, then openai-whisper (module 'whisper').
- Falls back to a deterministic MOCK that returns a single segment.
- transcribe(audio_path) -> dict: {'text': str, 'segments': List[{'start','end','text'}], 'language': str}
- export_srt(segments, out_path) writes an SRT file.
"""

from __future__ import annotations
import os
import logging
import contextlib
import wave
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


class STTAgent:
    def __init__(self, model_size: str = "tiny", force_mock: Optional[bool] = None, device: Optional[str] = None):
        if force_mock is None:
            force_mock = os.getenv("CW_STT_FORCE_MOCK", "0").lower() in ("1", "true", "yes")
        self.force_mock = bool(force_mock)
        self.model_size = model_size
        self.device = device

        self.impl_name: Optional[str] = None
        self.impl_module: Optional[Any] = None
        self.model: Optional[Any] = None

        if self.force_mock:
            logger.info("[STTAgent] CW_STT_FORCE_MOCK set -> running in MOCK mode.")
            self.impl_name = "mock"
            return

        # Try whisper_timestamped
        try:
            import whisper_timestamped as wst  # type: ignore
            self.impl_module = wst
            load = getattr(wst, "load_model", None)
            if callable(load):
                logger.info("[STTAgent] Loading whisper_timestamped model '%s' ...", self.model_size)
                self.model = load(self.model_size, device=self.device)
            self.impl_name = "whisper_timestamped"
            logger.info("[STTAgent] Using whisper_timestamped backend.")
            return
        except Exception as e:
            logger.debug("[STTAgent] whisper_timestamped unavailable: %s", e)

        # Try openai-whisper
        try:
            import whisper as wh  # type: ignore
            self.impl_module = wh
            load = getattr(wh, "load_model", None)
            if callable(load):
                logger.info("[STTAgent] Loading openai-whisper model '%s' ...", self.model_size)
                self.model = load(self.model_size, device=self.device)
            self.impl_name = "whisper"
            logger.info("[STTAgent] Using openai-whisper backend.")
            return
        except Exception as e:
            logger.debug("[STTAgent] openai-whisper unavailable: %s", e)

        # Neither backend loaded
        logger.warning("[STTAgent] No whisper backend available; falling back to MOCK STT.")
        self.impl_name = "mock"

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Always returns a dict with 'text', 'segments', 'language'.
        """

        # Mock mode
        if self.impl_name == "mock" or self.model is None:
            segs = [self._single_segment_from_wav(audio_path, "[MOCK TRANSCRIPT - STT not available]")]
            safe_texts = []
            for s in segs:
                if isinstance(s, dict) and "text" in s:
                    safe_texts.append(s["text"])
                elif isinstance(s, str):
                    safe_texts.append(s)
                elif s is not None:
                    safe_texts.append(str(s))
            full_text = " ".join(t for t in safe_texts if t).strip()
            return {"text": full_text, "segments": segs, "language": language or "und"}

        # Real backends
        try:
            if self.impl_name == "whisper_timestamped":
                if hasattr(self.model, "transcribe"):
                    res = self.model.transcribe(audio_path, language=language)
                else:
                    fn = getattr(self.impl_module, "transcribe", None)
                    res = fn(self.model, audio_path, language=language) if fn else {}
            elif self.impl_name == "whisper":
                if hasattr(self.model, "transcribe"):
                    res = self.model.transcribe(audio_path, language=language)
                else:
                    fn = getattr(self.impl_module, "transcribe", None)
                    res = fn(self.model, audio_path, language=language) if fn else {}
            else:
                res = {}
        except Exception as e:
            logger.exception("[STTAgent] Transcription failed for %s: %s", audio_path, e)
            segs = [self._single_segment_from_wav(audio_path, "[TRANSCRIBE ERROR]")]
            safe_texts = [seg["text"] for seg in segs if isinstance(seg, dict) and "text" in seg]
            return {"text": " ".join(safe_texts), "segments": segs, "language": language or "und"}

        raw_segments = []
        if isinstance(res, dict) and "segments" in res:
            raw_segments = res["segments"]
        elif isinstance(res, list):
            raw_segments = res
        elif isinstance(res, dict) and "text" in res:
            txt = str(res.get("text", "")).strip()
            segs = [self._single_segment_from_wav(audio_path, txt)]
            return {"text": txt, "segments": segs, "language": language or "und"}

        segments: List[Dict[str, Any]] = []
        for s in raw_segments:
            try:
                start = None
                end = None
                text = ""
                if isinstance(s, dict):
                    if "start" in s:
                        start = float(s["start"])
                    elif "start_ms" in s:
                        start = float(s["start_ms"]) / 1000.0
                    if "end" in s:
                        end = float(s["end"])
                    elif "end_ms" in s:
                        end = float(s["end_ms"]) / 1000.0
                    text = str(s.get("text", "")).strip()
                else:
                    text = str(s).strip()
                if start is None or end is None:
                    dur = self._wav_duration(audio_path)
                    start = 0.0 if start is None else start
                    end = dur if end is None else end
                segments.append({"start": start, "end": end, "text": text})
            except Exception:
                logger.debug("[STTAgent] Skipping malformed segment: %s", repr(s))
                continue

        if not segments:
            segments = [self._single_segment_from_wav(audio_path, "[EMPTY TRANSCRIPT]")]

        safe_texts = []
        for s in segments:
            if isinstance(s, dict) and "text" in s:
                safe_texts.append(s["text"])
            elif isinstance(s, str):
                safe_texts.append(s)
            elif s is not None:
                safe_texts.append(str(s))

        full_text = " ".join(t for t in safe_texts if t).strip()
        return {"text": full_text, "segments": segments, "language": language or "und"}

    def _wav_duration(self, path: str) -> float:
        try:
            with contextlib.closing(wave.open(path, "r")) as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / float(rate)
        except Exception:
            return 0.0

    def _single_segment_from_wav(self, path: str, text: str) -> Dict[str, Any]:
        dur = self._wav_duration(path)
        return {"start": 0.0, "end": dur, "text": text}

    def export_srt(self, segments: List[Dict[str, Any]], out_path: str) -> None:
        def fmt(t: float) -> str:
            ms = int((t - int(t)) * 1000)
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

        lines = []
        idx = 1
        for seg in segments:
            lines.append(str(idx))
            lines.append(f"{fmt(seg['start'])} --> {fmt(seg['end'])}")
            lines.append(seg.get("text", ""))
            lines.append("")
            idx += 1

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
