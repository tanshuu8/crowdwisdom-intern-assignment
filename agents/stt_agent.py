"""
Robust STTAgent for CrowdWisdom project.

- Honors CW_STT_FORCE_MOCK env var or force_mock constructor arg.
- Supports three backends:
    1. whisper_timestamped (local)
    2. openai-whisper (local module "whisper")
    3. OpenAI API (if model_size == "openai")
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

        # --- OpenAI API mode ---
        if self.model_size.lower() == "openai":
            try:
                import openai  # type: ignore
                self.impl_name = "openai"
                self.impl_module = openai
                logger.info("[STTAgent] Using OpenAI API backend for STT.")
                return
            except Exception as e:
                logger.warning("[STTAgent] Failed to init OpenAI API backend: %s", e)

        # --- Try whisper_timestamped ---
        try:
            import whisper_timestamped as wst  # type: ignore
            self.impl_module = wst
            if hasattr(wst, "load_model"):
                logger.info("[STTAgent] Loading whisper_timestamped model '%s' ...", self.model_size)
                self.model = wst.load_model(self.model_size, device=self.device)
            self.impl_name = "whisper_timestamped"
            logger.info("[STTAgent] Using whisper_timestamped backend.")
            return
        except ImportError:
            logger.debug("[STTAgent] whisper_timestamped not installed.")
        except Exception as e:
            logger.warning("[STTAgent] whisper_timestamped failed: %s", e)

        # --- Try openai-whisper (local) ---
        try:
            import whisper as wh  # type: ignore
            self.impl_module = wh
            if hasattr(wh, "load_model"):
                logger.info("[STTAgent] Loading openai-whisper model '%s' ...", self.model_size)
                self.model = wh.load_model(self.model_size, device=self.device)
            self.impl_name = "whisper"
            logger.info("[STTAgent] Using openai-whisper backend.")
            return
        except ImportError:
            logger.debug("[STTAgent] openai-whisper not installed.")
        except Exception as e:
            logger.warning("[STTAgent] openai-whisper failed: %s", e)

        # --- Neither backend worked ---
        logger.warning("[STTAgent] No whisper backend available; falling back to MOCK STT.")
        self.impl_name = "mock"

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Always returns a dict with 'text', 'segments', 'language'."""

        # --- Mock mode ---
        if self.impl_name == "mock" or self.model is None:
            segs = [self._single_segment_from_wav(audio_path, "[MOCK TRANSCRIPT - STT not available]")]
            full_text = " ".join(s["text"] for s in segs if isinstance(s, dict))
            return {"text": full_text, "segments": segs, "language": language or "und"}

        # --- OpenAI API backend ---
        if self.impl_name == "openai":
            try:
                from openai import OpenAI
                client = OpenAI()
                with open(audio_path, "rb") as f:
                    resp = client.audio.transcriptions.create(
                        model="gpt-4o-mini-transcribe",
                        file=f
                    )
                text = resp["text"]
                dur = self._wav_duration(audio_path)
                segs = [{"start": 0.0, "end": dur, "text": text}]
                return {"text": text, "segments": segs, "language": language or "und"}
            except Exception as e:
                logger.exception("[STTAgent] OpenAI API transcription failed: %s", e)
                segs = [self._single_segment_from_wav(audio_path, "[OPENAI STT ERROR]")]
                return {"text": "[OPENAI STT ERROR]", "segments": segs, "language": "und"}

        # --- Local whisper backends ---
        try:
            if self.impl_name == "whisper_timestamped":
                res = self.model.transcribe(audio_path, language=language)
            elif self.impl_name == "whisper":
                res = self.model.transcribe(audio_path, language=language)
            else:
                res = {}
        except Exception as e:
            logger.exception("[STTAgent] Transcription failed for %s: %s", audio_path, e)
            segs = [self._single_segment_from_wav(audio_path, "[TRANSCRIBE ERROR]")]
            return {"text": " ".join(s["text"] for s in segs), "segments": segs, "language": language or "und"}

        # --- Normalize segments ---
        raw_segments = res.get("segments", []) if isinstance(res, dict) else []
        if isinstance(res, dict) and "text" in res and not raw_segments:
            txt = str(res.get("text", "")).strip()
            segs = [self._single_segment_from_wav(audio_path, txt)]
            return {"text": txt, "segments": segs, "language": res.get("language", language or "und")}

        segments: List[Dict[str, Any]] = []
        for s in raw_segments:
            try:
                start = float(s.get("start", 0.0))
                end = float(s.get("end", self._wav_duration(audio_path)))
                text = str(s.get("text", "")).strip()
                segments.append({"start": start, "end": end, "text": text})
            except Exception:
                logger.debug("[STTAgent] Skipping malformed segment: %s", repr(s))
                continue

        if not segments:
            segments = [self._single_segment_from_wav(audio_path, "[EMPTY TRANSCRIPT]")]

        full_text = " ".join(seg["text"] for seg in segments if seg.get("text")).strip()
        return {"text": full_text, "segments": segments, "language": res.get("language", language or "und")}

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
        for idx, seg in enumerate(segments, start=1):
            lines.append(str(idx))
            lines.append(f"{fmt(seg['start'])} --> {fmt(seg['end'])}")
            lines.append(seg.get("text", ""))
            lines.append("")

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
