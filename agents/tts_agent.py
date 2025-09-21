# agents/tts_agent.py
from pathlib import Path
from typing import Dict, Optional
from pydub import AudioSegment
from pydub.generators import Sine
import logging
import tempfile, os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TTSAgent:
    def __init__(self, out_dir: str = "outputs/audio", backend: str = "gtts"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend.lower()

    def synthesize(self, text: str, out_name: str, phonemes: Optional[str] = None) -> Dict:
        """
        Always use gTTS for Hebrew speech.
        Returns: {"path": str, "duration_ms": int, "sample_rate": 16000}
        """
        target = str(self.out_dir / out_name)
        input_for_tts = phonemes if phonemes and phonemes.strip() else text
        return self._synthesize_gtts(input_for_tts, target)

    def _synthesize_gtts(self, text: str, out_path: str) -> Dict:
        from gtts import gTTS

        # Google TTS sometimes uses 'iw' instead of 'he'
        lang_code = "he"
        try:
            tts = gTTS(text=text, lang=lang_code)
        except ValueError:
            lang_code = "iw"
            tts = gTTS(text=text, lang=lang_code)

        # Save to temporary MP3 (close handle before using with pydub)
        tmp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp_mp3.close()  # <-- CLOSE the file handle
        tts.save(tmp_mp3.name)

        # Convert to WAV using pydub
        seg = AudioSegment.from_file(tmp_mp3.name, format="mp3")
        seg = seg.set_frame_rate(16000).set_channels(1)
        seg.export(out_path, format="wav")

        # Now safe to delete
        os.unlink(tmp_mp3.name)

        return {"path": out_path, "duration_ms": len(seg), "sample_rate": 16000}

    def _synthesize_mock(self, text: str, out_path: str) -> Dict:
        """Fallback: simple beep if gTTS fails."""
        words = max(1, len(str(text).split()))
        tone = Sine(300).to_audio_segment(duration=300)
        silence = AudioSegment.silent(duration=200 * words)
        seg = tone + silence
        seg = seg.set_frame_rate(16000).set_channels(1)
        seg.export(out_path, format="wav")
        return {"path": out_path, "duration_ms": len(seg), "sample_rate": 16000}
