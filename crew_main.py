# crew_main.py
"""
Detailed orchestration for the intern assignment demo.
This file is intentionally verbose and production-minded:
- clear function decomposition
- robust error handling & guardrails
- structured logging and JSON metadata
- audio stitching and SRT export (if STT produces segments)
- configurable backends (mock vs real)
- per-turn artifact naming and metadata
- save transcript (txt + json), SRT, logs, and stitched audio
- easy to run: `python crew_main.py --turns 3 --stt-model tiny --tts-backend auto`

Requires:
 - agents package (nikud_agent, tts_agent, stt_agent, cs_agent, transcript_agent, logger_agent, supervisor_agent, client_agent)
 - pydub installed + ffmpeg on PATH
 - whisper installed for real STT (if used)
"""

import argparse
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from pydub import AudioSegment

# Agent imports (expect these modules to exist in agents/)
from agents.nikud_agent import NikudAgent
from agents.tts_agent import TTSAgent
from agents.stt_agent import STTAgent
from agents.cs_agent import CustomerServiceAgent
from agents.transcript_agent import TranscriptAgent
from agents.logger_agent import LoggerAgent
from agents.supervisor_agent import SupervisorAgent
from agents.client_agent import ClientAgent

# ---------- Configuration & constants ----------
BASE_DIR = Path.cwd()
OUTPUT_DIR = BASE_DIR / "outputs"
AUDIO_DIR = OUTPUT_DIR / "audio"
TRANS_DIR = OUTPUT_DIR / "transcripts"
LOG_DIR = OUTPUT_DIR / "logs"
METADATA_DIR = OUTPUT_DIR / "metadata"

for d in (OUTPUT_DIR, AUDIO_DIR, TRANS_DIR, LOG_DIR, METADATA_DIR):
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, encoding="utf-8")]
)
logger = logging.getLogger("crew_main")

# ---------- Utility helpers ----------

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def safe_write_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)

def ensure_wav_mono_16k(path: Path) -> Path:
    """
    Normalize audio to 16k mono WAV using pydub (ffmpeg must be installed).
    Returns path to normalized file (may overwrite).
    """
    seg = AudioSegment.from_file(path)
    seg = seg.set_frame_rate(16000).set_channels(1)
    seg.export(path, format="wav")
    return path

# ---------- SRT exporter ----------
def segments_to_srt(segments: List[Dict], out_path: Path):
    """
    segments: list of dicts with keys: start (s), end (s), text
    Creates a basic SRT file with 0-based segments.
    """
    def fmt_ts(seconds: float) -> str:
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        millis = int((td.total_seconds() - int(td.total_seconds())) * 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    lines = []
    for i, seg in enumerate(segments, start=1):
        start = seg.get("start", seg.get("start_time", 0.0))
        end = seg.get("end", seg.get("end_time", start + 1.0))
        text = seg.get("text", "").replace("\n", " ").strip()
        lines.append(f"{i}")
        lines.append(f"{fmt_ts(start)} --> {fmt_ts(end)}")
        lines.append(text)
        lines.append("")  # blank line
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote SRT: %s", out_path)

# ---------- Audio stitching ----------
def stitch_audio(audio_paths: List[Path], out_path: Path, pause_ms: int = 150) -> Optional[Path]:
    """
    Concatenate given audio files into a single WAV.
    Returns out_path if successful, else None.
    """
    segments = []
    for p in audio_paths:
        if p and p.exists():
            try:
                seg = AudioSegment.from_file(p)
                segments.append(seg)
            except Exception as e:
                logger.warning("Failed reading audio %s: %s", p, e)

    if not segments:
        logger.warning("No audio segments to stitch.")
        return None

    out = segments[0]
    for seg in segments[1:]:
        out += AudioSegment.silent(duration=pause_ms) + seg
    out = out.set_frame_rate(16000).set_channels(1)
    out.export(out_path, format="wav")
    logger.info("Stitched audio written: %s", out_path)
    return out_path

# ---------- Core orchestration per-turn ----------
def run_turn(
    turn_index: int,
    client_text: str,
    nikud_agent: NikudAgent,
    tts_agent: TTSAgent,
    stt_agent: STTAgent,
    cs_agent: CustomerServiceAgent,
    transcript_agent: TranscriptAgent,
    logger_agent: LoggerAgent,
) -> Dict:
    """
    Execute a single client->agent turn:
    - client_text -> nikud -> TTS -> client_wav
    - STT(client_wav) -> transcript_text
    - CS decides -> reply_text
    - nikud(reply_text) -> TTS -> agent_wav
    - Log and return metadata dict for this turn
    """
    meta = {"turn": turn_index, "client_text": client_text, "ts": now_iso()}
    logger.info("TURN %d: client_text=%s", turn_index, client_text)

    # 1) Client TTS
    client_v = nikud_agent.add_nikud(client_text)["vocalized"]
    client_wav = AUDIO_DIR / f"client_{turn_index+1}.wav"
    t1 = datetime.now()
    tts_res = tts_agent.synthesize(client_v, client_wav.name, phonemes=None)
    t2 = datetime.now()
    meta["client_audio"] = str(client_wav)
    meta["tts_client_duration_ms"] = tts_res.get("duration_ms")
    meta["tts_client_time_s"] = (t2 - t1).total_seconds()
    logger_agent.log({"role": "client_tts", "turn": turn_index, "text": client_text, "audio": str(client_wav)})

    # Normalize audio for STT
    try:
        ensure_wav_mono_16k(client_wav)
    except Exception as e:
        logger.warning("Failed to normalize client audio: %s", e)

    # 2) STT
    try:
        stt_out = stt_agent.transcribe(str(client_wav))
        stt_text = stt_out.get("text", "").strip()
        meta["stt_text"] = stt_text
        meta["stt_segments"] = stt_out.get("segments", [])
        logger_agent.log({"role": "stt", "turn": turn_index, "transcript": stt_text})
    except Exception as e:
        logger.exception("STT failed for turn %d: %s", turn_index, e)
        stt_text = ""
        meta["stt_error"] = str(e)

    # 3) Customer Service decision
    try:
        cs_resp = cs_agent.reply(stt_text or client_text)
        reply_text = cs_resp.get("reply", "מצטער, לא הבנתי — אפשר לחזור בבקשה?")
        meta["cs_action"] = cs_resp.get("action")
        meta["reply_text"] = reply_text
        logger_agent.log({"role": "cs_decision", "turn": turn_index, "action": cs_resp.get("action"), "text": reply_text})
    except Exception as e:
        logger.exception("CS agent failed: %s", e)
        reply_text = "מצטער, יש בעיה טכנית. נא נסה מאוחר יותר."
        meta["cs_error"] = str(e)

    # 4) Agent TTS
    try:
        reply_v = nikud_agent.add_nikud(reply_text)["vocalized"]
        agent_wav = AUDIO_DIR / f"agent_{turn_index+1}.wav"
        t3 = datetime.now()
        tts_res2 = tts_agent.synthesize(reply_v, agent_wav.name, phonemes=None)
        t4 = datetime.now()
        meta["agent_audio"] = str(agent_wav)
        meta["tts_agent_duration_ms"] = tts_res2.get("duration_ms")
        meta["tts_agent_time_s"] = (t4 - t3).total_seconds()
        logger_agent.log({"role": "agent_tts", "turn": turn_index, "text": reply_text, "audio": str(agent_wav)})
    except Exception as e:
        logger.exception("Agent TTS failed: %s", e)
        meta["tts_agent_error"] = str(e)
        agent_wav = None

    # 5) Transcript storage
    transcript_agent.add_turn("client", stt_text or client_text)
    transcript_agent.add_turn("agent", reply_text)

    meta["end_ts"] = now_iso()
    return meta

# ---------- Orchestration runner ----------
def run_conversation(
    max_turns: int,
    stt_model: str,
    tts_backend: str,
    use_real_phonikud: bool
) -> Dict:
    # Instantiate agents
    nikud = NikudAgent(use_real_phonikud=use_real_phonikud)
    tts = TTSAgent(out_dir=str(AUDIO_DIR), backend=tts_backend)
    stt = STTAgent(model_size=stt_model)
    cs = CustomerServiceAgent()
    transcript = TranscriptAgent(out_dir=str(TRANS_DIR))
    logger_agent = LoggerAgent(out_dir=str(LOG_DIR))
    supervisor = SupervisorAgent(asr_threshold=0.6, max_turns=max_turns)
    client = ClientAgent()  # scripted client with multiple utterances

    # metadata for run
    run_meta = {
        "started_at": now_iso(),
        "max_turns": max_turns,
        "stt_model": stt_model,
        "tts_backend": tts_backend,
        "use_real_phonikud": use_real_phonikud,
        "turns": []
    }

    audio_paths: List[Path] = []
    for turn in range(max_turns):
        if not supervisor.allow_new_turn(turn):
            logger.info("Supervisor prevented new turn: %d", turn)
            break

        client_text = client.next_utterance()
        if not client_text:
            logger.info("Client finished scripted utterances at turn %d", turn)
            break

        meta = run_turn(
            turn_index=turn,
            client_text=client_text,
            nikud_agent=nikud,
            tts_agent=tts,
            stt_agent=stt,
            cs_agent=cs,
            transcript_agent=transcript,
            logger_agent=logger_agent
        )

        run_meta["turns"].append(meta)

        # collect audio paths for stitching
        if meta.get("client_audio"):
            audio_paths.append(Path(meta["client_audio"]))
        if meta.get("agent_audio"):
            audio_paths.append(Path(meta["agent_audio"]))

        # check for end action from CS
        if meta.get("cs_action") in ("close", "goodbye"):
            logger.info("CS requested conversation close at turn %d", turn)
            break

    # Save transcripts & logs
    transcript_json = TRANS_DIR / f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    safe_write_json(transcript.conversation, transcript_json)
    logger.info("Saved transcript json: %s", transcript_json)

    # Save run metadata
    run_meta["finished_at"] = now_iso()
    run_meta_path = METADATA_DIR / f"run_meta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    safe_write_json(run_meta, run_meta_path)
    logger.info("Saved run metadata: %s", run_meta_path)

    # Persist logger agent logs
    logs_path = Path(logger_agent.save())
    logger.info("Saved logs: %s", logs_path)

    # Stitch audio into single conversation file
    stitched_path = OUTPUT_DIR / f"full_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    stitched = stitch_audio(audio_paths, stitched_path)
    if stitched:
        logger.info("Full conversation audio saved: %s", stitched)

    # Export SRT if STT provided timestamps
    # Use segments from first available turn that has them
    srt_candidate_segments = []
    for t in run_meta["turns"]:
        segs = t.get("stt_segments") or []
        if segs:
            # standardize segment list to dicts with start/end/text
            for s in segs:
                # whisper outputs 'start' and 'end' in seconds and 'text'
                if "start" in s and "end" in s and "text" in s:
                    srt_candidate_segments.append({"start": s["start"], "end": s["end"], "text": s["text"]})
            # break once we aggregated
    if srt_candidate_segments:
        srt_path = TRANS_DIR / f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt"
        segments_to_srt(srt_candidate_segments, srt_path)

    # Save final artifacts index
    artifacts = {
        "transcript_json": str(transcript_json),
        "logs": str(logs_path),
        "run_meta": str(run_meta_path),
        "stitched_audio": str(stitched) if stitched else None,
        "srt": str(srt_path) if srt_candidate_segments else None
    }
    safe_write_json(artifacts, METADATA_DIR / "artifacts_index.json")
    logger.info("Artifacts index saved.")

    return run_meta

# ---------- CLI entrypoint ----------
def parse_args():
    p = argparse.ArgumentParser(prog="crew_main.py", description="Run conversation demo with agents.")
    p.add_argument("--turns", type=int, default=3, help="Maximum number of client turns to run.")
    p.add_argument("--stt-model", type=str, default="tiny", help="Whisper model size (tiny, small, medium, large).")
    p.add_argument("--tts-backend", type=str, default="auto", help="TTS backend: auto, mock, pyttsx3, gtts.")
    p.add_argument("--phonikud", action="store_true", help="Use real phonikud if available.")
    return p.parse_args()

def main():
    args = parse_args()
    logger.info("Starting conversation run: turns=%s stt_model=%s tts_backend=%s phonikud=%s",
                args.turns, args.stt_model, args.tts_backend, args.phonikud)

    run_meta = run_conversation(
        max_turns=args.turns,
        stt_model=args.stt_model,
        tts_backend=args.tts_backend,
        use_real_phonikud=args.phonikud
    )

    logger.info("Run complete. Summary:\n%s", json.dumps(run_meta, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
