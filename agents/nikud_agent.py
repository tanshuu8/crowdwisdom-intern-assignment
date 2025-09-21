# agents/nikud_agent.py
"""
NikudAgent: integrates with the installed phonikud package.
Returns a dict: {"vocalized": ..., "phonemes": ...}

Behavior:
 - If phonikud.phonemize(text) exists, use it.
 - Else if phonikud.Phonemizer exists, instantiate it once and reuse P.phonemize(text).
 - Fallback: try other common entrypoints.
 - Final fallback: return a safe mock string so the pipeline doesn't crash.
 - Small LRU-like cache avoids repeated work for identical inputs.
"""

from typing import Dict, Optional
import logging
import subprocess
import shlex
from functools import lru_cache

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NikudAgent:
    def __init__(self, use_real_phonikud: bool = True, phonikud_cli_path: Optional[str] = None):
        """
        :param use_real_phonikud: Try to use the phonikud Python API if True.
        :param phonikud_cli_path: Optional path to a local phonikud CLI script/executable.
        """
        self.use_real = use_real_phonikud
        self.cli_path = phonikud_cli_path
        self._phonemizer_instance = None
        self._module = None
        if self.use_real:
            self._module = self._import_phonikud()

    # Public API
    def add_nikud(self, text: str) -> Dict:
        """
        Return dict with keys:
         - 'vocalized': string (nikudized / phoneme string / best-effort)
         - 'phonemes': optional phoneme hint string or None
        """
        if not text:
            return {"vocalized": "", "phonemes": None}

        # 1) try Python library implementation if available
        if self._module is not None:
            try:
                # use cached wrapper to avoid repeated heavy calls
                vocalized, phonemes = self._vocalize_using_python_lib_cached(text)
                return {"vocalized": vocalized, "phonemes": phonemes}
            except Exception as e:
                logger.warning("phonikud python API failed: %s", e)

        # 2) try CLI call if provided
        if self.cli_path:
            try:
                return self._vocalize_using_cli(text, self.cli_path)
            except Exception as e:
                logger.warning("phonikud CLI call failed: %s", e)

        # 3) fallback: mock
        return {"vocalized": self._vocalize_mock(text), "phonemes": None}

    # ------------------------------
    # Import helper
    # ------------------------------
    def _import_phonikud(self):
        """
        Import the installed phonikud module and return it, or None.
        """
        try:
            import phonikud as mod
            logger.info("Imported phonikud from: %s", getattr(mod, "__file__", "<built-in>"))
            return mod
        except Exception as e:
            logger.info("phonikud not importable: %s", e)
            return None

    # ------------------------------
    # Caching wrapper
    # ------------------------------
    @lru_cache(maxsize=1024)
    def _vocalize_using_python_lib_cached(self, text: str):
        # lru_cache requires args to be hashable -> we pass raw text only
        return self._vocalize_using_python_lib(text)

    # ------------------------------
    # Python-library integration
    # ------------------------------
    def _vocalize_using_python_lib(self, text: str):
        """
        Attempt various phonikud APIs:
         - phonemize(text)
         - Phonemizer().phonemize(text)
         - fallbacks by scanning candidate names
        Returns (vocalized_str, phonemes_or_none)
        """
        module = self._module
        if not module:
            raise RuntimeError("phonikud module not available")

        # Preferred: phonemize function
        if hasattr(module, "phonemize"):
            try:
                res = module.phonemize(text)
                # phonemize often returns a str (phoneme sequence); we put it into vocalized
                return str(res), None
            except Exception as e:
                logger.debug("phonemize call failed: %s", e)

        # Preferred: Phonemizer class
        if hasattr(module, "Phonemizer"):
            try:
                if self._phonemizer_instance is None:
                    P = getattr(module, "Phonemizer")
                    # instantiate with defaults
                    self._phonemizer_instance = P()
                # call instance.phonemize if present
                instance = self._phonemizer_instance
                if hasattr(instance, "phonemize"):
                    res = instance.phonemize(text)
                    return str(res), None
            except Exception as e:
                logger.debug("Phonemizer.phonemize failed: %s", e)

        # Try other common function names (best-effort)
        for fn_name in ("diacritize", "add_diacritics", "add_nikud", "nikud", "vocalize", "to_nikud"):
            if hasattr(module, fn_name):
                try:
                    fn = getattr(module, fn_name)
                    res = fn(text)
                    if isinstance(res, dict):
                        vocal = res.get("vocalized") or res.get("nikud") or str(res)
                        phon = res.get("phonemes")
                        return str(vocal), phon
                    if isinstance(res, (list, tuple)):
                        return str(res[0]), None
                    return str(res), None
                except Exception as e:
                    logger.debug("fallback %s failed: %s", fn_name, e)
                    continue

        # As a last resort, try to call any callable that looks promising
        for name in dir(module):
            lname = name.lower()
            if lname.startswith(("diac", "phon", "nik", "voca")):
                attr = getattr(module, name)
                if callable(attr):
                    try:
                        res = attr(text)
                        return str(res), None
                    except Exception:
                        continue

        raise RuntimeError("phonikud installed but no usable API found")

    # ------------------------------
    # CLI integration
    # ------------------------------
    def _vocalize_using_cli(self, text: str, cli_path: str) -> Dict:
        """
        If you cloned phonikud locally and it exposes a CLI, call it here.
        Example call (adjust flags according to the CLI):
          python path/to/phonikud_cli.py --text "..." --json
        """
        cmd = f'python {shlex.quote(cli_path)} --text {shlex.quote(text)} --json'
        logger.info("Calling phonikud CLI: %s", cmd)
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            raise RuntimeError(f"phonikud CLI error: {proc.stderr.strip()}")
        out = proc.stdout.strip()
        try:
            import json as _json
            parsed = _json.loads(out)
            vocalized = parsed.get("vocalized") or parsed.get("nikud") or parsed.get("text")
            phonemes = parsed.get("phonemes")
            return {"vocalized": vocalized, "phonemes": phonemes}
        except Exception:
            # CLI printed plain text
            return {"vocalized": out, "phonemes": None}

    # ------------------------------
    # Mock fallback
    # ------------------------------
    def _vocalize_mock(self, text: str) -> str:
        return text + " (ניקוד_המחשה)"


# -------------------------
# quick local test when run directly
# -------------------------
if __name__ == "__main__":
    a = NikudAgent(use_real_phonikud=True, phonikud_cli_path=None)
    samples = [
        "שלום, אבקש לבטל את המנוי שלי.",
        "האם תוכל לשלוח לי אישור לביטול במייל?"
    ]
    for s in samples:
        try:
            res = a.add_nikud(s)
            print("IN :", s)
            print("OUT:", res["vocalized"])
            print("PHO:", res["phonemes"])
            print("----")
        except Exception as e:
            print("ERROR:", e)
