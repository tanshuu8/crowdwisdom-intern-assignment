# inspect_phonikud.py
import importlib, inspect, sys

try:
    m = importlib.import_module("phonikud")
    print("phonikud module file:", getattr(m, "__file__", "built-in"))
    names = sorted([n for n in dir(m) if not n.startswith('_')])
    print("top-level names:", names)
    for cand in ("diacritize","add_diacritics","add_nikud","nikud","Diacritizer"):
        if hasattr(m, cand):
            obj = getattr(m, cand)
            print(f"\nFOUND: {cand} -> {type(obj)}")
            try:
                print("  signature:", inspect.signature(obj))
            except Exception:
                print("  signature: <not available>")
            doc = (getattr(obj, "__doc__", "") or "").strip().splitlines()
            if doc:
                print("  doc:", doc[0][:200])
except Exception as e:
    print("phonikud import failed:", e)
    sys.exit(1)
