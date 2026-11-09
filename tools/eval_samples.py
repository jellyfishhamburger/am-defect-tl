import glob
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from analysis.demo_app import available_runs, label_of, predict_one  # noqa: E402

samples = sorted(glob.glob("samples/D1/defect/*")) + sorted(glob.glob("samples/D1/nondefect/*"))
runs = available_runs()
rows = []
print(f"samples={len(samples)} runs={len(runs)}", flush=True)

for img in samples:
    true = "defect" if "/defect/" in img.replace("\\", "/") else "nondefect"
    rec = {"file": os.path.basename(img), "true": true, "p": {}}
    for r in runs:
        try:
            p = predict_one(r, img)["p_defect"]
        except Exception as e:
            p = None
            print("ERR", r.name, type(e).__name__, e, flush=True)
        rec["p"][r.name] = p
    rows.append(rec)
    preds = {k: ("defect" if (v or 0) >= 0.5 else "nondefect") for k, v in rec["p"].items() if v is not None}
    ok = sum(1 for v in preds.values() if v == true)
    print(f"{true:10s} {os.path.basename(img)} -> {ok}/{len(preds)} correct  "
          + " ".join(f"{k.split('__')[0][:6]}{'D2' if '__D2__' in k else 'D1'}={v:.2f}" for k, v in rec["p"].items() if v is not None),
          flush=True)

json.dump(rows, open(os.environ.get("TEMP", ".") + "/sample_eval.json", "w"), indent=1)

print("\n=== per-model accuracy on the 20 samples ===", flush=True)
for r in runs:
    ok = 0
    n = 0
    for rec in rows:
        p = rec["p"].get(r.name)
        if p is None:
            continue
        n += 1
        ok += (("defect" if p >= 0.5 else "nondefect") == rec["true"])
    print(f"{label_of(r):28s} {ok}/{n}", flush=True)
