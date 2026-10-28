"""Quick partial run for audit: only requested xi + low-score correction check."""
import sys, time
sys.path.insert(0, 'src')
from pl_predictor.evaluate.tune_dc_bp_xi import prepare_folds, _evaluate
from pl_predictor.models import scoreline

folds = prepare_folds()
for xi in (0.0010, 0.0018, 0.0030):
    rps_vals = []
    for f in folds:
        try:
            m = scoreline.fit_dixon_coles(f["train_df"], xi=xi)
            rps_vals.append(_evaluate(m, f["val_df"])["rps"])
        except Exception as exc:
            print(f"xi={xi} fold error: {exc}")
    if len(rps_vals) == len(folds):
        avg = sum(rps_vals) / len(rps_vals)
        print(f"xi={xi} RPS_mean={avg:.5f} (folds={len(rps_vals)})")
    else:
        print(f"xi={xi} INCOMPLETE: {len(rps_vals)}/{len(folds)} folds succeeded; skipping mean")
print(f"DONE at {time.strftime('%H:%M:%S')}")
