"""Fail CI if checked-in eval metrics differ from a fresh run (latency excluded)."""

import json
import sys

VOLATILE = {"latency_ms_p50", "latency_ms_p95"}


def metrics(path: str) -> dict:
    report = json.loads(open(path).read())
    return {
        name: {k: v for k, v in variant["metrics"].items() if k not in VOLATILE}
        for name, variant in report["variants"].items()
    }


checked_in, fresh = metrics(sys.argv[1]), metrics(sys.argv[2])
if checked_in != fresh:
    print("Checked-in results are stale.\nchecked in:", checked_in, "\nfresh:", fresh)
    sys.exit(1)
print("Checked-in eval metrics match a fresh run.")
