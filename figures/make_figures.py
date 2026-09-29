"""Regenerate every figure for the smile-smoothing-shootout paper.

    python make_figures.py             # all of them
    python make_figures.py fig9 fig10  # a subset

Outputs vector PDFs beside this file. fig6 (calibration cost) reads the
intraday results and fig7 (the exponent through the session) the
fitted-exponent panel; neither ships in the public repository, so both
are skipped when their input is missing.
"""
import runpy
import sys
import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SCRIPTS = sorted(os.path.basename(p) for p in glob.glob(os.path.join(HERE, "fig*.py")))

wanted = sys.argv[1:]
for s in SCRIPTS:
    if wanted and not any(s.startswith(w) for w in wanted):
        continue
    print("-- running", s, flush=True)
    try:
        runpy.run_path(os.path.join(HERE, s), run_name="__main__")
    except FileNotFoundError as e:
        # the intraday result CSVs are supplied on request, not released
        print("   skipped: input not found (%s)" % e.filename, flush=True)
