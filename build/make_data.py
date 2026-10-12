"""Pre-build the cached afterslip training runs used by notebooks 2.10, 2.22 and 3.01-3.06 (data/*.npz).

The notebooks build any missing cache themselves on first use; running this first just moves that wait up front.
usage: python build/make_data.py
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from bayeskit import afterslip as A  # noqa: E402

for label, call in [("single patch, 4,000 uniform draws", lambda: A.training_set_single(4000, seed=0)),
                    ("single patch, 4,000 Latin-hypercube draws", lambda: A.training_set_single(4000, seed=0, design="lhs")),
                    ("1-D fault, 1,500 friction profiles", lambda: A.training_set_fault(1500, seed=0))]:
    t0 = time.time()
    call()
    print(f"{label}: ready in {time.time() - t0:.0f} s")
