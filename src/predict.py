"""Usage: python src/predict.py <Ra> <Pr>     e.g. python src/predict.py 1e9 0.7"""
import sys
import joblib
import numpy as np
from rbc_re import ROOT


def predict(Ra, Pr):
    pack = joblib.load(ROOT / "models" / "re_model.joblib")
    (r0, r1), (p0, p1) = pack["domain"]["Ra"], pack["domain"]["Pr"]
    if not (r0 <= Ra <= r1 and p0 <= Pr <= p1):
        print(f"WARNING: outside training domain Ra [{r0:.0e},{r1:.0e}], Pr [{p0},{p1}] -> extrapolation")
    return 10 ** pack["model"].predict(np.array([[np.log10(Ra), np.log10(Pr)]]))[0]


if __name__ == "__main__":
    Ra, Pr = float(sys.argv[1]), float(sys.argv[2])
    print(f"Re(Ra={Ra:.3g}, Pr={Pr:.3g}) = {predict(Ra, Pr):.4g}")
