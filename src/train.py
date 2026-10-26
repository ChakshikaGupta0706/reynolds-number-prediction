"""Fit the final model on the full dataset, save it, and draw diagnostic plots.

Final model = power law in (Ra, Pr) + random-forest residual correction, chosen from the
cross-validation results (see results/cv_summary.csv).
"""
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from rbc_re import FEATS, ROOT, load, models

NAME = "PowerLaw + RF residual"


def main():
    raw, d = load()
    pl = LinearRegression().fit(d[FEATS].values, d.y.values)
    print(f"Training rows {len(d)}; power law: Re = {10**pl.intercept_:.4g} Ra^{pl.coef_[0]:.3f} Pr^{pl.coef_[1]:.3f}")
    model = models()[NAME].fit(d[FEATS].values, d.y.values)
    joblib.dump({"model": model, "features": FEATS, "name": NAME,
                 "power_law": {"A": float(10 ** pl.intercept_), "a_Ra": float(pl.coef_[0]), "b_Pr": float(pl.coef_[1])},
                 "domain": {"Ra": [float(d.Ra.min()), float(d.Ra.max())],
                            "Pr": [float(d.Pr.min()), float(d.Pr.max())]}}, ROOT / "models" / "re_model.joblib")

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    pred = 10 ** model.predict(d[FEATS].values)
    ax[0].loglog(d.Re, pred, "o", ms=5)
    lim = [d.Re.min() * .7, d.Re.max() * 1.4]
    ax[0].plot(lim, lim, "k-"); ax[0].set(xlabel="Re (DNS)", ylabel="Re (model, training fit)", title=NAME)
    Ra_g = np.logspace(5.5, 12.3, 200)
    for Pr_ in [0.021, 0.1, 1]:
        s = d[np.isclose(d.Pr, Pr_)]
        l, = ax[1].loglog(s.Ra, s.Re * s.Pr, "o", ms=5, label=f"Pr={Pr_} DNS")
        ax[1].loglog(Ra_g, Pr_ * 10 ** model.predict(np.c_[np.log10(Ra_g), np.full_like(Ra_g, np.log10(Pr_))]),
                     color=l.get_color())
    ax[1].set(xlabel="Ra", ylabel="Re·Pr", title="Re·Pr vs Ra"); ax[1].legend()
    Pr_g = np.logspace(-2, 2, 200)
    for Ra_ in [1e7, 1e8, 1e9, 1e10]:
        ax[2].loglog(Pr_g, 10 ** model.predict(np.c_[np.full_like(Pr_g, np.log10(Ra_)), np.log10(Pr_g)]),
                     label=f"Ra={Ra_:.0e}")
    s = d[np.isclose(d.Ra, 1e9)]
    ax[2].loglog(s.Pr, s.Re, "ko", label="DNS Ra=1e9")
    ax[2].set(xlabel="Pr", ylabel="Re", title="Re vs Pr"); ax[2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(ROOT / "results" / "figures" / "final_model_diagnostics.png", dpi=140)

    # coverage map of the training data
    fig, a = plt.subplots(figsize=(6.5, 4.5))
    for src, g in d.groupby("source"):
        a.scatter(g.Ra, g.Pr, label=f"{src} (n={len(g)})", s=30)
    a.set(xscale="log", yscale="log", xlabel="Ra", ylabel="Pr", title="Training data coverage"); a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(ROOT / "results" / "figures" / "data_coverage.png", dpi=140)
    print("saved model + figures")


if __name__ == "__main__":
    main()
