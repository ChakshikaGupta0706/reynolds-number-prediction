"""Cross-validation of all candidate models.

Schemes
  1. Repeated 5-fold CV (10 repeats)            -> interpolation skill
  2. Repeated 70/30 hold-out (50 splits)        -> comparison with earlier work
  3. Leave-one-Pr-out                           -> unseen Prandtl number
  4. Leave-one-Ra-decade-out (GroupKFold)       -> unseen Rayleigh range
  5. Train Ra<=1e9, test Ra>1e9                 -> extrapolation in Ra
  6. Train Pr<=1,  test Pr>1                    -> extrapolation in Pr

Outputs (results/): cv_summary.csv, cv_folds.csv, cv_predictions_<scheme>.csv, figures/*.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, RepeatedKFold, ShuffleSplit

from rbc_re import RNG, ROOT, FEATS, fit_predict, load, metrics, models

OUT = ROOT / "results"
(OUT / "figures").mkdir(parents=True, exist_ok=True)


def run_splits(model, d, splits):
    """Pooled out-of-fold prediction metrics + per-fold metrics."""
    folds, oof_sum, oof_n = [], np.zeros(len(d)), np.zeros(len(d))
    for tr, te in splits:
        p = fit_predict(model, d, d.index[tr], d.index[te])
        folds.append(metrics(d.y.values[te], p))
        oof_sum[te] += p
        oof_n[te] += 1
    oof = np.where(oof_n > 0, oof_sum / np.maximum(oof_n, 1), np.nan)
    return pd.DataFrame(folds), oof


def group_splits(groups):
    for g in pd.unique(groups):
        te = np.where(groups == g)[0]
        tr = np.where(groups != g)[0]
        yield tr, te


def main():
    _, d = load()
    d["Pr_grp"] = d.Pr.astype(str).values
    d["Ra_dec"] = np.floor(d.x_Ra.values).astype(int)
    idx = np.arange(len(d))
    schemes = {
        "5-fold x10 repeats": list(RepeatedKFold(n_splits=5, n_repeats=10, random_state=RNG).split(idx)),
        "70/30 hold-out x50": list(ShuffleSplit(50, test_size=0.3, random_state=RNG).split(idx)),
        "Leave-one-Pr-out": list(group_splits(d.Pr_grp.values)),
        "Leave-one-Ra-decade-out": list(group_splits(d.Ra_dec.values)),
        "Extrapolate Ra>1e9": [(np.where(d.Ra <= 1e9)[0], np.where(d.Ra > 1e9)[0])],
        "Extrapolate Pr>1": [(np.where(d.Pr <= 1)[0], np.where(d.Pr > 1)[0])],
    }
    summary, all_folds, preds = [], [], {}
    for sname, splits in schemes.items():
        for mname, m in models().items():
            print(f"{sname:28s} | {mname}", flush=True)
            f, oof = run_splits(m, d, splits)
            f.insert(0, "model", mname); f.insert(0, "scheme", sname)
            all_folds.append(f)
            te_all = np.unique(np.concatenate([te for _, te in splits]))
            pooled = metrics(d.y.values[te_all], oof[te_all])
            summary.append({"scheme": sname, "model": mname, "n_test_points": len(te_all),
                            "R2_log": pooled["R2_log"], "R2_Re": pooled["R2_Re"],
                            "MedAPE_%": pooled["MedAPE"], "MeanAPE_%": pooled["MeanAPE"],
                            "MaxAPE_%": pooled["MaxAPE"],
                            "fold_MedAPE_mean_%": f.MedAPE.mean(), "fold_MedAPE_std_%": f.MedAPE.std(ddof=0)})
            preds[(sname, mname)] = oof
    S = pd.DataFrame(summary)
    S.round(4).to_csv(OUT / "cv_summary.csv", index=False)
    pd.concat(all_folds).round(4).to_csv(OUT / "cv_folds.csv", index=False)

    # out-of-fold predictions for the main models, per scheme
    for sname in schemes:
        tab = d[["source", "Pr", "Ra", "Re"]].copy()
        for mname in models():
            tab[mname] = 10 ** preds[(sname, mname)]
        tab.round(2).to_csv(OUT / f"cv_predictions_{sname.replace(' ', '_').replace('/', '-').replace('>', 'gt')}.csv",
                            index=False)

    # figures ---------------------------------------------------------------
    names = list(models())
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    for ax, sname in zip(axs.ravel(), schemes):
        sub = S[S.scheme == sname].set_index("model").loc[names]
        ax.barh(range(len(names)), sub["MedAPE_%"], color=["#888"] * 4 + ["#1f77b4", "#d62728"])
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names if sname == list(schemes)[0] or
                                                           sname == list(schemes)[3] else [""] * 6, fontsize=8)
        ax.set_xscale("log"); ax.set_xlabel("median abs. % error in Re"); ax.set_title(sname)
        for i, v in enumerate(sub["MedAPE_%"]):
            ax.text(v, i, f" {v:.1f}", va="center", fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "figures" / "cv_model_comparison.png", dpi=140); plt.close()

    show = ["PowerLaw (log-linear)", "PowerLaw + RF residual", "PowerLaw + GP residual", "RandomForest (direct)"]
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.6), sharex=True, sharey=True)
    for ax, mname in zip(axs, show):
        oof = preds[("5-fold x10 repeats", mname)]
        sc = ax.scatter(d.Re, 10 ** oof, c=d.x_Pr, cmap="viridis", s=22)
        lim = [d.Re.min() * .7, d.Re.max() * 1.5]
        ax.plot(lim, lim, "k-", lw=1); ax.plot(lim, np.array(lim) * 1.2, "k:", lw=.8)
        ax.plot(lim, np.array(lim) / 1.2, "k:", lw=.8)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(mname, fontsize=10)
        ax.set_xlabel("Re (DNS)")
    axs[0].set_ylabel("Re predicted (5-fold CV, mean over repeats)")
    fig.colorbar(sc, ax=axs, label="log10 Pr", pad=0.01)
    plt.savefig(OUT / "figures" / "cv_parity_5fold.png", dpi=140, bbox_inches="tight"); plt.close()

    # per-point error vs Ra for leave-one-Pr-out, coloured by Pr
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for a, mname in zip(ax, ["PowerLaw (log-linear)", "PowerLaw + RF residual"]):
        p = 10 ** preds[("Leave-one-Pr-out", mname)]
        err = (p - d.Re) / d.Re * 100
        sc = a.scatter(d.Ra, err, c=d.x_Pr, cmap="viridis", s=22)
        a.axhline(0, color="k", lw=1); a.set_xscale("log"); a.set_xlabel("Ra"); a.set_title(f"Leave-one-Pr-out: {mname}")
    ax[0].set_ylabel("signed error in Re [%]"); fig.colorbar(sc, ax=ax, label="log10 Pr")
    plt.savefig(OUT / "figures" / "cv_error_vs_Ra_LOPO.png", dpi=140, bbox_inches="tight"); plt.close()
    print(S.round(2).to_string())


if __name__ == "__main__":
    main()
