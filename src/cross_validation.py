"""Cross-validation / generalisation study. Run from repo root:  python src/cross_validation.py
Writes results/*.csv and figures/*.png"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold, train_test_split
from rbc_re import *

raw, d = load()
d["Pr_grp"] = d.Pr.astype(str)
ms = models()
names = list(ms)
rows_cv, rows_split, rows_gen, per_pr = [], [], [], []
oof_lopo = {}

# 1) Repeated 5-fold CV (10 repeats = 50 fits per model)
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=RNG)
folds = list(rkf.split(d))
for n, m in ms.items():
    res = pd.DataFrame([metrics(d.y.values[te], fit_predict(m, d, tr, te)) for tr, te in folds])
    rows_cv.append({"model": n, **{f"{c} mean": res[c].mean() for c in ["R2_log", "MedAPE", "MeanAPE"]},
                    "R2_log std": res.R2_log.std(), "MedAPE std": res.MedAPE.std(), "MaxAPE (worst fold)": res.MaxAPE.max()})
cv = pd.DataFrame(rows_cv).set_index("model").round(4)
cv.to_csv(RESULTS / "cv_5fold_repeated.csv")

# 2) Single fixed 70/30 split, stratified by Pr so every Pr value is in both sets
tr_i, te_i = train_test_split(d.index, test_size=0.30, random_state=RNG, stratify=d.Pr_grp.where(d.Pr_grp.map(d.Pr_grp.value_counts()) > 1, "rare"))
for n, m in ms.items():
    p = fit_predict(m, d, tr_i, te_i)
    rows_split.append({"model": n, "n_train": len(tr_i), "n_test": len(te_i), **metrics(d.y.values[te_i], p)})
pd.DataFrame(rows_split).set_index("model").round(4).to_csv(RESULTS / "split_70_30.csv")

# 3) Generalisation tests: leave-one-Pr-out, Ra extrapolation, Pr extrapolation
for n, m in ms.items():
    p = np.zeros(len(d))
    for g in d.Pr_grp.unique():
        te = d.index[d.Pr_grp == g]; tr = d.index[d.Pr_grp != g]
        p[te] = fit_predict(m, d, tr, te)
    oof_lopo[n] = p
    r_l = metrics(d.y.values, p)
    out = {"model": n, "LOPO R2(log)": r_l["R2_log"], "LOPO MedAPE%": r_l["MedAPE"], "LOPO MaxAPE%": r_l["MaxAPE"]}
    for lab, mask in [("Ra>1e9", d.Ra > 1e9), ("Pr>1", d.Pr > 1), ("Ra<=1e7 held out", d.Ra <= 1e7)]:
        te = d.index[mask]; tr = d.index[~mask]
        out[f"extrap {lab} MedAPE%"] = metrics(d.y.values[te], fit_predict(m, d, tr, te))["MedAPE"]
        out[f"extrap {lab} n_test"] = len(te)
    rows_gen.append(out)
gen = pd.DataFrame(rows_gen).set_index("model").round(4)
gen.to_csv(RESULTS / "generalisation_tests.csv")

# Model selection (robustness, not random-split score)
sel_cols = ["LOPO MedAPE%", "extrap Ra>1e9 MedAPE%", "extrap Pr>1 MedAPE%"]
best = gen[sel_cols].mean(axis=1).idxmin()
(RESULTS / "selected_model.txt").write_text(f"{best}\n")

# 4) Per-Pr error of the selected model (leave-one-Pr-out predictions)
p = oof_lopo[best]
d["pred_lopo"] = 10 ** p
d["err_%"] = (d.pred_lopo - d.Re) / d.Re * 100
pp = d.groupby("Pr").agg(n=("Re", "size"), MedAPE=("err_%", lambda s: s.abs().median()), MaxAPE=("err_%", lambda s: s.abs().max()),
                         bias=("err_%", "mean")).round(2)
pp.to_csv(RESULTS / "per_Pr_error_LOPO.csv")
d[["source", "Pr", "Ra", "Re", "pred_lopo", "err_%"]].round(3).to_csv(RESULTS / "lopo_predictions.csv", index=False)

# Figures
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
x = np.arange(len(names)); w = .27
ax[0].bar(x - w, cv["MedAPE mean"], w, yerr=cv["MedAPE std"], label="Repeated 5-fold CV")
ax[0].bar(x, gen["LOPO MedAPE%"], w, label="Leave-one-Pr-out")
ax[0].bar(x + w, gen["extrap Ra>1e9 MedAPE%"], w, label="Extrapolate Ra>1e9")
ax[0].set_yscale("log"); ax[0].set_xticks(x); ax[0].set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=7)
ax[0].set_ylabel("Median abs. % error in Re"); ax[0].set_title("Model comparison"); ax[0].legend(fontsize=8)
ax[1].bar(pp.index.astype(str), pp.MedAPE); ax[1].set_xlabel("Pr (held out)"); ax[1].set_ylabel("Median abs. % error")
ax[1].set_title(f"Per-Pr error, {best}"); plt.setp(ax[1].get_xticklabels(), rotation=60, fontsize=7)
plt.tight_layout(); plt.savefig(FIGURES / "cv_model_comparison.png", dpi=140)

fig, ax = plt.subplots(figsize=(5.5, 5))
sc = ax.scatter(d.Re, d.pred_lopo, c=d.x_Pr, cmap="viridis", s=25); ax.set_xscale("log"); ax.set_yscale("log")
lim = [d.Re.min() * .7, d.Re.max() * 1.4]; ax.plot(lim, lim, "k-", lw=1)
ax.plot(lim, np.array(lim) * 1.2, "k:", lw=.8); ax.plot(lim, np.array(lim) / 1.2, "k:", lw=.8)
ax.set(xlabel="Re (DNS)", ylabel="Re predicted (leave-one-Pr-out)", title=best + "\n(dotted = ±20%)")
plt.colorbar(sc, label="log10 Pr"); plt.tight_layout(); plt.savefig(FIGURES / "parity_lopo.png", dpi=140)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print("Repeated 5x10 CV\n", cv.to_string(), "\n\n70/30 split\n", pd.read_csv(RESULTS / "split_70_30.csv").to_string(),
      "\n\nGeneralisation\n", gen.to_string(), "\n\nSelected:", best, "\n\nPer-Pr LOPO error\n", pp.to_string())
