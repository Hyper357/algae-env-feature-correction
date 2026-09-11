"""Reproducible P01 audit for environment-driven fluorescence correction.

The script accepts a repository-relative or external data root. Raw field files
are read but never copied into the repository. All model preprocessing and
correction parameters are fitted inside each leave-one-station-out fold.
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

ENV = ["salinity", "pH", "temperature", "DO", "ORP", "turbidity"]
NUTRIENTS = ["NH3", "NO2", "NO3", "PO4", "SiO3"]
EXC = [370, 470, 525, 570, 590, 600]
EMI = [580, 640, 680, 730]
EPS = 1e-9
SEED = 20260910


def read_csv_auto(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "gb18030", "utf-16"):
        try:
            return pd.read_csv(path, header=None, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            pass
    raise ValueError(f"Cannot decode {path}")


def station_eem_files(eem_dir: Path) -> dict[str, Path]:
    """Select one station-labelled EEM file; prefer LONG and exact metadata."""
    out: dict[str, Path] = {}
    for path in sorted(eem_dir.glob("*.csv")):
        if "站点文件" not in path.name:
            continue
        m = re.search(r"站点文件__([A-Z]\d{2})__", path.name)
        if not m:
            continue
        station = m.group(1)
        try:
            d = read_csv_auto(path)
            declared = str(d.iloc[2, 1]) if len(d) > 2 else ""
        except Exception:
            continue
        if station not in declared.upper():
            continue
        score = (2 if "LONG" in path.name.upper() else 1, -len(path.name))
        old = out.get(station)
        if old is None or score > (2 if "LONG" in old.name.upper() else 1, -len(old.name)):
            out[station] = path
    return out


def parse_eem(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = read_csv_auto(path)
    ex = pd.to_numeric(d.iloc[22, 1:], errors="coerce").to_numpy(dtype=float)
    em = pd.to_numeric(d.iloc[23:, 0], errors="coerce").to_numpy(dtype=float)
    mask_ex = np.isfinite(ex)
    mask_em = np.isfinite(em)
    ex = ex[mask_ex]
    em = em[mask_em]
    values = d.iloc[23:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    values = values[np.ix_(mask_em, mask_ex)]
    return ex, em, values


def extract_features(path: Path) -> tuple[dict[str, float], dict[str, object]]:
    ex, em, a = parse_eem(path)
    lookup = {(int(e), int(m)): float(a[np.where(em == m)[0][0], np.where(ex == e)[0][0]])
              for e in EXC for m in EMI if e in ex and m in em and m > e + 20}
    names = {}
    vals = {}
    for (e, m), value in lookup.items():
        name = f"I_ex{e}_em{m}"
        names[name] = value
        vals[name] = value
    for m in EMI[:3]:
        for e in [470]:
            if (e, m) in lookup and (e, 680) in lookup:
                vals[f"ratio470_log580_680" if m == 580 else f"ratio470_log{m}_680"] = np.log((lookup[(e, m)] + EPS) / (lookup[(e, 680)] + EPS))
    x = np.array(list(vals.values()), dtype=float)
    bad = (~np.isfinite(x)) | (x < 0) | (x > 1e7)
    for k, b in zip(list(vals), bad):
        if b:
            vals[k] = np.nan
    return vals, {"ex_min": float(ex.min()), "ex_max": float(ex.max()), "em_min": float(em.min()), "em_max": float(em.max()), "n_grid": int(a.size), "n_candidate": len(vals)}


def load_field(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path], dict[str, dict]]:
    xlsx = next(data_root.glob("*.xlsx"))
    main = pd.read_excel(xlsx, sheet_name=0, header=2)
    nutrients = pd.read_excel(xlsx, sheet_name=1, header=2)
    def col(pattern: str) -> str:
        return next(c for c in main.columns if pattern.lower() in str(c).lower())
    meta = pd.DataFrame({
        "station_id": main[col("站点")].astype(str).str.strip(),
        "temperature": pd.to_numeric(main[col("WaterTemp")], errors="coerce"),
        "pH": pd.to_numeric(main[col("pH")], errors="coerce"),
        "salinity": pd.to_numeric(main[col("Salinity")], errors="coerce"),
        "DO": pd.to_numeric(main[col("DO_mgL")], errors="coerce"),
        "ORP": pd.to_numeric(main[col("ORP")], errors="coerce"),
        "turbidity": pd.to_numeric(main[col("Turbidity")], errors="coerce"),
        "chlorophyll": pd.to_numeric(main[col("Chlorophyll")], errors="coerce"),
    })
    ncols = {"NH3": "NH3", "NO2": "NO2", "NO3": "NO3", "PO4": "PO4", "SiO3": "SiO3"}
    for target, pattern in ncols.items():
        source = next((c for c in nutrients.columns if pattern.lower() in str(c).lower()), None)
        vals = nutrients.set_index(nutrients.columns[0])[source] if source else pd.Series(dtype=float)
        meta[target] = meta.station_id.map(pd.to_numeric(vals, errors="coerce"))
    files = station_eem_files(data_root / "EEM整理")
    features = []
    info = {}
    for sid, path in files.items():
        try:
            v, i = extract_features(path)
            v["station_id"] = sid
            features.append(v); info[sid] = i
        except Exception as exc:
            info[sid] = {"error": str(exc)}
    fdf = pd.DataFrame(features)
    return meta, fdf, files, info


def correct_linear(xtr: np.ndarray, xte: np.ndarray, ztr: np.ndarray, zte: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    med = np.nanmedian(ztr)
    ztr = np.where(np.isfinite(ztr), ztr, med); zte = np.where(np.isfinite(zte), zte, med)
    model = LinearRegression().fit(ztr.reshape(-1, 1), xtr)
    z0 = float(np.median(ztr))
    ref = model.predict([[z0]])[0]
    return xtr - (model.predict(ztr.reshape(-1, 1)) - ref), xte - (model.predict(zte.reshape(-1, 1)) - ref), z0, model.coef_.ravel()


def prep_model(xtr: np.ndarray, xte: np.ndarray, ytr: np.ndarray, fixed_alpha: float | None = None):
    med = np.nanmedian(xtr, axis=0)
    xtr = np.where(np.isfinite(xtr), xtr, med); xte = np.where(np.isfinite(xte), xte, med)
    scaler = StandardScaler().fit(xtr)
    if fixed_alpha is None:
        model = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(scaler.transform(xtr), ytr)
    else:
        model = Ridge(alpha=fixed_alpha).fit(scaler.transform(xtr), ytr)
    return model.predict(scaler.transform(xte)), float(getattr(model, "alpha_", fixed_alpha or 1.0))


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"MAE": float(mean_absolute_error(y, pred)), "RMSE": float(mean_squared_error(y, pred) ** 0.5), "R2": float(r2_score(y, pred)), "Spearman_rho": float(spearmanr(y, pred).statistic)}


def loso(df: pd.DataFrame, feature_cols: list[str], env_cols: list[str] | None, mode: str, fixed_alpha: float | None = None) -> tuple[pd.DataFrame, dict[str, float], list[float]]:
    ids = df.station_id.to_numpy(); y = df.chlorophyll.to_numpy(float); x = df[feature_cols].to_numpy(float)
    z = df[env_cols].to_numpy(float) if env_cols else None
    rows = []; alphas = []
    for i, sid in enumerate(ids):
        train = np.arange(len(df)) != i
        xt, xe = x[train], x[[i]]
        # Imputation is fold-local and must precede the drift regression.
        xmed = np.nanmedian(xt, axis=0)
        xt = np.where(np.isfinite(xt), xt, xmed); xe = np.where(np.isfinite(xe), xe, xmed)
        if mode == "corrected":
            for j in range(z.shape[1]):
                xt, xe, _, _ = correct_linear(xt, xe, z[train, j], z[[i], j])
        elif mode == "concat":
            xt = np.column_stack([xt, z[train]]); xe = np.column_stack([xe, z[[i]]])
        elif mode == "environment_only":
            xt, xe = z[train], z[[i]]
        pred, alpha = prep_model(xt, xe, y[train], fixed_alpha)
        alphas.append(alpha); rows.append({"station_id": sid, "y_true": y[i], "y_pred": float(pred[0]), "error": float(y[i] - pred[0]), "mode": mode, "env": "+".join(env_cols or [])})
    out = pd.DataFrame(rows); return out, metrics(out.y_true.to_numpy(), out.y_pred.to_numpy()), alphas


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    raw = [c for c in df.columns if c.startswith("I_ex")]
    ratios = [c for c in df.columns if c.startswith("ratio")]
    # Total-intensity normalized shape features are fold-normalized downstream.
    shape = [f"shape_{c}" for c in raw]
    for c, s in zip(raw, shape): df[s] = df[c] / np.sqrt(np.nansum(np.square(df[raw].to_numpy(float)), axis=1))
    return {"raw_intensity": raw, "normalized_shape": shape, "mechanistic_ratio": ratios}


def save_plot(path: Path, dpi: int = 320):
    plt.tight_layout(); plt.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close()


def run(cfg: dict) -> dict:
    rng = np.random.default_rng(int(cfg.get("random_seed", SEED)))
    repo = Path(__file__).resolve().parents[1]
    data_root = Path(cfg["data_root"])
    if not data_root.is_absolute(): data_root = (repo / data_root).resolve()
    out = repo / "results"; figs = repo / "figures"; manifest_dir = repo / "data_manifest"; reports = repo / "reports"
    for p in (out, figs, manifest_dir, reports): p.mkdir(parents=True, exist_ok=True)
    meta, feat, files, info = load_field(data_root)
    fs = feature_sets(feat)
    merged = meta.merge(feat, on="station_id", how="left")
    manifest = meta.copy(); manifest["has_valid_eem"] = manifest.station_id.isin(files)
    manifest["eem_filename"] = manifest.station_id.map({k: v.name for k, v in files.items()})
    manifest["included_main_analysis"] = manifest.has_valid_eem & manifest.chlorophyll.notna()
    manifest["exclusion_reason"] = np.where(manifest.included_main_analysis, "", np.where(~manifest.has_valid_eem, "no station-labelled valid EEM", "missing chlorophyll"))
    manifest["missing_fields"] = manifest[["chlorophyll"] + ENV].isna().apply(lambda r: ";".join(r.index[r].tolist()), axis=1)
    manifest.to_csv(manifest_dir / "field_station_manifest.csv", index=False, encoding="utf-8-sig")
    analysis = merged[merged.included_main_analysis if "included_main_analysis" in merged else merged.station_id.isin(files)].copy()
    analysis = merged[merged.station_id.isin(files) & merged.chlorophyll.notna()].copy()
    raw = fs["raw_intensity"]; analysis = analysis.dropna(subset=raw).reset_index(drop=True)
    feature_df = analysis[["station_id"] + raw + fs["normalized_shape"] + fs["mechanistic_ratio"]]
    feature_df.to_csv(out / "derived_feature_summary.csv", index=False)
    # Full primary screening: raw features, all requested env variables.
    all_results = []; pred_tables = []
    for family, cols in fs.items():
        for env in ENV:
            sub = analysis.dropna(subset=[env]).reset_index(drop=True)
            for mode, label, ec in [("baseline", "fluorescence_only", None), ("concat", "fluorescence_plus_env", [env]), ("corrected", "corrected_fluorescence", [env]), ("environment_only", "environment_only", [env])]:
                pred, met, alphas = loso(sub, cols, ec, mode)
                pred["feature_family"] = family; pred_tables.append(pred)
                all_results.append({"feature_family": family, "environment": env, "model": label, "n": len(sub), **met, "median_alpha": float(np.median(alphas) if alphas else np.nan)})
    # Prespecified low-dimensional multivariable diagnostics for the primary raw family.
    for envs in [["salinity", "pH"], ["salinity", "pH", "temperature"], ENV]:
        sub = analysis.dropna(subset=envs).reset_index(drop=True)
        for mode, label in [("concat", "fluorescence_plus_env"), ("corrected", "corrected_fluorescence"), ("environment_only", "environment_only")]:
            pred, met, alphas = loso(sub, raw, envs, mode)
            pred["feature_family"] = "raw_intensity"; pred_tables.append(pred)
            all_results.append({"feature_family": "raw_intensity", "environment": "+".join(envs), "model": label, "n": len(sub), **met, "median_alpha": float(np.median(alphas) if alphas else np.nan)})
    baseline = pd.DataFrame(all_results); baseline.to_csv(out / "baseline_model_comparison.csv", index=False)
    baseline[baseline.model == "fluorescence_plus_env"].to_csv(out / "environment_variable_screening.csv", index=False)
    baseline[baseline.model.isin(["fluorescence_only", "corrected_fluorescence"])].to_csv(out / "environment_correction_comparison.csv", index=False)
    preds = pd.concat(pred_tables, ignore_index=True); preds.to_csv(out / "loso_predictions.csv", index=False)
    primary = analysis.dropna(subset=["salinity"]).reset_index(drop=True)
    raw_pred, raw_met, raw_a = loso(primary, raw, None, "baseline")
    corr_pred, corr_met, corr_a = loso(primary, raw, ["salinity"], "corrected")
    raw_pred = raw_pred.rename(columns={"y_pred": "pred_raw", "error": "error_raw"})
    corr_pred = corr_pred.rename(columns={"y_pred": "pred_corr", "error": "error_corr"})
    paired = raw_pred[["station_id", "y_true", "pred_raw", "error_raw"]].merge(corr_pred[["station_id", "pred_corr", "error_corr"]], on="station_id")
    paired.to_csv(out / "primary_loso_predictions.csv", index=False)
    # Permutation uses the exact primary raw-feature LOSO correction workflow.
    real_delta = raw_met["MAE"] - corr_met["MAE"]; perm = []
    z0 = primary.salinity.to_numpy(float).copy()
    for k in range(int(cfg.get("permutation_repeats", 1000))):
        p = primary.copy(); p["salinity"] = rng.permutation(z0)
        _, cm, _ = loso(p, raw, ["salinity"], "corrected")
        perm.append(real_delta if False else raw_met["MAE"] - cm["MAE"])
    perm_df = pd.DataFrame({"permutation": np.arange(len(perm)), "delta_mae": perm}); perm_df.to_csv(out / "permutation_salinity.csv", index=False)
    p_perm = (1 + int(np.sum(np.array(perm) >= real_delta))) / (len(perm) + 1)
    # Channel sensitivities from full-data descriptive fits, explicitly labelled non-LOSO mechanism diagnostics.
    sens = []
    for c in raw:
        fit = LinearRegression().fit(primary[["salinity"]], primary[c])
        sens.append({"feature": c, "environment": "salinity", "coefficient": float(fit.coef_[0]), "standardized_coefficient": float(fit.coef_[0] * primary.salinity.std() / primary[c].std()), "r2_descriptive": float(fit.score(primary[["salinity"]], primary[c]))})
    sens_df = pd.DataFrame(sens); sens_df.to_csv(out / "channel_salinity_sensitivity.csv", index=False); sens_df.to_csv(out / "channel_environment_sensitivity.csv", index=False)
    # Family-specific primary gains.
    fam = baseline[(baseline.environment == "salinity") & baseline.model.isin(["fluorescence_only", "corrected_fluorescence"])].copy(); fam.to_csv(out / "feature_family_comparison.csv", index=False)
    # Extreme sensitivity: predeclared top/bottom 1-3 by each audited variable.
    ext_rows = []
    for var in ["chlorophyll", "salinity", "turbidity", "pH"]:
        for tail in ["low", "high"]:
            order = primary[var].sort_values().index.tolist() if tail == "low" else primary[var].sort_values(ascending=False).index.tolist()
            for n in [1, 2, 3]:
                drop = set(primary.iloc[order[:n]].station_id); q = primary[~primary.station_id.isin(drop)].reset_index(drop=True)
                if len(q) < 8: continue
                _, bm, _ = loso(q, raw, None, "baseline"); _, cm, _ = loso(q, raw, ["salinity"], "corrected")
                ext_rows.append({"extreme_variable": var, "tail": tail, "removed_n": n, "removed_station_ids": ";".join(sorted(drop)), "n_remaining": len(q), "mae_raw": bm["MAE"], "mae_corrected": cm["MAE"], "delta_mae": bm["MAE"] - cm["MAE"], "relative_improvement": (bm["MAE"] - cm["MAE"]) / bm["MAE"]})
    ext = pd.DataFrame(ext_rows); ext.to_csv(out / "extreme_station_sensitivity.csv", index=False)
    # Figures.
    plt.figure(figsize=(5.4, 4.4)); plt.scatter(paired.y_true, paired.pred_raw, label="raw", alpha=.8); plt.scatter(paired.y_true, paired.pred_corr, label="corrected", alpha=.8); lo, hi = paired.y_true.min(), paired.y_true.max(); plt.plot([lo, hi], [lo, hi], "k--", lw=1); plt.xlabel("Observed chlorophyll"); plt.ylabel("LOSO predicted chlorophyll"); plt.legend(); save_plot(figs / "baseline_vs_corrected_scatter.png")
    gain = baseline[(baseline.feature_family == "raw_intensity") & (baseline.model.isin(["fluorescence_plus_env", "corrected_fluorescence"]))].copy(); plt.figure(figsize=(8, 4)); pv = gain.pivot(index="environment", columns="model", values="MAE"); pv.plot.bar(ax=plt.gca()); plt.ylabel("LOSO MAE"); plt.xlabel("Environment variable"); save_plot(figs / "environment_gain_bar.png")
    plt.figure(figsize=(5.5, 4)); plt.hist(perm_df.delta_mae, bins=30, color="#9aa7b2"); plt.axvline(real_delta, color="#b2182b", lw=2, label=f"observed ΔMAE={real_delta:.3g}"); plt.xlabel("ΔMAE = MAE raw − MAE corrected"); plt.ylabel("Permutation count"); plt.legend(); save_plot(figs / "permutation_salinity_delta_mae.png")
    plt.figure(figsize=(8, 4)); sens_df.sort_values("standardized_coefficient").plot.barh(x="feature", y="standardized_coefficient", ax=plt.gca(), legend=False); plt.xlabel("Standardized salinity coefficient"); plt.ylabel("Channel"); save_plot(figs / "channel_salinity_sensitivity.png")
    plt.figure(figsize=(8, 4)); plt.scatter(paired.station_id, np.abs(paired.error_raw), label="raw"); plt.scatter(paired.station_id, np.abs(paired.error_corr), label="corrected"); plt.xticks(rotation=90); plt.ylabel("Absolute LOSO error"); plt.xlabel("Station"); plt.legend(); save_plot(figs / "station_error_before_after.png")
    fg = fam.pivot(index="feature_family", columns="model", values="MAE"); plt.figure(figsize=(7, 4)); fg.plot.bar(ax=plt.gca()); plt.ylabel("LOSO MAE"); plt.xlabel("Feature family"); save_plot(figs / "feature_family_gain.png")
    # Decision rule is frozen before narrative: GREEN requires all listed gates.
    rel = real_delta / raw_met["MAE"] if raw_met["MAE"] else np.nan
    sensitivity_positive = bool((ext.delta_mae > 0).mean() >= 0.75) if len(ext) else False
    env_only = baseline[(baseline.feature_family == "raw_intensity") & (baseline.environment == "salinity") & (baseline.model == "environment_only")].iloc[0]
    corr_row = baseline[(baseline.feature_family == "raw_intensity") & (baseline.environment == "salinity") & (baseline.model == "corrected_fluorescence")].iloc[0]
    proxy_gap = float(env_only.MAE - corr_row.MAE)
    if rel >= .15 and corr_row.R2 >= baseline[(baseline.feature_family == "raw_intensity") & (baseline.environment == "salinity") & (baseline.model == "fluorescence_only")].iloc[0].R2 and p_perm < .05 and sensitivity_positive and proxy_gap > 0:
        decision = "GREEN"
    elif rel > 0 or p_perm < .1 or sensitivity_positive:
        decision = "YELLOW"
    else: decision = "RED"
    result = {"decision": decision, "n_stations": int(len(primary)), "raw_metrics": raw_met, "corrected_salinity_metrics": corr_met, "relative_mae_improvement": float(rel), "permutation_p": float(p_perm), "permutation_repeats": len(perm), "environment_only_salinity_mae": float(env_only.MAE), "proxy_gap_environment_only_minus_corrected": proxy_gap, "sensitivity_positive_fraction": float((ext.delta_mae > 0).mean()) if len(ext) else None, "selected_feature_family": "raw_intensity", "data_root_record": str(data_root.name)}
    (out / "P01_final_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report = repo / "reports" / "P01_environment_correction_audit.md"
    report.write_text(make_report(result, manifest, baseline, sens_df, ext), encoding="utf-8")
    (repo / "P01_DECISION.md").write_text(make_decision(result), encoding="utf-8")
    return result


def make_report(r: dict, manifest: pd.DataFrame, baseline: pd.DataFrame, sens: pd.DataFrame, ext: pd.DataFrame) -> str:
    return f'''# P01 Environment-Correction Audit

## 1. Executive Summary

P01 decision: **{r["decision"]}**. The analysis used {r["n_stations"]} F97 Pro field stations with valid station-labelled EEM and chlorophyll. Raw fluorescence MAE was {r["raw_metrics"]["MAE"]:.4g}; salinity-corrected fluorescence MAE was {r["corrected_salinity_metrics"]["MAE"]:.4g}, a relative change of {r["relative_mae_improvement"]:.2%}. The 1000-repeat salinity permutation p-value was {r["permutation_p"]:.4g}.

## 2. Research Question

Whether synchronous environmental state can explain and correct drift in limited-channel fluorescence features. This is a measurement-compensation audit, not a field algae-classification study.

## 3. Data Inventory

The source workbook contained {len(manifest)} field station rows. The frozen manifest is `data_manifest/field_station_manifest.csv`. Raw Excel/EEM files remain outside the repository. F-4700 laboratory files were not merged into this analysis.

## 4. Evidence Boundary

Chlorophyll is an independent validation endpoint. It is not an algae-composition truth label. No result here proves algae species or algae-group classification accuracy.

## 5. Feature Construction

Candidate channels used Ex 370, 470, 525, 570, 590, 600 nm and Em 580, 640, 680, 730 nm, with Em > Ex + 20 nm, finite non-negative values, and a fixed saturation ceiling. Families were raw intensities, total-intensity-normalized shape features, and mechanistic log-ratio features. The same feature rules were applied before LOSO; model scaling and imputation were fitted inside each fold.

## 6. Validation Protocol

Every station was held out once. The training fold alone determined missing-value medians, standardization, RidgeCV alpha, linear environment-drift coefficients, and training-median reference state. The test station was used only for prediction scoring.

## 7. Leakage Audit

- Station-level holdout is explicit and no replicate rows were treated as independent stations.
- The explicit correction fits one linear drift function per channel in the training fold only.
- Reference state `z0` is the training-fold median only.
- The chlorophyll endpoint is never included as a feature or environment variable.
- Environment-only and concatenation models are reported as diagnostics, not patent implementations.
- Channel sensitivity is a descriptive mechanism diagnostic and is not used for feature selection or decision tuning.

## 8. Baselines

`results/baseline_model_comparison.csv` retains fluorescence-only, fluorescence-plus-environment, explicit correction, and environment-only models for all requested environment variables and feature families.

## 9. Environment Variable Screening

The screening table is `results/environment_variable_screening.csv`. Salinity is evaluated as the prespecified primary candidate, not selected after inspecting the endpoint.

## 10. Explicit Environment Correction

For each channel, the fitted relation was `F_j = g_j(z) + residual`, with `F_corr = F - (g_j(z) - g_j(z0))`. Both single-variable and multi-variable correction code paths are represented in the reproducible implementation; the primary gate reports salinity correction and the screening reports the six prespecified single variables.

## 11. Permutation Test

Station salinity labels were permuted {r["permutation_repeats"]} times while EEM and chlorophyll pairing remained fixed. Observed ΔMAE was compared with the complete LOSO correction null distribution. The result is p = {r["permutation_p"]:.4g}; it is not treated as significant unless below 0.05.

## 12. Extreme Station Sensitivity

`results/extreme_station_sensitivity.csv` reports removal of the low/high 1–3 stations for chlorophyll, salinity, turbidity, and pH, followed by a new LOSO run. The positive-improvement fraction was {r["sensitivity_positive_fraction"]}.

## 13. Channel-Level Mechanism Analysis

`results/channel_salinity_sensitivity.csv` reports per-channel salinity slopes and standardized coefficients. This is a descriptive diagnostic of possible amplitude, channel-specific, spectral-shape, or ratio drift. It does not establish causality.

## 14. Negative Results

The environment-only salinity MAE was {r["environment_only_salinity_mae"]:.4g}. The difference between environment-only and corrected fluorescence MAE was {r["proxy_gap_environment_only_minus_corrected"]:.4g}; a small or negative gap would weaken the compensation interpretation.

## 15. Patent-Relevance Assessment

Only a simple, fold-fitted correction operator with stable independent-endpoint improvement can support a P02 implementation study. Feature concatenation alone is not evidence of the claimed compensation mechanism.

## 16. Limitations

The sample is small, field station coverage is limited, EEM instrument response is not cross-calibrated to F-4700, and no synchronous algae-composition truth is available. The station manifest and all exclusions must be preserved with future data revisions.

## 17. P01 Decision

**{r["decision"]}** under the frozen gates in the repository README and config. This status is a data-audit result, not a patentability opinion.

## 18. Recommended P02

If GREEN, freeze the operator, channels, reference-state definition, and implementation examples. If YELLOW, collect independent stations and preregister a confirmatory run before drafting a main claim. If RED, stop the environment-compensation route and retain the negative evidence.

## Claims We CAN Make

- The specified field EEM and environment records can be audited with station-level holdout.
- The data support or do not support a measurable association between environment state and the fluorescence-to-chlorophyll mapping, as quantified above.
- A training-fold-only explicit correction operator was implemented and tested.

## Claims We CANNOT Make

- We cannot claim that environment compensation improves algae species or algae-group classification accuracy.
- We cannot claim field algae-composition truth, causal ecological effects, or quantitative F97/F-4700 cross-instrument calibration.
- We cannot treat concatenating environment variables with fluorescence as the final patent solution.
'''


def make_decision(r: dict) -> str:
    return f'''# P01 Decision

## Status

`{r["decision"]}`

## Evidence

- Environment compensation effect: {'observed' if r["relative_mae_improvement"] > 0 else 'not observed'}; salinity-corrected raw-feature ΔMAE = `{r["raw_metrics"]["MAE"] - r["corrected_salinity_metrics"]["MAE"]:.6g}`.
- Primary variable: `salinity` was prespecified and evaluated first. Do not call it the winner without comparing `results/environment_variable_screening.csv`.
- Effect scale: relative MAE improvement `{r["relative_mae_improvement"]:.2%}`.
- Permutation stability: p = `{r["permutation_p"]:.6g}` from `{r["permutation_repeats"]}` permutations.
- Extreme-station sensitivity: positive-improvement fraction `{r["sensitivity_positive_fraction"]}`.
- Environment-only salinity MAE: `{r["environment_only_salinity_mae"]:.6g}`.

## Patent Route

This P01 result does not establish algae classification improvement. {'Proceed to P02 to freeze the correction operator and implementation examples.' if r["decision"] == 'GREEN' else 'Do not place the correction route in a main claim yet; collect independent validation or stop according to the limitations in the technical report.'}
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/p01.yaml")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config); cfg_path = cfg_path if cfg_path.is_absolute() else repo / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    result = run(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
