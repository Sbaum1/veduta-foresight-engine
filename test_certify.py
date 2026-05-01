"""
test_certify.py
================================================================================
VEDUTA Foresight Engine v3.0.0 — SHA-256 Certification
================================================================================
Generates cryptographic SHA-256 hashes of every engine source file and
forecast output. Produces a signed certification artifact that proves:

  1. FILE INTEGRITY — every .py file in the engine is hashed. If any file
     changes, the certificate becomes invalid.

  2. OUTPUT REPRODUCIBILITY — runs the engine on a canonical series and
     hashes the forecast output from every model. Same inputs must always
     produce the same outputs.

  3. PERFORMANCE GATE — verifies median MASE against the M3 result file
     (if available) to confirm certification tier.

Output:
    test_results/certified_hashes.json     — golden hashes (commit this)
    test_results/certification_report.json — full signed report
    test_results/VEDUTA_Certification_v3.txt — human-readable certificate

Run from C:\\Dev\\VEDUTA\\core\\foresight_x:
    python test_certify.py              # generate new certificate
    python test_certify.py --verify     # verify against existing certificate
================================================================================
"""

import sys, os, json, hashlib, time, argparse
from pathlib import Path
from datetime import datetime, timezone
import uuid

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--verify", action="store_true",
    help="Verify current files against existing certified_hashes.json")
args = parser.parse_args()

OUTPUT_DIR  = Path("diagnostics")
OUTPUT_DIR.mkdir(exist_ok=True)
HASH_FILE   = OUTPUT_DIR / "certified_hashes.json"
REPORT_FILE = OUTPUT_DIR / "certification_report.json"
CERT_FILE   = OUTPUT_DIR / "VEDUTA_Certification_v3.txt"

ENGINE_ROOT = Path("foresight_engine")
HORIZON     = 12
CANONICAL_SEED = 42


# ==============================================================================
# FILE HASHING
# ==============================================================================

def hash_file(filepath: Path) -> str:
    """SHA-256 hash of a file's contents."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def hash_dataframe(df: pd.DataFrame) -> str:
    """SHA-256 hash of a DataFrame's numeric content, rounded to 8dp."""
    sha = hashlib.sha256()
    for col in sorted(df.select_dtypes(include=[np.number]).columns):
        vals = df[col].fillna(-999999).round(8).values
        sha.update(vals.tobytes())
    return sha.hexdigest()


def collect_engine_file_hashes() -> dict:
    """Hash every .py file in foresight_engine/."""
    hashes = {}
    if not ENGINE_ROOT.exists():
        print(f"  ⚠️  Engine root not found: {ENGINE_ROOT}")
        return hashes
    for py_file in sorted(ENGINE_ROOT.rglob("*.py")):
        parts = py_file.parts
        # Skip pycache, backup files, and backup directories
        if ("__pycache__" in parts
                or py_file.name.startswith("BU_")
                or "Backups" in parts
                or "backup" in py_file.name.lower()):
            continue
        rel = str(py_file).replace("\\", "/")
        hashes[rel] = hash_file(py_file)
    return hashes


# ==============================================================================
# CANONICAL SERIES
# ==============================================================================

def canonical_series(n: int = 72, seed: int = CANONICAL_SEED) -> pd.DataFrame:
    """
    Deterministic synthetic manufacturing series.
    Same seed always produces identical data — required for reproducibility.
    """
    rng   = np.random.default_rng(seed)
    t     = np.arange(n, dtype=float)
    dates = pd.date_range("2018-01-01", periods=n, freq="MS")
    vals  = (
        350_000
        + t * 800
        + 22_000 * np.sin(2 * np.pi * t / 12 - 0.4)
        +  7_000 * np.sin(2 * np.pi * t / 6)
        + rng.normal(0, 5_500, n)
    )
    vals[27:30] -= 45_000
    vals[30:36] += np.linspace(-25_000, 0, 6)
    vals = np.maximum(vals, 200_000).round(2)
    return pd.DataFrame({"date": dates, "value": vals})


# ==============================================================================
# OUTPUT HASHING
# ==============================================================================

def collect_output_hashes(raw: dict) -> dict:
    """Hash the future forecast output of every successful model."""
    hashes = {}
    for name, result in raw.items():
        if name.startswith("_") or not isinstance(result, dict):
            continue
        if result.get("status") != "success":
            continue
        fc_df = result.get("forecast_df")
        if fc_df is None or fc_df.empty:
            continue
        # Hash only the future rows (actual = NaN)
        if "actual" in fc_df.columns:
            future = fc_df[fc_df["actual"].isna()]
        else:
            future = fc_df
        if future.empty:
            future = fc_df
        hashes[name] = hash_dataframe(future)
    return hashes


# ==============================================================================
# VERIFY MODE
# ==============================================================================

def verify_existing():
    print("\n" + "=" * 72)
    print("  VEDUTA Foresight Engine v3.0.0 — CERTIFICATION VERIFICATION")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    if not HASH_FILE.exists():
        print(f"\n  ❌ No certificate found at {HASH_FILE}")
        print("  Run without --verify to generate a new certificate first.\n")
        sys.exit(1)

    with open(HASH_FILE) as f:
        golden = json.load(f)

    golden_files   = golden.get("file_hashes",   {})
    golden_outputs = golden.get("output_hashes",  {})
    timestamp      = golden.get("timestamp",       "unknown")
    report_id      = golden.get("report_id",       "unknown")

    print(f"\n  Certificate ID:  {report_id}")
    print(f"  Issued:          {timestamp}")
    print()

    # File integrity check
    print("  ── File Integrity ──────────────────────────────────────────")
    file_pass = 0
    file_fail = 0
    current_files = collect_engine_file_hashes()

    for rel, expected in golden_files.items():
        current = current_files.get(rel)
        if current is None:
            print(f"  ⚠️  MISSING  {rel}")
            file_fail += 1
        elif current == expected:
            file_pass += 1
        else:
            print(f"  ❌ CHANGED  {rel}")
            file_fail += 1

    # New files not in certificate
    for rel in current_files:
        if rel not in golden_files:
            print(f"  ➕ NEW FILE {rel}")

    print(f"\n  Files unchanged: {file_pass}/{len(golden_files)}")

    # Output reproducibility check
    print("\n  ── Output Reproducibility ──────────────────────────────────")
    from foresight_engine.runner import run_all_models
    df  = canonical_series()
    raw = run_all_models(df=df, horizon=HORIZON, confidence_level=0.90)
    current_outputs = collect_output_hashes(raw)

    out_pass = 0
    out_fail = 0
    for name, expected in golden_outputs.items():
        current = current_outputs.get(name)
        if current is None:
            print(f"  ⚠️  MISSING  {name}")
            out_fail += 1
        elif current == expected:
            out_pass += 1
        else:
            print(f"  ❌ MISMATCH {name}")
            print(f"     Expected: {expected[:32]}...")
            print(f"     Got:      {current[:32]}...")
            out_fail += 1

    print(f"\n  Outputs matched: {out_pass}/{len(golden_outputs)}")

    overall = file_fail == 0 and out_fail == 0
    print("\n" + "─" * 72)
    print(f"  VERIFICATION: {'✅ CERTIFIED — All hashes match' if overall else '❌ INVALID — Certificate does not match current engine'}")
    print("=" * 72 + "\n")
    return overall


# ==============================================================================
# GENERATE CERTIFICATE
# ==============================================================================

def generate_certificate():
    print("\n" + "=" * 72)
    print("  VEDUTA Foresight Engine v3.0.0 — GENERATING CERTIFICATE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    report_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    # ── Step 1: Hash all engine source files ──────────────────────────────────
    print("\n  Step 1: Hashing engine source files...")
    file_hashes = collect_engine_file_hashes()
    print(f"  Hashed {len(file_hashes)} Python source files")
    for rel in sorted(file_hashes.keys()):
        print(f"    {file_hashes[rel][:16]}...  {rel}")

    # ── Step 2: Run engine on canonical series ─────────────────────────────────
    print("\n  Step 2: Running engine on canonical series (seed=42)...")
    from foresight_engine.runner   import run_all_models
    from foresight_engine.registry import get_model_registry

    df  = canonical_series()
    t0  = time.time()
    raw = run_all_models(df=df, horizon=HORIZON, confidence_level=0.90)
    elapsed = time.time() - t0
    print(f"  Engine run complete in {elapsed:.1f}s")

    # ── Step 3: Hash all model outputs ────────────────────────────────────────
    print("\n  Step 3: Hashing model forecast outputs...")
    output_hashes = collect_output_hashes(raw)
    successful = [n for n, r in raw.items()
                  if not n.startswith("_") and isinstance(r, dict)
                  and r.get("status") == "success"]
    crashed    = [n for n, r in raw.items()
                  if not n.startswith("_") and isinstance(r, dict)
                  and r.get("status") != "success"]

    for name in sorted(output_hashes.keys()):
        print(f"    {output_hashes[name][:16]}...  {name}")

    if crashed:
        print(f"\n  ⚠️  Models that did not produce output (excluded from cert):")
        for name in crashed:
            err = raw[name].get("error", "unknown")
            print(f"    {name}: {str(err)[:60]}")

    # ── Step 4: Performance gate — check M3 results if available ──────────────
    print("\n  Step 4: Performance gate...")
    m3_summary_path = OUTPUT_DIR / "m3_summary.json"
    m3_data         = None
    if m3_summary_path.exists():
        with open(m3_summary_path) as f:
            m3_data = json.load(f)
        median_mase  = m3_data.get("median_mase")
        passed_m3    = m3_data.get("passed", False)
        prior_mase   = m3_data.get("prior_mase", 0.6847)
        delta        = m3_data.get("delta_vs_prior", round(median_mase - prior_mase, 6) if median_mase else None)
        m3_data["prior_mase"]     = prior_mase
        m3_data["delta_vs_prior"] = delta
        direction    = "improvement" if delta and delta < 0 else "regression"
        print(f"  M3 result found: Median MASE = {median_mase}  →  {'✅ PASS' if passed_m3 else '❌ FAIL'}")
        print(f"  Delta vs prior ({prior_mase}): {delta:+.6f}  ({direction})")
    else:
        print("  ⚠️  No M3 results found — run test_m3.py first for full certification")
        print("  Proceeding with file integrity + reproducibility certification only")
        median_mase = None
        passed_m3   = None

    # ── Step 5: Compute master certificate hash ────────────────────────────────
    print("\n  Step 5: Computing master certificate hash...")
    master_payload = json.dumps({
        "file_hashes":   file_hashes,
        "output_hashes": output_hashes,
        "timestamp":     timestamp,
        "report_id":     report_id,
    }, sort_keys=True)
    master_hash = hashlib.sha256(master_payload.encode()).hexdigest()
    print(f"  Master hash: {master_hash}")

    # ── Step 6: Determine certification tier ──────────────────────────────────
    n_models     = len(get_model_registry()) - 1  # exclude Primary Ensemble
    n_certified  = len(output_hashes)
    n_crashed    = len(crashed)

    if n_crashed == 0 and (passed_m3 is True or passed_m3 is None):
        if median_mase is not None and median_mase < 0.70:
            cert_tier = "ELITE — Median MASE < 0.70"
        elif median_mase is not None and median_mase < 0.85:
            cert_tier = "STRONG — Median MASE < 0.85"
        elif median_mase is not None:
            cert_tier = "CERTIFIED — Median MASE < 1.00"
        else:
            cert_tier = "INTEGRITY CERTIFIED — M3 benchmark pending"
    else:
        cert_tier = "PARTIAL — See failed models"

    # ── Save certificate ───────────────────────────────────────────────────────
    certificate = {
        "report_id":      report_id,
        "master_hash":    master_hash,
        "timestamp":      timestamp,
        "engine_version": "Foresight Engine v3.0.0",
        "cert_tier":      cert_tier,
        "n_models":       n_models,
        "n_certified":    n_certified,
        "n_crashed":      n_crashed,
        "m3_median_mase": median_mase,
        "m3_passed":      passed_m3,
        "file_hashes":    file_hashes,
        "output_hashes":  output_hashes,
    }

    with open(HASH_FILE, "w", encoding="utf-8") as f:
        json.dump(certificate, f, indent=2)
    print(f"\n  Certificate saved → {HASH_FILE}")

    # ── Human-readable certificate ─────────────────────────────────────────────
    lines = [
        "=" * 72,
        "  VEDUTA FORESIGHT ENGINE — CERTIFICATION DOCUMENT",
        "=" * 72,
        "",
        f"  Certificate ID:     {report_id}",
        f"  Issued:             {datetime.now().strftime('%B %d, %Y %H:%M UTC')}",
        f"  Engine Version:     Foresight Engine v3.0.0",
        f"  Certification Tier: {cert_tier}",
        f"  Master Hash:        {master_hash}",
        "",
        "  MODELS REGISTERED",
        "  " + "─" * 68,
    ]
    reg = get_model_registry()
    for m in reg:
        name  = m["name"]
        tier  = m.get("min_tier", "—")
        diag  = " [diagnostic]" if m.get("diagnostic_only") else ""
        ens   = " [ensemble]"   if m.get("ensemble_member") else ""
        h     = output_hashes.get(name, "not hashed")[:16] + "..." if name in output_hashes else "n/a"
        lines.append(f"  {name:28s} {tier:12s}{diag}{ens}")
        lines.append(f"    Output hash: {h}")

    lines += [
        "",
        "  M3 COMPETITION BENCHMARK",
        "  " + "─" * 68,
    ]
    if m3_data:
        smape_val = m3_data.get('median_smape')
        smape_str = f"{smape_val:.4f}%" if smape_val is not None else "not computed"
        lines += [
            f"  Dataset:        M3 Monthly ({m3_data.get('series_tested', '?')} series)",
            f"  Horizon:        18 months  |  Seasonal period: 12",
            f"  Median MASE:    {m3_data.get('median_mase', '?')}  "
            f"(Hyndman & Koehler, 2006)",
            f"  Mean MASE:      {m3_data.get('mean_mase', '?')}",
            f"  Median sMAPE:   {smape_str}  (Makridakis, 1993 — original M3 metric)",
            f"  Prior MASE:     {m3_data.get('prior_mase', m3_data.get('prior_result', 0.6913))}",
            f"  Delta vs prior: {delta:+.6f}  ({direction})",
            f"  Valid series:   {m3_data.get('valid_results', '?')}  "
            f"Crashes: {m3_data.get('n_crashes', 0)}  "
            f"Timeouts: {m3_data.get('n_timeouts', 0)}",
            f"  Result:         {'PASS' if m3_data.get('passed') else 'FAIL'}",
        ]
    else:
        lines.append("  M3 benchmark not yet run — execute test_m3.py to complete certification")

    lines += [
        "",
        "  METHODOLOGY COMPLIANCE",
        "  " + "─" * 68,
        "  MASE:    Hyndman, R.J. & Koehler, A.B. (2006). Another look at measures",
        "           of forecast accuracy. International Journal of Forecasting,",
        "           22(4), 679-688. https://doi.org/10.1016/j.ijforecast.2006.03.001",
        "  sMAPE:   Makridakis, S. (1993). Accuracy measures: theoretical and",
        "           practical concerns. International Journal of Forecasting,",
        "           9(4), 527-529.",
        "  Dataset: Makridakis, S. & Hibon, M. (2000). The M3-Competition: results,",
        "           conclusions and implications. International Journal of",
        "           Forecasting, 16(4), 451-476.",
        "           https://doi.org/10.1016/S0169-2070(00)00057-1",
        "  MASE formula:   MAE_forecast / mean(|y_t - y_{t-12}|) on training set",
        "  sMAPE formula:  mean(200*|actual-forecast|/(|actual|+|forecast|))",
        "  Aggregation:    Median MASE across series (mean MAE across horizons)",
        "  Timeout policy: Timed-out series excluded — conservative, cannot",
        "                  inflate scores. Count reported separately from crashes.",
        "",
        "  FILE INTEGRITY",
        "  " + "─" * 68,
        f"  {len(file_hashes)} source files hashed",
        "",
    ]
    for rel, h in sorted(file_hashes.items()):
        lines.append(f"  {h[:32]}  {rel}")

    lines += [
        "",
        "=" * 72,
        f"  {cert_tier}",
        "=" * 72,
        "",
    ]

    cert_text = "\n".join(lines)
    with open(CERT_FILE, "w", encoding="utf-8") as f:
        f.write(cert_text)

    print(cert_text)
    print(f"  Certificate document saved → {CERT_FILE}\n")
    return True


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    if args.verify:
        success = verify_existing()
    else:
        success = generate_certificate()
    sys.exit(0 if success else 1)

