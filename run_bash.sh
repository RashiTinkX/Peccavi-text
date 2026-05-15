#!/usr/bin/env bash
# =============================================================================
#  run_bash.sh
#  PECCAVI-TEXT  ·  AIISC AI Integrity & Safety Consortium
#
#  Single self-contained entry point.
#  No external .py files needed — all three stages are embedded inline.
#
#  Pipeline stages (all run from this one file):
#    1. process_dataset   – load each CSV separately, normalise schema,
#                           run Auctor → Custos → Scriba per row,
#                           write per-dataset results
#    2. combine_results   – merge all per-dataset outputs into one CSV
#                           and one summary JSON
#    3. benchmark_eval    – run the full PECCAVI evaluation loop
#                           (run_peccavi → benchmarks.py criteria):
#                           effective score ≥ 0.85, AUC-ROC ≥ 0.90,
#                           readability ≥ 4.5/5
#
#  USAGE
#  ─────
#  All datasets in one folder:
#    ./run_bash.sh --data-dir /path/to/folder
#
#  Each dataset from its own folder:
#    ./run_bash.sh \
#      --arxiv-dir     /data/arxiv        \
#      --gutenberg-dir /data/gutenberg    \
#      --arctic-dir    /data/arctic       \
#      --c4-dir        /data/c4           \
#      --output-dir    ./results
#
#  Optional flags:
#    --sample   N       rows per dataset              (default: 25)
#    --theta    FLOAT   watermark strength θ           (default: 2.0)
#    --mode     eval|infer                             (default: infer)
#    --gens     N       benchmark generations          (default: 5)
#    --skip     a,b     comma-separated datasets to skip
#    --no-bench         skip benchmark stage (process + combine only)
#
#  Dataset filenames expected (must match exactly):
#    arxiv_5000.csv
#    gutenberg_chunks_5000.csv
#    arctic_submissions_5000_secondversion.csv
#    c4_multilingual_5000.csv
# =============================================================================
set -euo pipefail

#   Colours  
RED='\033[0;31m';  GREEN='\033[0;32m';  YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m';      RESET='\033[0m'

log()      { echo -e "${CYAN}[$(date +%H:%M:%S)]${RESET} $*"; }
ok()       { echo -e "  ${GREEN}✓${RESET}  $*"; }
warn()     { echo -e "  ${YELLOW}⚠${RESET}  $*"; }
die()      { echo -e "  ${RED}✗  FATAL:${RESET} $*" >&2; exit 1; }
banner()   { echo -e "\n${BOLD}  $*  ${RESET}"; }
skip_msg() { echo -e "  ${YELLOW}↷${RESET}  SKIPPED: $*"; }

#   Defaults  
DATA_DIR=""
ARXIV_DIR="";  GUTENBERG_DIR="";  ARCTIC_DIR="";  C4_DIR=""
OUTPUT_DIR="./peccavi_output"
SAMPLE=25
THETA=2.0
MODE="infer"
GENS=5
SKIP_LIST=""
RUN_BENCH=true

#   Argument parsing  
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)       DATA_DIR="$2";       shift 2 ;;
    --arxiv-dir)      ARXIV_DIR="$2";      shift 2 ;;
    --gutenberg-dir)  GUTENBERG_DIR="$2";  shift 2 ;;
    --arctic-dir)     ARCTIC_DIR="$2";     shift 2 ;;
    --c4-dir)         C4_DIR="$2";         shift 2 ;;
    --output-dir)     OUTPUT_DIR="$2";     shift 2 ;;
    --sample)         SAMPLE="$2";         shift 2 ;;
    --theta)          THETA="$2";          shift 2 ;;
    --mode)           MODE="$2";           shift 2 ;;
    --gens)           GENS="$2";           shift 2 ;;
    --skip)           SKIP_LIST="$2";      shift 2 ;;
    --no-bench)       RUN_BENCH=false;     shift 1 ;;
    --help|-h)
      grep "^#  " "$0" | sed 's/^#  //'
      exit 0 ;;
    *) die "Unknown argument: $1  (run with --help)" ;;
  esac
done

[[ "$MODE" =~ ^(eval|infer)$ ]] || die "--mode must be eval or infer (got: $MODE)"

# Fill per-dataset dirs from --data-dir if not individually set
if [[ -n "$DATA_DIR" ]]; then
  [[ -z "$ARXIV_DIR"     ]] && ARXIV_DIR="$DATA_DIR"
  [[ -z "$GUTENBERG_DIR" ]] && GUTENBERG_DIR="$DATA_DIR"
  [[ -z "$ARCTIC_DIR"    ]] && ARCTIC_DIR="$DATA_DIR"
  [[ -z "$C4_DIR"        ]] && C4_DIR="$DATA_DIR"
fi

#  Banner  
echo ""
echo -e "${BOLD}  ${RESET}"
echo -e "${BOLD}   PECCAVI-TEXT  ·  Unified Pipeline  (run_bash.sh)         ${RESET}"
echo -e "${BOLD}   AIISC AI Integrity & Safety Consortium                   ${RESET}"
echo -e "${BOLD}  ${RESET}"
echo ""
log "mode=${MODE}  sample=${SAMPLE}  theta=${THETA}  gens=${GENS}"
log "output → ${OUTPUT_DIR}"
[[ -n "$SKIP_LIST" ]] && warn "Skipping datasets: ${SKIP_LIST}"
$RUN_BENCH || warn "--no-bench: skipping PECCAVI benchmark stage"

# =============================================================================
#  STAGE 1 — process_dataset  (embedded Python)
#
#  Loads one dataset at a time, normalises its schema to a unified
#  'prompt' field, runs Auctor→Custos→Scriba per row (or stubs when
#  backbone modules are not yet importable), and writes:
#    <output-dir>/<dataset>/raw_sample.csv
#    <output-dir>/<dataset>/results.csv
#    <output-dir>/<dataset>/summary.json
# =============================================================================
banner "Stage 1 · process_dataset"

should_skip() { echo "$SKIP_LIST" | tr ',' '\n' | grep -qx "$1"; }

declare -A DS_DIRS=(
  [arxiv]="$ARXIV_DIR"
  [gutenberg]="$GUTENBERG_DIR"
  [arctic]="$ARCTIC_DIR"
  [c4]="$C4_DIR"
)
declare -A DS_FILES=(
  [arxiv]="arxiv_5000.csv"
  [gutenberg]="gutenberg_chunks_5000.csv"
  [arctic]="arctic_submissions_5000_secondversion.csv"
  [c4]="c4_multilingual_5000.csv"
)

# Validate all paths first
HAS_ANY=0
declare -A DS_PATHS=()
for ds in arxiv gutenberg arctic c4; do
  if should_skip "$ds"; then
    skip_msg "$ds"
    continue
  fi
  dir="${DS_DIRS[$ds]}"
  file="${DS_FILES[$ds]}"
  if [[ -z "$dir" ]]; then
    warn "$ds — no directory set (use --${ds}-dir or --data-dir); skipping"
    continue
  fi
  full="${dir}/${file}"
  if [[ ! -f "$full" ]]; then
    die "File not found: ${full}\n  Expected filename: ${file}"
  fi
  ok "${ds}  →  ${full}  ($(wc -l < "$full") lines)"
  DS_PATHS[$ds]="$full"
  HAS_ANY=1
done
[[ $HAS_ANY -eq 1 ]] || die "No datasets available. Check --data-dir / --skip."

mkdir -p "$OUTPUT_DIR"
PROCESSED=()
FAILED=()

# ── Per-dataset processor (inline Python)   
run_one_dataset() {
  local ds="$1"
  local csv_path="$2"
  local out_dir="${OUTPUT_DIR}/${ds}"
  mkdir -p "$out_dir"

  python3 - <<PYEOF
import sys, json, time, traceback, hashlib, statistics, random
from pathlib import Path
import pandas as pd

DS        = "${ds}"
CSV_PATH  = "${csv_path}"
OUT_DIR   = Path("${out_dir}")
SAMPLE    = ${SAMPLE}
THETA     = ${THETA}
MODE      = "${MODE}"

# ── Schema normalisers ────────────────────────────────────────────────────────
def load_arxiv(path, n):
    df = pd.read_csv(path).dropna(subset=["experiment"]).head(n).copy()
    df["prompt"] = df["experiment"].str.strip()
    df["source"] = "arxiv"
    df = df.rename(columns={"url": "meta_url", "timestamp": "meta_timestamp"})
    return df[["source","prompt","meta_url","meta_timestamp"]].reset_index(drop=True)

def load_gutenberg(path, n):
    df = pd.read_csv(path).dropna(subset=["experiment"]).head(n).copy()
    df["prompt"] = (df["experiment"]
                    .str.replace(r"\r\n|\r", " ", regex=True)
                    .str.strip().str[:600])
    df["source"] = "gutenberg"
    df = df.rename(columns={"book_id":"meta_book_id","chunk_id":"meta_chunk_id"})
    return df[["source","prompt","meta_book_id","meta_chunk_id"]].reset_index(drop=True)

def load_arctic(path, n):
    df = pd.read_csv(path).dropna(subset=["experiment"]).head(n).copy()
    df["prompt"] = df["experiment"].str.strip()
    df["source"] = "arctic"
    df = df.rename(columns={"id":"meta_post_id","author":"meta_author",
                             "subreddit":"meta_subreddit","created_at":"meta_created_at"})
    return df[["source","prompt","meta_post_id","meta_author",
               "meta_subreddit","meta_created_at"]].reset_index(drop=True)

def load_c4(path, n):
    df = pd.read_csv(path).dropna(subset=["experiment"]).head(n).copy()
    df["prompt"] = df["experiment"].str.strip()
    df["source"] = "c4"
    df = df.rename(columns={"language":"meta_language","url":"meta_url",
                             "timestamp":"meta_timestamp"})
    return df[["source","prompt","meta_language","meta_url","meta_timestamp"]].reset_index(drop=True)

LOADERS = {"arxiv":load_arxiv,"gutenberg":load_gutenberg,"arctic":load_arctic,"c4":load_c4}

# ── Stub pipeline (used when backbone/peccavi not importable) ─────────────────
class _StubBackbone:
    class tokenizer:
        vocab_size = 32000
        eos_token_id = 2
        @staticmethod
        def encode(t): return [hash(w) % 32000 for w in t.split()]
        @staticmethod
        def decode(ids, skip_special_tokens=True): return " ".join(str(i) for i in ids[:20])
    def generate(self, prompt, max_new_tokens=200, temperature=0.85):
        return {"text": "[STUB] " + " ".join(prompt.split()[:25]) + " ..."}

class _StubAuctor:
    def __init__(self, backbone, theta=2.0):
        self.theta = theta; self._bb = backbone
    def generate(self, prompt, max_tokens=200):
        return self._bb.generate(prompt, max_new_tokens=max_tokens)["text"] + f" [theta={self.theta}]"

class _StubCustos:
    def __init__(self, backbone): self._bb = backbone
    def watermark_score(self, text):
        import hashlib, random as rng
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % 10000
        rng.seed(seed); return round(rng.uniform(0.45, 0.95), 4)
    def effective_score(self, paraphrases):
        if not paraphrases: return 0.0
        return min(self.watermark_score(p) for p in paraphrases)
    def detect(self, text, threshold=0.52):
        score = self.watermark_score(text)
        return {"score": round(score,4), "is_watermarked": score>=threshold, "threshold": threshold}

class _StubScriba:
    def __init__(self, backbone, n_variants=5):
        self._bb = backbone; self.n = n_variants
    def paraphrase(self, text):
        words = text.split()
        return [" ".join(words[i:]+words[:i])[:200]+f" [para-{i+1}]" for i in range(self.n)]

def _load_peccavi(theta):
    try:
        from backbone.model import LLaMABackbone
        from peccavi.auctor import Auctor
        from peccavi.custos import Custos
        from peccavi.scriba import Scriba
        bb = LLaMABackbone(); a = Auctor(bb); a.theta = theta
        return a, Custos(bb), Scriba(bb, n_variants=5), False
    except ImportError:
        bb = _StubBackbone()
        return _StubAuctor(bb, theta), _StubCustos(bb), _StubScriba(bb), True

# ── Row evaluation ────────────────────────────────────────────────────────────
def evaluate_row(idx, row, auctor, custos, scriba):
    prompt = str(row["prompt"])
    out = {"row_id":idx, "source":row.get("source",""), "prompt_chars":len(prompt),
           "wm_score":None, "threshold":None, "is_watermarked":None,
           "robustness_pct":None, "resilience_score":None,
           "n_variants":None, "n_variants_detected":None,
           "elapsed_s":None, "error":None}
    try:
        t0 = time.perf_counter()
        wm_text   = auctor.generate(prompt, max_tokens=200)
        detection = custos.detect(wm_text) if hasattr(custos,"detect") else \
                    {"score":custos.watermark_score(wm_text),"threshold":0.52,
                     "is_watermarked":custos.watermark_score(wm_text)>=0.52}
        variants  = scriba.paraphrase(wm_text)
        var_scores = [custos.watermark_score(v) for v in variants]
        n_det = sum(1 for s in var_scores if s > detection["threshold"])
        out.update({
            "wm_text_preview":     wm_text[:250],
            "wm_score":            detection["score"],
            "threshold":           detection["threshold"],
            "is_watermarked":      bool(detection["is_watermarked"]),
            "robustness_pct":      round(n_det / max(len(variants),1) * 100, 1),
            "resilience_score":    round(min(var_scores),4) if var_scores else 0.0,
            "n_variants":          len(variants),
            "n_variants_detected": n_det,
            "elapsed_s":           round(time.perf_counter()-t0, 3),
        })
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out

# ── Main ──────────────────────────────────────────────────────────────────────
try:
    print(f"  [{DS}] loading from {CSV_PATH}")
    df = LOADERS[DS](CSV_PATH, SAMPLE)
    df = df[df["prompt"].str.len() > 10].reset_index(drop=True)
    print(f"  [{DS}] {len(df)} rows after filtering")
    df.to_csv(OUT_DIR / "raw_sample.csv", index=False)

    auctor, custos, scriba, stub_mode = _load_peccavi(THETA)
    if stub_mode:
        print(f"  [{DS}] STUB mode (backbone not available)")

    results, total = [], len(df)
    for i, (_, row) in enumerate(df.iterrows(), 1):
        if i == 1 or i % 10 == 0 or i == total:
            print(f"  [{DS}] row {i}/{total}")
        results.append(evaluate_row(i, row, auctor, custos, scriba))

    df_r = pd.DataFrame(results)
    df_r.to_csv(OUT_DIR / "results.csv", index=False)

    ok_rows = df_r[df_r["error"].isna()]
    summary = {
        "dataset": DS, "mode": MODE, "theta": THETA, "stub_mode": stub_mode,
        "total_rows": len(df_r), "rows_ok": len(ok_rows),
        "rows_errored": int(df_r["error"].notna().sum()),
        "metrics": {
            "mean_wm_score":       round(float(ok_rows["wm_score"].mean()),4)       if len(ok_rows) else None,
            "detected_pct":        round(float(ok_rows["is_watermarked"].mean())*100,1) if len(ok_rows) else None,
            "mean_robustness_pct": round(float(ok_rows["robustness_pct"].mean()),1) if len(ok_rows) else None,
            "mean_resilience":     round(float(ok_rows["resilience_score"].mean()),4) if len(ok_rows) else None,
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    m = summary["metrics"]
    print(f"\n  ┌─ {DS.upper()} ({'STUB' if stub_mode else 'LIVE'}) {'─'*35}")
    print(f"  │  rows ok        : {summary['rows_ok']}/{summary['total_rows']}")
    print(f"  │  mean wm score  : {m['mean_wm_score']}")
    print(f"  │  detected       : {m['detected_pct']} %")
    print(f"  │  robustness     : {m['mean_robustness_pct']} %")
    print(f"  │  resilience     : {m['mean_resilience']}")
    print(f"  └{'─'*45}\n")

except Exception:
    (OUT_DIR / "error.log").write_text(traceback.format_exc())
    print(traceback.format_exc(), file=sys.stderr)
    sys.exit(1)
PYEOF
}

# Run each dataset
for ds in arxiv gutenberg arctic c4; do
  [[ -z "${DS_PATHS[$ds]+x}" ]] && continue
  echo ""
  log "▶  Processing: ${BOLD}${ds}${RESET}"
  if run_one_dataset "$ds" "${DS_PATHS[$ds]}"; then
    ok "${ds} complete → ${OUTPUT_DIR}/${ds}/"
    PROCESSED+=("$ds")
  else
    warn "${ds} FAILED — see ${OUTPUT_DIR}/${ds}/error.log"
    FAILED+=("$ds")
  fi
done

# =============================================================================
#  STAGE 2 — combine_results  (embedded Python)
#
#  Reads all per-dataset results.csv + summary.json files,
#  concatenates them, and writes:
#    <output-dir>/combined_results.csv
#    <output-dir>/combined_summary.json
# =============================================================================
banner "Stage 2 · combine_results"

python3 - <<PYEOF
import json, sys
from pathlib import Path
import pandas as pd

OUT_DIR   = Path("${OUTPUT_DIR}")
DATASETS  = "${PROCESSED[*]}".split()   # only successfully processed ones

all_frames    = []
all_summaries = {}

for ds in DATASETS:
    ds_dir  = OUT_DIR / ds
    csv_f   = ds_dir / "results.csv"
    json_f  = ds_dir / "summary.json"
    if not csv_f.exists():
        print(f"  [combine] WARNING: {csv_f} not found — skipping {ds}")
        continue
    df = pd.read_csv(csv_f)
    df.insert(0, "dataset", ds)
    all_frames.append(df)
    if json_f.exists():
        all_summaries[ds] = json.loads(json_f.read_text())
    print(f"  [combine] {ds:<12} → {len(df)} rows")

if not all_frames:
    print("  [combine] ERROR: no results to combine", file=sys.stderr)
    sys.exit(1)

combined = pd.concat(all_frames, ignore_index=True)
combined.to_csv(OUT_DIR / "combined_results.csv", index=False)
print(f"  [combine] combined_results.csv → {len(combined)} total rows")

ok_rows = combined[combined["error"].isna()] if "error" in combined.columns else combined

def smean(col):
    if col in ok_rows.columns and len(ok_rows):
        return round(float(ok_rows[col].mean()), 4)
    return None

combined_summary = {
    "total_rows": len(combined),
    "rows_ok":    len(ok_rows),
    "rows_errored": int(combined["error"].notna().sum()) if "error" in combined.columns else 0,
    "overall_metrics": {
        "mean_wm_score":       smean("wm_score"),
        "detected_pct":        round(float(ok_rows["is_watermarked"].mean())*100,1) if len(ok_rows) else None,
        "mean_robustness_pct": smean("robustness_pct"),
        "mean_resilience":     smean("resilience_score"),
    },
    "per_dataset": all_summaries,
}
(OUT_DIR / "combined_summary.json").write_text(json.dumps(combined_summary, indent=2))
print(f"  [combine] combined_summary.json written")

# Print comparison table
hdr = f"  {'Dataset':<16} {'Rows':>5} {'WM Score':>10} {'Detected%':>11} {'Robust%':>9} {'Resilience':>12}"
sep = "  " + "─" * 67
print(f"\n{sep}\n{hdr}\n{sep}")
for ds, s in all_summaries.items():
    m    = s.get("metrics", {})
    stub = " [STUB]" if s.get("stub_mode") else ""
    print(f"  {(ds+stub):<16} {s.get('rows_ok','?'):>5} "
          f"{str(m.get('mean_wm_score','?')):>10} "
          f"{str(m.get('detected_pct','?'))+'%':>11} "
          f"{str(m.get('mean_robustness_pct','?'))+'%':>9} "
          f"{str(m.get('mean_resilience','?')):>12}")
ov = combined_summary["overall_metrics"]
print(sep)
print(f"  {'OVERALL':<16} {combined_summary['rows_ok']:>5} "
      f"{str(ov.get('mean_wm_score','?')):>10} "
      f"{str(ov.get('detected_pct','?'))+'%':>11} "
      f"{str(ov.get('mean_robustness_pct','?'))+'%':>9} "
      f"{str(ov.get('mean_resilience','?')):>12}")
print(sep + "\n")
PYEOF

ok "combined_results.csv + combined_summary.json written"

# =============================================================================
#  STAGE 3 — benchmark_eval  (embedded Python)
#
#  Implements benchmarks.py run_benchmarks() / _print_summary() logic.
#  Calls run_peccavi() from eval/watermark.py (Praeco → Auctor → Scriba
#  → Custos → Magister) for --gens generations, then evaluates against
#  the three success criteria from eval/benchmarks.py:
#    • effective_score_final  ≥ 0.85   (meets_85pct_retention)
#    • auc_roc                ≥ 0.90   (meets_90pct_auc)
#    • avg_readability        ≥ 4.5/5  (meets_readability_45)
#  Falls back to a deterministic stub when backbone is unavailable.
# =============================================================================
if ! $RUN_BENCH; then
  warn "Skipping benchmark stage (--no-bench)"
else

banner "Stage 3 · benchmark_eval  (benchmarks.py criteria)"

python3 - <<PYEOF
import json, sys, logging, random, hashlib, statistics, math, time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("AIISC.benchmark")

OUT_DIR = Path("${OUTPUT_DIR}")
GENS    = ${GENS}
THETA   = ${THETA}

# ── Try to import the real PECCAVI stack ─────────────────────────────────────
STUB_MODE = False
try:
    from backbone.model import LLaMABackbone
    from peccavi.praeco  import Praeco
    from peccavi.auctor  import Auctor
    from peccavi.scriba  import Scriba
    from peccavi.custos  import Custos
    from peccavi.magister import Magister
    from sklearn.metrics  import roc_auc_score
    import textstat
    logger.info("Real PECCAVI modules loaded ✓")

except ImportError as e:
    logger.warning(f"backbone/peccavi not importable ({e}) — running STUB benchmark")
    STUB_MODE = True

# ── Stub implementations (mirror real APIs exactly) ───────────────────────────
if STUB_MODE:
    PROMPT_BANK = [
        "Summarize recent AI safety research.",
        "Explain the importance of watermarking in AI-generated content.",
        "Describe the risks of large language models in misinformation.",
        "Write a short paragraph about responsible AI deployment.",
        "Discuss how adversarial robustness improves AI reliability.",
        "Explain what content authenticity means in the context of generative AI.",
        "Describe how reinforcement learning is used in AI alignment.",
    ]

    class _BB:
        class tokenizer:
            vocab_size = 32000
            eos_token_id = 2
            @staticmethod
            def encode(t): return [hash(w) % 32000 for w in t.split()]
            @staticmethod
            def decode(ids, **kw): return " ".join(str(i) for i in ids[:30])
        def generate(self, prompt, max_new_tokens=100, temperature=0.85):
            words = prompt.split()[:20]
            return {"text": " ".join(words) + " [stub-gen]"}

    def _wm_score_fn(token_id, seed):
        h = hashlib.sha256(f"{seed}:{token_id}".encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    class Praeco:
        def next_prompt(self):          return random.choice(PROMPT_BANK)
        def batch_prompts(self, n):     return random.choices(PROMPT_BANK, k=n)

    class Auctor:
        def __init__(self, bb, theta=2.0): self.backbone=bb; self.theta=theta
        def generate(self, prompt, max_tokens=100):
            return self.backbone.generate(prompt)["text"] + f" [θ={self.theta:.2f}]"

    class Scriba:
        def __init__(self, bb, n_variants=5): self.backbone=bb; self.n=n_variants
        def paraphrase(self, text):
            words=text.split()
            return [" ".join(words[i:]+words[:i])[:150]+f" [p{i+1}]" for i in range(self.n)]

    class Custos:
        def __init__(self, bb): self.backbone=bb
        def watermark_score(self, text):
            ids = self.backbone.tokenizer.encode(text)
            if not ids: return 0.0
            scores=[]
            for i,tid in enumerate(ids):
                seed=int(hashlib.sha256(("AIISC-KEY"+"".join(str(x) for x in ids[max(0,i-5):i])).encode()).hexdigest()[:8],16)
                scores.append(_wm_score_fn(tid, seed))
            return statistics.mean(scores)
        def effective_score(self, paraphrases):
            if not paraphrases: return 0.0
            return min(self.watermark_score(p) for p in paraphrases)

    class Magister:
        def __init__(self, bb, theta_init=2.0, alpha=0.05, **kw):
            self.backbone=bb; self.theta=theta_init; self.alpha=alpha; self.history=[]
        def update(self, text, s_eff):
            reward = 0.6*s_eff + 0.4*0.5
            baseline = sum(self.history)/len(self.history) if self.history else 0.0
            self.history.append(reward)
            ids = self.backbone.tokenizer.encode(text)
            grad = sum(_wm_score_fn(tid, i)*self.theta for i,tid in enumerate(ids)) / max(len(ids),1)
            self.theta = max(0.1, min(self.theta + self.alpha*grad*(reward-baseline), 10.0))
            return self.theta

    def roc_auc_score(labels, scores):
        pos=[s for l,s in zip(labels,scores) if l==1]
        neg=[s for l,s in zip(labels,scores) if l==0]
        pairs=sum(1 for p in pos for n in neg if p>n)
        return pairs/(len(pos)*len(neg)) if pos and neg else 0.5

    class textstat:
        @staticmethod
        def flesch_reading_ease(t):
            words=t.split(); sents=max(1,t.count(".")+t.count("!")+t.count("?"))
            syllables=sum(max(1,sum(1 for c in w if c.lower() in "aeiou")) for w in words) if words else 1
            return max(0.0, 206.835 - 1.015*(len(words)/sents) - 84.6*(syllables/max(len(words),1)))

    LLaMABackbone = _BB

# ── readability helper (mirrors eval/watermark.py) ────────────────────────────
def readability_score(text):
    fre = textstat.flesch_reading_ease(text)
    return round(1 + (fre / 100) * 4, 2)

# ── run_peccavi (mirrors eval/watermark.py exactly) ──────────────────────────
def run_peccavi(backbone, generations=10, n_paraphrases=5, verbose=True):
    praeco   = Praeco()
    auctor   = Auctor(backbone)
    scriba   = Scriba(backbone, n_variants=n_paraphrases)
    custos   = Custos(backbone)
    magister = Magister(backbone, theta_init=auctor.theta)

    history = []
    for gen in range(1, generations + 1):
        prompt           = praeco.next_prompt()
        auctor.theta     = magister.theta
        wm_text          = auctor.generate(prompt, max_tokens=100)
        paraphrases      = scriba.paraphrase(wm_text)
        s_eff            = custos.effective_score(paraphrases)
        original_score   = custos.watermark_score(wm_text)
        new_theta        = magister.update(wm_text, s_eff)
        record = {
            "generation":     gen,
            "theta":          round(new_theta, 4),
            "original_score": round(original_score, 4),
            "effective_score": round(s_eff, 4),
            "readability":    readability_score(wm_text),
        }
        history.append(record)
        if verbose:
            logger.info(f"Gen {gen:>3} | θ={new_theta:.4f} | "
                        f"S_orig={original_score:.4f} | S_eff={s_eff:.4f}")

    # AUC-ROC + FPR  (mirrors eval/watermark.py)
    logger.info("Computing AUC-ROC and false positive rate...")
    eval_prompts  = praeco.batch_prompts(20)
    human_texts   = [backbone.generate(p, max_new_tokens=100)["text"] for p in eval_prompts]
    wm_texts_eval = [auctor.generate(p, max_tokens=100) for p in eval_prompts]
    labels = [0]*20 + [1]*20
    scores = [custos.watermark_score(t) for t in human_texts + wm_texts_eval]
    auc    = roc_auc_score(labels, scores)
    threshold = 0.52
    fp  = sum(1 for t in human_texts if custos.watermark_score(t) >= threshold)
    fpr = fp / len(human_texts)

    first_eff = history[0]["effective_score"]
    last_eff  = history[-1]["effective_score"]
    improvement = (last_eff - first_eff) / max(first_eff, 1e-6) * 100
    avg_readability = round(sum(r["readability"] for r in history) / len(history), 2)

    return {
        "theta_final":                    history[-1]["theta"],
        "effective_score_final":          last_eff,
        "effective_score_improvement_pct": round(improvement, 2),
        "meets_85pct_retention":          last_eff >= 0.85,
        "auc_roc":                        round(auc, 4),
        "false_positive_rate":            round(fpr, 4),
        "meets_90pct_auc":                auc >= 0.90,
        "avg_readability":                avg_readability,
        "meets_readability_45":           avg_readability >= 4.5,
        "history":                        history,
    }

# ── _peccavi_summary  (mirrors benchmarks.py exactly) ────────────────────────
def _peccavi_summary(pec_out):
    return {
        "theta_final":          pec_out["theta_final"],
        "effective_score_final": pec_out["effective_score_final"],
        "improvement_pct":      pec_out["effective_score_improvement_pct"],
        "auc_roc":              pec_out["auc_roc"],
        "false_positive_rate":  pec_out["false_positive_rate"],
        "avg_readability":      pec_out["avg_readability"],
        "pass_retention":       pec_out["meets_85pct_retention"],
        "pass_auc":             pec_out["meets_90pct_auc"],
        "pass_readability":     pec_out["meets_readability_45"],
    }

# ── _print_summary  (mirrors benchmarks.py exactly) ──────────────────────────
def _print_summary(report):
    print("\n" + "═"*60)
    print("  PECCAVI BENCHMARK SUMMARY")
    print("═"*60)
    p = report["peccavi"]
    print(f"\n  θ_final           : {p['theta_final']}")
    print(f"  Effective Score   : {p['effective_score_final']:.4f}  "
          f"({'PASS' if p['pass_retention'] else 'FAIL'} ≥0.85)")
    print(f"  Improvement       : {p['improvement_pct']:.1f}%")
    print(f"  AUC-ROC           : {p['auc_roc']:.4f}  "
          f"({'PASS' if p['pass_auc'] else 'FAIL'} ≥0.90)")
    print(f"  False Positive    : {p['false_positive_rate']:.4f}")
    print(f"  Avg Readability   : {p['avg_readability']:.2f}/5  "
          f"({'PASS' if p['pass_readability'] else 'FAIL'} ≥4.5)")

    # Overall pass/fail
    all_pass = p["pass_retention"] and p["pass_auc"] and p["pass_readability"]
    status   = "\033[0;32mALL CRITERIA MET ✓\033[0m" if all_pass else "\033[0;31mSOME CRITERIA FAILED ✗\033[0m"
    print(f"\n  Overall           : {status}")
    print("═"*60 + "\n")

# ── run_benchmarks  (mirrors benchmarks.py exactly) ──────────────────────────
def run_benchmarks(backbone, output_path, verbose=True):
    report = {}
    print("\n" + "═"*60)
    print("  BENCHMARK: PECCAVI - Watermarking & Content Authenticity")
    print("═"*60)
    pec_out           = run_peccavi(backbone, generations=GENS, verbose=verbose)
    report["peccavi"] = _peccavi_summary(pec_out)
    report["history"] = pec_out["history"]
    report["stub_mode"] = STUB_MODE
    _print_summary(report)
    import os
    os.makedirs(str(Path(output_path).parent), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Results saved → {output_path}")
    return report

#  Entry point 
backbone    = LLaMABackbone()
output_path = str(OUT_DIR / "benchmark_results.json")
report      = run_benchmarks(backbone, output_path=output_path, verbose=True)

if STUB_MODE:
    print("  ℹ  Ran in STUB mode — scores are deterministic placeholders.")
    print("     With real backbone, import backbone.model and peccavi.*\n")
PYEOF

fi  # end --no-bench block

# =============================================================================
#  Final summary
# =============================================================================
echo ""
echo -e "${BOLD} ${RESET}"
echo -e "${BOLD}  Pipeline complete  ${RESET}"
printf  "${BOLD}  Processed : %-43s║${RESET}\n" "${PROCESSED[*]:-none}"
[[ ${#FAILED[@]} -gt 0 ]] && \
printf  "${BOLD}  Failed    : %-43s║${RESET}\n" "${FAILED[*]}"
printf  "${BOLD}   Output    : %-43s║${RESET}\n" "${OUTPUT_DIR}"
echo -e "${BOLD} ${RESET}"
echo ""
echo "Output files:"
find "$OUTPUT_DIR" \( -name "*.csv" -o -name "*.json" \) 2>/dev/null \
  | sort | while read -r f; do
      printf "  %-58s %s\n" "$f" "$(du -sh "$f" 2>/dev/null | cut -f1)"
    done
echo ""

[[ ${#FAILED[@]} -gt 0 ]] && exit 1 || exit 0