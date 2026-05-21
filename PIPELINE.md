# PECCAVI: Pipeline Architecture & Ideological Development

## 1. Overview

PECCAVI is a multi-agent LLM watermarking framework. Its goal is to embed an invisible, statistically detectable signal into AI-generated text such that:

1. The watermark survives adversarial paraphrase attacks
2. A detector can reliably distinguish watermarked from human text (AUC-ROC ≥ 0.90)
3. The watermarked text remains readable and coherent (readability ≥ 4.5/5)
4. The watermark is retained in ≥ 85% of paraphrased variants

The system is built around five agents — Praeco, Auctor, Scriba, Custos, and Magister — each with a distinct responsibility, coordinated through a shared LLM backbone.

---

## 2. The Watermarking Problem

### 2.1 Why Watermark LLM Output?

As LLMs become capable of producing human-indistinguishable text, the ability to verify whether a piece of text was machine-generated becomes critical for:

- Detecting AI-generated misinformation
- Academic integrity
- Content provenance and authenticity
- Regulatory compliance

### 2.2 The Core Challenge

Any watermarking scheme must solve three competing objectives simultaneously:

- **Detectability**: The signal must be strong enough for a detector to reliably find it
- **Robustness**: The signal must survive paraphrasing, word substitution, and reordering attacks
- **Invisibility**: The watermarked text must remain natural and readable

These objectives are in direct tension. A stronger signal is easier to detect but easier for an attacker to notice and remove. A more subtle signal preserves quality but is harder to detect reliably.

---

## 3. Theoretical Foundation

### 3.1 The Watermarked Distribution

PECCAVI implements a modified generation distribution:

```
p_w(x_t | x_{<t}, θ) ∝ p_LM(x_t | x_{<t}) · exp(θ · g(x_t, r_t))
```

Where:
- `p_LM` is the base language model distribution
- `θ` (theta) is the watermark strength parameter — the key variable learned by PECCAVI
- `g(x_t, r_t)` is the watermark score for token `x_t` given a context-derived random seed `r_t`
- The higher `θ`, the more aggressively green tokens are preferred

### 3.2 Green/Red List Partitioning

Each token is deterministically assigned a score in [0, 1] using a hash function:

```python
g(token_id, seed) = SHA256(f"{seed}:{token_id}") / 0xFFFFFFFF
```

This creates a soft green/red partitioning:
- **Green tokens**: g-score close to 1.0 — preferred during watermarked generation
- **Red tokens**: g-score close to 0.0 — suppressed during watermarked generation

The partitioning is **context-dependent** — the seed changes with each new context window (last 5 generated tokens), so different positions in the text have different green/red assignments. This prevents an attacker from learning a fixed green list.

### 3.3 Context Seed Derivation

```python
seed = SHA256(SECRET_KEY + str(context_ids[-5:]))
```

The seed depends only on the **generated token IDs** (not the prompt), using a rolling window of the last 5 tokens. This is critical — both the embedder (Auctor) and detector (Custos) must use the same seed derivation logic to align their green/red lists.

### 3.4 KGW vs SIR vs PECCAVI

The Kirchenbauer-Geiping-Wenner (KGW) scheme applies a fixed additive logit bias (`delta`) to all green-list tokens at every generation step. SIR (Selective Insertion with Randomness) improves on KGW by applying the bias only at high-entropy positions, reducing quality degradation. Both methods use a fixed, non-learned policy.

PECCAVI extends these with three compounding innovations:

1. **Tournament sampling with inline biased generation** — rather than adding a flat logit bias, PECCAVI samples K candidates (k=16) from the LM distribution and re-weights via `θ · (2g - 1)` before multinomial selection. This embeds a stronger signal without catastrophically suppressing high-quality red tokens.
2. **Context-adaptive θ via REINFORCE** — `θ(context) = θ_base + w · φ(prompt)`, where the weight vector `w` is learned jointly with `θ_base`. High-entropy prompts (creative writing) receive a higher θ; low-entropy prompts (factual Q&A) receive a gentler watermark that better preserves quality.
3. **Attack-aware policy learning** — the REINFORCE reward includes a back-translation survival term `ρ · S_survival`, where `S_survival` measures how much watermark signal survives an EN→FR→EN MarianMT round-trip attack applied *during training*. KGW and SIR have no learned policy and cannot optimise for post-attack detection.

---

## 4. System Architecture

### 4.1 Shared Backbone

**File**: `backbone/model.py`

All agents share a single `LLaMABackbone` instance. The backbone supports three modes:

| Backend | Model | Use Case |
|---|---|---|
| `transformers` | LLaMA-2-7b-chat-hf (4-bit) | Primary — local GPU inference |
| `openai` | GPT-4o | Baseline comparison |
| `anthropic` | Claude Sonnet | Baseline comparison |

The backbone is loaded once and shared across all agents to avoid redundant GPU memory allocation. 4-bit quantization (bitsandbytes NF4) reduces LLaMA-2-7B from 14GB to ~4GB, making it feasible on T4/A100 GPUs.

**NaN/Inf Safety**: 4-bit quantized models occasionally produce NaN or Inf logits for unusual token sequences. A `_NanInfClamp` LogitsProcessor is applied during every `model.generate()` call to prevent these from poisoning `torch.multinomial` sampling.

### 4.2 Agent: Praeco (Prompt Orchestrator)

**File**: `peccavi/praeco.py`

Praeco manages the prompt pool for training generations. It:
- Loads prompts from CSV datasets in `datasets/` (columns named `experiment` or `experiments`)
- Falls back to a built-in bank of 7 AI safety prompts if no datasets found
- Uses weighted random sampling — prompts can be scored to bias toward harder/more informative examples
- `next_prompt()` samples one prompt per training generation
- `batch_prompts(n)` samples n prompts for AUC-ROC evaluation

### 4.3 Agent: Auctor (Watermark Embedder)

**File**: `peccavi/auctor.py`

Auctor generates watermarked text using tournament sampling inline during autoregressive generation.

#### Ideological Evolution

**Version 1 — Post-hoc Refinement (20% coverage)**:
The original approach generated a full baseline text using `backbone.generate()`, then went back and replaced the last 20% of tokens using tournament sampling. Problems:
- Tokens after position i were generated assuming the original token i — replacing it retroactively broke coherence
- Only 20% of tokens carried the watermark signal, diluting detection

**Version 2 — Post-hoc Refinement (50% coverage)**:
Coverage increased to 50% to improve signal strength. Same coherence problem but worse, since more upstream tokens were being replaced.

**Version 3 — Post-hoc Refinement (30% coverage)**:
Reduced to 30% to improve readability. Still fundamentally broken — Custos was scoring all tokens including the 70% non-watermarked ones, diluting S_orig to ~0.50–0.54.

**Version 4 — Inline Autoregressive Generation (current)**:
The correct approach. Tournament sampling is applied at every token step during generation, not post-hoc:

```python
for _ in range(max_tokens):
    context_text = tokenizer.decode(prompt_ids + generated_ids)
    new_token, _ = self._tournament_sample(context_text, generated_ids)
    if new_token == eos_token_id:
        break
    generated_ids.append(new_token)
```

Each token is selected under the watermarked distribution before being committed to the sequence. This:
- Preserves the autoregressive coherence chain (no post-hoc replacement)
- Embeds the watermark signal in every token
- Allows Custos to score all tokens without dilution
- Produces S_orig of 0.60–0.70 vs 0.52–0.55 in earlier versions

#### Tournament Sampling Mechanism

```python
# 1. Get LM logits for next token
logits = model(context)[:, -1, :]
logits = nan_to_num(logits)  # safety clamp

# 2. Take top-k candidates (k=16)
top_k_logits, top_k_indices = torch.topk(logits, k=16)

# 3. Compute watermark-aware bias for each candidate
for token_id, base_logit in zip(top_k_indices, top_k_logits):
    g = watermark_score(token_id, seed)
    boost = theta * (2*g - 1)   # maps g in [0,1] to boost in [-θ, +θ]
    biased_logit = base_logit + boost

# 4. Softmax and multinomial sample
probs = softmax(biased_logits)
winner = multinomial(probs)
```

The bias `θ · (2g - 1)` shifts green tokens up by θ logit units and red tokens down by θ, creating a bimodal preference while still allowing red tokens to be selected if their base probability is high enough.

### 4.4 Agent: Scriba (Adversarial Paraphraser)

**File**: `peccavi/scriba.py`

Scriba attacks each watermarked text with N paraphrase variants to measure how much of the watermark signal survives. It uses three attack strategies:

**Lexical Attack**: Replaces words with WordNet synonyms. Preserves sentence structure but changes individual tokens — moderate watermark disruption.

**Syntactic Attack (Back-translation)**: Translates EN→FR via Helsinki-NLP/opus-mt-en-fr, then FR→EN via opus-mt-fr-en. The round-trip changes sentence structure while preserving meaning — strong watermark disruption since token sequences change significantly.

**Semantic Attack (LM Paraphrase)**: Uses the backbone LLM with prompts like "Rewrite the following using completely different words while preserving meaning." Generates semantically equivalent text with entirely different surface form — strongest watermark disruption.

The `paraphrase()` method generates N variants by randomly selecting from these three strategies for each variant.

**Why Scriba Matters**: Without adversarial paraphrasing during training, Magister would only optimise for embedding a strong watermark in the original text. The paraphrase attack forces θ to grow in a direction that produces signals robust enough to survive real-world attacks.

### 4.5 Agent: Custos (Watermark Detector)

**File**: `peccavi/custos.py`

Custos detects watermarks by computing the average watermark score across all tokens:

```
S(text) = (1/T) · Σ_t g(x_t, r_t)
```

Where the seed `r_t` is derived from the preceding generated tokens using the same logic as Auctor, ensuring the green/red lists align between embedding and detection.

**Detection**: `S ≥ 0.52` → watermarked; `S < 0.52` → human

**Effective Score**: `S_eff = min_i S(paraphrase_i)` — the worst-case score across all Scriba paraphrases. This measures robustness.

#### Ideological Evolution

**Version 1 — Full-text Scoring (original)**:
Scored all tokens. When Auctor only watermarked the last 30%, the first 70% contributed random scores (~0.50 each), diluting S_orig to ~0.52–0.54. AUC-ROC was limited because the gap between watermarked and human texts was only ~0.04.

**Version 2 — Partial Scoring (last 30%)**:
Custos was changed to only score the last 30% of tokens (matching Auctor's coverage). S_orig jumped to ~0.60–0.61 because the dilution was removed. However, this introduced a coordination assumption: Custos needed to know where the watermark was embedded, making the system vulnerable to attacks that target only the last 30%.

**Version 3 — Full-text Scoring with Inline Generation (current)**:
Custos reverted to scoring all tokens, but now Auctor watermarks all tokens via inline generation. No coordination assumption needed — the watermark covers the entire text uniformly.

### 4.6 Agent: Magister (Policy Learner)

**File**: `peccavi/magister.py`

Magister implements REINFORCE-style policy gradient to adapt θ over training generations.

#### REINFORCE Update

```
θ ← θ + α · ∇_θ log p_w(x) · advantage
```

Where:
- `α = 0.05` is the learning rate
- `advantage = reward - baseline`
- `baseline = 0.5` (fixed at chance level — reward above 0.5 means watermark is working)

**Policy Gradient Approximation**:
```
∇_θ log p_w(x) ≈ Σ_t g(x_t, r_t)
```
Summed over all generated tokens (since inline generation now watermarks all tokens).

**Composite Reward**:
```
r = λ · S_eff + ν · Q - μ · max(0, PPL_ratio - 1) + ρ · S_survival
```

Where:
- `S_eff = min_i S(paraphrase_i)` — worst-case score across Scriba's paraphrases (robustness signal)
- `Q` = quality score combining perplexity (`max(0, 1 - (ppl-1)/99)`) and BERTScore F1
- `PPL_ratio` = generated_ppl / reference_ppl — penalises fluency degradation (μ=0.0 in default config)
- `S_survival` — watermark signal surviving a MarianMT EN→FR→EN back-translation (new term)
- Tunable weights: `λ=0.5, ν=0.3, μ=0.0, ρ=0.2` in the attack-aware config

**Why S_eff not S_orig for reward?**: The updated reward uses S_eff (post-paraphrase) rather than S_orig (pre-paraphrase) because the goal is robustness. θ controls embedding strength, and stronger embedding correlates with better post-paraphrase survival. S_orig still appears implicitly since S_eff ≤ S_orig — the reward is zero if S_eff hits zero even if S_orig is high.

**Attack-aware survival term** (`ρ · S_survival`): During each REINFORCE update, Magister:
1. Back-translates the generated text through MarianMT (EN→FR→EN) — the same attack Scriba uses
2. Scores the back-translated text with the PECCAVI detector
3. Applies a sigmoid centred at z=2.0: `S_survival = σ(z - 2.0)` — so reward kicks in above the practical detection threshold
4. Adds `ρ · S_survival` to the composite reward

This makes the policy explicitly optimise for watermark signal that survives the most common real-world attack. KGW and SIR cannot do this — they have no policy to update.

**Adaptive θ** (`θ(context) = θ_base + w · φ(prompt)`): When `adaptive_theta=True`, Magister learns a weight vector `w ∈ ℝ^D` over prompt features φ (entropy, length, topic indicators). The REINFORCE update becomes:
```
θ_base ← θ_base + α · grad · advantage
w      ← w      + α · grad · advantage · φ(prompt)
```
This allows the watermark strength to scale with prompt difficulty — creative prompts get stronger watermarks, factual prompts get gentler ones that better preserve quality.

#### Ideological Evolution

**Version 1 — Moving Average Baseline**:
Used a moving average of past rewards as the baseline. Problem: as the system improved, the baseline tracked the improving reward, making advantages always near zero. θ collapsed to 0.1 (minimum).

**Version 2 — Discounted Returns**:
Applied discount factor γ=0.99 to accumulate multi-step returns. Problem: with single-step episodes (one generation = one reward), discounting added no information and the seq_len normalisation made updates too small (θ stuck near 2.0).

**Version 3 — Fixed Baseline, S_orig reward**:
Fixed baseline at 0.5 (chance level). When S_orig > 0.5, advantage is positive and θ increases. Simple, stable, and correctly incentivised. θ now grows reliably from 2.0 to 5.8+ over 20 generations.

**Version 4 — S_eff reward + PPL penalty (current default)**:
Switched reward signal from S_orig to S_eff (post-paraphrase minimum score) to incentivise robustness directly. Added optional μ·PPL_penalty term. The rolling 20-sample history baseline replaces the fixed 0.5 — more stable when θ has converged and advantages would otherwise oscillate.

**Version 5 — Attack-aware training (attack-aware config, `rho_survival > 0`)**:
Added MarianMT back-translation survival score ρ·S_survival to the composite reward. MarianMT models load lazily on CPU (no VRAM conflict with the LLM on GPU) and a sigmoid normalised z-score provides a differentiable signal above the detection threshold. This is the primary novel contribution for EMNLP 2026.

**Version 6 — Adaptive θ (feature-conditioned policy)**:
Added feature vector φ(prompt) and weight vector w so that θ(context) = θ_base + w·φ. The REINFORCE update now trains both θ_base and w simultaneously. High-entropy prompts receive higher θ; low-entropy prompts receive lower θ, improving the quality-detection tradeoff across diverse prompt types.

---

## 5. Training Pipeline

**Entry point**: `main.py --mode train`

**Flow**:

```
Praeco.next_prompt()
    ↓
Auctor.generate(prompt, max_tokens=100)       ← inline tournament sampling
    ↓
Custos.watermark_score(wm_text)               ← S_orig
    ↓
Scriba.paraphrase(wm_text, n=10)              ← 10 adversarial variants
    ↓
Custos.watermark_score(each paraphrase)       ← S_eff = min score
Custos.retention_rate(paraphrases, threshold=0.52)
    ↓
Magister.update(wm_text, S_orig, prompt)      ← θ update via REINFORCE
    ↓
repeat for G generations
    ↓
AUC-ROC evaluation (100 human + 100 watermarked texts)
    ↓
Summary report
```

**Config** (`configs/peccavi.yaml`):

| Parameter | Value | Purpose |
|---|---|---|
| `theta_init` | 2.0 | Starting watermark strength |
| `tournament_k` | 16 | Candidates per tournament step |
| `detection_threshold` | 0.52 | Score cutoff for watermark detection |
| `generations` | 50 | Training generations (increased from 20) |
| `alpha` | 0.05 | REINFORCE learning rate |
| `lambda_wm` | 0.5 | Watermark score weight in reward |
| `nu_quality` | 0.3 | Quality score weight in reward |
| `mu_ppl` | 0.0 | PPL penalty weight (disabled by default) |
| `rho_survival` | 0.0 | Back-translation survival weight (0.2 in attack-aware config) |
| `scriba_n_variants` | 5 | Paraphrases per training generation |
| `adaptive_theta` | false | Enable context-conditioned θ(prompt) |
| `n_eval_samples` | 200 | Texts for AUC-ROC evaluation |

**Experiment configs**:

| Config | Key difference | Purpose |
|---|---|---|
| `peccavi.yaml` | λ=0.5, ν=0.3, ρ=0.0 | Main PECCAVI baseline |
| `peccavi_attack_aware.yaml` | ρ=0.2, λ=0.5, ν=0.3 | Novel contribution: attack-aware training |
| `peccavi_high_nu.yaml` | λ=0.4, ν=0.6 | Quality-focused tradeoff variant |
| `kgw_baseline.yaml` | Fixed delta/gamma, no policy | KGW comparison |
| `sir_baseline.yaml` | Entropy-gated KGW, no policy | SIR comparison |
| `ablation_fixed_theta.yaml` | `adaptive_theta=false`, fixed θ | Ablation: no θ learning |
| `ablation_no_quality.yaml` | ν=0.0 | Ablation: watermark signal only |
| `ablation_no_watermark.yaml` | λ=0.0 | Ablation: quality signal only |

---

## 6. Evaluation Pipeline

**Entry point**: `main.py --mode eval`

Runs a shorter training loop (5 generations) then evaluates three success criteria:

### 6.1 Watermark Retention Rate

```
retention = |{paraphrases : S(p) ≥ 0.52}| / N_paraphrases
```

Target: ≥ 85%. Measures what fraction of Scriba's adversarial paraphrases are still detected as watermarked.

### 6.2 AUC-ROC

Generates 100 human texts (plain backbone) and 100 watermarked texts (Auctor). Scores all 200 with Custos. Computes AUC-ROC of the detector. Target: ≥ 0.90.

### 6.3 Readability

Flesch Reading Ease score mapped to 1–5 scale. Target: ≥ 4.5/5.

### 6.4 False Positive Rate

Fraction of human texts scoring ≥ 0.52 (incorrectly flagged as watermarked). Not a primary success criterion but tracked — high FPR indicates the threshold is too close to the natural mean of 0.50.

---

## 7. Key Design Decisions & Tradeoffs

| Decision | Chosen Approach | Tradeoff |
|---|---|---|
| Token selection | Tournament sampling (top-16) | Stronger signal than direct bias; slower than greedy |
| Generation strategy | Inline per-token loop | Correct coherence; ~3× slower than post-hoc |
| θ learning | Rolling-history baseline REINFORCE | Stable convergence; no multi-step credit assignment |
| Reward signal | S_eff (post-paraphrase) + survival | Robustness-incentivised; policy gradient noisier than S_orig |
| Attack-aware training | MarianMT EN→FR→EN on CPU | Zero VRAM cost; covers back-translation attack only |
| Adaptive θ | Linear feature policy θ_base + w·φ | Interpretable; linear may underfit complex prompts |
| Quantization | 4-bit NF4 (bitsandbytes) | Fits on 16GB GPU; slight quality degradation |
| Seed window | Last 5 generated tokens | Context-dependent lists; short enough to survive minor edits |
| Paraphrase attacks | Lexical + syntactic (MarianMT) + semantic | Coverage of major attack vectors; all destroy token-level signal |
| Baseline comparison | KGW + SIR (no learned policy) | Fair comparison; EWD (Christ et al. 2023) not yet implemented |

---

## 8. Known Limitations

**Paraphrase robustness ceiling**: All three token-level methods (KGW, SIR, PECCAVI) drop to near-zero watermark retention after back-translation and semantic paraphrase attacks. This is fundamental — when the token sequence changes, the context seeds change, and green/red assignments realign randomly. The signal lives in which tokens were chosen, not in what the text means. Attack-aware training (`rho_survival > 0`) partially addresses this by making the policy prefer token choices that are more likely to survive back-translation, but cannot overcome the ceiling entirely. Sentence-level or semantic watermarks would be more robust at the cost of detectability.

**Speed**: Inline generation requires one full 7B-parameter forward pass per token. Generating 100 tokens takes ~60–90 seconds on A100. The AUC-ROC evaluation (100 watermarked texts) dominates total runtime at ~90 minutes.

**No θ persistence**: θ resets to 2.0 at the start of each run. A checkpoint system would allow θ to carry over between sessions and accumulate improvement across multiple training runs.

**KV cache not used**: Because the generation loop decodes to text and re-tokenizes at each step (for seed derivation), the transformer's KV cache cannot be reused across steps. A token-ID-level implementation would be ~10× faster.

---

## 9. Experimental Results (Seeds 42, 123)

### 9.1 Main Comparison (Table 1)

| Metric | PECCAVI | KGW | SIR |
|---|---|---|---|
| AUC-ROC | **0.974** | 0.821 | 0.739 |
| TPR @ 1% FPR | **0.886** | 0.202 | — |
| PPL ratio | 1.508 | **1.190** | ~1.25 |
| GPT-4 quality (1–5) | 3.28 | **3.60** | ~3.4 |
| FPR @ z≥4 | ~0.01 | ~0.04 | ~0.06 |

PECCAVI achieves 4× higher TPR@1%FPR than KGW and 19% higher AUC-ROC. The quality tradeoff (3.28 vs 3.60) is addressed by the `peccavi_high_nu` variant (ν=0.6) which shifts priority toward text quality at the cost of some detection power.

### 9.2 Ablation Study (Seeds 42, 123 — in progress)

| Variant | AUC-ROC | Notes |
|---|---|---|
| PECCAVI (full) | 0.974 | All reward terms |
| ablation_fixed_θ | TBD (s123 ✅) | No θ learning — fixed baseline |
| ablation_no_quality | TBD (s123 ✅) | ν=0.0, watermark only |
| ablation_no_watermark | TBD (s123 ✅) | λ=0.0, quality only |

### 9.3 Attack-aware Training (Seed 7 — running)

`peccavi_attack_aware` with ρ=0.2 adds MarianMT survival to REINFORCE reward. Expected to show improved S_eff post-back-translation vs standard PECCAVI. First watermarking method to explicitly optimise post-attack detection via policy gradient.

### 9.4 Success Criteria Progress

| Metric | Target | Initial | Current |
|---|---|---|---|
| θ_final | — | 2.6 | **~2.4–5.8** (adaptive) |
| AUC-ROC | ≥ 0.90 | 0.82 | **0.974** ✅ |
| TPR @ 1% FPR | high | — | **0.886** |
| PPL ratio | ≤ 1.3 | — | 1.508 ⚠️ |
| GPT-4 quality | ≥ 3.5 | — | 3.28 ⚠️ |
| FPR @ z≥4 | ≤ 0.05 | 0.20 | **~0.01** ✅ |

---

## 10. EMNLP 2026 Research Contributions

**Submission deadline**: May 25, 2026

### Primary Contribution
**Attack-aware watermark policy learning**: PECCAVI is the first watermarking framework to incorporate back-translation survival into the REINFORCE training reward. The policy learns to prefer token choices that are stable under EN→FR→EN round-trip translation — a direct optimisation target that KGW and SIR cannot replicate due to their fixed (non-learned) policies.

### Secondary Contributions
1. **Context-adaptive θ**: Learned linear policy `θ(prompt) = θ_base + w·φ(prompt)` adapts watermark strength to prompt entropy, improving the quality-detection tradeoff across diverse prompt types.
2. **Multi-seed ablation study**: Two-seed (42, 123) ablations isolating each reward component quantify the contribution of the quality term, PPL penalty, and attack-aware survival term.
3. **Pareto frontier analysis**: θ sweep reveals the detection-quality frontier and shows PECCAVI dominates KGW/SIR at equal PPL cost — directly counters the "just use higher delta" objection.
4. **Backbone-agnostic generalization**: Mistral-7B-Instruct ablations verify the method generalises beyond LLaMA-2.

### Paper Error to Fix
The draft describes Auctor's generation strategy as "tournament sampling during speculative decoding." This is incorrect. The correct description is **inline biased sampling**: at each autoregressive step, the top-K candidates are drawn from the LM distribution and re-weighted via `exp(θ · g(token, seed))` before multinomial selection. No speculative decoding or draft model is involved.

### EMNLP Probability Estimate
- Main conference: ~50–55% (strong detection results; quality tradeoff and missing EWD baseline are weaknesses)
- Findings track: ~85% (solid empirical contribution, multi-seed ablations, novel attack-aware training)
