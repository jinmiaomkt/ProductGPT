# Revision plan — reviewer comments on `jmp_v11.tex`

Source: the coloured blocks in `Paper/jmp_v11.tex` (red = reviewer, blue = Jin,
brown = Ying, purple = Fanglin). Eleven reviewer comments, grouped below by what
they ask for rather than by where they appear.

The useful news: **most of these are already answered by work done in September
2026** (R26–R34, 456 runs). The table for each comment says what exists, what is
missing, and what it costs. Numbers cited here are in `EXPERIMENTS.md`.

---

## A. Benchmarks and evaluation (the easiest wins)

**A1. "Stronger benchmarks beyond RNNs/LSTMs."** *(Simulation section, red)*

Answered, and far more strongly than the comment asks. R30 ran a five-stage
tuning programme: 235 runs over four families (plain GRU, GRU encoder,
transformer + recency prior, stacked hybrid), each tuned on an equal budget,
confirmed on fresh seeds, with the holdout opened exactly once.

- Headline: **no family wins.** Eight cells span 0.0194 nats at a seed sd of
  0.0075 — inside the 0.02 adoption margin.
- The best cell is also the **smallest** model (1.25M parameters against the
  transformer's 7.93M).
- This replaces "we beat an LSTM" with "we tuned four families to their own
  frontiers and they tie", which is a much harder claim to attack.

**Missing:** nothing computational. Writing only — a benchmark table with
parameters, seconds per epoch and the tuning budget per family.

**A2. "Evaluation plan is unclear; metrics should match prediction goals
(AUC for binary vs hit rate for categorical)."** *(Empirical Results, red;
Fanglin: "easy fix"; Jin: group-averaged AUC)*

The design is implemented and disciplined — the Lu & Kannan 2×2 (in-sample vs
out-of-sample customers × calibration vs holdout campaigns), campaign-27
validation, `split_seed=33`, holdout opened once at stage 4. The summariser
already reports macro-F1, macro-AUPRC, hit rate and revenue MAE per cell.

**Missing:** (i) a figure/table making the 2×2 explicit — currently it lives in
code comments; (ii) group-averaged AUC for the 9-class task, as Jin proposed;
(iii) one sentence per metric tying it to a decision the firm makes.
**Cost:** half a day of writing plus a small addition to the summariser.

**A3. "How are simulation errors managed in the recursive generative loop?"**
*(Objective Function, red)*

Planned in detail as R35: a rollout engine (model samples a decision → the
gacha resolves it under published rates and pity → inventory, IPT and campaign
advance), behind a validation gate: free-running vs teacher-forced calibration,
aggregate trajectory match against campaigns 28–30, and drift over horizon.
Calibration is already measured teacher-forced (ECE 0.010–0.019), which is the
precondition.

**Missing:** the engine itself. **Cost:** ~2 days of implementation, inference
only.

---

## B. Embeddings — index vs attribute, and their validation

**B1. "Compare index and attribute-based approaches, combine the two embedding
types, evaluate the impact of limited feature availability."** *(Triplet
section, red; Ying and Fanglin both endorse)*

Partly answered. The code has both: `product_id_embed` (index) and the 34-column
attribute table, plus two vocabulary levels (pooled 44 tokens vs 118
per-product). Stage 4 measured the vocabulary axis at five seeds and produced
what is really the "limited feature availability" result:

| Family | pooled (level 6) | per-product (level 7) | cost of product identity |
|---|---|---|---|
| plain GRU | 0.8856 | 0.9033 | +0.0177 |
| GRU encoder | 0.8904 | 0.8944 | +0.0040 |
| hybrid | 0.8958 | 0.8962 | +0.0004 |
| transformer | 0.9049 | 0.8925 | **−0.0124** |

The ordering is the finding: **the more attention an architecture carries, the
cheaper product-level resolution becomes** — free for the hybrid, beneficial for
the transformer, expensive only for the purely recurrent model. Jin's framing in
the source ("index suffices if the seller only revives old LTPs; attributes are
needed for new LTPs") is exactly the right economic reading and should go in.

**Missing:** the attribute-only cell (`--no-product-id`) at level 7, which
completes the 2×2 of index / attributes / both. **Cost:** 10 runs, ~4 GPU-hours.

**B2. "Expand the discussion of embeddings and their validation."** *(red, twice;
Jin: use the perception map; Fanglin: more validation results)*

Answered. Stage 5 produced a rotation-invariant validation of the learned
product space — nearest-neighbour purity against chance:

| Attribute | chance | transformer | hybrid | plain GRU |
|---|---|---|---|---|
| rarity | 0.408 | 0.911 | 0.907 | 0.666 |
| figure vs weapon | 0.514 | 0.992 | 0.977 | 0.641 |
| limited vs standard | 0.514 | 0.909 | 0.923 | 0.746 |
| element | 0.170 | 0.278 | 0.280 | **0.142** |

Jin's hypothesis holds: products of the same category cluster, and 3-star items
cluster together (rarity purity 0.91 against 0.41 chance). The attention models
organise by **commercial** attributes; the plain GRU barely organises at all.

**Caveat that must travel with it (R34):** this is what the model groups
together when predicting, not a preference or substitution map. Presenting it
as a perception map would overclaim.

**Cost:** writing only; the figure exists as coordinates in
`results/r30/stage5/embmap_*.json`.

---

## C. Architecture choices the reviewer questioned

**C1. "Consider cross-attention to pool information instead of concatenating to
triplets; the dynamic importance of each source mode is interesting."**
*(Attention section, red; Fanglin: show it was an intentional choice)*

Answered with evidence rather than recollection. We built the cross-attention
variant (offer tokens attending to inventory tokens) and measured it repeatedly:
R27 found it worth ≈ 0, and stage 4's ablation puts the entire stock path at
**0.0137 nats** — inside the adoption margin — with the token/cross-attention
form *worse* than simple per-product counts (0.8998 vs 0.8890).

So the paper can say: we implemented the reviewer's suggestion, measured it
against the concatenated triplet at equal budget, and it did not pay. That is
the strongest possible form of "intentional choice".

**Bonus for "dynamic importance of each source mode":** the gated hybrid records
the mean gate weight on the recurrent branch, which is a readable measure of how
much the model leans on recency versus retrieval. Worth reporting.

**Missing:** nothing computational; one paragraph plus the ablation table.

**C2. "How are stopping rules modelled?"** *(Objective Function, red; Fanglin
proposes an end-of-day stop decision and Raluca's search-gap paper)*

Already implemented — and it *is* Fanglin's proposal. The `_IPT` revision drops
ad-hoc no-draw rows and inserts one synthetic NotBuy at the end of every 24-hour
interval, so decision 9 means "no purchase within a 24h interval" rather than
end-of-sequence. Termination is by padding; there is no EOS.

Stage 5 then shows the stopping decision behaves differently from purchases:
NotBuy is 40.7% of occasions, and the architecture ordering **reverses** between
the two (recurrence wins NotBuy by 0.079, attention wins purchases by 0.042).
That is a substantive answer to "how is stopping modelled", not a definitional
one.

**Missing:** the comparison against the *old* stopping rule is not run on
corrected data, and the search-gap framing needs to be written. **Cost:**
writing, plus optionally 10 runs on the pre-IPT data for the contrast.

**C3. "How is heterogeneity across players accounted for?"** *(Objective
Function, red)*

This is the weakest point in the current draft and now has a real answer.
Today: user embeddings exist but were **off** (`USER_EMB=0`) in every reported
run; heterogeneity enters only implicitly through the sequence state. R34 s7
quantifies the cost: unmodelled heterogeneity in satiation strength attenuates
the pooled estimate by 26–54% at realistic dispersion. A customer fixed effect
does not fix it and makes the kernel worse (s2, s7).

There is also a design constraint worth stating: half the evaluation is
out-of-sample customers, so any per-customer quantity must be **predictable from
that customer's own history**, not a free parameter.

**Missing:** a hierarchical or latent-class version, and the simulation (S8)
that checks whether the population distribution is recoverable before we trust
it. **Cost:** S8 is ~2 CPU-hours; the model change is ~1 day plus 25 runs.

---

## D. Framing and positioning

**D1. "Explain the connection between loot-box mechanics and limited-time
products more clearly."** *(Overview, red; Ying: position for broader LTP and
use loot boxes as the application)*

Writing task, and our identification work gives it a spine. The obstacle we
documented — **assortment rotation makes substitution unidentifiable** — is not
a loot-box property at all. It applies to any limited-time-product market where
offers persist for weeks: fashion drops, seasonal menus, flash sales, game
banners. Loot boxes are then the *illustration* of a general problem, which is
exactly Ying's requested positioning.

**D2. "Generalizability: loot boxes are probabilistic, many LTPs rely on
transparent feature-based utility, where the triplet data may not apply."**
*(Overview, red; Jin: address generalizability and triplet/duplet tokens)*

The honest answer separates two things the comment merges:

- The **triplet token** (offer, obtained, previous decision) needs a realised
  *outcome* stream, which probabilistic products have and deterministic ones do
  not. For a transparent LTP the obtained stream degenerates to "what you chose",
  and the triplet collapses to a **duplet** — which the codebase already
  supports (`gen2_lp_duplet`). That is a clean generalisability statement with
  an implementation behind it.
- The **probabilistic nature** is what gives us identification leverage the
  deterministic case lacks: conditional on how many times a customer pulled,
  *which* products they received is the machine's draw. R34 exploits exactly
  this to identify duplicate weights. Worth turning the reviewer's concern into
  a contribution.

**D3. "More detail about the specific empirical context."** *(Data, red; Jin: a
3×3 diagram; Ying: model-free evidence of perceived luckiness)*

We now have concrete calendar facts from the aggregate extraction: 30 campaigns,
49 distinct products ever offered, **3–4 offered per campaign**, median **45
occasions per customer per campaign**. Those numbers also carry the
identification argument, so they earn their place.

**Missing:** Ying's model-free luckiness evidence (does a customer who just lost
a 50/50 behave differently?) is not done. **Cost:** ~half a day, aggregates only.

---

## Priority and sequencing

| # | Task | Type | Cost |
|---|---|---|---|
| 1 | Benchmark table + evaluation-design figure + AUC/hit-rate fix (A1, A2) | writing + small | 1 day |
| 2 | Cross-attention-vs-triplet paragraph and ablation table (C1) | writing | 0.5 day |
| 3 | Embedding validation section from the stage-5 map, with the R34 caveat (B2) | writing | 0.5 day |
| 4 | Stopping-rule section from the IPT design + per-class reversal (C2) | writing | 0.5 day |
| 5 | Attribute-only cell to complete index/attribute/both (B1) | 10 runs | 4 GPU-h |
| 6 | S8 heterogeneity simulation, then the hierarchical model (C3) | 2 CPU-h + 25 runs | 2 days |
| 7 | R35 rollout engine and revenue link (A3 and "business value") | build | 2 days |
| 8 | Loot-box ↔ LTP framing and duplet generalisation (D1, D2) | writing | 1 day |
| 9 | Model-free luckiness evidence (D3) | analysis | 0.5 day |

Items 1–4 need no computation and close five reviewer comments. Item 7 closes
the two that matter most for a marketing audience: recursive-loop error control,
and the link from prediction to revenue.
