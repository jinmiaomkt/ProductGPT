"""R45 step 2: generation on chopped histories (EXPERIMENTS.md, R45 step 2).

    python scripts/r45_generate.py             # every arm (GPU)
    python scripts/r45_generate.py --smoke     # every arm on 12 customers, seed 1 (CPU)

R43 step 2's protocol and frame, unchanged (scripts/r43_generate.py): the
outsample_users_holdout_period cell, 300 customers drawn with default_rng(0), 8
replicates, chunks of 6, the environment fitted on 600 training customers'
calibration rows, the split-half band (120 splits, seed 3), rollout seed 1000 +
training seed. Every rollout is cropped to R36i's 384-row frame before it is
scored, so the real set, the band and R43's yardsticks are shared and R43's
numbers can be quoted alongside.

WHAT IS CHOPPED: only what the model reads. Each customer's first k active
campaigns are removed from the model's input by censor6.ChopView (calendar,
state zero: the computed state restarts, and the flag is set for every chopped
customer), exactly as part A trained and tested Z and ZF. The game keeps the
full history: the environment's pity is warmed on the full real prefix
(rollout's pity_from). Row t of a chopped history is row s + t of the full one;
the generated rows are placed back there and cropped to the frame, so every arm
is scored against the same real rows.

ARMS, training seeds 1-3, each free running and teacher forced:
    F   k = 0        b39_g6w384    R43's G6: the reference
    F   k = 1, 2, 4  b39_g6w384    a full-history model handed a censored history
    Z   k = 1, 2, 4  r45a_Z_k<k>   trained and generating on censored histories
    ZF  k = 1, 2, 4  r45a_ZF_k<k>  the same, with the censoring flag
The rules are read by scripts/r45_step2_read.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "gen5_multistream", REPO / "gen6_streaming", REPO / "simulation",
          REPO / "shared", REPO / "scripts"):
    sys.path.insert(0, str(p))

import config5  # noqa: E402
import seq_metrics as sm  # noqa: E402
import dataset_multistream as dsm  # noqa: E402
from baselines import real_sequences  # noqa: E402
from censor6 import ChopView, collate_prefix  # noqa: E402
from gacha_env import AcquisitionEnv, ProductTable  # noqa: E402
from r36c_generate import collate_chunk, score  # noqa: E402
from r36i_rerun import evaluate, integrity, pooled_counts, summarise  # noqa: E402
from r43_generate import (CELL, DATA_KEYS, FIX_COMMIT, HOLDOUT_FROM, MAX_EVENTS,  # noqa: E402
                          NULL_SEED, PREFIX_Q95, REF, align, git, item_len, to_frame)
from rollout import first_row_of_campaign, load_checkpoint, rollout  # noqa: E402

RUNS = Path("/storage/home/jinmiao/ProductGPT/runs")
F_TAG = "b39_g6w384"
DEPTHS = (1, 2, 4)
SEEDS = (1, 2, 3)
ROWS = ("lto", "obtained", "prev_decision", "label", "campaign")
# R44's frozen gen-6 configuration: every checkpoint must share it with F
SAME_AS_F = DATA_KEYS + ("window", "d_model", "d_ff", "N", "num_heads", "lr", "dropout",
                         "weight_decay", "grad_accum", "warmup_frac", "chunk", "batch_size",
                         "max_events", "model_rev", "camp_fe", "max_users")
ABSENT = {"chop": "none", "censor_flag": False, "sched_sampling": 0.0}   # R44 predates R45


def arms():
    """(name, tag, depth, teacher_forced), free running first."""
    out = []
    for tf in (False, True):
        pre = "tf" if tf else "fr"
        out.append((f"{pre}_F_k0", F_TAG, 0, tf))
        for k in DEPTHS:
            out += [(f"{pre}_F_k{k}", F_TAG, k, tf), (f"{pre}_Z_k{k}", f"r45a_Z_k{k}", k, tf),
                    (f"{pre}_ZF_k{k}", f"r45a_ZF_k{k}", k, tf)]
    return out


def expected(tag: str) -> dict:
    """The chop settings a checkpoint's cfg must carry."""
    if tag == F_TAG:
        return {"chop": "none", "censor_flag": False}
    k = int(tag.rsplit("_k", 1)[1])
    return {"chop": "calendar", "chop_depth": k, "chop_state": "zero", "chop_where": "both",
            "censor_flag": tag.startswith("r45a_ZF_")}


def find_ckpt(runs: Path, tag: str, seed: int) -> Path:
    hits = sorted(runs.glob(f"*_{tag}_s{seed}/gen6_streaming/hpcc/best.pt"))
    if len(hits) != 1:
        raise SystemExit(f"{tag} seed {seed}: expected one checkpoint, found {len(hits)}")
    return hits[0]


def chop_batch(cell, idx, k: int, val_from: int):
    """The model's input for customers `idx` of the full-history cell: each history
    without its first k active campaigns (state zero, flagged). Returns the batch,
    the chop point s and the chopped length per customer."""
    view = ChopView(cell, "calendar" if k else "none", k, "zero", val_from)
    items = [view[int(i)] for i in idx]
    b = collate_prefix(items, collate_chunk(items, range(len(items))))
    s = np.array([int(it["chopped"]) for it in items])
    lz = np.array([int(it["lto"].size(0)) for it in items])
    return b, s, lz


def check_chop(bz, b6, s, lz, full_len, owns_z, owns6) -> list:
    """Row t of the chopped batch must be row s + t of the full one, every row
    after the chop must be kept, and each holdout must start s rows earlier. A
    customer with no holdout row has none in either batch (first_row_of_campaign
    then returns each batch's own length; nothing is generated for them)."""
    bad = []
    for c in range(len(s)):
        o, L = int(s[c]), int(lz[c])
        if o + L != int(full_len[c]):
            bad.append(f"customer {c}: {o} chopped + {L} kept != {int(full_len[c])} rows")
        for key in ROWS:
            if not torch.equal(bz[key][c, :L], b6[key][c, o:o + L]):
                bad.append(f"customer {c}: {key}")
        has6 = bool((b6["campaign"][c] >= HOLDOUT_FROM).any())
        hasz = bool((bz["campaign"][c] >= HOLDOUT_FROM).any())
        if has6 != hasz:
            bad.append(f"customer {c}: holdout rows in one batch only")
        elif has6 and int(owns_z[c]) + o != int(owns6[c]):
            bad.append(f"customer {c}: holdout starts at {int(owns_z[c])} + {o}, not {int(owns6[c])}")
    return bad


def unchop(r: dict, b6: dict, s: np.ndarray, lz: np.ndarray, n_rep: int):
    """A rollout on chopped histories, placed back on the full histories. Rows
    before each chop keep their real values: nothing is generated there. Returns
    the full-history result and how many generated or live rows did not land on
    the full history's live rows."""
    lab6 = b6["label"].cpu().numpy()
    B, S6 = lab6.shape
    alive = np.repeat(lab6 != 0, n_rep, axis=0)
    dec = np.zeros((B * n_rep, S6), dtype=np.int64)
    obt = np.repeat(b6["obtained"].cpu().numpy(), n_rep, axis=0)
    lto = np.repeat(b6["lto"].cpu().numpy(), n_rep, axis=0)
    bad = 0
    for j in range(B * n_rep):
        c = j // n_rep
        o, L = int(s[c]), int(lz[c])
        dec[j, o:o + L] = r["decisions"][j, :L]
        obt[j, o:o + L] = r["obtained"][j, :L]
        bad += int(not np.array_equal(r["alive"][j, :L], alive[j, o:o + L]))
        bad += int(not np.array_equal(r["lto"][j, :L], lto[j, o:o + L]))
        bad += int((r["decisions"][j, L:] > 0).sum() + r["alive"][j, L:].sum())
    return {"decisions": dec, "alive": alive, "obtained": obt, "lto": lto}, bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(RUNS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="the k = 0 and k = 2 arms on 12 customers x 2 replicates, training "
                         "seed 1, after loading every checkpoint. A wiring check; its numbers "
                         "mean nothing.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    runs = Path(a.runs_dir)
    out_path = Path(a.out or f"results/r45/{'smoke_' if a.smoke else ''}step2.json")

    if a.smoke:
        n_users, n_rep, chunk, n_boot, seeds = 12, 2, 6, 20, (1,)
    else:
        n_users, n_rep, chunk, n_boot, seeds = 300, 8, 6, 120, SEEDS
        if a.device != "cuda":
            raise SystemExit("the full run needs a GPU; use --smoke for a wiring check")

    # ------------------------------------------------------------- the code
    head = git("rev-parse", "--short", "HEAD").stdout.strip()
    if git("merge-base", "--is-ancestor", FIX_COMMIT, "HEAD").returncode != 0:
        raise SystemExit(f"HEAD {head} does not contain the rollout fix {FIX_COMMIT}")
    dirty = git("status", "--porcelain", "--untracked-files=no", "--", "simulation", "scripts",
                "gen5_multistream", "gen6_streaming", "shared").stdout.strip()
    print(f"[code] HEAD {head}, contains {FIX_COMMIT}; "
          f"{'UNCOMMITTED CHANGES:' + chr(10) + dirty if dirty else 'clean'}")
    if dirty and not a.smoke:
        raise SystemExit("uncommitted changes to the code this job runs")

    # ---------------------------------------------- all 21 checkpoints, checked
    tags = sorted({t for _, t, _, _ in arms()})
    paths = {(t, s): find_ckpt(runs, t, s) for t in tags for s in SEEDS}
    cfgs = {key: torch.load(str(p), map_location="cpu", weights_only=False)["cfg"]
            for key, p in paths.items()}
    print("[ckpt]")
    for (t, s), c in cfgs.items():
        ref = cfgs[(F_TAG, s)]
        diff = [k for k in SAME_AS_F if c.get(k) != ref.get(k)]
        want = dict(expected(t), seed=s, gen=6, window=384, max_events=None, model_rev=2,
                    camp_fe=True, sched_sampling=0.0)
        got = {k: c.get(k, ABSENT.get(k, "<absent>")) for k in want}
        got["sched_sampling"] = float(c.get("sched_sampling") or 0.0)
        got["camp_fe"] = bool(c.get("camp_fe"))
        wrong = {k: got[k] for k in want if got[k] != want[k]}
        print(f"  {t}_s{s}: {paths[(t, s)].parent.parent.parent.name}  chop {got['chop']} "
              f"{c.get('chop_depth', '')} {c.get('chop_state', '')}  censor_flag "
              f"{got['censor_flag']}  lr {c.get('lr')}")
        if diff:
            raise SystemExit(f"{t}_s{s} differs from {F_TAG}_s{s} on {diff}")
        if wrong:
            raise SystemExit(f"{t}_s{s}: {wrong}, expected {want}")
    if a.smoke:
        for key, p in paths.items():
            load_checkpoint(p, device="cpu", legacy_camp_bias=False)
        print(f"[ckpt] all {len(paths)} checkpoints load")

    # ---------------------------------------------------------- the frame
    from train5_multistream import build_loaders
    ref_path = sorted(runs.glob(f"*_{REF[0]}_s{REF[1]}/*/hpcc/best.pt"))
    if len(ref_path) != 1:
        raise SystemExit(f"the frame's checkpoint {REF[0]}_s{REF[1]}: found {len(ref_path)}")
    ref_cfg = torch.load(str(ref_path[0]), map_location="cpu", weights_only=False)["cfg"]
    diff = [k for k in DATA_KEYS if ref_cfg.get(k) != cfgs[(F_TAG, 1)].get(k)]
    if diff:
        raise SystemExit(f"the frame's configuration differs from {F_TAG}_s1 on {diff}")
    cfg5 = dict(ref_cfg)
    cfg5["max_events"] = MAX_EVENTS
    dsm.set_product_range(cfg5["first_prod_id"], cfg5["last_prod_id"])
    print(f"[frame] {REF[0]}_s{REF[1]}'s configuration at context {MAX_EVENTS} (R36i, R43)")
    tr_dl, _, tests, _ = build_loaders(cfg5)
    cell5 = tests[CELL].dataset
    train_ds = tr_dl.dataset
    pick = np.sort(np.random.default_rng(0).permutation(len(cell5))[:n_users])
    uids = [str(cell5[i]["uid"]) for i in pick]
    uid_sha = hashlib.sha256("\n".join(uids).encode()).hexdigest()[:16]
    print(f"[cell] {CELL}: {len(cell5)} customers, scoring {len(pick)} "
          f"(customer-list sha256 {uid_sha})")

    fit_items = [train_ds[i] for i in range(min(len(train_ds), 600))]
    vocab = ProductTable.from_xlsx(config5.feature_file_path(cfg5))
    env = AcquisitionEnv(vocab).fit(fit_items, holdout_from=HOLDOUT_FROM)
    print(f"[env] calibrated on {len(fit_items)} training customers (frame build)")

    chunks = [pick[i:i + chunk] for i in range(0, len(pick), chunk)]
    batches = [collate_chunk(cell5, ch) for ch in chunks]
    owns = [first_row_of_campaign(b["campaign"], HOLDOUT_FROM) for b in batches]
    lens = [np.array([item_len(cell5, i) for i in ch]) for ch in chunks]
    F_real = np.vstack([score(real_sequences(b, start=int(st.min())), st)
                        for b, st in zip(batches, owns)])
    scaler = sm.Scaler.fit(F_real)
    Zr = scaler(F_real)
    null = sm.null_band(Zr, n_boot=n_boot, seed=NULL_SEED)
    q95 = float(np.quantile(null, 0.95))
    real_counts = sum(pooled_counts(b["label"].cpu().numpy(), b["label"].cpu().numpy() != 0)
                      for b in batches)
    same_as_prefix = (not a.smoke) and abs(q95 - PREFIX_Q95) < 1e-9
    print(f"[real] {F_real.shape[0]} sequences, {int(real_counts.sum()):,} holdout rows")
    print(f"[band] split-half null, {n_boot} splits: q95 {q95:.7f}"
          + ("" if a.smoke else f"; R36i/R43 q95 {PREFIX_Q95:.7f} "
             + ("REPRODUCED" if same_as_prefix else "NOT REPRODUCED")))

    # ------------------------------------------- the full-history build
    cfg6 = dict(cfgs[(F_TAG, 1)])
    val_from = int(cfg6["val_from"])
    _, _, tests6, _ = build_loaders(cfg6)
    cell6 = tests6[CELL].dataset
    where6 = {str(cell6[i]["uid"]): i for i in range(len(cell6))}
    if any(u not in where6 for u in uids):
        raise SystemExit("some of the frame's customers are not in the full-history cell")
    chunks6 = [np.array([where6[str(cell5[i]["uid"])] for i in ch]) for ch in chunks]
    offs = [np.array([align(cell5, cell6, i5, i6) for i5, i6 in zip(c5, c6)])
            for c5, c6 in zip(chunks, chunks6)]
    batches6 = [collate_chunk(cell6, ch) for ch in chunks6]
    owns6 = [first_row_of_campaign(b["campaign"], HOLDOUT_FROM) for b in batches6]
    F6 = []
    for b6, b5, o, L, st in zip(batches6, batches, offs, lens, owns):
        fr, outside = to_frame(real_sequences(b6, start=0), b5, o, L, 1)
        if outside:
            raise SystemExit(f"real rows outside the frame: {outside}")
        F6.append(score(fr, st))
    if not np.array_equal(np.vstack(F6), F_real, equal_nan=True):
        raise SystemExit("the full-history build does not reproduce the frame's real functionals")
    print(f"[full] {len(cell6)} customers in the full-history cell; cropped real functionals "
          f"IDENTICAL to the frame's")

    # ------------------------------------------- the chopped inputs, checked
    chopped = {}
    print("[chop] calendar, state zero, flag on every chopped customer; the cut stops at "
          f"campaign {val_from}")
    lens6 = [np.array([item_len(cell6, i) for i in ch]) for ch in chunks6]
    for k in (0,) + DEPTHS:
        per = [chop_batch(cell6, ch, k, val_from) for ch in chunks6]
        owns_z = [first_row_of_campaign(bz["campaign"], HOLDOUT_FROM) for bz, _, _ in per]
        bad = []
        for (bz, s, lz), b6, L6, oz, o6 in zip(per, batches6, lens6, owns_z, owns6):
            bad += check_chop(bz, b6, s, lz, L6, oz, o6)
            if k == 0 and not all(torch.equal(bz[key], b6[key]) for key in ROWS):
                bad.append("k = 0 changed the batch")
        if bad:
            raise SystemExit(f"k = {k}: chopped rows are not the full rows: {bad[:5]}")
        s_all = np.concatenate([s for _, s, _ in per])
        cens = int(sum(int(bz["censored"].sum()) for bz, _, _ in per))
        in_frame = int(sum(((s - o) > 0).sum() for (_, s, _), o in zip(per, offs)))
        chopped[k] = (per, owns_z)
        print(f"  k = {k}: {int((s_all > 0).sum())} of {len(s_all)} customers chopped, "
              f"{cens} flagged; rows removed: mean {s_all.mean():.0f}, max {int(s_all.max())}; "
              f"{in_frame} customers lose rows inside the frame")

    # ------------------------------------------------------------ the arms
    out = {"cell": CELL, "head": head, "smoke": a.smoke, "n_users": int(len(pick)),
           "n_rep": n_rep, "chunk": chunk, "max_events": MAX_EVENTS,
           "customer_sha256": uid_sha, "functionals": sm.functional_names(),
           "null_band": {"n_boot": n_boot, "seed": NULL_SEED, "q95": q95,
                         "mean": float(null.mean()), "sd": float(null.std(ddof=1)),
                         "r36i_q95": PREFIX_Q95, "reproduces_r36i": same_as_prefix},
           "real": {"shares": (real_counts / real_counts.sum()).tolist(),
                    "holdout_rows": int(real_counts.sum())},
           "checkpoints": {f"{t}_s{s}": str(p) for (t, s), p in paths.items()},
           "arms": {}, "problems": []}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw = {"real": F_real}
    npz = out_path.with_name(out_path.stem + "_functionals.npz")

    def save() -> None:
        out_path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
        np.savez_compressed(npz, functionals=np.array(sm.functional_names()), n_rep=n_rep,
                            customer_sha256=uid_sha, **raw)

    def generate(model, k: int, ci: int, sd: int, tf: bool):
        (bz, s, lz), oz = chopped[k][0][ci], chopped[k][1][ci]
        r = rollout(model, bz, env, vocab, start=int(oz.min()), n_rep=n_rep, seed=1000 + sd,
                    device=a.device, teacher_forced=tf,
                    pity_from=(batches6[ci], int(owns6[ci].min())))
        full, bad = unchop(r, batches6[ci], s, lz, n_rep)
        ig = integrity(full, batches6[ci], owns6[ci], n_rep, tf)
        fr, outside = to_frame(full, batches[ci], offs[ci], lens[ci], n_rep)
        return fr, ig, outside + bad

    print("\n[model arms]")
    for name, tag, k, tf in arms():
        if a.smoke and k not in (0, 2):
            continue
        for sd in seeds:
            model, _ = load_checkpoint(paths[(tag, sd)], device=a.device, legacy_camp_bias=False)
            if name == "fr_F_k0" and sd == seeds[0]:
                # the chop path at k = 0 must be R43's path, draw for draw
                r43 = rollout(model, batches6[0], env, vocab, start=int(owns6[0].min()),
                              n_rep=n_rep, seed=1000 + sd, device=a.device)
                full, bad = unchop(rollout(
                    model, chopped[0][0][0][0], env, vocab, start=int(chopped[0][1][0].min()),
                    n_rep=n_rep, seed=1000 + sd, device=a.device,
                    pity_from=(batches6[0], int(owns6[0].min()))),
                    batches6[0], chopped[0][0][0][1], chopped[0][0][0][2], n_rep)
                # compared as scored, in the frame: the plain rollout also writes the
                # last outcome into the padding row after a history, which unchop drops
                fa, oa = to_frame(r43, batches[0], offs[0], lens[0], n_rep)
                fb, ob = to_frame(full, batches[0], offs[0], lens[0], n_rep)
                same = (not bad and not oa and not ob
                        and np.array_equal(full["decisions"], r43["decisions"])
                        and all(np.array_equal(fa[key], fb[key]) for key in fa))
                print(f"  [gate] k = 0 through the chop path = R43's path on chunk 0: "
                      f"{'IDENTICAL' if same else 'DIFFERENT'}")
                if not same:
                    raise SystemExit("the chop path at k = 0 is not R43's rollout -- refusing")
            t0 = time.time()
            F, cnt = [], np.zeros(9, dtype=np.int64)
            integ = {"rows_alive": 0, "rows_mismatched": 0, "real_rows_altered": 0,
                     "rows_outside_frame": 0}
            for ci, st in enumerate(owns):
                r, ig, outside = generate(model, k, ci, sd, tf)
                F.append(score(r, st))
                cnt += pooled_counts(r["decisions"], r["alive"])
                for key, v in ig.items():
                    integ[key] += v
                integ["rows_outside_frame"] += outside
            raw[f"{name}__{sd}"] = np.vstack(F)
            rec = evaluate(Zr, scaler(raw[f"{name}__{sd}"]), n_boot)
            rec.update(shares=(cnt / max(cnt.sum(), 1)).tolist(), integrity=integ,
                       seconds=round(time.time() - t0), tag=tag, depth=k, teacher_forced=tf)
            if rec["null_q95"] != q95:
                out["problems"].append(f"{name} {sd}: band {rec['null_q95']} is not {q95}")
            if integ["rows_mismatched"] or integ["real_rows_altered"] or integ["rows_outside_frame"]:
                out["problems"].append(f"{name} {sd}: integrity {integ}")
            if integ["rows_alive"] != n_rep * int(real_counts.sum()):
                out["problems"].append(f"{name} {sd}: {integ['rows_alive']} live rows, "
                                       f"expected {n_rep * int(real_counts.sum())}")
            out["arms"].setdefault(name, {"runs": {}})["runs"][str(sd)] = rec
            lo, hi = rec["auc_ci"]
            print(f"  {name:<10} {sd:>2}  energy {rec['energy']:+.3f}  z {rec['z']:>6.1f}  "
                  f"{'INSIDE ' if rec['inside_band'] else 'outside'}  AUC {rec['auc']:.3f} "
                  f"[{lo:.3f}, {hi:.3f}]  NotBuy {rec['shares'][8]:.3f}  "
                  f"top {', '.join(rec['top_functionals'][:3])}  "
                  f"mismatch {integ['rows_mismatched']} altered {integ['real_rows_altered']} "
                  f"outside {integ['rows_outside_frame']}  ({rec['seconds']}s)", flush=True)
            save()
            del model
            if a.device == "cuda":
                torch.cuda.empty_cache()

    # ---------------------------------------------------------------- report
    for v in out["arms"].values():
        v["summary"] = summarise(v["runs"])
    save()
    print(f"\n=== R45 step 2: {len(pick)} customers x {n_rep} replicates, band q95 {q95:.4f} ===")
    print(f"{'arm':<10}{'n':>3}{'energy':>9}{'sd':>8}{'min':>8}{'max':>8}{'in':>4}{'AUC':>7}"
          f"{'NotBuy':>8}  top functionals")
    for nm, v in out["arms"].items():
        s = v["summary"]
        print(f"{nm:<10}{s['n']:>3}{s['energy_mean']:>+9.3f}{s['energy_sd']:>8.3f}"
              f"{s['energy_min']:>+8.3f}{s['energy_max']:>+8.3f}{s['inside_band']:>4}"
              f"{s['auc_mean']:>7.3f}{s['shares_mean'][8]:>8.3f}  "
              f"{', '.join(s['top_functionals'][:3])}")
    print(f"{'real':<10}{'':>47}{out['real']['shares'][8]:>8.3f}")
    if a.smoke:
        print("\n(smoke run: wiring only; the numbers mean nothing)")
    if out["problems"]:
        print("\nPROBLEMS:")
        for p in out["problems"]:
            print("  " + p)
    print(f"\nwrote {out_path} and {npz} ({len(raw) - 1} arm runs + real)")
    if out["problems"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
