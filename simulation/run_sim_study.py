"""
R34: run the four identification experiments and write the results.

    python simulation/run_sim_study.py --experiment s1          # recovery + power curve
    python simulation/run_sim_study.py --experiment s2          # FALSE POSITIVE test
    python simulation/run_sim_study.py --experiment s3          # separability
    python simulation/run_sim_study.py --experiment s4          # offer schedule
    python simulation/run_sim_study.py --experiment all --quick # smoke run, minutes

Results go to results/r34/<experiment>.json (git-ignored). Nothing is read from
PRODUCTGPT_DATA: every panel here is synthetic.

The gate for R33 (pre-registered in EXPERIMENTS.md):
  S1 must recover kappa -- Spearman > 0.6 on the off-diagonal at our N; AND
  S2's false-positive kernel must be absent, or removed by a customer fixed
  effect. Otherwise the estimated kernel is reported as descriptive, not
  structural.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_sim import SimConfig, fit, kernel_metrics, simulate  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "r34"


def base_cfg(quick: bool) -> SimConfig:
    return SimConfig(n_products=24, n_elements=6, n_customers=400 if quick else 1000,
                     n_occasions=120 if quick else 300, campaign_len=15, taste_sd=0.0)


EPOCHS: dict = {}          # set from --epochs


def run_fit(data, quick: bool, **kw) -> dict:
    ep = EPOCHS.get("n") or (60 if quick else 250)
    return fit(data, epochs=ep, batch=128 if quick else 256, **kw)


def report(name: str, rows: list[dict], keys: list[str]) -> None:
    print(f"\n{name}")
    w = max(len(str(r.get(keys[0], ''))) for r in rows) if rows else 10
    print("  " + "  ".join(k.rjust(max(10, len(k))) for k in keys))
    for r in rows:
        print("  " + "  ".join(
            (f"{r[k]:>{max(10, len(k))}.3f}" if isinstance(r.get(k), float)
             else str(r.get(k, "")).rjust(max(10, len(k)))) for k in keys))
    del w


# --------------------------------------------------------------------------
def s1_recovery(quick: bool) -> dict:
    """Can the kernel and the half-life be recovered, and with how many customers?"""
    cfg0 = base_cfg(quick)
    sizes = [200, 400] if quick else [250, 500, 1000, 2000]
    rows = []
    for n in sizes:
        cfg = replace(cfg0, n_customers=n, seed=100 + n)
        d = simulate(cfg)
        r = run_fit(d, quick, kernel="learned", seed=1)
        m = kernel_metrics(r["kappa_hat"], d.kappa_true, d.elem_of)
        rows.append({"n_customers": n, "occasions": cfg.n_occasions,
                     "spearman": m["spearman_offdiag"], "elem_lift": m["same_elem_lift"],
                     "procrustes": m["procrustes"], "half_life_hat": r["half_life_hat"],
                     "half_life_true": cfg.half_life, "lam_hat": r["lam_hat"],
                     "lam_true": cfg.lam, "nll": r["nll"]})
        print(f"  n={n:>5}  spearman {m['spearman_offdiag']:+.3f}  "
              f"lift {m['same_elem_lift']:.2f}  half-life {r['half_life_hat']:.1f} "
              f"(true {cfg.half_life})")
    report("S1 recovery (power curve)", rows,
           ["n_customers", "spearman", "elem_lift", "procrustes", "half_life_hat", "lam_hat"])
    return {"rows": rows}


def s2_false_positive(quick: bool) -> dict:
    """THE test. No substitution in the DGP, but customers differ in taste.
    Does a learned kernel invent off-diagonal structure, and does a customer
    fixed effect remove it?"""
    cfg0 = base_cfg(quick)
    rows = []
    for taste_sd, label in ((0.0, "homogeneous (reference)"), (1.0, "heterogeneous")):
        cfg = replace(cfg0, kappa_identity=True, taste_sd=taste_sd, seed=200)
        d = simulate(cfg)
        for fe in (False, True):
            r = run_fit(d, quick, kernel="learned", customer_fe=fe, seed=1)
            m = kernel_metrics(r["kappa_hat"], d.kappa_true, d.elem_of)
            rows.append({"dgp": label, "customer_fe": str(fe), "elem_lift": m["same_elem_lift"],
                         "offdiag_mass": m["offdiag_mass"], "diag_mean": m["diag_mean"],
                         "nll": r["nll"]})
            print(f"  {label:<26} FE={str(fe):<5}  element lift {m['same_elem_lift']:.3f}  "
                  f"off-diagonal mass {m['offdiag_mass']:.3f}")
    report("S2 false positive (true kernel is the identity; lift 1.0 = no invented structure)",
           rows, ["dgp", "customer_fe", "elem_lift", "offdiag_mass", "diag_mean"])
    return {"rows": rows}


def s3_separability(quick: bool) -> dict:
    """Substitution, homogeneous customers. A heterogeneity-only model must not
    be able to mimic it: compare on a held-out later period."""
    cfg = replace(base_cfg(quick), taste_sd=0.0, seed=300)
    d = simulate(cfg)
    split = int(0.7 * cfg.n_occasions)
    rows = []
    for kernel, fe, label in (("learned", False, "kernel, no FE"),
                              ("identity", True, "identity kernel + customer FE"),
                              ("identity", False, "identity kernel, no FE")):
        r = run_fit(d, quick, kernel=kernel, customer_fe=fe, t_hi=split, seed=1)
        rows.append({"model": label, "train_nll": r["nll"], "held_out_nll": r["held_out_nll"]})
        print(f"  {label:<30} held-out NLL {r['held_out_nll']:.4f}")
    report("S3 separability (held-out occasions; lower is better)", rows,
           ["model", "train_nll", "held_out_nll"])
    return {"rows": rows, "split_occasion": split}


def s4_schedule(quick: bool) -> dict:
    """Does identification depend on the banner rotation? Compare the real
    campaign-style rotation with offers redrawn every occasion."""
    grid = ([(False, 15, "campaign len 15"), (True, 15, "randomised offers")] if quick else
            [(False, 30, "campaign len 30"), (False, 15, "campaign len 15"),
             (False, 5, "campaign len 5"), (True, 15, "randomised offers")])
    rows = []
    for randomized, clen, label in grid:
        cfg = replace(base_cfg(quick), randomized_schedule=randomized,
                      campaign_len=clen, seed=400)
        d = simulate(cfg)
        r = run_fit(d, quick, kernel="learned", seed=1)
        m = kernel_metrics(r["kappa_hat"], d.kappa_true, d.elem_of)
        rows.append({"schedule": label, "campaign_len": clen,
                     "spearman": m["spearman_offdiag"],
                     "elem_lift": m["same_elem_lift"], "half_life_hat": r["half_life_hat"]})
        print(f"  {label:<20} spearman {m['spearman_offdiag']:+.3f}  "
              f"lift {m['same_elem_lift']:.2f}", flush=True)
    report("S4 offer schedule (how fast must offers rotate for kappa to be identified?)",
           rows, ["schedule", "spearman", "elem_lift", "half_life_hat"])
    return {"rows": rows}


def s0_diagnostic(quick: bool) -> dict:
    """Before the four experiments: is a weak kernel estimate a training-budget
    problem or an identification problem?

    Fits one panel for a long budget, recording recovery as it goes, and
    compares three likelihoods at the end:
        learned kernel   what the estimator can reach
        TRUE kappa       the oracle -- the best any kernel could do here
        identity kappa   no substitution at all
    If the oracle barely beats identity, the data carry almost no information
    about kappa and no amount of training will recover it. That is an
    identification finding, and it is the answer R34 exists to produce.
    """
    import torch                                            # local: keeps the CLI light
    from kernel_sim import KernelModel

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    budget = EPOCHS.get("n") or (150 if quick else 1200)

    def anchors_for(data, kappa_true=None, epochs=None):
        """NLL with kappa frozen at the identity, or at the truth."""
        out = {}
        off = torch.as_tensor(data.offers, device=dev)
        de = torch.as_tensor(data.decisions, device=dev)
        ac = torch.as_tensor(data.acquired, device=dev)
        for label in ("identity", "true"):
            torch.manual_seed(0)
            mod = KernelModel(data.cfg.n_products, data.elem_of, data.cfg.n_elements,
                              kernel="identity").to(dev)
            if label == "true":
                tk = torch.as_tensor(data.kappa_true, dtype=torch.float32, device=dev)
                mod.kappa = lambda k=tk: k
            opt = torch.optim.Adam(mod.parameters(), lr=0.05)
            r2 = np.random.default_rng(0)
            vals = []
            for _ in range(epochs or max(1, budget // 4)):
                idx = torch.as_tensor(r2.choice(data.cfg.n_customers,
                                                min(256, data.cfg.n_customers),
                                                replace=False), device=dev)
                loss = mod.nll(off, de[idx], ac[idx], uid=idx, t_lo=20)
                opt.zero_grad()
                loss.backward()
                opt.step()
                vals.append(float(loss.detach()))
            out[label] = float(np.mean(vals[-10:]))
        return out

    # ---- part A: how much is the kernel worth, as satiation gets stronger? ----
    # If the oracle (true kappa) cannot beat the identity kernel, the panel
    # carries no information about kappa and no estimator can recover it. This
    # is the detection threshold, and it is a property of the DESIGN, not of us.
    print("  A. oracle gap vs satiation strength (identity NLL - true-kappa NLL)")
    lam_rows = []
    for lam in ([1.0, 3.0] if quick else [0.5, 1.0, 2.0, 4.0]):
        cfg_l = replace(base_cfg(quick), lam=lam, seed=100)
        d_l = simulate(cfg_l)
        an = anchors_for(d_l)
        gap = an["identity"] - an["true"]
        nb = float((d_l.decisions == 8).mean())
        lam_rows.append({"lam": lam, "identity_nll": an["identity"], "true_nll": an["true"],
                         "oracle_gap": gap, "notbuy_share": nb})
        print(f"     lam {lam:>4}  identity {an['identity']:.4f}  true {an['true']:.4f}  "
              f"gap {gap:+.4f}  (NotBuy share {nb:.2f})", flush=True)

    best = max(lam_rows, key=lambda r: r["oracle_gap"])
    print(f"\n  B. training curve at lam={best['lam']} (largest oracle gap, "
          f"{best['oracle_gap']:+.4f})")
    cfg = replace(base_cfg(quick), lam=best["lam"], seed=100)
    d = simulate(cfg)
    offers = torch.as_tensor(d.offers, device=dev)
    dec = torch.as_tensor(d.decisions, device=dev)
    acq = torch.as_tensor(d.acquired, device=dev)
    curve, rng = [], np.random.default_rng(0)

    torch.manual_seed(0)
    model = KernelModel(cfg.n_products, d.elem_of, cfg.n_elements, kernel="learned").to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for ep in range(budget + 1):
        idx = torch.as_tensor(rng.choice(cfg.n_customers, min(256, cfg.n_customers),
                                         replace=False), device=dev)
        loss = model.nll(offers, dec[idx], acq[idx], uid=idx, t_lo=20)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if ep % max(1, budget // 12) == 0 or ep == budget:
            with torch.no_grad():
                m = kernel_metrics(model.kappa().cpu().numpy(), d.kappa_true, d.elem_of)
            curve.append({"epoch": ep, "nll": float(loss.detach()),
                          "spearman": m["spearman_offdiag"], "elem_lift": m["same_elem_lift"],
                          "half_life_hat": float(torch.exp(model.log_hl))})
            print(f"  ep {ep:>5}  nll {float(loss.detach()):.4f}  "
                  f"spearman {m['spearman_offdiag']:+.3f}  lift {m['same_elem_lift']:.2f}  "
                  f"half-life {float(torch.exp(model.log_hl)):.1f}", flush=True)

    # equal budget to the learned fit, or "learned beats the oracle" would just
    # mean the oracle was trained less
    anchors = anchors_for(d, epochs=budget)
    learned = float(np.mean([c["nll"] for c in curve[-2:]]))
    anchors["learned"] = learned
    gap = anchors["identity"] - anchors["true"]
    print(f"\n  anchors at lam={best['lam']}: identity {anchors['identity']:.4f}  "
          f"true {anchors['true']:.4f}  learned {learned:.4f}")
    print(f"  oracle gap (identity - true kappa) = {gap:+.4f} nats: how much the "
          f"kernel is worth here at all")
    print(f"  learned - true = {learned - anchors['true']:+.4f}: how much of it the "
          f"estimator captured")
    report("S0 oracle gap by satiation strength", lam_rows,
           ["lam", "identity_nll", "true_nll", "oracle_gap", "notbuy_share"])
    return {"lam_sweep": lam_rows, "curve": curve, "anchors": anchors,
            "oracle_gap": gap, "lam_used": best["lam"]}


def s5_parametric(quick: bool) -> dict:
    """THE follow-up to s4. The free kernel failed because it asks the data for
    P-squared similarities it does not contain. A PARAMETRIC kernel asks for two
    numbers instead:

        kappa(j,p)  proportional to  exp(theta_self * 1[j=p] + theta_attr * 1[same element])

    Same substitution story, same marketing content (McAlister's attribute
    satiation), 2 free parameters instead of 24x24. If the design supports that,
    the answer to "can we model substitution?" is yes-with-structure rather than
    no. Truth: theta_self = kappa_self, theta_attr = kappa_same_elem.

    Run under BOTH schedules, so the parametric kernel is judged where the free
    one failed (campaign rotation) and where it succeeded (randomised).
    """
    rows = []
    for randomized, label in ((False, "campaign rotation"), (True, "randomised offers")):
        cfg = replace(base_cfg(quick), randomized_schedule=randomized, campaign_len=15,
                      seed=500)
        d = simulate(cfg)
        split = int(0.7 * cfg.n_occasions)
        for kern in ("identity", "attr", "learned"):
            r = run_fit(d, quick, kernel=kern, t_hi=split, seed=1)
            m = kernel_metrics(r["kappa_hat"], d.kappa_true, d.elem_of)
            row = {"schedule": label, "kernel": kern, "free_params":
                   {"identity": 0, "attr": 2, "learned": cfg.n_products ** 2}[kern],
                   "held_out_nll": r["held_out_nll"], "spearman": m["spearman_offdiag"],
                   "elem_lift": m["same_elem_lift"], "half_life_hat": r["half_life_hat"],
                   "tier_hat": [round(x, 3) for x in r["tier_hat"]]}
            if kern == "attr":
                row["theta_self_hat"], row["theta_self_true"] = r["attr_self"], cfg.kappa_self
                row["theta_attr_hat"], row["theta_attr_true"] = r["attr_same"], cfg.kappa_same_elem
                print(f"  {label:<20} attr    theta_self {r['attr_self']:.2f} "
                      f"(true {cfg.kappa_self})  theta_attr {r['attr_same']:.2f} "
                      f"(true {cfg.kappa_same_elem})  held-out {r['held_out_nll']:.4f}",
                      flush=True)
            else:
                print(f"  {label:<20} {kern:<7} held-out {r['held_out_nll']:.4f}  "
                      f"spearman {m['spearman_offdiag']:+.3f}", flush=True)
            rows.append(row)
    report("S5 parametric kernel vs free kernel (held-out NLL; lower is better)", rows,
           ["schedule", "kernel", "free_params", "held_out_nll", "spearman", "elem_lift"])
    print("  true tier weights:", list(base_cfg(quick).tier_weights),
          "-- compare with tier_hat in the json (are duplicate weights identified?)")
    return {"rows": rows, "tier_true": list(base_cfg(quick).tier_weights)}


def s6_real_calendar(quick: bool) -> dict:
    """S6, two questions the earlier experiments left open.

    A. HOW STRONG must substitution be for OUR calendar to detect it? The runs
       so far fixed the true kernel's shape and varied the schedule. Here the
       schedule is fixed at our real one (campaigns of 45 occasions) and the
       true same-attribute weight is swept. The estimator is the parametric
       kernel, which is the only one with a chance.
    B. Do CAMPAIGN-LEVEL demand shocks corrupt the decay? Time since acquisition
       and time since the campaign ended advance together, so a calendar shock
       can masquerade as memory decay. Fit with and without campaign fixed
       effects and compare the recovered half-life.
    """
    rows = []
    print("  A. how strong must substitution be, under OUR 45-occasion calendar?")
    for true_attr in ([0.5, 3.0] if quick else [0.5, 1.5, 3.0, 5.0]):
        cfg = replace(base_cfg(quick), campaign_len=45, kappa_same_elem=true_attr, seed=600)
        d = simulate(cfg)
        r = run_fit(d, quick, kernel="attr", seed=1)
        rows.append({"experiment": "A substitution strength", "true_theta_attr": true_attr,
                     "theta_attr_hat": r["attr_same"], "theta_self_hat": r["attr_self"],
                     "ratio": r["attr_same"] / true_attr if true_attr else float("nan"),
                     "half_life_hat": r["half_life_hat"]})
        print(f"     true theta_attr {true_attr:>4}  ->  estimated {r['attr_same']:>6.2f}  "
              f"({r['attr_same'] / true_attr:>5.0%} of truth)", flush=True)

    print("  B. do campaign shocks corrupt the decay?")
    for shock in ([0.0, 0.5] if quick else [0.0, 0.3, 0.6]):
        cfg = replace(base_cfg(quick), campaign_len=45, camp_shock_sd=shock, seed=601)
        d = simulate(cfg)
        for camp_fe in (False, True):
            r = run_fit(d, quick, kernel="attr", campaign_fe=camp_fe, seed=1)
            rows.append({"experiment": "B campaign shocks", "shock_sd": shock,
                         "campaign_fe": str(camp_fe), "half_life_hat": r["half_life_hat"],
                         "half_life_true": cfg.half_life,
                         "theta_attr_hat": r["attr_same"]})
            print(f"     shock sd {shock:>4}  campaign FE {str(camp_fe):<5}  "
                  f"half-life {r['half_life_hat']:>5.1f} (true {cfg.half_life})", flush=True)
    report("S6 (A) substitution strength / (B) campaign shocks", rows,
           ["experiment", "true_theta_attr", "theta_attr_hat", "shock_sd", "campaign_fe",
            "half_life_hat"])
    return {"rows": rows}


def s7_heterogeneity(quick: bool) -> dict:
    """S7: are the kernel parameters the same for every customer?

    Everything so far assumed one satiation strength for the whole population.
    Here the DGP gives each customer their own lam_i (log-normal), and a
    HOMOGENEOUS estimator is fitted to it. Two questions:
      - how biased is the single lam-hat relative to the population mean?
      - does unmodelled heterogeneity in the STATE-DEPENDENCE parameter create
        false substitution structure, the way preference heterogeneity did
        in S2?
    A customer fixed effect is included as the standard remedy, to see whether
    it helps when the heterogeneity is in the slope rather than the intercept.
    """
    rows = []
    for sd in ([0.0, 0.6] if quick else [0.0, 0.3, 0.6, 1.0]):
        cfg = replace(base_cfg(quick), campaign_len=45, lam_sd=sd, seed=700)
        d = simulate(cfg)
        pop_mean = float(np.mean(d.lam_true))
        for fe in (False, True):
            r = run_fit(d, quick, kernel="learned", customer_fe=fe, seed=1)
            m = kernel_metrics(r["kappa_hat"], d.kappa_true, d.elem_of)
            rows.append({"lam_sd": sd, "customer_fe": str(fe), "lam_hat": r["lam_hat"],
                         "lam_pop_mean": pop_mean, "elem_lift": m["same_elem_lift"],
                         "offdiag_mass": m["offdiag_mass"], "half_life_hat": r["half_life_hat"]})
            print(f"  lam sd {sd:>4}  FE {str(fe):<5}  lam-hat {r['lam_hat']:>5.2f} "
                  f"(population mean {pop_mean:.2f})  element lift {m['same_elem_lift']:>5.2f}",
                  flush=True)
    report("S7 heterogeneous satiation, homogeneous estimator", rows,
           ["lam_sd", "customer_fe", "lam_hat", "lam_pop_mean", "elem_lift", "offdiag_mass"])
    return {"rows": rows}


EXPERIMENTS = {"s0": s0_diagnostic, "s1": s1_recovery, "s2": s2_false_positive,
               "s3": s3_separability, "s4": s4_schedule, "s5": s5_parametric, "s6": s6_real_calendar,
               "s7": s7_heterogeneity}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=[*EXPERIMENTS, "all"])
    ap.add_argument("--quick", action="store_true", help="small panels, for a smoke run")
    ap.add_argument("--epochs", type=int, default=None, help="override the training budget")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = leave alone)")
    a = ap.parse_args()
    if a.epochs:
        EPOCHS["n"] = a.epochs
    if a.threads:
        import torch
        torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    names = list(EXPERIMENTS) if a.experiment == "all" else [a.experiment]
    for name in names:
        print(f"\n{'=' * 78}\n{name.upper()}  ({'quick' if a.quick else 'full'})\n{'=' * 78}")
        t0 = time.time()
        res = EXPERIMENTS[name](a.quick)
        res["minutes"] = round((time.time() - t0) / 60, 2)
        res["config"] = asdict(base_cfg(a.quick))
        res["quick"] = a.quick
        path = OUT / f"{name}{'_quick' if a.quick else ''}.json"
        path.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
        print(f"\n  {res['minutes']:.2f} min -> {path}")


if __name__ == "__main__":
    main()
