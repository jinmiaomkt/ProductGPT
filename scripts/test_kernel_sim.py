"""
Correctness checks for the R34 simulation harness. Run before trusting any
recovery number it prints.

    python scripts/test_kernel_sim.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "simulation"))
from kernel_sim import (NOTBUY, KernelModel, SimConfig, fit, kernel_metrics,  # noqa: E402
                        offdiag, simulate, spearman, true_kernel)

FAILED = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def main() -> None:
    print("R34 simulation harness -- correctness checks\n")

    cfg = SimConfig(n_products=12, n_elements=3, n_customers=200, n_occasions=60,
                    campaign_len=10, seed=1)
    d = simulate(cfg)

    print("1. data generating process")
    check("decisions in range", d.decisions.min() >= 0 and d.decisions.max() <= NOTBUY,
          f"min {d.decisions.min()} max {d.decisions.max()}")
    check("acquired ids valid", bool(((d.acquired == -1) |
                                      ((d.acquired >= 0) & (d.acquired < cfg.n_products))).all()))
    n_items = (d.acquired >= 0).sum(axis=2)
    one = (d.decisions % 2 == 0) & (d.decisions != NOTBUY)
    ten = (d.decisions % 2 == 1)
    check("1-draw rows acquire exactly 1", bool((n_items[one] == 1).all()))
    check("10-draw rows acquire exactly 10", bool((n_items[ten] == 10).all()))
    check("NotBuy rows acquire nothing", bool((n_items[d.decisions == NOTBUY] == 0).all()))
    share = (d.decisions == NOTBUY).mean()
    check("NotBuy share is plausible", 0.15 < share < 0.95, f"{share:.3f}")
    check("kappa rows sum to 1", np.allclose(d.kappa_true.sum(1), 1.0))

    print("\n2. the state recursion (brute force vs the recursion the DGP uses)")
    rho, tw = cfg.rho(), np.asarray(cfg.tier_weights)
    i = 7
    R_brute = np.zeros((cfg.n_occasions, cfg.n_products))
    owned = np.zeros(cfg.n_products, dtype=int)
    weights = []                                   # (tau, product, g)
    for t in range(cfg.n_occasions):
        acc = np.zeros(cfg.n_products)
        for tau, p, g in weights:
            acc[p] += g * rho ** (t - tau)
        R_brute[t] = acc
        for p in d.acquired[i, t][d.acquired[i, t] >= 0]:
            weights.append((t, int(p), tw[min(owned[p], len(tw) - 1)]))
            owned[p] += 1
    R_rec = np.zeros((cfg.n_occasions, cfg.n_products))
    R, owned2 = np.zeros(cfg.n_products), np.zeros(cfg.n_products, dtype=int)
    for t in range(cfg.n_occasions):
        R_rec[t] = R
        for p in d.acquired[i, t][d.acquired[i, t] >= 0]:
            R[p] += tw[min(owned2[p], len(tw) - 1)]
            owned2[p] += 1
        R *= rho
    err = np.abs(R_brute - R_rec).max()
    check("recursion equals the explicit kernel sum", err < 1e-9, f"max abs err {err:.2e}")

    print("\n3. the estimator's state matches the DGP's")
    model = KernelModel(cfg.n_products, d.elem_of, cfg.n_elements, kernel="identity",
                        n_tiers=len(cfg.tier_weights))
    with torch.no_grad():
        model.log_hl.fill_(float(np.log(cfg.half_life)))
        # g = [1, .6, .35] -> sigmoid steps .6 and .35/.6
        model.tier_raw.copy_(torch.log(torch.tensor([0.6 / 0.4, (0.35 / 0.6) / (1 - 0.35 / 0.6)])))
    g_hat = model.tier().detach().numpy()
    check("tier weights parameterise correctly", np.allclose(g_hat, tw, atol=1e-6),
          f"{np.round(g_hat, 4).tolist()}")
    # replay the estimator's own state update and compare with the numpy DGP
    with torch.no_grad():
        rho_t = torch.exp(-np.log(2.0) / torch.exp(model.log_hl))
        g = model.tier()
        Rm = torch.zeros(1, cfg.n_products)
        own = torch.zeros(1, cfg.n_products, dtype=torch.long)
        acq = torch.as_tensor(d.acquired[i:i + 1])
        seq = []
        for t in range(cfg.n_occasions):
            seq.append(Rm.clone().numpy()[0])
            Rm, own = model.advance(Rm, own, acq[:, t], g, rho_t)
    err2 = np.abs(np.stack(seq) - R_rec).max()
    check("estimator state == DGP state", err2 < 1e-5, f"max abs err {err2:.2e}")
    dup = int((np.diff(np.sort(d.acquired[i][d.acquired[i] >= 0].reshape(-1))) == 0).sum())
    check("the panel contains duplicate copies (the case that broke)", dup > 0,
          f"{dup} duplicate acquisitions for this customer")

    print("\n4. satiation is visible in the simulated data")
    hi = SimConfig(**{**cfg.__dict__, "lam": 3.0, "seed": 2})
    lo = SimConfig(**{**cfg.__dict__, "lam": 0.0, "seed": 2})
    s_hi, s_lo = simulate(hi), simulate(lo)
    nb_hi = (s_hi.decisions == NOTBUY).mean()
    nb_lo = (s_lo.decisions == NOTBUY).mean()
    check("stronger satiation => more NotBuy", nb_hi > nb_lo, f"{nb_hi:.3f} vs {nb_lo:.3f}")
    ident = SimConfig(**{**cfg.__dict__, "kappa_identity": True})
    check("kappa_identity gives exactly I", np.allclose(true_kernel(ident, s_hi.elem_of),
                                                        np.eye(cfg.n_products)))

    print("\n5. metrics behave")
    k = d.kappa_true
    m_self = kernel_metrics(k, k, d.elem_of)
    check("perfect recovery scores 1", abs(m_self["spearman_offdiag"] - 1.0) < 1e-9,
          f"spearman {m_self['spearman_offdiag']:.4f}")
    check("element lift > 1 on the true kernel", m_self["same_elem_lift"] > 2,
          f"lift {m_self['same_elem_lift']:.2f}")
    rng = np.random.default_rng(0)
    noise = rng.random(k.shape)
    noise /= noise.sum(1, keepdims=True)
    m_noise = kernel_metrics(noise, k, d.elem_of)
    check("noise scores near 0", abs(m_noise["spearman_offdiag"]) < 0.15,
          f"spearman {m_noise['spearman_offdiag']:.4f}")
    check("noise finds no element structure", 0.7 < m_noise["same_elem_lift"] < 1.4,
          f"lift {m_noise['same_elem_lift']:.2f}")

    print("\n6. fitting reduces the loss")
    res = fit(d, kernel="learned", epochs=40, batch=128, seed=0, device="cpu")
    first, last = np.mean(res["history"][:5]), np.mean(res["history"][-5:])
    check("Adam reduces NLL", last < first, f"{first:.4f} -> {last:.4f}")
    check("kappa-hat rows sum to 1", np.allclose(res["kappa_hat"].sum(1), 1.0, atol=1e-5))

    print()
    if FAILED:
        print(f"{len(FAILED)} CHECK(S) FAILED: {', '.join(FAILED)}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
