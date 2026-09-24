"""
R34: can the inventory kernel be identified? Data generating process, estimator
and recovery metrics.

THE OBJECT UNDER TEST (R33). A customer's stock of product j at occasion t is

    S_t(j) = sum_{tau < t}  kappa(j, p_tau) * g(copy_tau) * phi(t - tau)

    kappa  row-stochastic product-product similarity: acquiring p satiates j
    phi    decay, here 0.5 ** (elapsed / half_life)
    g      weight of the k-th copy of a product (duplicates are not variety)

Satiation enters utility as -lam * S, so a customer who already holds close
substitutes of what a banner offers is less likely to pull on it.

WHY A SIMULATION. On real data kappa's off-diagonal mass has two possible
sources: genuine substitution (state dependence) or stable taste differences
between customers that a flexible kernel can absorb (heterogeneity). That is
Heckman's initial-conditions problem in modern dress, and Dube, Hitsch & Rossi
(2010) is the standard of proof. Here we know the truth, so we can measure
whether the estimator finds it -- and, in S2, whether it invents it.

The decision space matches the real data: 4 banners x {1 draw, 10 draws} plus
NotBuy = 9 classes, index 8 = NotBuy.

Everything is synthetic. No real customer data is read by this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_DECISIONS = 9
NOTBUY = 8


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
@dataclass
class SimConfig:
    # world
    n_products: int = 44
    n_elements: int = 7            # the attribute that drives substitution
    n_banners: int = 4
    featured_per_banner: int = 2   # one 5-star + one 4-star, as in the game
    # panel
    n_customers: int = 1000
    n_occasions: int = 300
    campaign_len: int = 21         # banners rotate every campaign
    randomized_schedule: bool = False   # S4: break the rotation
    # truth
    half_life: float = 30.0        # occasions
    lam: float = 1.0               # satiation strength
    kappa_self: float = 3.0        # logit weight on j == p
    kappa_same_elem: float = 1.5   # logit weight on same element
    kappa_identity: bool = False   # S2: exactly no substitution, kappa = I
    tier_weights: tuple = (1.0, 0.6, 0.35)   # 1st, 2nd, 3rd+ copy
    taste_sd: float = 0.0          # sd of customer element-level taste (heterogeneity)
    beta: float = 1.0              # sensitivity to banner attractiveness
    c1: float = -1.0               # intercept, 1-draw
    c10: float = -2.5              # intercept, 10-draw
    # gacha
    p_five: float = 0.01
    p_four: float = 0.10
    seed: int = 0

    def rho(self) -> float:
        return float(0.5 ** (1.0 / self.half_life))


@dataclass
class SimData:
    """Everything an estimator may see, plus the truth it must not see."""
    offers: np.ndarray        # (T, n_banners, featured) product ids
    decisions: np.ndarray     # (N, T) int, 0..8
    acquired: np.ndarray      # (N, T, 10) product ids, -1 = empty slot
    elem_of: np.ndarray       # (P,) element of each product
    kappa_true: np.ndarray    # (P, P) row-stochastic
    taste_true: np.ndarray    # (N, n_elements) customer element taste
    base_true: np.ndarray     # (P,) population product taste
    cfg: SimConfig = field(default_factory=SimConfig)


# --------------------------------------------------------------------------
# data generating process
# --------------------------------------------------------------------------
def true_kernel(cfg: SimConfig, elem_of: np.ndarray) -> np.ndarray:
    """Row-stochastic kappa: strong self-weight, moderate same-element weight."""
    if cfg.kappa_identity:
        return np.eye(len(elem_of))
    same = (elem_of[:, None] == elem_of[None, :]).astype(float)
    logits = cfg.kappa_self * np.eye(len(elem_of)) + cfg.kappa_same_elem * same
    logits -= logits.max(axis=1, keepdims=True)
    k = np.exp(logits)
    return k / k.sum(axis=1, keepdims=True)


def build_schedule(cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    """(T, n_banners, featured) product ids.

    Default: products rotate through banners campaign by campaign, the way a
    live-service calendar works -- that rotation is the exogenous variation the
    kernel is identified from. With randomized_schedule, offers are redrawn
    every occasion (S4's comparison).
    """
    T, B, F_ = cfg.n_occasions, cfg.n_banners, cfg.featured_per_banner
    offers = np.zeros((T, B, F_), dtype=np.int64)
    if cfg.randomized_schedule:
        for t in range(T):
            pick = rng.choice(cfg.n_products, size=B * F_, replace=False)
            offers[t] = pick.reshape(B, F_)
        return offers
    n_camp = int(np.ceil(T / cfg.campaign_len))
    order = rng.permutation(cfg.n_products)
    cursor = 0
    for c in range(n_camp):
        pick = np.array([order[(cursor + i) % cfg.n_products] for i in range(B * F_)])
        cursor += B * F_
        lo, hi = c * cfg.campaign_len, min((c + 1) * cfg.campaign_len, T)
        offers[lo:hi] = pick.reshape(B, F_)
    return offers


def simulate(cfg: SimConfig) -> SimData:
    """Draw a synthetic panel. Vectorised over customers, looped over time."""
    rng = np.random.default_rng(cfg.seed)
    P, N, T, B = cfg.n_products, cfg.n_customers, cfg.n_occasions, cfg.n_banners
    elem_of = np.arange(P) % cfg.n_elements
    kappa = true_kernel(cfg, elem_of)
    offers = build_schedule(cfg, rng)

    base = rng.normal(0.0, 1.0, size=P)
    taste = rng.normal(0.0, cfg.taste_sd, size=(N, cfg.n_elements)) if cfg.taste_sd > 0 \
        else np.zeros((N, cfg.n_elements))
    theta = base[None, :] + taste[:, elem_of]                    # (N,P) customer taste

    # 3-star pool: everything not featured this campaign is a possible filler.
    rho = cfg.rho()
    R = np.zeros((N, P))          # decayed, g-weighted acquisition history
    owned = np.zeros((N, P), dtype=np.int64)
    decisions = np.zeros((N, T), dtype=np.int64)
    acquired = np.full((N, T, 10), -1, dtype=np.int64)
    tw = np.asarray(cfg.tier_weights, dtype=float)

    for t in range(T):
        stock = R @ kappa.T                                      # (N,P)
        util = np.zeros((N, N_DECISIONS))
        attract = np.zeros((N, B))
        for b in range(B):
            idx = offers[t, b]                                   # (F,)
            attract[:, b] = (theta[:, idx] - cfg.lam * stock[:, idx]).mean(axis=1)
            util[:, 2 * b] = cfg.c1 + cfg.beta * attract[:, b]
            util[:, 2 * b + 1] = cfg.c10 + cfg.beta * attract[:, b]
        util[:, NOTBUY] = 0.0
        gumbel = rng.gumbel(size=(N, N_DECISIONS))
        choice = np.argmax(util + gumbel, axis=1)
        decisions[:, t] = choice

        buying = choice != NOTBUY
        if buying.any():
            who = np.flatnonzero(buying)
            banner = choice[who] // 2
            n_draws = np.where(choice[who] % 2 == 0, 1, 10)
            for slot in range(10):
                active = who[n_draws > slot]
                if active.size == 0:
                    break
                b_a = banner[n_draws > slot]
                u = rng.random(active.size)
                # rarity: featured 5-star, featured 4-star, else a 3-star filler
                item = np.where(u < cfg.p_five, offers[t, b_a, 0],
                        np.where(u < cfg.p_five + cfg.p_four, offers[t, b_a, -1],
                                 rng.integers(0, P, size=active.size)))
                acquired[active, t, slot] = item
                w = tw[np.minimum(owned[active, item], len(tw) - 1)]
                R[active, item] += w
                owned[active, item] += 1
        R *= rho

    return SimData(offers=offers, decisions=decisions, acquired=acquired, elem_of=elem_of,
                   kappa_true=kappa, taste_true=taste, base_true=base, cfg=cfg)


# --------------------------------------------------------------------------
# estimator
# --------------------------------------------------------------------------
class KernelModel(nn.Module):
    """The R33 stock, estimated. `kernel`:
         identity  kappa = I            (L0, the control)
         attr      kappa fixed by the element table  (L1)
         learned   kappa = softmax(E Wq (E Wk)^T / sqrt(d))  (L2, QKV)
       `customer_fe` adds per-customer element tastes -- the heterogeneity
       control whose effect on kappa-hat is the point of S2.
    """

    def __init__(self, n_products: int, elem_of: np.ndarray, n_elements: int,
                 kernel: str = "learned", d: int = 16, n_tiers: int = 3,
                 customer_fe: bool = False, n_customers: int = 0,
                 learn_decay: bool = True, init_half_life: float = 20.0):
        super().__init__()
        self.P, self.kernel = n_products, kernel
        self.register_buffer("elem_of", torch.as_tensor(elem_of, dtype=torch.long))
        same = (elem_of[:, None] == elem_of[None, :]).astype(np.float32)
        self.register_buffer("same_elem", torch.as_tensor(same))
        if kernel == "learned":
            self.emb = nn.Parameter(torch.randn(n_products, d) * 0.1)
            self.wq = nn.Linear(d, d, bias=False)
            self.wk = nn.Linear(d, d, bias=False)
            self.d = d
        elif kernel == "attr":
            self.attr_self = nn.Parameter(torch.tensor(1.0))
            self.attr_same = nn.Parameter(torch.tensor(0.5))
        elif kernel != "identity":
            raise ValueError(kernel)
        self.log_hl = nn.Parameter(torch.tensor(float(np.log(init_half_life))),
                                   requires_grad=learn_decay)
        self.tier_raw = nn.Parameter(torch.zeros(n_tiers - 1))   # copies 2..n, via sigmoid
        self.log_lam = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.base = nn.Parameter(torch.zeros(n_products))
        self.c1 = nn.Parameter(torch.tensor(-1.0))
        self.c10 = nn.Parameter(torch.tensor(-2.0))
        self.fe = nn.Parameter(torch.zeros(n_customers, n_elements)) if customer_fe else None

    def kappa(self) -> torch.Tensor:
        if self.kernel == "identity":
            return torch.eye(self.P, device=self.base.device)
        if self.kernel == "attr":
            logits = self.attr_self * torch.eye(self.P, device=self.base.device) \
                + self.attr_same * self.same_elem
            return torch.softmax(logits, dim=1)
        q, k = self.wq(self.emb), self.wk(self.emb)
        return torch.softmax(q @ k.T / np.sqrt(self.d), dim=1)

    def tier(self) -> torch.Tensor:
        """g = [1, s1, s1*s2, ...]: weakly decreasing, first copy fixed at 1."""
        steps = torch.sigmoid(self.tier_raw)
        return torch.cat([torch.ones(1, device=steps.device), torch.cumprod(steps, 0)])

    def advance(self, R: torch.Tensor, owned: torch.Tensor, items: torch.Tensor,
                g: torch.Tensor, rho: torch.Tensor):
        """Fold one occasion's acquisitions into the state, then decay it.

        items: (N, slots) product ids, -1 = empty. Copies acquired in the SAME
        10-pull must be weighted 1st, 2nd, 3rd... -- counting them all as the
        holding before the occasion would over-weight duplicates, which is
        exactly what `g` is meant to discount. `rank` is how many earlier slots
        of this row already held the same product.
        """
        valid = items >= 0
        if valid.any():
            same = (items.unsqueeze(2) == items.unsqueeze(1))              # (N,S,S)
            earlier = torch.tril(torch.ones(items.shape[1], items.shape[1],
                                            device=items.device, dtype=torch.bool), -1)
            rank = (same & earlier).sum(dim=2)                             # (N,S)
            rows, slots = torch.nonzero(valid, as_tuple=True)
            ids = items[rows, slots]
            copy = (owned[rows, ids] + rank[rows, slots]).clamp(max=len(g) - 1)
            R = R.index_put((rows, ids), g[copy], accumulate=True)
            owned = owned.index_put((rows, ids), torch.ones_like(ids), accumulate=True)
        return R * rho, owned

    def nll(self, offers: torch.Tensor, decisions: torch.Tensor, acquired: torch.Tensor,
            uid: Optional[torch.Tensor] = None, t_lo: int = 0,
            t_hi: Optional[int] = None) -> torch.Tensor:
        """Mean negative log-likelihood per occasion. Tensors:
             offers    (T,B_banners,F) long      decisions (N,T) long
             acquired  (N,T,10) long, -1 pad     uid       (N,) long for the FE

        The state is always replayed from t=0; only occasions in [t_lo, t_hi)
        enter the loss. That is how the early occasions serve as burn-in, and
        how a later window serves as a held-out period for models that carry a
        customer fixed effect (which cannot be held out across customers).
        """
        dev = self.base.device
        N, T = decisions.shape
        kappa, g, rho = self.kappa(), self.tier(), torch.exp(-np.log(2.0) / torch.exp(self.log_hl))
        lam = F.softplus(self.log_lam)
        theta = self.base[None, :].expand(N, self.P)
        if self.fe is not None and uid is not None:
            theta = theta + self.fe[uid][:, self.elem_of]

        R = torch.zeros(N, self.P, device=dev)
        owned = torch.zeros(N, self.P, device=dev, dtype=torch.long)
        total, n_obs = torch.zeros((), device=dev), 0
        for t in range(T):
            stock = R @ kappa.T
            util = torch.zeros(N, N_DECISIONS, device=dev)
            for b in range(offers.shape[1]):
                idx = offers[t, b]
                a = (theta[:, idx] - lam * stock[:, idx]).mean(dim=1)
                util[:, 2 * b] = self.c1 + self.beta * a
                util[:, 2 * b + 1] = self.c10 + self.beta * a
            if t_lo <= t < (T if t_hi is None else t_hi):
                total = total + F.cross_entropy(util, decisions[:, t], reduction="sum")
                n_obs += N
            # update the state with what was acquired AT t (seen only from t+1)
            R, owned = self.advance(R, owned, acquired[:, t], g, rho)
        return total / max(n_obs, 1)


def fit(data: SimData, kernel: str = "learned", customer_fe: bool = False,
        epochs: int = 150, lr: float = 0.05, batch: int = 256, device: str = "auto",
        burn_in: int = 20, t_hi: Optional[int] = None, seed: int = 0,
        verbose: bool = False) -> dict:
    """Maximum likelihood by Adam over customer minibatches.

    Occasions [burn_in, t_hi) are the training window; with t_hi set, the rest
    of the panel is left for `held_out_nll`.
    """
    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available())
                       else ("cuda" if device == "cuda" else "cpu"))
    torch.manual_seed(seed)
    cfg = data.cfg
    model = KernelModel(cfg.n_products, data.elem_of, cfg.n_elements, kernel=kernel,
                        n_tiers=len(cfg.tier_weights), customer_fe=customer_fe,
                        n_customers=cfg.n_customers).to(dev)
    offers = torch.as_tensor(data.offers, device=dev)
    dec = torch.as_tensor(data.decisions, device=dev)
    acq = torch.as_tensor(data.acquired, device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    N = cfg.n_customers
    hist = []
    for ep in range(epochs):
        idx = torch.as_tensor(rng.choice(N, size=min(batch, N), replace=False), device=dev)
        loss = model.nll(offers, dec[idx], acq[idx], uid=idx, t_lo=burn_in, t_hi=t_hi)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        hist.append(float(loss.detach()))
        if verbose and (ep % 25 == 0 or ep == epochs - 1):
            print(f"    epoch {ep:>4}  nll {float(loss):.4f}  "
                  f"half-life {float(torch.exp(model.log_hl)):.1f}")
    with torch.no_grad():
        out = {
            "nll": float(np.mean(hist[-10:])),
            "kappa_hat": model.kappa().cpu().numpy(),
            "half_life_hat": float(torch.exp(model.log_hl)),
            "lam_hat": float(F.softplus(model.log_lam)),
            "tier_hat": model.tier().cpu().numpy().tolist(),
            "history": hist,
            "attr_self": float(model.attr_self) if kernel == "attr" else None,
            "attr_same": float(model.attr_same) if kernel == "attr" else None,
            "model": model,
            "device": str(dev),
        }
        if t_hi is not None:
            out["held_out_nll"] = held_out_nll(model, data, t_hi, dev)
    return out


def held_out_nll(model: KernelModel, data: SimData, t_lo: int,
                 device: torch.device, chunk: int = 512) -> float:
    """Mean NLL over occasions >= t_lo for every customer, state replayed from 0."""
    offers = torch.as_tensor(data.offers, device=device)
    dec = torch.as_tensor(data.decisions, device=device)
    acq = torch.as_tensor(data.acquired, device=device)
    tot, n = 0.0, 0
    with torch.no_grad():
        for lo in range(0, data.cfg.n_customers, chunk):
            idx = torch.arange(lo, min(lo + chunk, data.cfg.n_customers), device=device)
            v = model.nll(offers, dec[idx], acq[idx], uid=idx, t_lo=t_lo)
            tot += float(v) * len(idx)
            n += len(idx)
    return tot / max(n, 1)


# --------------------------------------------------------------------------
# recovery metrics
# --------------------------------------------------------------------------
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12:
        return float("nan")      # a constant off-diagonal (e.g. the identity) has no ranking
    ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def offdiag(m: np.ndarray) -> np.ndarray:
    return m[~np.eye(m.shape[0], dtype=bool)]


def procrustes_error(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised Procrustes distance between two product maps (rows of a, b)."""
    a = a - a.mean(0, keepdims=True)
    b = b - b.mean(0, keepdims=True)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    s = np.linalg.svd(a.T @ b, compute_uv=False).sum()
    return float(max(0.0, 2.0 - 2.0 * s))


def kernel_metrics(kappa_hat: np.ndarray, kappa_true: np.ndarray,
                   elem_of: np.ndarray) -> dict:
    """How well does kappa-hat reproduce the truth, and does it find the
    element structure? `same_elem_lift` is the ratio of mean off-diagonal mass
    inside an element to outside it -- 1.0 means no structure found."""
    oh, ot = offdiag(kappa_hat), offdiag(kappa_true)
    same = (elem_of[:, None] == elem_of[None, :])
    off = ~np.eye(len(elem_of), dtype=bool)
    inside = kappa_hat[same & off].mean()
    outside = kappa_hat[~same & off].mean()
    return {
        "spearman_offdiag": spearman(oh, ot),
        "same_elem_lift": float(inside / outside) if outside > 0 else float("inf"),
        "offdiag_mass": float(kappa_hat[off].sum(axis=None) / kappa_hat.shape[0]),
        "diag_mean": float(np.diag(kappa_hat).mean()),
        "procrustes": procrustes_error(kappa_hat, kappa_true),
    }
