"""
Does the attention inventory NEST the classical models? A demonstration with no
loot boxes, no probabilistic goods -- ordinary brand choice.

THE CLAIM. The stock the attention inventory carries,

    S_t(j) = sum_{tau < t} kappa(j, c_tau) * phi(t - tau),

reduces to Guadagni & Little's (1983) loyalty variable when kappa = I and phi is
geometric, and to McAlister's (1982) accumulated-attribute satiation when kappa
is the attribute-similarity matrix. Attention supplies a LEARNED kappa (a
softmax over product embeddings) and a LEARNED phi (a recency bias). So the
architecture does not add a mechanism; it estimates what the classics fix in
advance.

THE TEST. Generate ordinary panel data in which one of the classics is true by
construction, fit the learned model, and ask whether it recovers the classical
object -- the smoothing constant, or the attribute structure.

    E1  truth = Guadagni-Little   -> is alpha recovered, and kappa ~ identity?
    E2  truth = McAlister         -> is the attribute structure recovered?
    E3  same truth as E2, but the assortment ROTATES instead of being fully
        available -> identification collapses, which is the general form of the
        limited-time-product problem (EXPERIMENTS.md R34).

Households choose one brand from those available, or nothing. What they choose
is what they get: no lottery anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BrandConfig:
    n_brands: int = 12
    n_attributes: int = 4          # e.g. flavour families; brands share them
    n_households: int = 800
    n_occasions: int = 200
    availability: str = "all"      # "all" (supermarket) | "rotating" (LTP)
    avail_size: int = 4            # brands on offer at once when rotating
    rotate_every: int = 20
    alpha: float = 0.8             # TRUE smoothing: phi(delta) = alpha^delta
    gamma: float = 1.5             # coefficient on the stock
    kernel: str = "identity"       # TRUTH: "identity" = GL, "attr" = McAlister
    attr_weight: float = 2.0       # off-diagonal strength when kernel = "attr"
    taste_sd: float = 0.5
    price_sd: float = 0.3
    price_beta: float = -1.0
    seed: int = 0

    def half_life(self) -> float:
        return float(np.log(0.5) / np.log(self.alpha))


@dataclass
class BrandData:
    choices: np.ndarray            # (N,T) brand index, or n_brands for no-purchase
    avail: np.ndarray              # (T, n_brands) bool
    prices: np.ndarray             # (N,T,n_brands)
    attr_of: np.ndarray            # (n_brands,)
    kappa_true: np.ndarray         # (n_brands, n_brands), row-stochastic
    base_true: np.ndarray
    cfg: BrandConfig = field(default_factory=BrandConfig)


def true_kernel(cfg: BrandConfig, attr_of: np.ndarray) -> np.ndarray:
    J = cfg.n_brands
    if cfg.kernel == "identity":
        return np.eye(J)
    same = (attr_of[:, None] == attr_of[None, :]).astype(float)
    logits = 3.0 * np.eye(J) + cfg.attr_weight * same
    logits -= logits.max(1, keepdims=True)
    k = np.exp(logits)
    return k / k.sum(1, keepdims=True)


def simulate(cfg: BrandConfig) -> BrandData:
    rng = np.random.default_rng(cfg.seed)
    J, N, T = cfg.n_brands, cfg.n_households, cfg.n_occasions
    attr_of = np.arange(J) % cfg.n_attributes
    kappa = true_kernel(cfg, attr_of)
    base = rng.normal(0.0, 0.7, size=J)
    taste = rng.normal(0.0, cfg.taste_sd, size=(N, cfg.n_attributes))
    theta = base[None, :] + taste[:, attr_of]

    avail = np.ones((T, J), dtype=bool)
    if cfg.availability == "rotating":
        avail[:] = False
        order = rng.permutation(J)
        cur = 0
        for t in range(T):
            if t % cfg.rotate_every == 0:
                pick = [order[(cur + i) % J] for i in range(cfg.avail_size)]
                cur += cfg.avail_size
            avail[t, pick] = True

    prices = rng.normal(0.0, cfg.price_sd, size=(N, T, J))
    R = np.zeros((N, J))
    choices = np.zeros((N, T), dtype=np.int64)
    for t in range(T):
        stock = R @ kappa.T
        v = theta + cfg.gamma * stock + cfg.price_beta * prices[:, t]
        v = np.where(avail[t][None, :], v, -1e9)
        v = np.concatenate([v, np.zeros((N, 1))], axis=1)      # no-purchase = 0
        c = np.argmax(v + rng.gumbel(size=(N, J + 1)), axis=1)
        choices[:, t] = c
        bought = c < J
        R *= cfg.alpha
        R[np.flatnonzero(bought), c[bought]] += 1.0
    return BrandData(choices=choices, avail=avail, prices=prices, attr_of=attr_of,
                     kappa_true=kappa, base_true=base, cfg=cfg)


class LearnedInventory(nn.Module):
    """The attention inventory: a LEARNED kernel and a LEARNED decay.

    kernel='identity' with a fixed alpha is exactly Guadagni-Little;
    kernel='attr' with alpha fixed at 1 is exactly McAlister attribute
    satiation. 'learned' estimates both.
    """

    def __init__(self, cfg: BrandConfig, attr_of: np.ndarray, kernel: str = "learned",
                 learn_decay: bool = True, alpha_init: float = 0.5, d: int = 8,
                 customer_fe: bool = False):
        super().__init__()
        J = cfg.n_brands
        self.J, self.kernel = J, kernel
        same = (attr_of[:, None] == attr_of[None, :]).astype(np.float32)
        self.register_buffer("same_attr", torch.as_tensor(same))
        self.register_buffer("attr_of", torch.as_tensor(attr_of, dtype=torch.long))
        # S8: per-household attribute tastes, i.e. the heterogeneity the DGP has
        # and E2b showed the kernel absorbs when the estimator lacks it.
        self.hh_taste = (nn.Parameter(torch.zeros(cfg.n_households, cfg.n_attributes))
                         if customer_fe else None)
        if kernel == "learned":
            self.emb = nn.Parameter(torch.randn(J, d) * 0.1)
            self.wq, self.wk = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
            self.d = d
        elif kernel == "attr":
            self.attr_self = nn.Parameter(torch.tensor(3.0))
            self.attr_same = nn.Parameter(torch.tensor(0.5))
        self.logit_alpha = nn.Parameter(torch.logit(torch.tensor(alpha_init)),
                                        requires_grad=learn_decay)
        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.base = nn.Parameter(torch.zeros(J))
        self.price_beta = nn.Parameter(torch.tensor(0.0))
        self.nobuy = nn.Parameter(torch.tensor(0.0))

    def kappa(self) -> torch.Tensor:
        eye = torch.eye(self.J, device=self.base.device)
        if self.kernel == "identity":
            return eye
        if self.kernel == "attr":
            return torch.softmax(self.attr_self * eye + self.attr_same * self.same_attr, 1)
        q, k = self.wq(self.emb), self.wk(self.emb)
        return torch.softmax(q @ k.T / np.sqrt(self.d), dim=1)

    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_alpha)

    def nll(self, choices, avail, prices, hh=None) -> torch.Tensor:
        N, T = choices.shape
        J, dev = self.J, self.base.device
        kappa, alpha = self.kappa(), self.alpha()
        base = self.base[None, :].expand(N, J)
        if self.hh_taste is not None and hh is not None:
            base = base + self.hh_taste[hh][:, self.attr_of]
        R = torch.zeros(N, J, device=dev)
        total = torch.zeros((), device=dev)
        for t in range(T):
            stock = R @ kappa.T
            v = base + self.gamma * stock + self.price_beta * prices[:, t]
            v = torch.where(avail[t][None, :], v, torch.full_like(v, -1e9))
            v = torch.cat([v, self.nobuy.expand(N, 1)], dim=1)
            total = total + F.cross_entropy(v, choices[:, t], reduction="sum")
            c = choices[:, t]
            hit = (c < J)
            R = R * alpha
            if hit.any():
                rows = torch.nonzero(hit, as_tuple=True)[0]
                R = R.index_put((rows, c[rows]), torch.ones(len(rows), device=dev),
                                accumulate=True)
        return total / (N * T)


def fit(data: BrandData, kernel: str = "learned", learn_decay: bool = True,
        epochs: int = 300, lr: float = 0.05, batch: int = 256, seed: int = 0,
        alpha_init: float = 0.5, device: str = "cpu", customer_fe: bool = False) -> dict:
    dev = torch.device(device)
    torch.manual_seed(seed)
    cfg = data.cfg
    m = LearnedInventory(cfg, data.attr_of, kernel=kernel, learn_decay=learn_decay,
                         alpha_init=alpha_init, customer_fe=customer_fe).to(dev)
    ch = torch.as_tensor(data.choices, device=dev)
    av = torch.as_tensor(data.avail, device=dev)
    pr = torch.as_tensor(data.prices, dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    hist = []
    for ep in range(epochs):
        idx = torch.as_tensor(rng.choice(cfg.n_households, min(batch, cfg.n_households),
                                         replace=False), device=dev)
        loss = m.nll(ch[idx], av, pr[idx], hh=idx)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step()
        hist.append(float(loss.detach()))
    with torch.no_grad():
        k = m.kappa().cpu().numpy()
        out = {"nll": float(np.mean(hist[-10:])), "kappa_hat": k,
               "alpha_hat": float(m.alpha()), "gamma_hat": float(m.gamma),
               "price_beta_hat": float(m.price_beta),
               "half_life_hat": float(np.log(0.5) / np.log(max(float(m.alpha()), 1e-6))),
               "offdiag_mass": float(k[~np.eye(cfg.n_brands, dtype=bool)].sum()
                                     / cfg.n_brands),
               "history": hist}
        if kernel == "attr":
            out["attr_same_hat"] = float(m.attr_same)
    return out
