"""The R46 ladder as ONE functional choice model.

Utilities U_j for the nine decisions (NotBuy, index 8, is the base) are built
from the state blocks of state.py and a flat parameter vector theta, so the
fit, the exact Hessian, the per-customer scores and the forward simulator all
call the same function.

    rung   adds                                            parameters
    M0     constants only (the marginal)                   asc
    M1     previous-decision constants only (Markov-1)     lag
    R0     constants + the offer + campaign phase          asc, fig, rr_a, rr_b, unoff_b, wep, phase
    R1     R0 with the lag matrix in place of asc          lag replaces asc
    R1b    R1 + habit persistence (Roy et al.)             rho
    R2     R1 + run structure (Bawa; Seetharaman-Chintagunta)  bawa, runs
    R3     R2 + decayed stocks (Guadagni-Little; Seetharaman)  a_dec, a_out, loy, inc, tenl, out
    R4     R3 + holdings, pity, guarantee                  cop_f, cop_w, elem, owned, pity0-3, guar, wloss
    R5     R4 in a buy -> banner -> pack nested logit      nest
    R6     R5 with K latent classes                        d_asc, d_inc, d_loy, d_nb, cls

The lag matrix makes constants redundant (every row has exactly one previous
decision), so rungs with `lag` carry no `asc`; in R6 the class deviations of
the first class are fixed at zero for the same reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch

RUNGS = ("M0", "M1", "R0", "R1", "R1b", "R2", "R3", "R4", "R5", "R6")

ALT_BANNER = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, -1])
BUY = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=torch.float64)
TEN = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float64)
BAN = torch.stack([(ALT_BANNER == b).to(torch.float64) for b in range(4)])   # (4,9)


def flags(rung: str) -> Dict[str, bool]:
    i = RUNGS.index(rung)
    f = dict(asc=rung in ("M0", "R0"), lag=rung == "M1" or i >= RUNGS.index("R1"),
             offer=i >= RUNGS.index("R0"), rho=rung == "R1b",
             runs=i >= RUNGS.index("R2"), stocks=i >= RUNGS.index("R3"),
             hold=i >= RUNGS.index("R4"), nest=i >= RUNGS.index("R5"),
             classes=rung == "R6")
    return f


@dataclass
class Spec:
    rung: str
    K: int = 1
    names: List[str] = field(default_factory=list)
    shapes: List[Tuple[int, ...]] = field(default_factory=list)

    def __post_init__(self):
        f = flags(self.rung)
        if not f["classes"]:
            self.K = 1
        add = lambda n, *s: (self.names.append(n), self.shapes.append(tuple(s)))
        if f["asc"]:
            add("asc", 8)
        if f["lag"]:
            add("lag", 10, 8)
        if f["offer"]:
            add("fig", 11); add("rr_a", 2); add("rr_b", 2); add("unoff_b", 1)
            add("wep", 6); add("phase", 3)
        if f["rho"]:
            add("rho", 1)
        if f["runs"]:
            add("bawa", 2); add("runs", 3)
        if f["stocks"]:
            add("a_dec", 1); add("a_out", 1); add("loy", 1); add("inc", 1)
            add("tenl", 1); add("out", 3)
        if f["hold"]:
            add("cop_f", 3); add("cop_w", 2); add("elem", 1); add("owned", 1)
            for b in range(4):
                add(f"pity{b}", 3)
            add("guar", 1); add("wloss", 2)
        if f["nest"]:
            add("nest", 2)
        if f["classes"]:
            K1 = self.K - 1
            add("d_asc", K1, 8); add("d_inc", K1); add("d_loy", K1); add("d_nb", K1)
            add("cls", K1)
        self.flags = f

    @property
    def size(self) -> int:
        return int(sum(int(np.prod(s)) for s in self.shapes))

    def unpack(self, theta: torch.Tensor) -> Dict[str, torch.Tensor]:
        out, o = {}, 0
        for n, s in zip(self.names, self.shapes):
            k = int(np.prod(s))
            out[n] = theta[o:o + k].reshape(s)
            o += k
        return out

    def offsets(self) -> Dict[str, Tuple[int, int]]:
        out, o = {}, 0
        for n, s in zip(self.names, self.shapes):
            k = int(np.prod(s))
            out[n] = (o, o + k)
            o += k
        return out

    def init(self, *, marginal: np.ndarray | None = None) -> torch.Tensor:
        th = torch.zeros(self.size, dtype=torch.float64)
        off = self.offsets()
        logit = lambda p: float(np.log(p / (1 - p)))
        if "asc" in off and marginal is not None:
            a, b = off["asc"]
            th[a:b] = torch.as_tensor(np.log(marginal[:8] / marginal[8]))
        if "lag" in off and marginal is not None:
            a, b = off["lag"]
            th[a:b] = torch.as_tensor(np.tile(np.log(marginal[:8] / marginal[8]), 10))
        for n, v in (("a_dec", logit(0.7)), ("a_out", logit(0.9)), ("rho", logit(0.05))):
            if n in off:
                th[off[n][0]] = v
        if "nest" in off:
            th[off["nest"][0]] = logit(0.9)
            th[off["nest"][0] + 1] = logit(0.9)
        return th


# ─────────────────────────────── pieces ───────────────────────────────
def decay(X: torch.Tensor, alpha: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """L_t = alpha L_(t-1) + (1 - alpha) X_t, L_(-1) = 0. Closed form inside a chunk."""
    B, S, F = X.shape
    i = torch.arange(chunk, dtype=X.dtype)
    E = (i[:, None] - i[None, :]).clamp(min=0)
    low = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool))
    out, carry = [], X.new_zeros(B, F)
    for s0 in range(0, S, chunk):
        c = min(chunk, S - s0)
        W = torch.where(low[:c, :c], alpha ** E[:c, :c], torch.zeros((), dtype=X.dtype)) * (1 - alpha)
        part = torch.einsum("ij,bjf->bif", W, X[:, s0:s0 + c])
        pw = alpha ** torch.arange(1, c + 1, dtype=X.dtype)
        part = part + pw[None, :, None] * carry[:, None, :]
        out.append(part)
        carry = part[:, -1]
    return torch.cat(out, 1)


def nested_logp(U: torch.Tensor, lam_top: torch.Tensor, lam_ban: torch.Tensor) -> torch.Tensor:
    """Three-level nested logit: buy or not -> banner -> single or ten-pull."""
    lead = U.shape[:-1]
    Ub = U[..., :8].reshape(*lead, 4, 2)
    lse_b = torch.logsumexp(Ub / lam_ban, -1)
    iv_b = lam_ban * lse_b
    lse_top = torch.logsumexp(iv_b / lam_top, -1)
    iv_buy = lam_top * lse_top
    u9 = U[..., 8]
    den = torch.logaddexp(u9, iv_buy)
    lp = (iv_buy - den)[..., None, None] + (iv_b / lam_top - lse_top[..., None])[..., None] \
        + (Ub / lam_ban - lse_b[..., None])
    return torch.cat([lp.reshape(*lead, 8), (u9 - den)[..., None]], -1)


def utilities(spec: Spec, P: Dict[str, torch.Tensor], X: Dict[str, torch.Tensor]) -> torch.Tensor:
    """(K,B,S,9) utilities; K = 1 unless the rung has latent classes."""
    f = spec.flags
    B, S = X["prev_idx"].shape
    dt = torch.float64
    U = torch.zeros(B, S, 9, dtype=dt)
    z1 = lambda v: torch.cat([v, torch.zeros(*v.shape[:-1], 1, dtype=dt)], -1)
    if f["asc"]:
        U = U + z1(P["asc"])
    if f["lag"]:
        U = U + z1(P["lag"][X["prev_idx"]])
    if f["offer"]:
        fa, fb = X["off_a"], X["off_b"]
        va = fa[..., :11] @ P["fig"] + fa[..., 11:] @ P["rr_a"]
        vb = fb[..., :11] @ P["fig"] + fb[..., 11:] @ P["rr_b"] + X["unoff_b"][..., 0] * P["unoff_b"][0]
        vw = X["off_w"] @ P["wep"]
        U = U + va[..., None] * BAN[1] + vb[..., None] * BAN[2] + vw[..., None] * BAN[3]
        U = U + (X["phase"] @ P["phase"])[..., None] * BUY
    if f["runs"]:
        mlb = (ALT_BANNER[None, None, :] == X["last_banner"][..., None]).to(dt)
        U = U + (X["bawa"] @ P["bawa"])[..., None] * mlb
        r = X["runs"]
        U = U + (r[..., 0] * P["runs"][0] + r[..., 1] * P["runs"][1])[..., None] * BUY \
            + (r[..., 2] * P["runs"][2])[..., None] * TEN
    LD = None
    if f["stocks"]:
        a_d, a_o = torch.sigmoid(P["a_dec"][0]), torch.sigmoid(P["a_out"][0])
        LD = decay(X["ind"][..., :6], a_d)
        LO = decay(X["ind"][..., 6:9], a_o)
        U = U + P["loy"][0] * (LD[..., :4] @ BAN)
        U = U + (P["inc"][0] * LD[..., 4] + LO @ P["out"])[..., None] * BUY \
            + (P["tenl"][0] * LD[..., 5])[..., None] * TEN
    if f["hold"]:
        pit = X["pity"]
        v0 = pit[..., 0, :] @ P["pity0"]
        va = (X["copies_a"] @ P["cop_f"] + X["elem_a"][..., 0] * P["elem"][0]
              + X["guar"][..., 0] * P["guar"][0] + pit[..., 1, :] @ P["pity1"])
        vb = (X["copies_b"] @ P["cop_f"] + X["elem_b"][..., 0] * P["elem"][0]
              + X["guar"][..., 0] * P["guar"][0] + pit[..., 2, :] @ P["pity2"])
        vw = X["copies_w"] @ P["cop_w"] + X["wloss"] @ P["wloss"] + pit[..., 3, :] @ P["pity3"]
        U = U + v0[..., None] * BAN[0] + va[..., None] * BAN[1] + vb[..., None] * BAN[2] \
            + vw[..., None] * BAN[3] + (X["owned"][..., 0] * P["owned"][0])[..., None] * BUY
    U = U[None]
    if f["classes"] and spec.K > 1:
        zk = lambda v: torch.cat([torch.zeros(1, *v.shape[1:], dtype=dt), v], 0)
        d_asc, d_inc, d_loy, d_nb = zk(P["d_asc"]), zk(P["d_inc"]), zk(P["d_loy"]), zk(P["d_nb"])
        dU = z1(d_asc)[:, None, None, :] \
            + (d_inc[:, None, None] * LD[None, ..., 4] + d_nb[:, None, None] * X["runs"][None, ..., 0])[..., None] * BUY \
            + d_loy[:, None, None, None] * (LD[..., :4] @ BAN)[None]
        U = U + dU
    return U


def log_probs(spec: Spec, theta: torch.Tensor, X: Dict[str, torch.Tensor]
              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """(K,B,S,9) per-class log choice probabilities and (K,) log class weights."""
    P = spec.unpack(theta)
    U = utilities(spec, P, X)
    if spec.flags["nest"]:
        lt = torch.sigmoid(P["nest"][0])
        lb = lt * torch.sigmoid(P["nest"][1])
        lp = nested_logp(U, lt, lb)
    else:
        lp = torch.log_softmax(U, -1)
    if spec.flags["rho"]:
        rho = torch.sigmoid(P["rho"][0])
        pi = X["prev_idx"]
        hab = torch.zeros(*pi.shape, 10, dtype=torch.float64).scatter_(-1, pi[..., None], 1.0)[..., 1:]
        # no previous decision -> no habit mass, so the row is plain MNL
        r = rho * (pi > 0).to(torch.float64)[..., None]
        lp = torch.log(((1 - r)[None] * lp.exp() + (r * hab)[None]).clamp_min(1e-300))
    if spec.flags["classes"] and spec.K > 1:
        logpi = torch.log_softmax(torch.cat([torch.zeros(1, dtype=torch.float64), P["cls"]]), 0)
    else:
        logpi = torch.zeros(1, dtype=torch.float64)
    return lp, logpi


def customer_ll(spec: Spec, theta: torch.Tensor, X: Dict[str, torch.Tensor],
                y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(B,) log-likelihood of each customer's masked rows (a mixture integrates the class)."""
    lp, logpi = log_probs(spec, theta, X)
    ly = lp.gather(-1, (y.clamp(1, 9) - 1)[None, ..., None].expand(lp.shape[0], -1, -1, 1))[..., 0]
    llk = (ly * mask[None].to(ly.dtype)).sum(-1)                       # (K,B)
    return torch.logsumexp(logpi[:, None] + llk, 0)


def predictive(spec: Spec, theta: torch.Tensor, X: Dict[str, torch.Tensor],
               y: torch.Tensor, valid: torch.Tensor, *, prior_only: bool = False) -> torch.Tensor:
    """(B,S,9) one-step predictive log probabilities.

    A latent-class model filters its class weights from the customer's own
    earlier rows, pi_k(t) ∝ pi_k prod_{s<t} P_k(y_s) -- the predictive density
    given history. No parameter is trained on the customer.
    """
    lp, logpi = log_probs(spec, theta, X)
    if lp.shape[0] == 1:
        return lp[0]
    if prior_only:
        return torch.logsumexp(logpi[:, None, None, None] + lp, 0)
    ly = lp.gather(-1, (y.clamp(1, 9) - 1)[None, ..., None].expand(lp.shape[0], -1, -1, 1))[..., 0]
    ly = ly * valid[None].to(ly.dtype)
    cum = torch.cumsum(ly, -1) - ly                                   # exclusive: rows < t
    lw = logpi[:, None, None] + cum
    lw = lw - torch.logsumexp(lw, 0, keepdim=True)
    return torch.logsumexp(lw[..., None] + lp, 0)
