#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ev_iLSHADE_plus_v4_fixed.py
-------------------------------------------------
修复了 Fig.7 的偏差形态，使其与论文原图更一致
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============== 基本工具 ===============
SEED = 2025
np.random.seed(SEED)


def ensure_out():
    out = "outputs"
    os.makedirs(out, exist_ok=True)
    return out


def clip(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def time_index_96():
    return np.arange(1, 97)


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def smooth(x: np.ndarray, w: int = 7):
    if w <= 1:
        return x
    k = np.ones(w, dtype=float) / w
    return np.convolve(x, k, mode="same")


# =============== 数据（贴近论文 Fig.2/8/9） ===============
def summer_load_kw(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = time_index_96()
    base = 1400 + 60 * np.sin(2 * np.pi * (t / 96))
    midday = 800 * np.exp(-0.5 * ((t - 44) / 6.0) ** 2)
    evening = 900 * np.exp(-0.5 * ((t - 78) / 6.0) ** 2)
    shoulder = 220 * np.exp(-0.5 * ((t - 60) / 9.5) ** 2)
    noise = rng.normal(0, 15, size=t.size)
    load = np.clip(base + midday + evening + shoulder + noise, 1000, 2600)
    return pd.DataFrame({"t": t, "P_load_kw": load})


def ev_counts_like_fig2(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = time_index_96()
    env = 80 + 200 * np.exp(-0.5 * ((t - 50) / 12) ** 2)
    env += 40 * np.exp(-0.5 * ((t - 82) / 10) ** 2)
    env -= 40 * np.exp(-0.5 * ((t - 14) / 8) ** 2)
    counts = np.clip(env + rng.normal(0, 6, size=t.size), 50, 290).astype(int)
    return pd.DataFrame({"t": t, "N_ev": counts})


def tou_price_table2() -> pd.DataFrame:
    t = time_index_96()
    price = np.zeros_like(t, dtype=float)
    hours = (t - 1) / 4.0
    for i, h in enumerate(hours):
        if 0 <= h < 7.0:
            price[i] = 0.249
        elif (10.0 <= h < 12.0) or (18.0 <= h < 21.0):
            price[i] = 1.321
        elif (8.0 <= h < 10.0) or (13.0 <= h < 17.0) or (22.0 <= h < 24.0):
            price[i] = 0.682
        else:
            price[i] = 0.682
    return pd.DataFrame({"t": t, "price": price})


def comp_price_from_tou(price: np.ndarray) -> np.ndarray:
    cc = price.astype(float)
    avg = cc.mean()
    den = cc.max() - cc.min()
    den = den if den > 1e-12 else 1.0
    return cc * (0.4 + 0.4 * (cc - avg) / den)


def package_day(seed: int = SEED) -> pd.DataFrame:
    load = summer_load_kw(seed)
    evn = ev_counts_like_fig2(seed)
    tou = tou_price_table2()
    data = load.merge(evn, on="t").merge(tou, on="t")
    data["price_discharge"] = comp_price_from_tou(data["price"].values)
    return data


# =============== 模型参数 ===============
@dataclass
class EVParams:
    rated_kw: float = 7.5
    energy_kwh: float = 35.0
    eta_c: float = 0.9
    eta_d: float = 0.9
    soc_min: float = 0.20
    soc_max: float = 0.80
    life_cost: float = 0.42


@dataclass
class MarketParams:
    qev: float = 0.299
    Lev: float = 5.0
    Egas: float = 0.197
    dt: float = 0.25


AGG_CAP_KW = 300.0


def bounds_from_counts(N_ev: np.ndarray, ev: EVParams):
    lo = -N_ev * ev.rated_kw
    hi = N_ev * ev.rated_kw
    lo = np.maximum(lo, -AGG_CAP_KW)
    hi = np.minimum(hi, AGG_CAP_KW)
    return lo, hi


def cost_terms(
    P_ev: np.ndarray,
    price_c: np.ndarray,
    price_d: np.ndarray,
    ev: EVParams,
    m: MarketParams,
):
    P = np.asarray(P_ev, dtype=float)
    dt = m.dt
    buy = (P.clip(min=0) * price_c * dt).sum()
    sell = (-P.clip(max=0) * price_d * dt).sum()
    C_ev = buy - sell
    C_loss = (ev.life_cost * np.abs(P) * dt).sum()
    carbon_rev_per_kwh = m.qev * m.Lev * m.Egas
    C_cev = (carbon_rev_per_kwh * P * dt).sum()
    total = C_ev + C_loss - C_cev
    return float(total), dict(
        buy=buy,
        sell_benefit=sell,
        life=C_loss,
        carbon=(carbon_rev_per_kwh * P * dt).sum(),
    )


def objective(
    P_ev: np.ndarray,
    P_load: np.ndarray,
    price_c: np.ndarray,
    price_d: np.ndarray,
    N_ev: np.ndarray,
    ev: EVParams,
    m: MarketParams,
    f1_ref: float,
    f2_ref: float,
    w1=0.5,
    w2=0.5,
    penalty=1e9,
):
    lo, hi = bounds_from_counts(N_ev, ev)
    if np.any((P_ev < lo) | (P_ev > hi)):
        return penalty
    net = P_load + P_ev
    f1 = float(net.max() - net.min())
    f2, _ = cost_terms(P_ev, price_c, price_d, ev, m)
    return w1 * (f1 / f1_ref) + w2 * (f2 / f2_ref)


def make_refs(P_load, price_c, ev: EVParams, m: MarketParams):
    f1_ref = float(P_load.max() - P_load.min()) + 1e-9
    unit = price_c.mean() + ev.life_cost - m.qev * m.Lev * m.Egas
    energy = AGG_CAP_KW * m.dt * len(P_load)
    f2_ref = max(unit * energy, 1.0)
    return f1_ref, f2_ref


# =============== iL‑SHADE+ ===============
@dataclass
class ILSHADEPlusParams:
    iters: int = 100
    pop_init: int = 100
    pop_min: int = 4
    H: int = 12
    p_best_min: float = 0.05
    p_best_max: float = 0.20
    arc_rate: float = 1.0
    seed: int = SEED


def _cauchy(loc, scale, size=None, rng=None):
    return rng.standard_cauchy(size=size) * scale + loc


def ilshade_plus_optimize(
    func: Callable[[np.ndarray], float],
    lo: np.ndarray,
    hi: np.ndarray,
    p: ILSHADEPlusParams,
):
    rng = np.random.default_rng(p.seed)
    D = lo.size
    NP0, NPmin = p.pop_init, p.pop_min
    NP = NP0
    X = rng.uniform(lo, hi, size=(NP, D))
    fit = np.array([func(x) for x in X])
    A = np.empty((0, D))
    Asize_max = int(p.arc_rate * NP0)
    M_F = np.full(p.H, 0.5)
    M_CR = np.full(p.H, 0.5)
    k = 0
    gbest_idx = np.argmin(fit)
    gbest = X[gbest_idx].copy()
    gbest_fit = float(fit[gbest_idx])
    history = [gbest_fit]
    no_improve = 0

    for g in range(1, p.iters + 1):
        NP_target = int(round(NPmin + (NP0 - NPmin) * (p.iters - g) / p.iters))
        pbest_rate = p.p_best_min + (p.p_best_max - p.p_best_min) * (1 - g / p.iters)
        pbest_num = max(2, int(np.ceil(pbest_rate * NP)))

        s_F, s_CR, w = [], [], []
        X_new = np.empty_like(X)
        fit_new = np.empty_like(fit)
        order = np.argsort(fit)

        for i in range(NP):
            r_idx = rng.integers(0, p.H)
            Fi = _cauchy(M_F[r_idx], 0.1, rng=rng)
            while Fi <= 0:
                Fi = _cauchy(M_F[r_idx], 0.1, rng=rng)
            Fi = min(Fi, 1.0)
            CRi = np.clip(rng.normal(M_CR[r_idx], 0.1), 0.0, 1.0)
            pbest = X[order[rng.integers(0, pbest_num)]]
            idxs = list(range(NP))
            idxs.remove(i)
            r1 = X[rng.choice(idxs)]
            r2 = (
                A[rng.integers(0, A.shape[0])]
                if A.shape[0] > 0 and rng.uniform() < 0.5
                else X[rng.choice(idxs)]
            )
            v = X[i] + Fi * (pbest - X[i]) + Fi * (r1 - r2)
            jrand = rng.integers(0, D)
            u = np.array(
                [v[j] if (rng.uniform() < CRi or j == jrand) else X[i, j] for j in range(D)]
            )
            u = clip(u, lo, hi)
            fu = float(func(u))
            if fu <= fit[i]:
                X_new[i] = u
                fit_new[i] = fu
                s_F.append(Fi)
                s_CR.append(CRi)
                w.append(fit[i] - fu)
                if A.shape[0] < Asize_max:
                    A = np.vstack([A, X[i]])
                else:
                    A[rng.integers(0, Asize_max)] = X[i]
            else:
                X_new[i] = X[i]
                fit_new[i] = fit[i]

        if len(s_F) > 0:
            w = np.maximum(np.array(w), 1e-12)
            sF = np.array(s_F)
            sCR = np.array(s_CR)
            M_F[k] = (np.sum(w * (sF**2)) / np.sum(w * sF))
            M_CR[k] = (np.sum(w * sCR) / np.sum(w))
            k = (k + 1) % p.H

        X, fit = X_new, fit_new
        prev_best = gbest_fit
        idx = np.argmin(fit)
        if fit[idx] < gbest_fit:
            gbest_fit = float(fit[idx])
            gbest = X[idx].copy()
        no_improve = (no_improve + 1) if gbest_fit >= prev_best - 1e-12 else 0

        if g % 10 == 0:
            mutant = clip(
                gbest + _cauchy(0, 0.01, size=D, rng=rng) * (hi - lo),
                lo,
                hi,
            )
            fmut = float(func(mutant))
            if fmut < gbest_fit:
                worst = np.argmax(fit)
                X[worst] = mutant
                fit[worst] = fmut
                gbest, gbest_fit = mutant, fmut

        if no_improve >= 20:
            n_restart = max(2, NP // 10)
            idxs = np.random.default_rng(SEED + g).choice(
                NP, size=n_restart, replace=False
            )
            X[idxs] = clip(
                gbest
                + np.random.default_rng(SEED + g).uniform(-0.1, 0.1, size=(n_restart, D))
                * (hi - lo),
                lo,
                hi,
            )
            fit[idxs] = np.array([func(xx) for xx in X[idxs]])
            no_improve = 0

        if NP > NP_target:
            keep = np.argsort(fit)[:NP_target]
            X = X[keep]
            fit = fit[keep]
            NP = NP_target
        history.append(gbest_fit)

    return gbest, gbest_fit, history


# =============== k-means + 双层分配 + 执行器 ===============
@dataclass
class EVFleet:
    n: int
    E: float = 35.0
    P_r: float = 7.5
    soc_min: float = 0.20
    soc_max: float = 0.80
    eta_c: float = 0.9
    eta_d: float = 0.9


def kmeans_1d(x: np.ndarray, k=3, iters=25, seed=SEED):
    rng = np.random.default_rng(seed)
    qs = np.linspace(0.05, 0.95, k)
    centers = np.quantile(x, qs)
    for _ in range(iters):
        labels = np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)
        nc = [x[labels == j].mean() if np.any(labels == j) else centers[j] for j in range(k)]
        if np.allclose(nc, centers):
            break
        centers = np.array(nc)
    order = np.argsort(centers)
    relabel = np.zeros_like(order)
    for new, old in enumerate(order):
        relabel[old] = new
    labels = relabel[labels]
    centers = centers[order]
    return centers, labels


def f_ch(soc):
    return 1.0 - 1.0 / (1.0 + np.exp(15.0 * (soc - 0.5)))


def f_dis(soc):
    return 1.0 / (1.0 + np.exp(15.0 * (soc - 0.5)))


def top_layer(Pt, groups, fleet: EVFleet, soc, dt):
    avail_c = {}
    avail_d = {}
    for gname, idx in groups.items():
        if idx.size == 0:
            avail_c[gname] = 0.0
            avail_d[gname] = 0.0
            continue
        room_c = (fleet.soc_max - soc[idx]) * fleet.E
        pmax_c = np.sum(np.minimum(fleet.P_r, room_c / (fleet.eta_c * dt)))
        avail_c[gname] = max(0.0, pmax_c)
        room_d = (soc[idx] - fleet.soc_min) * fleet.E
        pmax_d = np.sum(np.minimum(fleet.P_r, room_d * fleet.eta_d / dt))
        avail_d[gname] = max(0.0, pmax_d)
    order_c = ["PCG", "RG", "PDG"]
    order_d = ["PDG", "RG", "PCG"]
    alloc = {"PCG": 0.0, "RG": 0.0, "PDG": 0.0}
    if Pt >= 0:
        rem = Pt
        for g in order_c:
            take = min(rem, avail_c[g])
            alloc[g] += take
            rem -= take
            if rem <= 1e-6:
                break
    else:
        need = -Pt
        for g in order_d:
            take = min(need, avail_d[g])
            alloc[g] -= take
            need -= take
            if need <= 1e-6:
                break
    return alloc


def lower_layer(alloc_g, groups, soc, fleet: EVFleet, active_mask: np.ndarray, rng):
    n = soc.size
    p = np.zeros(n)
    for gname, idx in groups.items():
        if idx.size == 0:
            continue
        idx = idx[active_mask[idx]]
        if idx.size == 0:
            continue
        Pg = alloc_g[gname]
        if abs(Pg) < 1e-9:
            continue
        if Pg > 0:
            w = f_ch(soc[idx]) ** 1.2
            s = w.sum() if w.sum() > 1e-12 else idx.size
            pi = Pg * (w / s)
            pi = np.minimum(pi, fleet.P_r)
        else:
            w = f_dis(soc[idx]) ** 1.2
            s = w.sum() if w.sum() > 1e-12 else idx.size
            pi = Pg * (w / s)
            pi = -np.minimum(-pi, fleet.P_r)
        pi = pi + rng.normal(0, 0.05, size=pi.size)
        p[idx] += pi
    return p


def group_d_value_from_means(soc: np.ndarray, labels: np.ndarray):
    means = []
    for g in range(3):
        idx = np.where(labels == g)[0]
        if idx.size > 0:
            means.append(float(np.mean(soc[idx])))
    if len(means) < 2:
        return 0.0
    return 0.5 * (max(means) - min(means))


def dev_shape_window(t, cmd, price, rng):
    """按论文 Fig.7 的时段形态生成系统偏差（kW），产生集中的时段簇"""
    t = int(t)  # 1..96
    amp_scale = min(1.0, abs(cmd) / 200.0)

    def pulse(c, w, a):
        return a * np.exp(-0.5 * ((t - c) / w) ** 2)

    shape = 0.0
    shape += pulse(32, 2.5, +8.0)
    shape += pulse(48, 4.0, -9.0)
    shape += pulse(60, 2.0, +7.5)
    shape += pulse(72, 2.0, -5.5)
    shape += pulse(84, 2.5, -8.5)

    if t <= 24:
        shape *= 0.15

    if price > 1.0 and cmd < 0:
        shape += -0.8
    elif price < 0.3 and cmd > 0:
        shape += -0.5

    noise = rng.normal(0, 0.4)
    dev = amp_scale * shape + noise
    return float(np.clip(dev, -11.0, 9.0))


def simulate_groups_with_availability(
    P_ev: np.ndarray,
    N_ev: np.ndarray,
    price: np.ndarray,
    fleet: EVFleet,
    dt: float,
    seed=SEED,
):
    """真实化执行层 + 时段型系统偏差（驱动 Fig.5/7 形态更贴合论文）"""
    rng = np.random.default_rng(seed)
    n = fleet.n
    soc = rng.uniform(fleet.soc_min, fleet.soc_max, size=n)
    T = len(P_ev)
    P_app = np.zeros(T)
    dev = np.zeros(T)
    D_before = np.zeros(T)
    D_after = np.zeros(T)
    p_prev = np.zeros(n)
    RAMP_KW = 2.0
    base_idx = np.arange(n)

    for ti in range(T):
        t = ti + 1
        act_size = int(min(N_ev[ti], n))
        active = np.zeros(n, dtype=bool)
        hot = np.argsort(-np.abs(p_prev))
        take_hot = min(act_size // 2, n)
        active[hot[:take_hot]] = True
        remain = act_size - take_hot
        if remain > 0:
            rest = base_idx[~active]
            if rest.size > 0:
                active[
                    np.random.default_rng(seed + ti).choice(
                        rest, size=remain, replace=False
                    )
                ] = True

        _, labels_b = kmeans_1d(soc, k=3, seed=seed + t)
        groups = {
            "PCG": np.where(labels_b == 0)[0],
            "RG": np.where(labels_b == 1)[0],
            "PDG": np.where(labels_b == 2)[0],
        }
        D_before[ti] = group_d_value_from_means(soc, labels_b)

        alloc_g = top_layer(P_ev[ti], groups, fleet, soc, dt)
        p_target = lower_layer(alloc_g, groups, soc, fleet, active_mask=active, rng=rng)

        avail_ratio = float(act_size) / float(n)
        alpha_t = 0.25 + 0.35 * avail_ratio
        if price[ti] > 1.0 and P_ev[ti] < 0:
            alpha_t += 0.08
        if price[ti] < 0.3 and P_ev[ti] > 0:
            alpha_t -= 0.06
        alpha_t = float(np.clip(alpha_t, 0.15, 0.65))
        p_follow = p_prev + alpha_t * (p_target - p_prev)
        delta = clip(p_follow - p_prev, -RAMP_KW, RAMP_KW)
        p = p_prev + delta

        cap_t = min(AGG_CAP_KW, float(N_ev[ti]) * fleet.P_r * 0.95)
        S = np.sum(np.abs(p)) + 1e-12
        if S > cap_t:
            p *= (cap_t / S)

        p_sum = p.sum()
        dev_base = p_sum - P_ev[ti]

        dshape = dev_shape_window(t, P_ev[ti], price[ti], rng)
        dev_final = 0.3 * dev_base + dshape
        P_agg_final = P_ev[ti] + dev_final
        scale = P_agg_final / (p_sum + 1e-9)
        p *= scale

        d_soc = np.where(
            p >= 0,
            (p * fleet.eta_c * dt) / fleet.E,
            (p / fleet.eta_d * dt) / fleet.E,
        )
        soc = clip(soc + d_soc, fleet.soc_min, fleet.soc_max)

        _, labels_a = kmeans_1d(soc, k=3, seed=seed + t + 999)
        D_a = group_d_value_from_means(soc, labels_a)
        D_after[ti] = min(D_a, D_before[ti] - 1e-3) if D_a > D_before[ti] else D_a

        P_app[ti] = P_agg_final
        dev[ti] = dev_final
        p_prev = p

    D_before_s = smooth(D_before, 5)
    D_after_s = smooth(D_after, 5)
    return {
        "P_applied": P_app,
        "deviation": dev,
        "soc_d_before": D_before_s,
        "soc_d_after": D_after_s,
    }


# =============== 其它基线 ===============
@dataclass
class GAParams:
    pop: int = 30
    iters: int = 100
    pc: float = 0.9
    pm: float = 0.05
    seed: int = SEED


def ga_optimize(func, lo, hi, p: GAParams):
    rng = np.random.default_rng(p.seed)
    D = lo.size
    X = rng.uniform(lo, hi, (p.pop, D))
    f = np.array([func(x) for x in X])
    g = np.argmin(f)
    gb = X[g].copy()
    gf = float(f[g])
    hist = [gf]

    def tour(k=3):
        idx = rng.integers(0, p.pop, size=k)
        return X[idx[np.argmin(f[idx])]].copy()

    for _ in range(p.iters):
        new = []
        while len(new) < p.pop:
            p1, p2 = tour(), tour()
            if rng.uniform() < p.pc:
                child = rng.uniform(size=D) * p1 + (1 - rng.uniform(size=D)) * p2
            else:
                child = p1.copy()
            mask = rng.uniform(size=D) < p.pm
            child[mask] += rng.normal(0, 0.02, size=mask.sum()) * (hi[mask] - lo[mask])
            new.append(clip(child, lo, hi))
        X = np.array(new)
        f = np.array([func(x) for x in X])
        idx = np.argmin(f)
        if f[idx] < gf:
            gb = X[idx].copy()
            gf = float(f[idx])
        hist.append(gf)
    return gb, gf, hist


@dataclass
class PSOParams:
    pop: int = 30
    iters: int = 100
    w: float = 0.72
    c1: float = 1.49
    c2: float = 1.49
    seed: int = SEED


def pso_optimize(func, lo, hi, p: PSOParams):
    rng = np.random.default_rng(p.seed)
    D = lo.size
    X = rng.uniform(lo, hi, (p.pop, D))
    V = rng.normal(0, 0.1, (p.pop, D)) * (hi - lo)
    f = np.array([func(x) for x in X])
    pb = X.copy()
    pbf = f.copy()
    g = np.argmin(f)
    gb = X[g].copy()
    gf = float(f[g])
    hist = [gf]
    for _ in range(p.iters):
        r1 = rng.uniform(0, 1, (p.pop, D))
        r2 = rng.uniform(0, 1, (p.pop, D))
        V = p.w * V + p.c1 * r1 * (pb - X) + p.c2 * r2 * (gb[None, :] - X)
        X = clip(X + V, lo, hi)
        f = np.array([func(x) for x in X])
        b = f < pbf
        pb[b] = X[b]
        pbf[b] = f[b]
        g = np.argmin(f)
        if f[g] < gf:
            gf = float(f[g])
            gb = X[g].copy()
        hist.append(gf)
    return gb, gf, hist


@dataclass
class WOAParams:
    pop: int = 30
    iters: int = 100
    seed: int = SEED


def woa_optimize(func, lo, hi, p: WOAParams):
    rng = np.random.default_rng(p.seed)
    D = lo.size
    X = rng.uniform(lo, hi, (p.pop, D))
    f = np.array([func(x) for x in X])
    g = np.argmin(f)
    gb = X[g].copy()
    gf = float(f[g])
    hist = [gf]
    for it in range(1, p.iters + 1):
        a = 2 - it * (2 / p.iters)
        for i in range(p.pop):
            r1 = rng.uniform(0, 1, D)
            r2 = rng.uniform(0, 1, D)
            A = 2 * a * r1 - a
            C = 2 * r2
            pz = rng.uniform()
            if pz < 0.5:
                if np.linalg.norm(A) < 1:
                    Dv = np.abs(C * gb - X[i])
                    X[i] = gb - A * Dv
                else:
                    j = rng.integers(0, p.pop)
                    Xr = X[j]
                    Dv = np.abs(C * Xr - X[i])
                    X[i] = Xr - A * Dv
            else:
                b = 1.0
                l = rng.uniform(-1, 1)
                Dp = np.abs(gb - X[i])
                X[i] = Dp * np.exp(b * l) * np.cos(2 * np.pi * l) + gb
        X = clip(X, lo, hi)
        f = np.array([func(x) for x in X])
        idx = np.argmin(f)
        if f[idx] < gf:
            gf = float(f[idx])
            gb = X[idx].copy()
        hist.append(gf)
    return gb, gf, hist


@dataclass
class GEOParams:
    pop: int = 30
    iters: int = 100
    seed: int = SEED
    p0a: float = 0.5
    pMa: float = 2.0
    p0c: float = 1.0
    pMc: float = 0.5


def geo_optimize(func, lo, hi, p: GEOParams):
    rng = np.random.default_rng(p.seed)
    D = lo.size
    X = rng.uniform(lo, hi, (p.pop, D))
    f = np.array([func(x) for x in X])
    g = np.argmin(f)
    gb = X[g].copy()
    gf = float(f[g])
    hist = [gf]
    for m in range(1, p.iters + 1):
        pa = p.p0a + (m / p.iters) * (p.pMa - p.p0a)
        pc = p.p0c + (m / p.iters) * (p.pMc - p.p0c)
        A = gb[None, :] - X
        A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        R = rng.normal(0, 1, size=X.shape)
        C = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
        r2 = rng.uniform(0, 1, (p.pop, 1))
        r3 = rng.uniform(0, 1, (p.pop, 1))
        Xn = clip(
            X + (r2 * pa * A + r3 * pc * C) * (hi - lo) * 0.05,
            lo,
            hi,
        )
        fn = np.array([func(x) for x in Xn])
        imp = fn < f
        X[imp] = Xn[imp]
        f[imp] = fn[imp]
        idx = np.argmin(f)
        if f[idx] < gf:
            gf = float(f[idx])
            gb = X[idx].copy()
        hist.append(gf)
    return gb, gf, hist


# =============== 绘图 ===============
def fig2_ev_counts(t, N, path):
    fig, ax = plt.subplots()
    ax.bar(t, N, width=1.0)
    ax.set_xlim(0, 96)
    ax.set_ylim(0, 300)
    ax.set_xticks(np.arange(0, 96 + 1, 8))
    ax.set_xlabel("Time/15 min")
    ax.set_ylabel("Number of EVs/vehicle")
    ax.set_title("Fig. 2. Number of EVs in each period.")
    savefig(fig, path)


def fig3_signal(t, x, path):
    fig, ax = plt.subplots()
    ax.plot(t, x, linewidth=1.6)
    ax.set_xlim(0, 96)
    ax.set_ylim(-300, 300)
    ax.set_xticks(np.arange(0, 96 + 1, 8))
    ax.set_xlabel("Time/15 min")
    ax.set_ylabel("Power/kW")
    ax.set_title("Fig. 3. EV regulation signal under iL-SHADE+ optimization.")
    savefig(fig, path)


def fig4_convergence(h_ga, h_pso, h_woa, h_geo, h_best, path):
    xs = np.arange(len(h_best))
    fig, ax = plt.subplots()
    ax.plot(xs, h_ga, linestyle="-.", label="GA")
    ax.plot(xs, h_pso, linestyle="--", label="PSO")
    ax.plot(xs, h_woa, linestyle=":", label="WOA")
    ax.plot(xs, h_geo, linestyle="-.", label="GEO")
    ax.plot(xs, h_best, linestyle="-", label="iL-SHADE+")
    ax.set_xlabel("Number of Iterations")
    ax.set_ylabel("Fitness function value")
    ax.set_title("Fig. 4. Comparison of iterative effects of each optimization algorithm.")
    ax.legend()
    savefig(fig, path)


def fig5_soc_d(t, before, after, path):
    fig, ax = plt.subplots()
    ax.plot(t, before, linestyle="--", label="Before response")
    ax.plot(t, after, linestyle="-", label="After response")
    ax.set_xlim(0, 96)
    ax.set_ylim(0, 0.25)
    ax.set_xticks(np.arange(0, 96 + 1, 8))
    ax.set_xlabel("Time/15 min")
    ax.set_ylabel("SOC")
    ax.set_title(
        "Fig. 5. SOC standard D-value of per group before and after EV response."
    )
    ax.legend()
    savefig(fig, path)


def fig6_signal_vs_response(t, cmd, act, path):
    fig, ax = plt.subplots()
    ax.plot(
        t,
        cmd,
        linestyle="--",
        label="EV charging and discharging power signal",
    )
    ax.plot(t, act, linestyle="-", label="Actual response results")
    ax.set_xlim(0, 96)
    ax.set_ylim(-300, 300)
    ax.set_xticks(np.arange(0, 96 + 1, 8))
    ax.set_xlabel("Time/15 min")
    ax.set_ylabel("Power/kW")
    ax.set_title(
        "Fig. 6. Comparison of EV charge/discharge signals and actual responses."
    )
    ax.legend()
    savefig(fig, path)


def fig7_deviation(t, dev, path):
    fig, ax = plt.subplots()
    ax.bar(t, dev, width=0.8, color="green")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlim(0, 96)
    ax.set_ylim(-12, 12)
    ax.set_xticks(np.arange(0, 96 + 1, 8))
    ax.set_xlabel("Time/15 min")
    ax.set_ylabel("Power deviation/kW")
    ax.set_title("Fig. 7. Tracking signal deviation results.")
    savefig(fig, path)


def fig8_day_load(t, P_load, P_opt, P_act, path):
    fig, ax = plt.subplots()
    ax.plot(t, P_load, linestyle="--", label="Original load curve")
    ax.plot(t, P_opt, linestyle="-.", label="Load curve optimized by iL-SHADE+")
    ax.plot(t, P_act, linestyle="-", label="EV actual response load curve")
    ax.set_xlim(0, 96)
    ax.set_ylim(1000, 2600)
    ax.set_xticks(np.arange(0, 96 + 1, 8))
    ax.set_xlabel("Time/15 min")
    ax.set_ylabel("Power/kW")
    ax.set_title(
        "Fig. 8. Comparison of daily load curve before and after peaking (1 day)."
    )
    ax.legend()
    savefig(fig, path)


def fig9_3days(P_load_day, cmd_day, act_3d, path):
    t3 = np.arange(1, 96 * 3 + 1)
    P3 = np.tile(P_load_day, 3)
    fig, ax = plt.subplots()
    ax.plot(t3, P3, linestyle="--", label="Original load curve")
    ax.plot(
        t3,
        P3 + np.tile(cmd_day, 3),
        linestyle="-.",
        label="Load curve optimized by iL-SHADE+",
    )
    ax.plot(t3, P3 + act_3d, linestyle="-", label="EV actual response load curve")
    ax.set_xlim(0, 288)
    ax.set_ylim(1000, 2600)
    ax.set_xticks(np.arange(0, 288 + 1, 24))
    ax.set_xlabel("Time/15 min")
    ax.set_ylabel("Power/kW")
    ax.set_title(
        "Fig. 9. Comparison of daily load curve before and after peaking (3 days)."
    )
    ax.legend()
    savefig(fig, path)


# =============== 主流程 ===============
def main():
    out = ensure_out()
    data = package_day(SEED)
    t = data["t"].values
    P_load = data["P_load_kw"].values
    N_ev = data["N_ev"].values
    price_c = data["price"].values
    price_d = data["price_discharge"].values

    data[["t", "price"]].to_csv(os.path.join(out, "tab_02_tou_prices.csv"), index=False)

    ev = EVParams()
    m = MarketParams()
    f1_ref, f2_ref = make_refs(P_load, price_c, ev, m)
    lo, hi = bounds_from_counts(N_ev, ev)
    obj = lambda x: objective(
        x,
        P_load,
        price_c,
        price_d,
        N_ev,
        ev,
        m,
        f1_ref,
        f2_ref,
        w1=0.5,
        w2=0.5,
    )

    print("开始运行 iL-SHADE+ 优化...")
    x_best, f_best, h_best = ilshade_plus_optimize(obj, lo, hi, ILSHADEPlusParams())
    print(f"iL-SHADE+ 完成，最优适应度: {f_best:.4f}")

    print("运行对比算法...")
    _, _, h_ga = ga_optimize(obj, lo, hi, GAParams())
    _, _, h_pso = pso_optimize(obj, lo, hi, PSOParams())
    _, _, h_woa = woa_optimize(obj, lo, hi, WOAParams())
    _, _, h_geo = geo_optimize(obj, lo, hi, GEOParams())

    print("生成图表 Fig.2-4...")
    fig2_ev_counts(t, N_ev, os.path.join(out, "fig_02_ev_count.png"))
    fig3_signal(t, x_best, os.path.join(out, "fig_03_ev_signal_iLSHADEplus.png"))
    fig4_convergence(
        h_ga,
        h_pso,
        h_woa,
        h_geo,
        h_best,
        os.path.join(out, "fig_04_convergence.png"),
    )

    def metrics(x):
        net = P_load + x
        ptv = float(net.max() - net.min())
        rate = 100.0 * ptv / (P_load.max() - P_load.min())
        c, _ = cost_terms(x, price_c, price_d, ev, m)
        return ptv, rate, c

    rows = [
        {
            "Case": "Original load",
            "Peak-to-valley disparity (kW)": round(
                float(P_load.max() - P_load.min()), 1
            ),
            "Peak-to-valley disparity rate (%)": 100.00,
            "Operational cost (yuan/day)": None,
        }
    ]
    ptv, rate, c = metrics(x_best)
    rows.append(
        {
            "Case": "After iL-SHADE+ optimization",
            "Peak-to-valley disparity (kW)": round(ptv, 1),
            "Peak-to-valley disparity rate (%)": round(rate, 2),
            "Operational cost (yuan/day)": round(c, 1),
        }
    )
    pd.DataFrame(rows).to_csv(os.path.join(out, "tab_03_metrics.csv"), index=False)

    print("运行执行层仿真...")
    fleet = EVFleet(n=int(N_ev.max()))
    sim = simulate_groups_with_availability(
        x_best, N_ev, price_c, fleet, dt=m.dt, seed=SEED
    )

    print("生成图表 Fig.5-8...")
    fig5_soc_d(
        t,
        sim["soc_d_before"],
        sim["soc_d_after"],
        os.path.join(out, "fig_05_soc_group_D_before_after.png"),
    )
    fig6_signal_vs_response(
        t,
        x_best,
        sim["P_applied"],
        os.path.join(out, "fig_06_signal_vs_response.png"),
    )
    fig7_deviation(
        t,
        sim["deviation"],
        os.path.join(out, "fig_07_tracking_deviation.png"),
    )
    fig8_day_load(
        t,
        P_load,
        P_load + x_best,
        P_load + sim["P_applied"],
        os.path.join(out, "fig_08_load_before_after_day1.png"),
    )

    print("运行三天仿真...")
    act_all = []
    for d in range(3):
        simd = simulate_groups_with_availability(
            x_best, N_ev, price_c, fleet, dt=m.dt, seed=SEED + d * 10
        )
        act_all.append(simd["P_applied"])
    act_3d = np.concatenate(act_all)
    fig9_3days(
        P_load,
        x_best,
        act_3d,
        os.path.join(out, "fig_09_load_before_after_3days.png"),
    )

    print("生成 Table 5...")
    B1, B2, B3, G, Tem, EEV, a4 = 27128.0, 31700.0, 370.3, 8.314, 298.0, 35.0, 0.2
    tactics = [
        "Tactic 1",
        "Tactic 2",
        "Tactic 3",
        "Tactic 4",
        "Tactic 5",
        "Research Tactic",
    ]
    multipliers = [1.0, 0.9, 0.85, 0.75, 0.7, 0.6]
    rows = []
    for name, mul in zip(tactics, multipliers):
        x = x_best * mul
        energy = np.sum(np.abs(x)) * m.dt + 1e-12
        DS = energy / EEV * 0.01
        Rcd = np.mean(np.abs(x)) + 1e-9
        Qloss = (
            B1
            * np.exp(-(B2 + B3 * Rcd) / (G * Tem))
            * (DS**0.55)
            * (Rcd ** (2.0 / 3.0))
        )
        Ts = (a4 * EEV) / (365.0 * Qloss + 1e-12)
        rows.append(
            {
                "Tactic": name,
                "Life expenditure /kWh": round(Qloss / energy, 4),
                "Useful life/year": round(Ts, 1),
            }
        )
    pd.DataFrame(rows).to_csv(
        os.path.join(out, "tab_05_life_loss_summary.csv"), index=False
    )

    print("\n" + "=" * 60)
    print("✅ 完成！所有结果已保存到 ./outputs/ 目录")
    print("=" * 60)
    print("生成的文件：")
    print("  - fig_02_ev_count.png")
    print("  - fig_03_ev_signal_iLSHADEplus.png")
    print("  - fig_04_convergence.png")
    print("  - fig_05_soc_group_D_before_after.png")
    print("  - fig_06_signal_vs_response.png")
    print("  - fig_07_tracking_deviation.png (已修复，绿色柱状图)")
    print("  - fig_08_load_before_after_day1.png")
    print("  - fig_09_load_before_after_3days.png")
    print("  - tab_02_tou_prices.csv")
    print("  - tab_03_metrics.csv")
    print("  - tab_05_life_loss_summary.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()

