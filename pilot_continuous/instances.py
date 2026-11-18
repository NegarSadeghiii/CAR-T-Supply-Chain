"""Network data and instance generator for the continuous-arrival pilot.

The network (hospitals c1..c10, facilities m1 Philadelphia, m2 Houston, m3 Los Angeles, transport times and unit
costs, CVM, HC, T_LS, T_MFE, T_QC and the Liu transition rates) is read from the same workbook as
CART_doe_main.ipynb (CART_instance_30patients_cap4.xlsx). The workbook's ND, horizon, line counts and patients are
NOT used: they are set per pilot setting.

Patients arrive continuously: on every day d = 1..D the number of new patients is Poisson(rate[d]). By default
rate[d] = annual demand / 365 on every day; daily_rates() builds a nonhomogeneous vector (e.g. a demand surge).
"""
import re
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

# ----------------------------------------------------------------------------------------------------------------
# Generator settings (defaults; run_pilot.py passes its own values)
# ----------------------------------------------------------------------------------------------------------------
REGION_SHARES = {"East": 0.60, "Central": 0.13, "West": 0.27}
# demand weights per hospital (Wan et al. 2026, total orders per hospital)
DEMAND_WEIGHT = {"Boston": 24, "New York": 29, "Philadelphia": 14, "Baltimore": 23, "Durham": 12, "Chicago": 14,
                 "Houston": 14, "Denver": 30, "Los Angeles": 37, "Palo Alto": 28}


# ----------------------------------------------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------------------------------------------
def load_network(workbook):
    """Read the fixed network data from the workbook (data_only=True: cached formula values)."""
    wb = openpyxl.load_workbook(workbook, data_only=True)

    def table(sheet, first_row, ncols, prefix):
        out = []
        for r in wb[sheet].iter_rows(min_row=first_row, max_col=ncols, values_only=True):
            if not (isinstance(r[0], str) and re.fullmatch(prefix + r"\d+", r[0])):
                break
            out.append(r)
        return out

    settings = {str(lab): val for lab, val, *_ in wb["Settings"].iter_rows(min_row=4, max_col=3, values_only=True) if lab}
    get = lambda start: next(v for k, v in settings.items() if k.startswith(start))
    net = dict(workbook=str(workbook), T_LS=int(get("T_LS")), T_MFE=int(get("T_MFE")), T_QC=int(get("T_QC")),
               HC=float(get("HC")))
    fac = table("Facilities", 4, 5, "m")
    net["M"] = [r[0] for r in fac]
    net["M_CITY"] = {r[0]: r[1] for r in fac}
    net["CVM"] = {r[0]: float(r[4]) for r in fac}
    sites = table("Sites", 4, 5, "c")
    net["C"] = [r[0] for r in sites]
    net["CITY"] = {r[0]: r[1] for r in sites}
    net["REGION"] = {r[0]: r[4] for r in sites}
    net["PAIRED_H"] = {r[0]: r[3] for r in sites}
    out_rows = table("Transport_out", 4, 7, "c")
    ret_rows = table("Transport_return", 4, 7, "m")
    net["DIST1"] = {(r[0], r[1]): float(r[3]) for r in out_rows}
    net["TT1"] = {(r[0], r[1], r[2]): int(r[5]) for r in out_rows}
    net["U1"] = {(r[0], r[1], r[2]): float(r[6]) for r in out_rows}
    net["TT2"] = {(r[0], r[1], r[2]): int(r[5]) for r in ret_rows}
    net["U2"] = {(r[0], r[1], r[2]): float(r[6]) for r in ret_rows}
    net["J"] = sorted({k[2] for k in net["TT1"]} | {k[2] for k in net["TT2"]})
    net["MODE"] = {"j1": "fast", "j2": "slow"}
    net["q"] = {str(r[0]).split(":")[0]: float(r[1])
                for r in wb["Clinical"].iter_rows(min_row=4, max_row=6, max_col=2, values_only=True)}
    net["PI_SHEET"] = [r for r in wb["Pi"].iter_rows(min_row=5, max_col=7, values_only=True) if isinstance(r[0], (int, float))]
    net["NEAREST"] = {c: min(net["M"], key=lambda m: net["DIST1"][c, m]) for c in net["C"]}
    city = lambda c: net["CITY"][c].split(",")[0]
    net["CITY_SHORT"] = {c: city(c) for c in net["C"]}
    assert set(DEMAND_WEIGHT) == set(net["CITY_SHORT"].values()), "demand weights must cover the ten hospitals"
    return net


def compute_pi(q, tau_max, pi_sheet=None):
    """pi[g, tau, s] = (P^tau)[g, s], P ordered (R, U, D), eq. (21). Cross-checked against the Pi sheet (1e-6)."""
    P = np.array([[1 - q["q_RU"] - q["q_RD"], q["q_RU"], q["q_RD"]], [0.0, 1 - q["q_UD"], q["q_UD"]], [0.0, 0.0, 1.0]])
    pi, Pk = {}, np.eye(3)
    for tau in range(tau_max + 1):
        for gi, g in enumerate("RU"):
            for si, s in enumerate("RUD"):
                pi[g, tau, s] = float(Pk[gi, si])
        Pk = Pk @ P
    if pi_sheet:
        dev = max(abs(pi[g, int(r[0]), s] - v) for r in pi_sheet if int(r[0]) <= tau_max
                  for (g, s), v in zip([("R", "R"), ("R", "U"), ("R", "D"), ("U", "R"), ("U", "U"), ("U", "D")], r[1:7]))
        assert dev <= 1e-6, f"pi does not match the Pi sheet ({dev})"
    return pi


# ----------------------------------------------------------------------------------------------------------------
# Instance generator
# ----------------------------------------------------------------------------------------------------------------
def daily_rates(D, annual_demand, surge_factor=1.0, surge_start=None, surge_end=None):
    """Arrival rate (patients/day) for each day 1..D: annual_demand / 365, multiplied by surge_factor on days
    surge_start..surge_end (inclusive). Without a surge this is the constant rate used by P1-P4."""
    base = annual_demand / 365.0
    rate = {d: base for d in range(1, D + 1)}
    if surge_start is not None:
        for d in range(surge_start, surge_end + 1):
            rate[d] = base * surge_factor
    return rate


def generate_instance(net, D, annual_demand, urgent_prob, seed, region_shares=REGION_SHARES,
                      demand_weight=DEMAND_WEIGHT, max_patients=None, rate=None):
    """Continuous arrivals over days 1..D.

    rate: dict day -> expected patients that day (see daily_rates). None = annual_demand / 365 on every day, which
    reproduces exactly the instances of P1-P4.
    One numpy.random.default_rng(seed) stream, drawn day by day: the number of new patients on day d is
    Poisson(rate[d]); then for each new patient, in turn: region (independent draw with region_shares),
    hospital within the region (probability proportional to demand_weight), urgent (probability urgent_prob).
    Patients are numbered p1, p2, ... in generation order, which is also collection-day order.
    max_patients keeps only the first patients (used by the smoke test only).
    """
    rng = np.random.default_rng(seed)
    regions = list(region_shares)
    reg_p = np.array([region_shares[r] for r in regions], float)
    assert abs(reg_p.sum() - 1) < 1e-9, "region shares must sum to 1"
    sites_of = {r: [c for c in net["C"] if net["REGION"][c] == r] for r in regions}
    w_of = {r: np.array([demand_weight[net["CITY_SHORT"][c]] for c in sites_of[r]], float) for r in regions}
    if rate is None:
        rate = daily_rates(D, annual_demand)
    rows = []
    for d in range(1, D + 1):
        for _ in range(int(rng.poisson(rate[d]))):
            reg = regions[rng.choice(len(regions), p=reg_p)]
            site = sites_of[reg][rng.choice(len(sites_of[reg]), p=w_of[reg] / w_of[reg].sum())]
            urgent = bool(rng.random() < urgent_prob)
            rows.append({"collection day": d, "site": site, "hospital": net["PAIRED_H"][site], "city": net["CITY"][site],
                         "region": reg, "class": "U" if urgent else "R"})
    df = pd.DataFrame(rows, columns=["collection day", "site", "hospital", "city", "region", "class"])
    if max_patients is not None:
        df = df.head(max_patients)
    df.insert(0, "p", [f"p{i}" for i in range(1, len(df) + 1)])
    df.attrs.update(D=D, annual_demand=annual_demand, urgent_prob=urgent_prob, seed=seed)
    return df


def save_instance(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_instance(path):
    return pd.read_csv(path)


def make_data(net, patients, ND, lines, D):
    """Model data for one setting. Horizon NT = D + ND + 2."""
    d = dict(net)
    d["ND"], d["D"] = int(ND), int(D)
    d["NT"] = int(D) + int(ND) + 2
    d["FCAP"] = dict(zip(net["M"], [int(x) for x in lines]))
    d["P"] = list(patients["p"])
    d["A"] = dict(zip(patients["p"], patients["collection day"].astype(int)))
    d["SITE"] = dict(zip(patients["p"], patients["site"]))
    d["HOSP"] = dict(zip(patients["p"], patients["hospital"]))
    d["G"] = dict(zip(patients["p"], patients["class"]))
    pnum = lambda p: int(p[1:])
    d["ORDER"] = sorted(d["P"], key=lambda p: (d["A"][p], pnum(p)))     # FIFO order: collection day, then index
    d["PI"] = compute_pi(net["q"], d["NT"], net["PI_SHEET"])
    return d


def nearest_shares(net, region_shares=REGION_SHARES, demand_weight=DEMAND_WEIGHT):
    """Expected share of patients whose nearest facility (by distance) is m."""
    share = {m: 0.0 for m in net["M"]}
    for r, s in region_shares.items():
        sites = [c for c in net["C"] if net["REGION"][c] == r]
        wsum = sum(demand_weight[net["CITY_SHORT"][c]] for c in sites)
        for c in sites:
            share[net["NEAREST"][c]] += s * demand_weight[net["CITY_SHORT"][c]] / wsum
    return share


def expected_loads(net, lines, rate_per_day, region_shares=REGION_SHARES, demand_weight=DEMAND_WEIGHT):
    """Expected load at a given daily arrival rate: overall and per facility (nearest facility by distance)."""
    T = net["T_MFE"]
    lines = dict(zip(net["M"], lines))
    share = nearest_shares(net, region_shares, demand_weight)
    div = lambda x, y: x / y if y else float("inf")
    out = {"overall": div(rate_per_day * T, sum(lines.values()))}
    for m in net["M"]:
        out[f"{m} {net['M_CITY'][m].split(',')[0]}"] = div(rate_per_day * share[m] * T, lines[m])
    return out


def utilisation(net, patients, lines, D, annual_demand, region_shares=REGION_SHARES, demand_weight=DEMAND_WEIGHT,
                rate=None):
    """rho overall and per facility (nearest facility by distance), expected (from the mean daily rate) and realised.

    overall:      patients per day * T_MFE / total lines
    facility m:   patients whose nearest facility is m, per day * T_MFE / lines of m
    """
    T = net["T_MFE"]
    lines = dict(zip(net["M"], lines))
    # expected arrivals per day: mean of the daily rate vector (constant rate = annual demand / 365)
    rate = float(np.mean(list(rate.values()))) if rate is not None else annual_demand / 365.0
    share = nearest_shares(net, region_shares, demand_weight)
    n = len(patients)
    div = lambda x, y: x / y if y else float("inf")
    out = {"rho overall (expected)": div(rate * T, sum(lines.values())),
           "rho overall (realised)": div(n / D * T, sum(lines.values()))}
    for m in net["M"]:
        cnt = int((patients["site"].map(net["NEAREST"]) == m).sum())
        out[f"rho {m} {net['M_CITY'][m].split(',')[0]} (expected)"] = div(rate * share[m] * T, lines[m])
        out[f"rho {m} {net['M_CITY'][m].split(',')[0]} (realised)"] = div(cnt / D * T, lines[m])
    return out
