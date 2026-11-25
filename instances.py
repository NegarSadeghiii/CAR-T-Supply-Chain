"""Fixed network data and DOE instance generation (same logic as the earlier lambda-based DOE).

Network (read-only, data_only=True) from the Houston 4 / T_MFE 7 workbook CART_instance_30patients_cap4.xlsx:
hospitals c1..c10, facilities m1 Philadelphia, m2 Houston, m3 Los Angeles, transport times and unit costs,
CVM, HC, T_LS, T_MFE, T_QC and the Liu transition rates. Per run only three things are overwritten:
facility lines (factor C), ND (factor A) and the patients. Horizon NT = last collection day + ND + 2.

Patients: for each demand level N and replication r one numpy.random.default_rng(1000*N + r) stream, drawn in
this order:
  1. hospitals, region by region (East, Central, West; counts round(0.60 N), round(0.13 N), rest), each patient
     with probability proportional to the Wan et al. (2026) demand weights;
  2. collection days, uniform integers on 1..10;
  3. a random permutation; the urgent patients are the first round(share * N) of it, so the 10% set is inside
     the 20% set, which is inside the 30% set.
Patients are numbered p1..pN by collection day (ties in generation order); the number only breaks FIFO ties.
The same patients are used for every ND, capacity layout and urgent share with that N and r.
"""
import functools
import os
import re
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE_WORKBOOK = ROOT / "CART_instance_30patients_cap4.xlsx"     # Houston 4 / T_MFE 7

# ---- design (factors A-D) ---------------------------------------------------------------------------------------
ND_LEVELS = [41, 52]                                              # A
N_LEVELS = [30, 60]                                               # B: patients collected over days 1..10
LAYOUTS = {"C1": (10, 10, 10), "C2": (4, 10, 10), "C3": (4, 4, 10), "C4": (4, 4, 4)}   # C: (Phil, Houston, LA) lines
SHARES = [0.10, 0.20, 0.30]                                       # D: urgent share
REPS = list(range(1, 11))
COLLECTION_DAYS = 10
REGION_SHARE = {"East": 0.60, "Central": 0.13}                    # West = the rest
DEMAND_WEIGHT = {"Boston": 24, "New York": 29, "Philadelphia": 14, "Baltimore": 23, "Durham": 12, "Chicago": 14,
                 "Houston": 14, "Denver": 30, "Los Angeles": 37, "Palo Alto": 28}
pnum = lambda p: int(p[1:])


@functools.lru_cache(maxsize=None)
def load_base(path=str(BASE_WORKBOOK)):
    wb = openpyxl.load_workbook(path, data_only=True)

    def table(sheet, first_row, ncols, prefix):
        out = []
        for r in wb[sheet].iter_rows(min_row=first_row, max_col=ncols, values_only=True):
            if not (isinstance(r[0], str) and re.fullmatch(prefix + r"\d+", r[0])):
                break
            out.append(r)
        return out

    settings = {str(lab): val for lab, val, *_ in wb["Settings"].iter_rows(min_row=4, max_col=3, values_only=True) if lab}
    get = lambda start: next(v for k, v in settings.items() if k.startswith(start))
    b = dict(T_LS=int(get("T_LS")), T_MFE=int(get("T_MFE")), T_QC=int(get("T_QC")), HC=float(get("HC")))
    fac = table("Facilities", 4, 5, "m")
    b["M"] = [r[0] for r in fac]
    b["M_CITY"] = {r[0]: r[1] for r in fac}
    b["CVM"] = {r[0]: float(r[4]) for r in fac}
    sites = table("Sites", 4, 5, "c")
    b["C"] = [r[0] for r in sites]
    b["CITY"] = {r[0]: r[1] for r in sites}
    b["REGION"] = {r[0]: r[4] for r in sites}
    b["PAIRED_H"] = {r[0]: r[3] for r in sites}
    out_rows = table("Transport_out", 4, 7, "c")
    ret_rows = table("Transport_return", 4, 7, "m")
    b["DIST1"] = {(r[0], r[1]): float(r[3]) for r in out_rows}
    b["TT1"] = {(r[0], r[1], r[2]): int(r[5]) for r in out_rows}
    b["U1"] = {(r[0], r[1], r[2]): float(r[6]) for r in out_rows}
    b["TT2"] = {(r[0], r[1], r[2]): int(r[5]) for r in ret_rows}
    b["U2"] = {(r[0], r[1], r[2]): float(r[6]) for r in ret_rows}
    b["J"] = sorted({k[2] for k in b["TT1"]} | {k[2] for k in b["TT2"]})
    b["MODE"] = {"j1": "fast", "j2": "slow"}
    q = {str(r[0]).split(":")[0]: float(r[1]) for r in wb["Clinical"].iter_rows(min_row=4, max_row=6, max_col=2, values_only=True)}
    Pm = np.array([[1 - q["q_RU"] - q["q_RD"], q["q_RU"], q["q_RD"]], [0.0, 1 - q["q_UD"], q["q_UD"]], [0.0, 0.0, 1.0]])
    tau_max = COLLECTION_DAYS + max(ND_LEVELS) + 2
    pi, Pk = {}, np.eye(3)
    for tau in range(tau_max + 1):
        for gi, g in enumerate("RU"):
            for si, s in enumerate("RUD"):
                pi[g, tau, s] = float(Pk[gi, si])
        Pk = Pk @ Pm
    dev = 0.0
    for r in wb["Pi"].iter_rows(min_row=5, max_col=7, values_only=True):
        if not isinstance(r[0], (int, float)):
            break
        for (g, s), v in zip([("R", "R"), ("R", "U"), ("R", "D"), ("U", "R"), ("U", "U"), ("U", "D")], r[1:7]):
            dev = max(dev, abs(pi[g, int(r[0]), s] - v))
    assert dev <= 1e-6, f"pi does not match the Pi sheet ({dev})"
    b["PI"] = pi
    b["NEAREST"] = {c: min(b["M"], key=lambda m: b["DIST1"][c, m]) for c in b["C"]}
    b["CITY_SHORT"] = {c: b["CITY"][c].split(",")[0] for c in b["C"]}
    assert set(DEMAND_WEIGHT) == set(b["CITY_SHORT"].values())
    return b


def region_counts(N):
    nE, nC = round(REGION_SHARE["East"] * N), round(REGION_SHARE["Central"] * N)
    return {"East": nE, "Central": nC, "West": N - nE - nC}


def generate(N, r):
    base = load_base()
    rng = np.random.default_rng(1000 * N + r)
    hosp, region = [], []
    for reg, n in region_counts(N).items():
        sites = [c for c in base["C"] if base["REGION"][c] == reg]
        w = np.array([DEMAND_WEIGHT[base["CITY_SHORT"][c]] for c in sites], float)
        hosp += list(rng.choice(sites, size=n, p=w / w.sum()))
        region += [reg] * n
    days = rng.integers(1, COLLECTION_DAYS + 1, size=N)
    perm = rng.permutation(N)
    rank = np.empty(N, int)
    rank[perm] = np.arange(N)
    order = sorted(range(N), key=lambda i: (days[i], i))
    rows = []
    for k, i in enumerate(order, start=1):
        rows.append({"p": f"p{k}", "collection day": int(days[i]), "site": hosp[i], "hospital": base["PAIRED_H"][hosp[i]],
                     "city": base["CITY"][hosp[i]], "region": region[i], "permutation rank": int(rank[i]) + 1,
                     **{f"urgent at {int(round(s * 100))}%": "U" if rank[i] < round(s * N) else "R" for s in SHARES}})
    return pd.DataFrame(rows)


def instance_path(out_dir, N, r):
    return Path(out_dir) / "instances" / f"patients_N{N}_r{r:02d}.csv"


def write_instances(out_dir):
    """Generate every (N, r) patient list once and save it; existing files are kept (and reused by the runs)."""
    for N in N_LEVELS:
        for r in REPS:
            path = instance_path(out_dir, N, r)
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            df = generate(N, r)
            cnt = df.region.value_counts()
            assert (cnt.get("East", 0), cnt.get("Central", 0), cnt.get("West", 0)) == tuple(region_counts(N).values())
            for s in SHARES:
                assert (df[f"urgent at {int(round(s * 100))}%"] == "U").sum() == round(s * N)
            u = {s: set(df[df[f"urgent at {int(round(s * 100))}%"] == "U"].p) for s in SHARES}
            assert u[0.10] <= u[0.20] <= u[0.30]
            df.to_csv(path, index=False)


@functools.lru_cache(maxsize=None)
def patients(out_dir, N, r):
    return pd.read_csv(instance_path(out_dir, N, r))


def make_data(out_dir, ND, N, layout, share, r):
    """Model data for one instance (one combination of factors A-D and one replication)."""
    base = load_base()
    pat = patients(str(out_dir), N, r)
    gcol = f"urgent at {int(round(share * 100))}%"
    d = dict(base)
    d.update(ND=ND, NT=int(pat["collection day"].max()) + ND + 2, FCAP=dict(zip(base["M"], LAYOUTS[layout])),
             P=list(pat.p), A=dict(zip(pat.p, pat["collection day"].astype(int))), SITE=dict(zip(pat.p, pat.site)),
             HOSP=dict(zip(pat.p, pat.hospital)), G=dict(zip(pat.p, pat[gcol])))
    d["ORDER"] = sorted(d["P"], key=lambda p: (d["A"][p], pnum(p)))     # FIFO order: collection day, then index
    return d
