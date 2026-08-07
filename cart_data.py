"""
Data layer for the survival-aware i-SHIPMENT scheduling study.

Responsibilities
----------------
1. Parse ``Data200_profileA.dat`` (AMPL-style .dat used by the baseline
   i-SHIPMENT Pyomo notebook) into plain Python dicts.
2. Build scaled patient cohorts (100 / 200 / 500 patients) by tiling the
   50-patient base cohort, preserving the tier mix and the leukapheresis-site
   distribution, and re-spreading arrivals over the planning window so that
   *demand density* grows with N.
3. Hold the FIXED clinical calibration (tiers, survival curve, sigma lookup,
   clinical-loss weights). Nothing here is refit anywhere in the study.

The baseline notebook (``i-SHIPMENT_Pyomo.ipynb``) is never modified or
imported; this module only reads its data file.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, field, asdict

# --------------------------------------------------------------------------
# FIXED CALIBRATION  (gamma = 1, eta = 42, kappa = 1) -- do not refit
# --------------------------------------------------------------------------
GAMMA = 1.0          # survival-model shape
ETA = 42             # survival reference horizon (days)
KAPPA = 1.0          # survival-model scale

TIER_MIX = {"H": 0.30, "M": 0.40, "L": 0.30}     # 30% high / 40% med / 30% low
W_RISK = {"H": 0.15, "M": 0.05, "L": 0.02}       # per-eta deterioration rate
RHO = {"H": 3.0, "M": 2.0, "L": 1.0}             # clinical-loss weights

TIER_ORDER = ("H", "M", "L")

# Turnaround-time support.  TRT = TLS + TT1 + TMFE + TQC + TT3 + HOLD
#   min = 1 + 1 + 7 + 7 + 1 + 0 = 17 days   (all-air, no hold)
#   max = ND = 42 days                      (eq. 31, relaxed from 18)
D_MIN, D_MAX = 17, 42
D_RANGE = tuple(range(D_MIN, D_MAX + 1))

# Planning horizon of the baseline model.
HORIZON = 130
# Latest admissible leukapheresis arrival: the therapy must still be delivered
# inside the 130-day horizon when the full ND = 42 turnaround is used.
ARRIVAL_WINDOW = (1, HORIZON - D_MAX)            # -> days 1..88

SEED = 0


def survival(tier: str, t: float) -> float:
    """S_u(t) = (1 - w_u) ** (t / eta), with gamma = kappa = 1."""
    return (1.0 - W_RISK[tier]) ** (KAPPA * (t / ETA) ** GAMMA)


def sigma_table() -> dict:
    """sigma[d][u] for integer days d = 17..42 -- precomputed once, reused."""
    return {d: {u: survival(u, d) for u in TIER_ORDER} for d in D_RANGE}


# --------------------------------------------------------------------------
# .dat parsing
# --------------------------------------------------------------------------
def _strip_comments(text: str) -> str:
    return "\n".join(line.split("#")[0] for line in text.splitlines())


def parse_dat(path: str) -> dict:
    """Parse the AMPL-style .dat file used by the baseline notebook."""
    text = _strip_comments(open(path).read())
    blocks = [b.strip() for b in text.split(";") if b.strip()]

    data: dict = {"sets": {}, "params": {}, "scalars": {}}
    for block in blocks:
        tokens = block.split()
        if tokens[0] == "set":
            name = tokens[1]
            body = tokens[2:]
            if body and body[0] == ":=":
                body = body[1:]
            data["sets"][name] = body
        elif tokens[0] == "param":
            name = tokens[1]
            body = tokens[2:]
            if body and body[0] == ":=":
                body = body[1:]
            if len(body) == 1:                      # scalar parameter
                data["scalars"][name] = float(body[0])
                continue
            # Indexed parameter: trailing token of every record is the value;
            # arity is inferred from the total token count.
            entries = []
            # Records are (k_1 ... k_n, value); infer n from the first record by
            # scanning until a numeric token that is followed by a non-numeric
            # token or end-of-record. Simpler: the .dat uses a fixed arity per
            # param, determined by the known signature below.
            arity = {"CIM": 1, "CVM": 1, "FCAP": 1, "TT1": 1, "TT3": 1,
                     "U1": 3, "U3": 3, "INC": 3}[name]
            step = arity + 1
            for i in range(0, len(body), step):
                rec = body[i:i + step]
                key = tuple(rec[:arity])
                entries.append((key if arity > 1 else key[0], float(rec[-1])))
            data["params"][name] = dict(entries)
    return data


# --------------------------------------------------------------------------
# Network constants shared by every phase (never re-fit)
# --------------------------------------------------------------------------
@dataclass
class Network:
    c: list                 # leukapheresis sites
    h: list                 # hospitals
    j: list                 # transport modes (j1 = air, j2 = ground)
    m: list                 # manufacturing sites
    CIM: dict               # capital investment per facility
    CVM: dict               # fixed variable cost per facility
    FCAP: dict              # concurrent capacity per facility
    TT1: dict               # transport time LS -> MS by mode
    TT3: dict               # transport time MS -> hospital by mode
    U1: dict                # unit transport cost (c, m, j)
    U3: dict                # unit transport cost (m, h, j)
    TLS: float              # leukapheresis duration
    TAD: float              # administration duration
    TMFE: int = 7           # manufacturing duration excl. QC
    TQC: int = 7            # QC duration
    C_material: float = 10476.0
    CQC: float = 9312.0
    FMIN: float = 0.0
    FMAX: float = 1.0
    horizon: int = HORIZON

    @property
    def hospital_of(self) -> dict:
        """CON12-CON15: hospital h_k is co-located with leukapheresis site c_k."""
        return {ci: hi for ci, hi in zip(self.c, self.h)}


def load_network(dat_path: str) -> Network:
    d = parse_dat(dat_path)
    return Network(
        c=d["sets"]["c"], h=d["sets"]["h"], j=d["sets"]["j"], m=d["sets"]["m"],
        CIM=d["params"]["CIM"], CVM=d["params"]["CVM"],
        FCAP={k: int(v) for k, v in d["params"]["FCAP"].items()},
        TT1={k: int(v) for k, v in d["params"]["TT1"].items()},
        TT3={k: int(v) for k, v in d["params"]["TT3"].items()},
        U1=d["params"]["U1"], U3=d["params"]["U3"],
        TLS=d["scalars"]["TLS"], TAD=d["scalars"]["TAD"],
        FMIN=d["scalars"]["FMIN"], FMAX=d["scalars"]["FMAX"],
    )


def base_cohort(dat_path: str):
    """The 50-patient base cohort as [(patient_id, c, arrival_day), ...]."""
    d = parse_dat(dat_path)
    inc = d["params"]["INC"]
    order = d["sets"]["p"]
    rec = {}
    for (p, c, t), v in inc.items():
        if v > 0.5:
            rec[p] = (c, int(t))
    return [(p, rec[p][0], rec[p][1]) for p in order]


# --------------------------------------------------------------------------
# Scaled instance generation
# --------------------------------------------------------------------------
@dataclass
class Patient:
    pid: str
    c: str          # leukapheresis site
    h: str          # co-located hospital
    t0: int         # leukapheresis arrival day (INC[p, c, t0] = 1)
    tier: str       # 'H' | 'M' | 'L'


@dataclass
class Instance:
    n: int
    mult: int
    seed: int
    patients: list = field(default_factory=list)
    arrival_window: tuple = ARRIVAL_WINDOW

    @property
    def density(self) -> float:
        lo, hi = self.arrival_window
        return self.n / (hi - lo + 1)

    def to_records(self):
        return [asdict(p) for p in self.patients]


def assign_tiers(pids, seed=SEED):
    """Fixed 30/40/30 H/M/L split of the base cohort (seeded, reproducible)."""
    rng = random.Random(seed)
    shuffled = list(pids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_h = int(round(n * TIER_MIX["H"]))
    n_m = int(round(n * TIER_MIX["M"]))
    tiers = {}
    for i, p in enumerate(shuffled):
        tiers[p] = "H" if i < n_h else ("M" if i < n_h + n_m else "L")
    return tiers


def build_instance(dat_path: str, mult: int, seed: int = SEED,
                   jitter: int = 3) -> Instance:
    """
    Tile the 50-patient base cohort ``mult`` times.

    * tier mix (30/40/30) and leukapheresis-site distribution are inherited
      patient-by-patient from the base cohort, so they are preserved exactly;
    * arrival days are rescaled into the admissible window [1, 88] and each
      replica is given a small seeded jitter, so N patients are spread over the
      SAME window -> arrival DENSITY grows linearly with N (this is what loads
      the factories);
    * costs, transport, FCAP and the network structure are untouched.
    """
    base = base_cohort(dat_path)
    tiers = assign_tiers([p for p, _, _ in base], seed=seed)
    lo, hi = ARRIVAL_WINDOW
    days = [t for _, _, t in base]
    d_lo, d_hi = min(days), max(days)
    scale = (hi - lo) / (d_hi - d_lo)

    rng = random.Random(seed + 1)
    hosp = {"c1": "h1", "c2": "h2", "c3": "h3", "c4": "h4"}
    patients = []
    for r in range(mult):
        for (pid, c, t) in base:
            t_scaled = lo + (t - d_lo) * scale
            off = 0 if mult == 1 and r == 0 else rng.randint(-jitter, jitter)
            day = int(min(hi, max(lo, round(t_scaled + off))))
            new_id = pid if mult == 1 else f"{pid}r{r}"
            patients.append(Patient(new_id, c, hosp[c], day, tiers[pid]))
    return Instance(n=len(patients), mult=mult, seed=seed, patients=patients)


def save_instance(inst: Instance, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"n": inst.n, "mult": inst.mult, "seed": inst.seed,
                   "arrival_window": list(inst.arrival_window),
                   "density_per_day": inst.density,
                   "patients": inst.to_records()}, f, indent=1)
