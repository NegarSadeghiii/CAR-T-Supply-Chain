"""
WSC 2026 — CAR-T Supply Chain Paper
Generate all publication-quality figures.

Palette: deep navy #2C3E6B | muted teal #3A7D8C | warm amber #D4882A
         light grey #B0B8C1 | white background
Font: serif (Times New Roman where available)

Output → WSC2026_Draft/
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)))

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY   = '#2C3E6B'
TEAL   = '#3A7D8C'
AMBER  = '#D4882A'
GREY   = '#B0B8C1'
WHITE  = '#FFFFFF'
RED_X  = '#C0392B'

# Urgency colours
C_HIGH = NAVY
C_MED  = TEAL
C_LOW  = AMBER

# Model colours
C_CCP = NAVY
C_TS  = TEAL

# ── Global rcParams ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size':          11,
    'axes.titlesize':     12,
    'axes.labelsize':     11,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    10,
    'figure.facecolor':   WHITE,
    'axes.facecolor':     WHITE,
    'axes.edgecolor':     '#444444',
    'axes.linewidth':     0.8,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.25,
    'grid.linestyle':     '--',
    'grid.color':         GREY,
    'lines.linewidth':    2.0,
    'figure.dpi':         150,
})

# ── Data ──────────────────────────────────────────────────────────────────────

SIGMA = [0.10, 0.20, 0.30, 0.40]

# Out-of-sample violation rates (%). None = infeasible.
VIOL_CCP_N15 = [31.0, 41.8, None, None]
VIOL_TS_N15  = [2.7,  15.7, 34.0, None]
VIOL_CCP_N50 = [73.5, 84.0, None, None]
VIOL_TS_N50  = [10.0, 44.1, 74.6, None]

# τ-sensitivity N=15 (costs in $M, None = infeasible)
TAU_VALS  = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
TAU_CCP15 = [6.3392, 6.3392, 6.3427, 6.3629, 6.3892, None]
TAU_TS15  = [6.3393, 6.3401, 6.3453, 6.3588, 6.3908, 6.4111]

# Normalise to τ=0.0 baseline
_base = TAU_TS15[0]
TAU_CCP_NORM = [v / _base if v is not None else None for v in TAU_CCP15]
TAU_TS_NORM  = [v / _base for v in TAU_TS15]

# Per-urgency expediting (from patient-level model output)
URG_GROUPS = ['High', 'Medium', 'Low']
URG_PEXP   = [8.8, 3.0, 22.1]     # P(expedite) %
URG_COLORS = [C_HIGH, C_MED, C_LOW]
# Expediting cost = π × P(exp): π_H=8000, π_M=4000, π_L=1500
PI_MAP = {'High': 8000, 'Medium': 4000, 'Low': 1500}
URG_EXPCOST = [PI_MAP[g] * p / 100 for g, p in zip(URG_GROUPS, URG_PEXP)]

# Gantt data (N=15 representative schedule)
GANTT = [
    {'facility': 'Facility 1', 'fcap': 4, 'patients': [
        {'id': 'p7',  'group': 'high',   'start': 3,  'end': 10, 'expedite': False},
        {'id': 'p2',  'group': 'medium', 'start': 4,  'end': 11, 'expedite': False},
        {'id': 'p4',  'group': 'medium', 'start': 8,  'end': 15, 'expedite': False},
        {'id': 'p25', 'group': 'high',   'start': 12, 'end': 19, 'expedite': True},
        {'id': 'p8',  'group': 'medium', 'start': 13, 'end': 20, 'expedite': False},
        {'id': 'p46', 'group': 'high',   'start': 14, 'end': 21, 'expedite': True},
        {'id': 'p12', 'group': 'low',    'start': 18, 'end': 25, 'expedite': False},
        {'id': 'p43', 'group': 'medium', 'start': 20, 'end': 27, 'expedite': True},
        {'id': 'p19', 'group': 'low',    'start': 22, 'end': 29, 'expedite': False},
    ]},
    {'facility': 'Facility 2', 'fcap': 4, 'patients': [
        {'id': 'p38', 'group': 'low',    'start': 14, 'end': 21, 'expedite': False},
    ]},
    {'facility': 'Facility 3', 'fcap': 10, 'patients': []},   # spare capacity
]

GROUP_COLOR = {'high': C_HIGH, 'medium': C_MED, 'low': C_LOW}

SIGMA_STR = [f'{s:.2f}' for s in SIGMA]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: save figure
# ─────────────────────────────────────────────────────────────────────────────
def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches='tight', dpi=300, facecolor=WHITE)
    plt.close(fig)
    print(f'  Saved: {path}')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Process Flow
# ─────────────────────────────────────────────────────────────────────────────
def fig_process_flow():
    from matplotlib.patches import Ellipse
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis('off')

    # Fixed, equal box dimensions and uniform spacing
    BOX_W = 1.50
    BOX_H = 0.62
    GAP   = 0.42          # gap between boxes
    STEP  = BOX_W + GAP   # 1.92 per slot
    X0    = 0.85          # left edge of first box centre
    Y     = 2.60          # main process row y

    steps = [
        ('Leukapheresis',    r'$T_{LS} = 1$ day',                     NAVY+'22'),
        ('Outbound\nTransport', r'$TT_1 \in \{1,4\}$ days',           NAVY+'22'),
        ('Manufacturing',    r'$T_{MF} \sim \mathrm{LogN}(\mu,\sigma_L^2)$', AMBER+'44'),
        ('Quality\nControl', r'$T_{QC} = 7$ days',                    NAVY+'22'),
        ('Return\nTransport', r'$TT_3 \in \{1,4\}$ days',             TEAL+'44'),
        ('Infusion',         r'(patient)',                             '#2E7D3222'),
    ]

    xcs = [X0 + i * STEP for i in range(len(steps))]

    for (title, sub, fc), xc in zip(steps, xcs):
        rect = FancyBboxPatch((xc - BOX_W/2, Y - BOX_H/2), BOX_W, BOX_H,
                              boxstyle='round,pad=0.06', lw=1.2,
                              edgecolor='#333333', facecolor=fc, zorder=3)
        ax.add_patch(rect)
        ax.text(xc, Y + 0.09, title, ha='center', va='center',
                fontsize=9.0, fontweight='bold', zorder=4)
        ax.text(xc, Y - 0.20, sub, ha='center', va='center',
                fontsize=7.5, color='#555555', zorder=4)

    # Arrows between boxes
    for i in range(len(steps) - 1):
        x_start = xcs[i]   + BOX_W/2 + 0.04
        x_end   = xcs[i+1] - BOX_W/2 - 0.04
        ax.annotate('', xy=(x_end, Y), xytext=(x_start, Y),
                    arrowprops=dict(arrowstyle='->', color='#333333',
                                   lw=1.5, mutation_scale=14), zorder=5)

    # Stage 1 bracket — spans leukapheresis through return transport (steps 0–4)
    s1_x0 = xcs[0] - BOX_W/2
    s1_x1 = xcs[4] + BOX_W/2
    s1_y  = Y - BOX_H/2 - 0.20
    ax.annotate('', xy=(s1_x1, s1_y), xytext=(s1_x0, s1_y),
                arrowprops=dict(arrowstyle='<->', color=NAVY, lw=1.4))
    ax.text((s1_x0 + s1_x1) / 2, s1_y - 0.16,
            'Stage 1: facility and route selection (column generation)',
            ha='center', fontsize=9, color=NAVY, fontstyle='italic')

    # Stage 2 dashed oval — encloses ONLY the Return Transport box (step 4)
    rt_xc = xcs[4]
    ell = Ellipse((rt_xc, Y), BOX_W + 0.42, BOX_H + 0.34,
                  linewidth=2.0, edgecolor=AMBER, facecolor='none',
                  linestyle='--', zorder=6)
    ax.add_patch(ell)

    # Label below the diagram with an arrow pointing up to the oval
    label_y = s1_y - 0.48
    ax.annotate(
        r'Stage 2: if $T_{MF} > K_i$, switch ground$\to$air ($\delta$ days saved, cost $\pi_p$)',
        xy=(rt_xc, Y - BOX_H/2 - 0.17),        # tip of arrow at oval bottom
        xytext=(rt_xc, label_y),                # label sits below Stage 1 bracket
        ha='center', va='top',
        fontsize=8.5, color=AMBER, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=AMBER, lw=1.4, shrinkA=2),
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                  edgecolor=AMBER, alpha=0.95, linewidth=1.2),
        zorder=7,
    )

    # Auto-scale axes to content
    x_total = xcs[-1] + BOX_W/2 + 0.3
    ax.set_xlim(0, x_total)
    ax.set_ylim(label_y - 0.3, Y + BOX_H/2 + 0.5)

    ax.set_title('CAR-T Cell Therapy Supply Chain — Process Flow',
                 fontsize=13, pad=8, fontweight='bold')
    _save(fig, 'fig_process_flow.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Violation Bar Chart (two panel groups: N=15 | N=50)
# ─────────────────────────────────────────────────────────────────────────────
def fig_violation_bar():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.0), sharey=True)
    fig.subplots_adjust(wspace=0.08, top=0.88, bottom=0.22)

    def _draw_panel(ax, viol_ccp, viol_ts, title):
        n_s   = len(SIGMA)
        x     = np.arange(n_s)
        w     = 0.32
        gap   = 0.04

        ccp_vals = [v if v is not None else 0 for v in viol_ccp]
        ts_vals  = [v if v is not None else 0 for v in viol_ts]

        b1 = ax.bar(x - w/2 - gap/2, ccp_vals, w,
                    label='CCP ($\\tau=0.4$)', color=C_CCP, alpha=0.88,
                    hatch='//', edgecolor=C_CCP, linewidth=0.6)
        b2 = ax.bar(x + w/2 + gap/2, ts_vals,  w,
                    label='Two-stage stochastic', color=C_TS, alpha=0.88,
                    edgecolor=C_TS, linewidth=0.6)

        # Value labels + n/a markers
        for bar, orig in zip(b1, viol_ccp):
            h = bar.get_height()
            if orig is None:
                ax.text(bar.get_x() + bar.get_width()/2, 1.5,
                        'n/a', ha='center', va='bottom', fontsize=8,
                        color=GREY, fontstyle='italic')
            elif h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 1.2,
                        f'{h:.0f}', ha='center', va='bottom', fontsize=9)

        for bar, orig in zip(b2, viol_ts):
            h = bar.get_height()
            if orig is None:
                ax.text(bar.get_x() + bar.get_width()/2, 1.5,
                        'n/a', ha='center', va='bottom', fontsize=8,
                        color=GREY, fontstyle='italic')
            elif h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 1.2,
                        f'{h:.0f}', ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels([f'$\\sigma_L={s:.2f}$' for s in SIGMA], fontsize=10)
        ax.set_xlabel('Manufacturing Variability $\\sigma_L$')
        ax.set_title(title, fontsize=12, pad=6)
        ax.set_ylim(0, 98)
        return b1, b2

    b1, b2 = _draw_panel(ax1, VIOL_CCP_N15, VIOL_TS_N15, '$N = 15$ patients')
    _draw_panel(ax2, VIOL_CCP_N50, VIOL_TS_N50, '$N = 50$ patients')

    ax1.set_ylabel('Out-of-sample Deadline Violation Rate (\\%)')

    # Per-panel legends — added after both panels are drawn
    for ax in (ax1, ax2):
        ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99),
                  framealpha=0.9, fontsize=10)

    # Footnote sits inside the bottom margin (below xlabel, above figure edge)
    fig.text(0.5, 0.06,
             '$n = 2{,}000$ held-out scenarios. "n/a" = model infeasible.',
             ha='center', fontsize=9, color='#666666', fontstyle='italic')

    fig.suptitle('Out-of-sample Deadline Violation Rate: CCP vs.\ Two-stage Stochastic',
                 fontsize=13, fontweight='bold')
    _save(fig, 'fig_violation_bar.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — τ-Sensitivity
# ─────────────────────────────────────────────────────────────────────────────
def fig_tau_sensitivity():
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    tau_arr = np.array(TAU_VALS)
    ccp_arr = np.array([v if v is not None else np.nan for v in TAU_CCP_NORM])
    ts_arr  = np.array(TAU_TS_NORM)

    # Two-stage — solid
    ax.plot(tau_arr, ts_arr, 'o-', color=C_TS, lw=2.2, ms=7, zorder=4,
            label='Two-stage stochastic (always feasible)')

    # CCP — dashed, only feasible points
    feas_mask = ~np.isnan(ccp_arr)
    ax.plot(tau_arr[feas_mask], ccp_arr[feas_mask], 's--',
            color=C_CCP, lw=2.2, ms=7, zorder=4,
            label='CCP (dashed = feasible)')

    # Infeasible ✕ markers
    infeas_tau = tau_arr[~feas_mask]
    if len(infeas_tau) > 0:
        # Project CCP trend to the infeasible points for visual
        last_feasible_y = ccp_arr[feas_mask][-1]
        for it in infeas_tau:
            ax.plot(it, last_feasible_y + 0.0005, 'x',
                    color=RED_X, ms=14, mew=3, zorder=6)
            ax.annotate('CCP infeasible',
                        xy=(it, last_feasible_y + 0.0005),
                        xytext=(it - 0.25, last_feasible_y + 0.003),
                        fontsize=9, color=RED_X, fontstyle='italic',
                        arrowprops=dict(arrowstyle='->', color=RED_X,
                                        lw=1.2, shrinkB=4))

    ax.set_xlabel('Deadline Tightness Parameter $\\tau$')
    ax.set_ylabel('Normalised Expected Total Cost\n(relative to $\\tau = 0$)')
    ax.set_title('Cost vs.\ Deadline Tightness\n'
                 '($N=15$, $\\sigma_L = 0.20$)', pad=8)
    ax.set_xticks(tau_arr)
    ax.set_xticklabels([f'{t:.1f}' for t in tau_arr])
    ax.legend(framealpha=0.9, loc='upper left')

    _save(fig, 'fig_tau_sensitivity.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Urgency Analysis (two panels)
# ─────────────────────────────────────────────────────────────────────────────
def fig_urgency_analysis():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    fig.subplots_adjust(wspace=0.32)

    x = np.arange(len(URG_GROUPS))
    w = 0.48

    def _label_bars(ax, bars, fmt='{}'):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                    fmt.format(f'{h:.1f}'),
                    ha='center', va='bottom', fontsize=9)

    # Panel A — P(expedite)
    b1 = ax1.bar(x, URG_PEXP, w, color=URG_COLORS, alpha=0.88,
                 edgecolor='white', linewidth=0.8)
    _label_bars(ax1, b1, '{}\\%')
    ax1.set_xticks(x); ax1.set_xticklabels(URG_GROUPS)
    ax1.set_xlabel('Urgency Class')
    ax1.set_ylabel('Expediting Probability (\\%)')
    ax1.set_title('P(Stage 2 Expediting\nActivated)', pad=6)
    ax1.set_ylim(0, max(URG_PEXP) * 1.35)

    # Panel B — Expected expediting cost
    b2 = ax2.bar(x, URG_EXPCOST, w, color=URG_COLORS, alpha=0.88,
                 edgecolor='white', linewidth=0.8)
    for bar, val in zip(b2, URG_EXPCOST):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'\\${val:.0f}',
                 ha='center', va='bottom', fontsize=9)
    ax2.set_xticks(x); ax2.set_xticklabels(URG_GROUPS)
    ax2.set_xlabel('Urgency Class')
    ax2.set_ylabel('Expected Expediting Cost (\\$ per patient)')
    ax2.set_title('Expected Expediting Cost\nper Patient', pad=6)
    ax2.set_ylim(0, max(URG_EXPCOST) * 1.35)

    # Shared legend
    patches = [mpatches.Patch(color=c, alpha=0.88, label=g)
               for c, g in zip(URG_COLORS, URG_GROUPS)]
    fig.legend(handles=patches, loc='lower center', ncol=3,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle('Per-Patient Expediting Analysis by Urgency Class\n'
                 '(Two-stage model, $N=15$, $\\sigma_L = 0.20$)',
                 fontsize=12, fontweight='bold', y=1.02)

    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(bottom=0.18)
    _save(fig, 'fig_urgency_analysis.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Gantt Chart
# ─────────────────────────────────────────────────────────────────────────────
def fig_gantt():
    # Flatten rows with facility separators
    rows = []
    fac_bands = []
    y = 0
    for fac_data in GANTT:
        y_start = y
        for p in fac_data['patients']:
            rows.append({**p, 'facility': fac_data['facility'],
                         'fcap': fac_data['fcap'], 'y': y})
            y += 1
        if not fac_data['patients']:
            # empty facility: still reserve one row for label
            rows.append({'id': '—', 'group': 'low', 'start': 0, 'end': 0,
                         'expedite': False, 'facility': fac_data['facility'],
                         'fcap': fac_data['fcap'], 'y': y, 'empty': True})
            y += 1
        fac_bands.append((fac_data['facility'], y_start, y - 1,
                          fac_data['fcap']))

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(10, n_rows * 0.70 + 2.0))
    BAR_H = 0.62

    for row in rows:
        if row.get('empty'):
            ax.text(0.5, row['y'], '(no patients assigned)',
                    va='center', ha='left', fontsize=8.5,
                    color=GREY, fontstyle='italic')
            continue

        color   = GROUP_COLOR[row['group']]
        dur     = row['end'] - row['start']
        mid     = row['start'] + dur / 2

        # Main bar
        ax.barh(row['y'], dur, left=row['start'], height=BAR_H,
                color=color, alpha=0.80, edgecolor='white', linewidth=0.7)

        # Hatched overlay if expediting activated
        if row['expedite']:
            ax.barh(row['y'], dur, left=row['start'], height=BAR_H,
                    color='none', edgecolor=GREY, hatch='///',
                    linewidth=0.4, alpha=0.9, zorder=4)

        # Day label: inside for wide bars, outside for narrow bars (8pt)
        label_text = f"D{row['start']}–{row['end']}"
        if dur > 3:
            ax.text(mid, row['y'], label_text,
                    ha='center', va='center', fontsize=8,
                    color='white', fontweight='bold')
        else:
            ax.text(row['end'] + 0.4, row['y'], label_text,
                    ha='left', va='center', fontsize=8, color='#333333')

    # Facility background bands + divider lines with labels above each group
    band_colors = [NAVY + '0A', TEAL + '0A', AMBER + '0A']
    xmax = max((r['end'] for r in rows if not r.get('empty')), default=35) + 5
    for i, (fname, y0, y1, fcap) in enumerate(fac_bands):
        ax.axhspan(y0 - 0.5, y1 + 0.5, color=band_colors[i % 3], zorder=0)
        if i > 0:
            # Divider line between facility groups
            ax.axhline(y0 - 0.5, color='#888888', lw=0.9, zorder=1)
            # Label just above the divider (y0 - 0.55 is visually above divider
            # on inverted axis; clip_on=False lets it render over the axes frame)
            ax.text(0, y0 - 0.55, f'{fname}  (capacity {fcap})',
                    ha='left', va='bottom', fontsize=9, clip_on=False,
                    color=NAVY, fontweight='bold')
        else:
            # First facility: label inside the band, just above the first row
            # (y0 + 0.45 on inverted axis = above y0=0, but inside axes limits)
            ax.text(0, y0 - 0.02, f'{fname}  (capacity {fcap})',
                    ha='left', va='bottom', fontsize=9, clip_on=False,
                    color=NAVY, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              edgecolor='none', alpha=0.75))

    # FCAP capacity line per facility (dashed red, no redundant text label)
    for fac_data in GANTT:
        fcap = fac_data['fcap']
        pats = [r for r in rows if r['facility'] == fac_data['facility']
                and not r.get('empty')]
        if len(pats) >= fcap:
            cap_y = pats[fcap - 1]['y'] + BAR_H / 2 + 0.10
            ax.plot([0, xmax - 2], [cap_y, cap_y],
                    color=RED_X, lw=1.6, linestyle='--', alpha=0.85, zorder=5)

    # Axes
    ax.set_yticks([r['y'] for r in rows])
    ax.set_yticklabels([r['id'] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(-1, xmax)
    ax.set_xlabel('Manufacturing Day')
    ax.set_title('Manufacturing Schedule — Gantt Chart\n'
                 '($N=15$, representative scenario; dashed red = facility capacity limit)',
                 pad=8, fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.25, linestyle='--', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)

    # Legend
    legend_patches = [
        mpatches.Patch(color=C_HIGH, alpha=0.80, label='High urgency'),
        mpatches.Patch(color=C_MED,  alpha=0.80, label='Medium urgency'),
        mpatches.Patch(color=C_LOW,  alpha=0.80, label='Low urgency'),
        mpatches.Patch(facecolor='white', edgecolor=GREY, hatch='///',
                       label='Stage 2 expediting activated'),
    ]
    ax.legend(handles=legend_patches, loc='lower right',
              framealpha=0.95, fontsize=9)

    _save(fig, 'fig_gantt.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Table 3 CSV + LaTeX snippet
# ─────────────────────────────────────────────────────────────────────────────
def save_table3():
    ref = 6.3892   # CCP cost at N=15, σ=0.10

    # OOS costs at N=15 and N=50
    COST_CCP = {15: 6.3892, 50: 8.8480}
    COST_TS  = {15: 6.3852, 50: 8.8287}

    rows = [
        (15, 0.10, 31.0,  2.7,  f'{COST_CCP[15]/ref:.3f}', f'{COST_TS[15]/ref:.3f}'),
        (15, 0.20, 41.8,  15.7, '—', '—'),
        (15, 0.30, None,  34.0, '—', '—'),
        (15, 0.40, None,  None, '—', '—'),
        (50, 0.10, 73.5,  10.0, f'{COST_CCP[50]/ref:.3f}', f'{COST_TS[50]/ref:.3f}'),
        (50, 0.20, 84.0,  44.1, '—', '—'),
        (50, 0.30, None,  74.6, '—', '—'),
        (50, 0.40, None,  None, '—', '—'),
    ]

    # CSV
    path = os.path.join(OUT, 'table3_results.csv')
    with open(path, 'w') as f:
        f.write('N,sigma_L,CCP_viol_%,2S_viol_%,CCP_norm_cost,2S_norm_cost\n')
        for r in rows:
            ccp_v = 'n/a' if r[2] is None else str(r[2])
            ts_v  = 'n/a' if r[3] is None else str(r[3])
            f.write(f'{r[0]},{r[1]},{ccp_v},{ts_v},{r[4]},{r[5]}\n')
        f.write(f'# Costs normalised to CCP at N=15 sigma=0.10 (${ref:.4f}M)\n')
        f.write('# n/a = model infeasible at this configuration\n')
    print(f'  Saved: {path}')

    # LaTeX
    print('\n  === LaTeX Table 3 snippet ===')
    print(r'  \begin{table}[t]')
    print(r'  \caption{Out-of-sample validation results ($n=2{,}000$ scenarios)}')
    print(r'  \label{tab:results}')
    print(r'  \centering')
    print(r'  \begin{tabular}{@{}ccrrrr@{}}')
    print(r'  \toprule')
    print(r'  $N$ & $\sigma_L$ & \multicolumn{2}{c}{Violation Rate (\%)} & \multicolumn{2}{c}{Normalised Cost} \\')
    print(r'  \cmidrule(lr){3-4} \cmidrule(lr){5-6}')
    print(r'       &            & CCP    & Two-Stage & CCP    & Two-Stage \\')
    print(r'  \midrule')
    for r in rows:
        ccp_v = r'n/a$^*$' if r[2] is None else f'{r[2]:.1f}'
        ts_v  = r'n/a$^*$' if r[3] is None else f'{r[3]:.1f}'
        print(f'  {r[0]} & {r[1]:.2f} & {ccp_v} & {ts_v} & {r[4]} & {r[5]} \\\\')
    print(r'  \bottomrule')
    print(r'  \multicolumn{6}{l}{\footnotesize $^*$Model infeasible at this configuration.} \\')
    print(r'  \multicolumn{6}{l}{\footnotesize Costs normalised to CCP baseline at $N=15$, $\sigma_L=0.10$.} \\')
    print(r'  \end{tabular}')
    print(r'  \end{table}')


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating WSC 2026 paper figures...\n')
    fig_process_flow()
    fig_violation_bar()
    fig_tau_sensitivity()
    fig_urgency_analysis()
    fig_gantt()
    save_table3()
    print('\nDone. All files in WSC2026_Draft/')
