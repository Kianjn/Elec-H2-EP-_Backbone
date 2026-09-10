"""Append figure helper + 11 plot cells to visualization.ipynb."""
from pathlib import Path

import nbformat as nbf

path = Path(r"c:\Users\kjafarinejad\Models\Project01\Now\visualization.ipynb")
nb = nbf.read(path, as_version=4)

# Drop the trailing placeholder markdown if present
if nb.cells and nb.cells[-1].cell_type == "markdown" and "Nothing is plotted yet" in "".join(nb.cells[-1].source):
    nb.cells.pop()


def md(src: str):
    nb.cells.append(nbf.v4.new_markdown_cell(src.strip() + "\n"))


def code(src: str):
    cell = nbf.v4.new_code_cell(src.strip() + "\n")
    cell["outputs"] = []
    cell["execution_count"] = None
    nb.cells.append(cell)


md(
    """
## Figures

Each block is one chart. Re-run the data cells above, then run a figure cell on its own.
"""
)

code(
    r'''
# Figure helpers — refresh investments, shared drawing tools
from matplotlib import patheffects as pe
from matplotlib.patches import Patch

CASE_ORDER = [ck for ck in ("sp", "me", "gsp", "gh2") if ck in inv_tables]
HALO = [pe.withStroke(linewidth=3.4, foreground=PAPER)]

AID_TO_SLOT = {aid: slot for slot, aid in SLOT_TO_AGENT.items()}


def _has_disaggregated_cap(case_dir: Path, aid: str) -> bool:
    summary_path = case_dir / "Agent_Summary.csv"
    if summary_path.is_file():
        ids = set(pd.read_csv(summary_path)["AgentID"].astype(str))
        if aid in ids:
            return True
    merged_path = case_dir / "Merged_Capacities.csv"
    if merged_path.is_file():
        slots = set(pd.read_csv(merged_path)["Slot"].astype(str))
        if AID_TO_SLOT.get(aid) in slots:
            return True
    return False


def refresh_investments() -> None:
    for case_key in list(investments):
        rows = []
        for risk in RISK_ORDER:
            if risk not in investments[case_key]:
                continue
            d = run_dir(risk, case_key)
            inv = load_investments(d)
            if case_key in ("gsp", "gh2") and "sp" in investments and risk in investments["sp"]:
                for aid in INVESTABLE:
                    if not _has_disaggregated_cap(d, aid):
                        inv[aid] = float(investments["sp"][risk][aid])
            investments[case_key][risk] = inv
            row = {"risk_folder": risk, "risk_label": RISK_FOLDERS[risk]}
            for aid in INVESTABLE:
                row[aid] = float(inv[aid])
            rows.append(row)
        if rows:
            inv_tables[case_key] = pd.DataFrame(rows).set_index("risk_folder")


refresh_investments()
print("Investments (MW, new capacity)")
for ck in CASE_ORDER:
    print(f"\n{CASE_SHORT[ck]}")
    print(inv_tables[ck][INVESTABLE].round(1).to_string())


def new_grid(nrows: int, ncols: int, *, width=10.8, height=7.2, top=0.70, bottom=0.14, wspace=0.28, hspace=0.42):
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), squeeze=False)
    fig.patch.set_facecolor(PAPER)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
    for ax in axes.ravel():
        ax.set_facecolor(PAPER)
    return fig, axes


def inv_mw(case_key: str, risk: str, aid: str) -> float:
    if case_key not in inv_tables or risk not in inv_tables[case_key].index:
        return np.nan
    return float(inv_tables[case_key].loc[risk, aid])


def grouped_bars(ax, categories, series_keys, values, color_of):
    n_s = len(series_keys)
    x = np.arange(len(categories), dtype=float)
    width = min(0.15, 0.76 / max(n_s, 1))
    offsets = (np.arange(n_s) - (n_s - 1) / 2.0) * width
    all_vals = []
    for i, sk in enumerate(series_keys):
        vals = [float(v) if _finite(v) else np.nan for v in values[sk]]
        all_vals.extend(vals)
        ax.bar(
            x + offsets[i], vals, width=width * 0.90,
            color=color_of(sk), zorder=3, linewidth=0, align="center",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontproperties=FONT_SANS, fontsize=8.5, color=INK_SOFT)
    finite = [v for v in all_vals if _finite(v)]
    hi = max(finite) if finite else 1.0
    ax.set_ylim(0, (hi * 1.18) if hi > 0 else 1.0)
    return finite


def fig_legend(fig, handles, labels, ncol: int):
    fig.legend(
        handles, labels, loc="lower center", ncol=ncol,
        bbox_to_anchor=(0.5, 0.085), frameon=False,
        prop=FONT_SANS, fontsize=8.5, handlelength=1.1, columnspacing=1.4,
    )


def price_duration(df: pd.DataFrame, col: str):
    p = df[col].to_numpy(dtype=float)
    w = df["W"].to_numpy(dtype=float)
    mask = np.isfinite(p) & np.isfinite(w)
    p, w = p[mask], w[mask]
    order = np.argsort(-p)
    p_sorted = p[order]
    w_sorted = w[order]
    cum = np.cumsum(w_sorted)
    return cum / cum[-1], p_sorted


def draw_price_duration(ax, case_key: str, market: str):
    col = PRICE_COLS[market]
    means = []
    for risk in loaded_risks(case_key):
        df = prices[case_key][risk]
        x, y = price_duration(df, col)
        ax.plot(x, y, color=RISK_COLORS[risk], lw=2.15, zorder=3, solid_capstyle="round")
        mu = w_mean_price(df, col)
        means.append((risk, mu, y[0]))
    for risk, mu, peak in means:
        ax.plot([], [], color=RISK_COLORS[risk], lw=2.15, label=f"{RISK_FOLDERS[risk]}   {mu:.1f}")
    finish_ax(ax, f"{MARKET_LABELS[market]}  (EUR / MWh)")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Share of weighted hours", fontproperties=FONT_SANS, fontsize=9.5, color=INK_SOFT)
    ax.axhline(0, color=RULE, lw=0.7, zorder=1)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5, prop=FONT_SANS, handlelength=1.4)
    return means


def commodity_metrics(case_key: str, risk: str) -> dict:
    ao = ao_full[case_key][risk]
    pr = prices[case_key][risk]
    wt = slot_weights(pr)
    n = len(pr)

    def col_or(name, fallback=None):
        if name in ao.columns:
            return ao[name].to_numpy(dtype=float)
        return fallback

    q_e = col_or("Cons_Elec_01_d")
    q_h = col_or("Prod_H2_Green_h2_out")
    if q_h is None:
        q_h = col_or("Offtaker_Green_h2_in")
    if len(D_EP_FLAT) == n:
        q_p = D_EP_FLAT
    else:
        q_p = ep_qty_vector(ao)

    p_e = pr["Elec_Price"].to_numpy(dtype=float)
    p_h = pr["H2_Price"].to_numpy(dtype=float)
    p_p = pr["EP_Price"].to_numpy(dtype=float)

    cwap_e = cwap_from_qty_price(q_e, p_e, wt) if q_e is not None else w_mean_price(pr, "Elec_Price")
    cwap_h = cwap_from_qty_price(q_h, p_h, wt) if q_h is not None else w_mean_price(pr, "H2_Price")
    cwap_p = cwap_from_qty_price(q_p, p_p, wt)

    def exp_qty(q, p):
        if q is None:
            return np.nan, np.nan
        qq = np.maximum(np.asarray(q, dtype=float), 0.0)
        pp = np.asarray(p, dtype=float)
        mask = np.isfinite(qq) & np.isfinite(pp) & np.isfinite(wt)
        return float(np.sum(wt[mask] * pp[mask] * qq[mask])), float(np.sum(wt[mask] * qq[mask]))

    e_exp, e_qty = exp_qty(q_e, p_e)
    h_exp, h_qty = exp_qty(q_h, p_h)
    p_exp, p_qty = exp_qty(q_p, p_p)
    return {
        "cwap_elec": cwap_e, "cwap_h2": cwap_h, "cwap_ep": cwap_p,
        "exp_elec": e_exp, "exp_h2": h_exp, "exp_ep": p_exp,
        "qty_elec": e_qty, "qty_h2": h_qty, "qty_ep": p_qty,
        "wmean_elec": w_mean_price(pr, "Elec_Price"),
        "wmean_h2": w_mean_price(pr, "H2_Price"),
        "wmean_ep": w_mean_price(pr, "EP_Price"),
    }


commodity: dict[str, pd.DataFrame] = {}
for ck in CASE_FOLDERS:
    rows = []
    for risk in RISK_ORDER:
        if risk not in prices.get(ck, {}):
            continue
        r = commodity_metrics(ck, risk)
        r.update(risk_folder=risk, risk_label=RISK_FOLDERS[risk])
        rows.append(r)
    if rows:
        commodity[ck] = pd.DataFrame(rows).set_index("risk_folder")

print("\nFigure helpers ready.")
'''
)

md("### 1 — Green investment by market design")

code(
    r'''
# 1. Green investments for each entry point, per risk case
fig, axes = new_grid(2, 2, width=11.0, height=7.6, top=0.69, bottom=0.16)
handles = [Patch(facecolor=RISK_COLORS[r], edgecolor="none") for r in RISK_ORDER]

for ax, aid in zip(axes.ravel(), INVESTABLE):
    scale, unit = INVESTABLE_SCALE[aid], INVESTABLE_UNITS[aid]
    values = {
        rk: [inv_mw(ck, rk, aid) * scale for ck in CASE_ORDER]
        for rk in RISK_ORDER
    }
    grouped_bars(ax, [CASE_SHORT[ck] for ck in CASE_ORDER], RISK_ORDER, values, lambda r: RISK_COLORS[r])
    finish_ax(ax, f"{INVESTABLE_LABELS[aid]}   ({unit} new)")

fig_legend(fig, handles, [RISK_FOLDERS[r] for r in RISK_ORDER], ncol=5)
add_chrome(
    fig,
    kicker="Capacity",
    title="Green investment, by market design",
    subtitle="Each cluster is one entry point. Bars run from risk-neutral to CVaR 0.8. Solar and wind in GW; electrolyzer and green ammonia plant in MW of new capacity.",
    source="Agent_Summary.csv and Merged_Capacities.csv  ·  existing seed capacity excluded",
)
rows = []
for ck in CASE_ORDER:
    for rk in RISK_ORDER:
        rec = {"case": CASE_SHORT[ck], "risk": RISK_FOLDERS[rk]}
        for aid in INVESTABLE:
            rec[f"{aid}_{INVESTABLE_UNITS[aid]}"] = inv_mw(ck, rk, aid) * INVESTABLE_SCALE[aid]
        rows.append(rec)
save_figure(fig, "01_green_investment_by_case", pd.DataFrame(rows))
plt.show()
'''
)

md("### 2 — Green investment by risk setting")

code(
    r'''
# 2. Green investments for each risk case, per entry point
fig, axes = new_grid(2, 2, width=11.0, height=7.6, top=0.69, bottom=0.16)
handles = [Patch(facecolor=CASE_COLORS[ck], edgecolor="none") for ck in CASE_ORDER]

for ax, aid in zip(axes.ravel(), INVESTABLE):
    scale, unit = INVESTABLE_SCALE[aid], INVESTABLE_UNITS[aid]
    values = {
        ck: [inv_mw(ck, rk, aid) * scale for rk in RISK_ORDER]
        for ck in CASE_ORDER
    }
    grouped_bars(ax, risk_xlabels(), CASE_ORDER, values, lambda c: CASE_COLORS[c])
    finish_ax(ax, f"{INVESTABLE_LABELS[aid]}   ({unit} new)")

fig_legend(fig, handles, [CASE_SHORT[ck] for ck in CASE_ORDER], ncol=4)
add_chrome(
    fig,
    kicker="Capacity",
    title="Green investment, by risk setting",
    subtitle="Same new capacity as the previous chart, regrouped. Each cluster is a risk case; colours are the four entry points.",
    source="Agent_Summary.csv and Merged_Capacities.csv  ·  existing seed capacity excluded",
)
rows = []
for rk in RISK_ORDER:
    for ck in CASE_ORDER:
        rec = {"risk": RISK_FOLDERS[rk], "case": CASE_SHORT[ck]}
        for aid in INVESTABLE:
            rec[f"{aid}_{INVESTABLE_UNITS[aid]}"] = inv_mw(ck, rk, aid) * INVESTABLE_SCALE[aid]
        rows.append(rec)
save_figure(fig, "02_green_investment_by_risk", pd.DataFrame(rows))
plt.show()
'''
)

md("### 3 — Electricity prices, social planner")

code(
    r'''
# 3. Price of electricity of SP per risk case
fig, ax = new_figure(width=9.2, height=5.6)
means = draw_price_duration(ax, "sp", "elec")
mean_txt = "   ·   ".join(f"{RISK_FOLDERS[r]} {m:.1f}" for r, m, _ in means)
add_chrome(
    fig,
    kicker="Social planner  ·  electricity",
    title="The power-price duration curve, risk by risk",
    subtitle=f"Hourly electricity prices stacked from highest to lowest, weighted by representative-day hours. Means (EUR/MWh): {mean_txt}.",
    source="Market_Prices.csv  ·  social_planner_results",
)
save_figure(fig, "03_sp_electricity_prices", pd.DataFrame(
    {"risk": [RISK_FOLDERS[r] for r, m, _ in means], "wmean_EUR_per_MWh": [m for r, m, _ in means]}
))
plt.show()
'''
)

md("### 4 — Hydrogen prices, social planner")

code(
    r'''
# 4. Price of hydrogen of SP per risk case
fig, ax = new_figure(width=9.2, height=5.6)
means = draw_price_duration(ax, "sp", "H2")
mean_txt = "   ·   ".join(f"{RISK_FOLDERS[r]} {m:.1f}" for r, m, _ in means)
add_chrome(
    fig,
    kicker="Social planner  ·  hydrogen",
    title="Hydrogen prices under complete risk trading",
    subtitle=f"Duration curves of hourly H2 prices. Means (EUR/MWh): {mean_txt}.",
    source="Market_Prices.csv  ·  social_planner_results",
)
save_figure(fig, "04_sp_hydrogen_prices", pd.DataFrame(
    {"risk": [RISK_FOLDERS[r] for r, m, _ in means], "wmean_EUR_per_MWh": [m for r, m, _ in means]}
))
plt.show()
'''
)

md("### 5 — Ammonia prices, social planner")

code(
    r'''
# 5. Price of ammonia of SP per risk case
fig, ax = new_figure(width=9.2, height=5.6)
means = draw_price_duration(ax, "sp", "EP")
mean_txt = "   ·   ".join(f"{RISK_FOLDERS[r]} {m:.1f}" for r, m, _ in means)
add_chrome(
    fig,
    kicker="Social planner  ·  ammonia",
    title="Ammonia prices when risk is traded completely",
    subtitle=f"Duration curves of hourly end-product prices. Means (EUR/MWh_EP): {mean_txt}.",
    source="Market_Prices.csv  ·  social_planner_results",
)
save_figure(fig, "05_sp_ammonia_prices", pd.DataFrame(
    {"risk": [RISK_FOLDERS[r] for r, m, _ in means], "wmean_EUR_per_MWh": [m for r, m, _ in means]}
))
plt.show()
'''
)

md("### 6 — Electricity prices, market exposure")

code(
    r'''
# 6. Price of electricity of ME per risk case
fig, ax = new_figure(width=9.2, height=5.6)
means = draw_price_duration(ax, "me", "elec")
mean_txt = "   ·   ".join(f"{RISK_FOLDERS[r]} {m:.1f}" for r, m, _ in means)
add_chrome(
    fig,
    kicker="Market exposure  ·  electricity",
    title="Power prices when firms bear their own risk",
    subtitle=f"Duration curves of hourly electricity prices under incomplete risk trading. Means (EUR/MWh): {mean_txt}.",
    source="Market_Prices.csv  ·  market_exposure_results",
)
save_figure(fig, "06_me_electricity_prices", pd.DataFrame(
    {"risk": [RISK_FOLDERS[r] for r, m, _ in means], "wmean_EUR_per_MWh": [m for r, m, _ in means]}
))
plt.show()
'''
)

md("### 7 — Hydrogen prices, market exposure")

code(
    r'''
# 7. Price of hydrogen of ME per risk case
fig, ax = new_figure(width=9.2, height=5.6)
means = draw_price_duration(ax, "me", "H2")
mean_txt = "   ·   ".join(f"{RISK_FOLDERS[r]} {m:.1f}" for r, m, _ in means)
add_chrome(
    fig,
    kicker="Market exposure  ·  hydrogen",
    title="Hydrogen prices under incomplete risk trading",
    subtitle=f"Duration curves of hourly H2 prices. Means (EUR/MWh): {mean_txt}.",
    source="Market_Prices.csv  ·  market_exposure_results",
)
save_figure(fig, "07_me_hydrogen_prices", pd.DataFrame(
    {"risk": [RISK_FOLDERS[r] for r, m, _ in means], "wmean_EUR_per_MWh": [m for r, m, _ in means]}
))
plt.show()
'''
)

md("### 8 — Ammonia prices, market exposure")

code(
    r'''
# 8. Price of ammonia of ME per risk case
fig, ax = new_figure(width=9.2, height=5.6)
means = draw_price_duration(ax, "me", "EP")
mean_txt = "   ·   ".join(f"{RISK_FOLDERS[r]} {m:.1f}" for r, m, _ in means)
add_chrome(
    fig,
    kicker="Market exposure  ·  ammonia",
    title="Ammonia prices when green firms cannot share risk",
    subtitle=f"Duration curves of hourly end-product prices. Means (EUR/MWh_EP): {mean_txt}.",
    source="Market_Prices.csv  ·  market_exposure_results",
)
save_figure(fig, "08_me_ammonia_prices", pd.DataFrame(
    {"risk": [RISK_FOLDERS[r] for r, m, _ in means], "wmean_EUR_per_MWh": [m for r, m, _ in means]}
))
plt.show()
'''
)

md("### 9 — Risk-adjusted cost of electricity, hydrogen and ammonia")

code(
    r'''
# 9. Total risk-adjusted cost of electricity, hydrogen & ammonia — SP vs ME
pair = [ck for ck in ("sp", "me") if ck in commodity]
fig, axes = new_grid(1, 3, width=11.2, height=5.4, top=0.68, bottom=0.16, wspace=0.32)
markets = [
    ("cwap_elec", "Electricity", "EUR / MWh"),
    ("cwap_h2", "Hydrogen", "EUR / MWh"),
    ("cwap_ep", "Ammonia", "EUR / MWh_EP"),
]
x = np.arange(len(RISK_ORDER))
for ax, (col, name, unit) in zip(axes.ravel(), markets):
    for ck in pair:
        ys = series_for(commodity[ck], col)
        ax.plot(
            x, ys, color=CASE_COLORS[ck], lw=2.2, marker="o", markersize=6.2,
            markerfacecolor=PAPER, markeredgewidth=1.8, zorder=3, label=CASE_SHORT[ck],
        )
        for i, v in enumerate(ys):
            if not _finite(v):
                continue
            ax.text(x[i], v, f"  {v:.1f}", color=CASE_COLORS[ck], fontsize=7.4,
                    va="bottom", ha="left", fontproperties=FONT_SANS, path_effects=HALO)
    ax.set_xticks(x, risk_xlabels())
    finish_ax(ax, f"{name}   ({unit})")

handles = [Line2D([0], [0], color=CASE_COLORS[ck], lw=2.2, marker="o", markerfacecolor=PAPER, markeredgewidth=1.6) for ck in pair]
fig_legend(fig, handles, [CASE_SHORT[ck] for ck in pair], ncol=2)
add_chrome(
    fig,
    kicker="Consumer and chain unit costs",
    title="Electricity, hydrogen and ammonia — SP against ME",
    subtitle="Quantity-weighted prices (CWAP): power demand × power price; electrolyzer H2 output × H2 price; inelastic ammonia demand × EP price. Scenario- and hour-weighted.",
    source="Agent_Objectives_Per_Timestep.csv  ·  Market_Prices.csv",
)
rows = []
for ck in pair:
    for rk in RISK_ORDER:
        if rk not in commodity[ck].index:
            continue
        r = commodity[ck].loc[rk]
        rows.append({
            "case": CASE_SHORT[ck], "risk": RISK_FOLDERS[rk],
            "elec_EUR_per_MWh": r["cwap_elec"], "h2_EUR_per_MWh": r["cwap_h2"],
            "ammonia_EUR_per_MWh": r["cwap_ep"],
        })
save_figure(fig, "09_sp_me_commodity_costs", pd.DataFrame(rows))
plt.show()
'''
)

md("### 10 — CWAP across all four entry points")

code(
    r'''
# 10. CWAP — electricity, hydrogen, ammonia × all entry points
fig, axes = new_grid(1, 3, width=11.2, height=5.6, top=0.68, bottom=0.16, wspace=0.32)
markets = [
    ("cwap_elec", "Electricity", "EUR / MWh"),
    ("cwap_h2", "Hydrogen", "EUR / MWh"),
    ("cwap_ep", "Ammonia", "EUR / MWh_EP"),
]
x = np.arange(len(RISK_ORDER))
cases = [ck for ck in CASE_ORDER if ck in commodity]
for ax, (col, name, unit) in zip(axes.ravel(), markets):
    for ck in cases:
        ys = series_for(commodity[ck], col)
        ax.plot(
            x, ys, color=CASE_COLORS[ck], lw=2.15, marker="o", markersize=5.6,
            markerfacecolor=PAPER, markeredgewidth=1.6, zorder=3,
        )
    ax.set_xticks(x, risk_xlabels())
    finish_ax(ax, f"{name}   ({unit})")

handles = [Line2D([0], [0], color=CASE_COLORS[ck], lw=2.15, marker="o", markerfacecolor=PAPER, markeredgewidth=1.5) for ck in cases]
fig_legend(fig, handles, [CASE_SHORT[ck] for ck in cases], ncol=4)
add_chrome(
    fig,
    kicker="CWAP",
    title="Consumer-weighted prices, all four institutions",
    subtitle="Quantity-weighted average price of electricity (load), hydrogen (electrolyzer output, or the hour-weighted market price where the chain is merged), and ammonia (inelastic EP demand).",
    source="Agent_Objectives_Per_Timestep.csv  ·  Market_Prices.csv",
)
rows = []
for ck in cases:
    for rk in RISK_ORDER:
        if rk not in commodity[ck].index:
            continue
        r = commodity[ck].loc[rk]
        rows.append({
            "case": CASE_SHORT[ck], "risk": RISK_FOLDERS[rk],
            "elec_CWAP": r["cwap_elec"], "h2_CWAP": r["cwap_h2"], "ammonia_CWAP": r["cwap_ep"],
        })
save_figure(fig, "10_cwap_all_cases", pd.DataFrame(rows))
plt.show()
'''
)

md("### 11 — Grey ammonia cost versus ME green ammonia")

code(
    r'''
# 11. Grey ammonia MC, then ME ammonia price per risk — mandate + risk premium
if "me" not in commodity:
    print("Skipping grey/green ammonia gap — ME not loaded.")
else:
    me_ep = series_for(commodity["me"], "cwap_ep")
    labels = ["Grey\nexpected MC"] + risk_xlabels()
    values = [GREY_MC_EXPECTED] + me_ep
    colors = [REF] + [RISK_COLORS[r] for r in RISK_ORDER]
    rn = me_ep[0] if me_ep else np.nan
    hi_ra = me_ep[-1] if me_ep else np.nan

    fig, ax = new_figure(width=9.4, height=5.6)
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.62, zorder=3, linewidth=0)
    ax.axhline(GREY_MC_EXPECTED, color=REF, ls=(0, (3, 2.5)), lw=1.05, zorder=2)
    for bar, val in zip(bars, values):
        if not _finite(val):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2, val,
            f"{val:.1f}", ha="center", va="bottom", fontsize=8.5,
            color=INK, fontproperties=FONT_SANS_SB, clip_on=False,
        )
    ax.set_xticks(x, labels)
    finish_ax(ax, "EUR / MWh_EP")
    finite = [v for v in values if _finite(v)]
    ax.set_ylim(0, max(finite) * 1.22)

    gap_rn = rn - GREY_MC_EXPECTED if _finite(rn) else np.nan
    gap_risk = hi_ra - rn if _finite(hi_ra) and _finite(rn) else np.nan
    add_chrome(
        fig,
        kicker="Market exposure  ·  ammonia",
        title="The green–grey ammonia gap, then the risk premium",
        subtitle=(
            f"Grey expected marginal cost is {GREY_MC_EXPECTED:.1f} EUR/MWh_EP. "
            f"Risk-neutral ME ammonia is {gap_rn:.1f} above that — the 42% H2-GC mandate and green scarcity. "
            f"Moving from RN to CVaR 0.8 adds {gap_risk:.1f} more — the risk premium on top."
        ),
        source="Grey MC from Data/data.yaml  ·  ME ammonia CWAP from Market_Prices.csv",
    )
    table = pd.DataFrame({
        "series": ["Grey expected MC"] + [RISK_FOLDERS[r] for r in RISK_ORDER],
        "ammonia_EUR_per_MWh_EP": values,
        "vs_grey": [v - GREY_MC_EXPECTED for v in values],
        "vs_RN": [np.nan] + [v - rn for v in me_ep],
    })
    save_figure(fig, "11_me_ammonia_grey_gap", table)
    plt.show()
'''
)

nbf.write(nb, path)
print(f"wrote {path}  cells={len(nb.cells)}")
