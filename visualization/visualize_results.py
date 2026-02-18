"""
Results visualization: Social Planner vs ADMM market exposure.
Run individual cells (# %%) to produce specific graphs.
Figures are saved to visualization/figures/
"""
# %%
# Imports and paths
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
_script_dir = Path(__file__).resolve().parent
PROJECT = _script_dir.parent
SP_DIR = PROJECT / 'social_planner_results'
ADMM_DIR = PROJECT / 'market_exposure_results'
OUTPUT_DIR = _script_dir / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)

# Modern style: clean, high-contrast, publication-ready
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'axes.edgecolor': '#333',
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.linestyle': '-',
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.framealpha': 0.95,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'Social Planner': '#2563eb',
    'ADMM': '#dc2626',
    'Diff': '#059669',
    'Diff_neg': '#b91c1c',
    'Accent': '#7c3aed',
}

def save_fig(fig, name):
    fig.savefig(OUTPUT_DIR / f'{name}.png', dpi=150, bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT_DIR / f'{name}.pdf', bbox_inches='tight', facecolor='white')
    print(f"Saved: {OUTPUT_DIR / name}")

# %%
# 1. Market price differences (Social Planner vs ADMM)
sp_prices = pd.read_csv(SP_DIR / 'Market_Prices.csv')
admm_prices = pd.read_csv(ADMM_DIR / 'Market_Prices.csv')
merged = sp_prices.merge(admm_prices, on='Time', suffixes=('_SP', '_ADMM'))
markets = ['Elec_Price', 'H2_Price', 'Elec_GC_Price', 'H2_GC_Price', 'EP_Price']
labels = ['Electricity', 'Hydrogen', 'Electricity GC', 'H2 GC', 'End Product']
for col in markets:
    merged[f'diff_{col}'] = merged[f'{col}_SP'] - merged[f'{col}_ADMM']

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()
for i, (col, lbl) in enumerate(zip(markets, labels)):
    ax = axes[i]
    diff = merged[f'diff_{col}']
    ax.fill_between(merged['Time'], 0, diff, where=diff >= 0, color=COLORS['Diff'], alpha=0.5)
    ax.fill_between(merged['Time'], 0, diff, where=diff < 0, color=COLORS['Diff_neg'], alpha=0.5)
    ax.plot(merged['Time'], diff, color='#374151', linewidth=1.5)
    ax.axhline(0, color='#6b7280', linestyle='-', linewidth=0.8)
    ax.set_title(lbl)
    ax.set_xlabel('Time (hour)')
    ax.set_ylabel('Price diff (SP − ADMM)')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
    # Insight: summary stats
    mean_d, max_abs = diff.mean(), diff.abs().max()
    ax.text(0.02, 0.98, f'Mean: {mean_d:.2e}\nMax |diff|: {max_abs:.2e}', transform=ax.transAxes,
            fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
axes[-1].axis('off')
diff_cols = [f'diff_{c}' for c in markets]
mean_row = merged[diff_cols].mean().values
maxabs_row = merged[diff_cols].abs().max().values
cell_text = [[f'{v:.2e}' for v in mean_row], [f'{v:.2e}' for v in maxabs_row]]
axes[-1].table(cellText=cell_text, rowLabels=['Mean diff', 'Max |diff|'], colLabels=labels,
               loc='center', cellLoc='center')
axes[-1].set_title('Summary across markets')
plt.suptitle('Market price differences: Social Planner vs ADMM', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
save_fig(fig, '01_price_differences')
plt.show()

# %%
# 2. Quantity differences (Social Planner vs ADMM)
sp_agents = pd.read_csv(SP_DIR / 'Agent_Summary.csv')
admm_agents = pd.read_csv(ADMM_DIR / 'Agent_Quantities_Final.csv')
merged_qty = sp_agents.merge(admm_agents, left_on='Agent', right_on='AgentID', how='outer')
merged_qty['Agent'] = merged_qty['Agent'].fillna(merged_qty['AgentID'])

def admm_primary(row):
    if pd.isna(row.get('EP_net_sum')) and pd.isna(row.get('elec_net_sum')) and pd.isna(row.get('H2_net_sum')):
        return np.nan
    g = str(row.get('Group', ''))
    if 'offtaker' in g or 'Offtaker' in str(row.get('Type', '')):
        return row.get('EP_net_sum', np.nan) or 0
    if 'power' in g or 'Power' in str(row.get('Type', '')):
        return row.get('elec_net_sum', np.nan) or 0
    if 'H2' in str(row.get('Group', '')) or 'H2Prod' in str(row.get('Type', '')):
        return row.get('H2_net_sum', np.nan) or 0
    return row.get('EP_net_sum', row.get('elec_net_sum', np.nan)) or 0

merged_qty['ADMM_primary'] = merged_qty.apply(admm_primary, axis=1)
merged_qty['Total_Quantity'] = pd.to_numeric(merged_qty['Total_Quantity'], errors='coerce')
merged_qty['diff'] = merged_qty['Total_Quantity'] - merged_qty['ADMM_primary']
valid = merged_qty.dropna(subset=['Total_Quantity', 'ADMM_primary'])

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(valid))
w = 0.38
bars1 = ax.bar(x - w/2, valid['Total_Quantity'], w, label='Social Planner', color=COLORS['Social Planner'], alpha=0.9, edgecolor='white', linewidth=1.2)
bars2 = ax.bar(x + w/2, valid['ADMM_primary'], w, label='ADMM', color=COLORS['ADMM'], alpha=0.9, edgecolor='white', linewidth=1.2)
ax.set_xticks(x)
ax.set_xticklabels(valid['Agent'], rotation=45, ha='right')
ax.set_ylabel('Quantity')
ax.set_title('Agent quantities: Social Planner vs ADMM')
ax.legend(loc='upper right', frameon=True)
# Add diff annotation
for i, (spv, adv) in enumerate(zip(valid['Total_Quantity'], valid['ADMM_primary'])):
    d = spv - adv
    if abs(d) > 0.01 * max(valid['Total_Quantity'].abs().max(), valid['ADMM_primary'].abs().max()):
        ax.annotate(f'{d:+.0f}', xy=(i, max(spv, adv)), ha='center', va='bottom', fontsize=8, color='#374151')
plt.tight_layout()
save_fig(fig, '02_quantity_comparison')
plt.show()

# %%
# 3. Price evolution per timestep (both cases)
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()
for i, (col, lbl) in enumerate(zip(markets, labels)):
    ax = axes[i]
    ax.plot(sp_prices['Time'], sp_prices[col], color=COLORS['Social Planner'], linewidth=2, label='Social Planner', alpha=0.9)
    ax.plot(admm_prices['Time'], admm_prices[col], color=COLORS['ADMM'], linewidth=1.5, label='ADMM', alpha=0.85, linestyle='--')
    ax.set_title(lbl)
    ax.set_xlabel('Time (hour)')
    ax.set_ylabel('Price')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(bottom=0)
axes[-1].axis('off')
plt.suptitle('Market price evolution over time', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
save_fig(fig, '03_price_evolution')
plt.show()

# %%
# 4. ADMM convergence: primal and dual residuals
conv = pd.read_csv(ADMM_DIR / 'ADMM_Convergence.csv')
conv = conv.replace([np.inf, -np.inf], np.nan)
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()
market_pairs = [('elec', 'Electricity'), ('H2', 'Hydrogen'), ('elec_GC', 'Elec GC'), ('H2_GC', 'H2 GC'), ('EP', 'End Product')]
for i, (key, lbl) in enumerate(market_pairs):
    ax = axes[i]
    prim, dual = f'{key}_primal', f'{key}_dual'
    if prim in conv.columns and dual in conv.columns:
        pmax = conv[prim].max()
        dmax = conv[dual].max()
        ax.semilogy(conv['iter'], conv[prim].fillna(pmax if pd.notna(pmax) else 1), '-', color=COLORS['Social Planner'], label='Primal', linewidth=2)
        ax.semilogy(conv['iter'], conv[dual].fillna(dmax if pd.notna(dmax) else 1), '--', color=COLORS['ADMM'], label='Dual', linewidth=1.5)
    ax.set_title(lbl)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Residual (log)')
    ax.legend(loc='upper right', fontsize=9)
    final_prim = conv[prim].iloc[-1] if prim in conv.columns else np.nan
    if pd.notna(final_prim):
        ax.text(0.98, 0.02, f'Final: {final_prim:.2e}', transform=ax.transAxes, fontsize=8, ha='right', va='bottom')
axes[-1].axis('off')
plt.suptitle('ADMM convergence: primal and dual residuals per market', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
save_fig(fig, '04_admm_convergence')
plt.show()

# %%
# 5. ADMM market history (price and imbalance evolution)
history_files = ['Electricity_Market_History', 'Hydrogen_Market_History', 'Electricity_GC_Market_History',
                 'H2_GC_Market_History', 'End_Product_Market_History']
labels_h = ['Electricity', 'Hydrogen', 'Elec GC', 'H2 GC', 'End Product']
fig, axes = plt.subplots(2, 1, figsize=(12, 9))
colors_h = plt.cm.tab10(np.linspace(0, 1, len(labels_h)))
for j, (f, lbl) in enumerate(zip(history_files, labels_h)):
    p = ADMM_DIR / f'{f}.csv'
    if p.exists():
        h = pd.read_csv(p)
        axes[0].plot(h['iter'], h['price_mean'], label=lbl, linewidth=2, color=colors_h[j])
        axes[1].plot(h['iter'], h['imb_mean'], label=lbl, linewidth=1.5, color=colors_h[j])
axes[0].set_title('Price mean per iteration', fontweight='bold')
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Price')
axes[0].legend(loc='upper right')
axes[1].set_title('Imbalance mean per iteration', fontweight='bold')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Imbalance')
axes[1].legend(loc='upper right')
axes[1].axhline(0, color='#6b7280', linestyle='--', linewidth=1)
plt.suptitle('ADMM market history: price and imbalance convergence', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save_fig(fig, '05_admm_market_history')
plt.show()

# %%
# 6. ADMM per-market net quantities (stacked view for insight)
fig, ax = plt.subplots(figsize=(11, 6))
qty_cols = ['elec_net_sum', 'H2_net_sum', 'elec_GC_net_sum', 'H2_GC_net_sum', 'EP_net_sum']
col_names = ['Electricity', 'H2', 'Elec GC', 'H2 GC', 'End Product']
admm_agents_plot = admm_agents.set_index('AgentID')[qty_cols]
admm_agents_plot.columns = col_names
admm_agents_plot.plot(kind='bar', ax=ax, width=0.8, edgecolor='white', linewidth=1)
ax.set_title('ADMM agent quantities by market')
ax.set_xlabel('Agent')
ax.set_ylabel('Net quantity')
ax.axhline(0, color='#6b7280', linestyle='-', linewidth=0.8)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Market', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
plt.tight_layout()
save_fig(fig, '06_admm_agent_quantities')
plt.show()

# %%
# 7. Price heatmap: time-of-day pattern for all 5 markets
fig, axes = plt.subplots(5, 2, figsize=(12, 14))
for i, (col, lbl) in enumerate(zip(markets, labels)):
    for j, (df, title) in enumerate([(sp_prices, 'Social Planner'), (admm_prices, 'ADMM')]):
        ax = axes[i, j]
        t = df['Time'].values
        hour = (t - 1) % 24
        day = (t - 1) // 24
        pivot_data = pd.DataFrame({'hour': hour, 'day': day, 'price': df[col]})
        pivot = pivot_data.pivot_table(values='price', index='hour', columns='day', aggfunc='mean')
        im = ax.imshow(pivot.values.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax.set_xlabel('Hour of day')
        ax.set_ylabel('Day index')
        ax.set_title(f'{lbl}: {title}')
        ax.set_xticks(np.arange(0, 24, 4))
        plt.colorbar(im, ax=ax, label='Price')
plt.suptitle('Price heatmaps by market (hour × day)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
save_fig(fig, '07_price_heatmap')
plt.show()
