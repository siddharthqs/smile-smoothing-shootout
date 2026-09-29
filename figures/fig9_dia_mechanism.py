"""fig9 -- why DIA crosses between pillars and the other names do not.

Between two admissible eSSVI slices the surface takes the parameter
midpoint. The Hendriks--Martini floor there is psi_1 * p < psi_mid, and
p exceeds 1 as soon as rho moves. So a crossing needs a small psi
increment and a rho step on the same adjacent pair. The figure shows
the fitted rho ladder, slice by slice, for DIA's median-|drho| capture
and the SPY capture from the same session: DIA's skew parameter jumps
between neighbouring expiries; SPY's drifts. The printed medians are the
numbers the text quotes.

Source: results/fits.csv and results/arbitrage.csv (`study --panel daily`).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _style import RESULTS, use_paper_style, save, note, res

use_paper_style()

G = ['essvi_sg05', 'essvi_g05', 'essvi_gfree']
KEY = ['ticker', 'date', 'time', 'model']
TICKS = ['SPY', 'QQQ', 'IWM', 'DIA', 'META', 'AMD', 'TSLA', 'GOOGL']

f = pd.read_csv(res('fits.csv'),
                usecols=KEY + ['status', 'tenor_years', 'essvi_theta',
                               'essvi_psi', 'essvi_rho'])
f = f[(f.status == 'ok') & f.model.isin(G)].sort_values(KEY + ['tenor_years'])

rows = []
for key, g in f.groupby(KEY):
    ps, rh = g.essvi_psi.values, g.essvi_rho.values
    if len(ps) < 3:
        continue
    rows.append(dict(zip(KEY, key),
                     min_dpsi=100 * (np.diff(ps) / ps[:-1]).min(),
                     max_drho=np.abs(np.diff(rh)).max()))
lad = pd.DataFrame(rows)
a = pd.read_csv(res('arbitrage.csv'), usecols=KEY + ['status', 'calendar_crossings'])
a = a[a.status.str.startswith('ok')]
m = lad.merge(a, on=KEY)
m['cross'] = m.calendar_crossings > 0

# ── the example session: DIA's median-|drho| capture on the free joint arm
dia = m[(m.ticker == 'DIA') & (m.model == 'essvi_gfree')].sort_values('max_drho')
ex = dia.iloc[len(dia) // 2]
sess = (ex.date, ex.time)

fig, axA = plt.subplots(1, 1, figsize=(4.4, 2.9))
fig.subplots_adjust(left=0.15, right=0.97, top=0.96, bottom=0.25)

# (a) rho ladders
CO = {'DIA': '#C08A00', 'SPY': '#0B4F8A'}
for t, mk in [('SPY', 's'), ('DIA', '^')]:
    g = f[(f.ticker == t) & (f.date == sess[0]) & (f.time == sess[1]) & (f.model == 'essvi_gfree')]
    axA.plot(g.tenor_years, g.essvi_rho, '-', color=CO[t], lw=1.0, marker=mk, ms=3.4,
             mfc='white', mew=0.9, label='%s, %d pillars' % (t, len(g)), zorder=3)
axA.set_xscale('log')
axA.set_xlabel('tenor (years, log scale)', fontsize=8)
axA.set_ylabel(r'fitted $\rho_i$ per slice', fontsize=8.5)
axA.legend(loc='upper left', fontsize=7)

n_cross = int(m.cross.sum())
note(fig, 'Session %s %s, G-eSSVI with $\\gamma$ free.' % (sess[0], '%04d' % int(sess[1])), y=0.015)

save(fig, 'fig9_dia_mechanism.pdf')
print('example session:', sess, ' name-day/arm pairs %d, crossed %d' % (len(m), n_cross))
for c in (False, True):
    d = m[m.cross == c].groupby('model')[['min_dpsi', 'max_drho']].median()
    print('  crossed' if c else '  clean  ', 'min dpsi %.1f-%.1f%%  max|drho| %.2f-%.2f'
          % (d.min_dpsi.min(), d.min_dpsi.max(), d.max_drho.min(), d.max_drho.max()))
for t in TICKS:
    d = m[m.ticker == t]
    print('  %-6s med max|drho| %.3f  med min dpsi %.1f%%  cross %.0f%%'
          % (t, d.max_drho.median(), d.min_dpsi.median(), 100 * d.cross.mean()))
