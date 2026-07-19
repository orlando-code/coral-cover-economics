# Speaker notes — Coral Cover Economics presentation

Open slides: `docs/presentation/index.html` in Chrome (press **S** for speaker view, **F** for fullscreen).

Serve locally if images don't load from `file://`:

```bash
cd docs/presentation && python -m http.server 8765
# → http://localhost:8765
```

## Timing (~20–25 min + questions)

| Slides | Topic | ~min |
|--------|-------|------|
| 1 | Title | 0.5 |
| 2–3 | Motivation + pipeline | 3 |
| 4 | Data | 2 |
| 5–7 | HBB + parameterization | 4 |
| 8–10 | Projection skill + limitations | 5 |
| 11 | Bright spots (optional skip) | 1.5 |
| 12–14 | Scenarios + economics + results | 5 |
| 15–18 | Claims + appendix + close | 3 |

## Key numbers to remember

### Ecology (forward CV, Reef Check repeat sites)

- **Selected spec:** `eco_parsimonious_trend_interact` — forward R² **0.410** at cutoff 2011 (test ≤ 2018)
- **HBB vs site mean:** +3.2 pp R²; vs Sully linear baseline +2.1 pp
- **Projection skill (047):** HBB level R² ~**0.36–0.43** by cutoff; **only model with positive ΔR² vs persistence** (~+0.06 to +0.13)
- **Within-site R² median:** **negative** for all models (~−1.0)
- **Bleaching era (2016–2018):** level R² drops to ~0.20–0.37

### Economics (dashboard run, linear Chen model, 3 sectors)

Baseline reef-linked value: **~$54B**

| Scenario | 2050 loss | 2100 loss | % baseline |
|----------|-----------|-----------|------------|
| RCP 4.5 | $20.2B | $27.8B | 38% → 52% |
| RCP 8.5 | $25.9B | $40.5B | 48% → 76% |

Tourism alone RCP8.5 2100 linear: **~$34B** loss.

**Not trillions** — be precise. Say "tens of billions" or use the conditional framing.

## The one sentence to repeat

> "We report **scenario-conditional economic exposure** assuming historical ecology–environment relationships hold under climate forcing — not validated forecasts of realized losses."

## If challenged on Sully / bright spots

- Bright spots = **in-sample positive residuals**, not demonstrated resilience
- Sully 2050/2100 = **scenario extrapolation**, not forward-validated at repeat sites
- Our contribution: **rolling-origin skill** through bleaching eras

## If challenged on using economics at all

- Economics layer is **transparent sensitivity** (Chen elasticity is literature-based)
- Rankings and **scenario spreads** (RCP45 vs 85) are more defensible than point totals
- Appendix holds validation figures; dollar figures are **Layer 2 × Layer 3**

## Before presenting

- [ ] Open slides once, check all images load
- [ ] Decide: skip bright-spots slide if audience is non-ecological?
- [ ] Update title slide with your name / date
- [ ] Optional: open dashboard tab for live demo (Map / GDP pages)
