## Plan: Update Figure 3 Panel A (trace distance plots)

Adjust Figure 3 Panel A so the left subplot shows trace distance vs $\Omega$ for the *old* protocol with an overlaid *new-protocol* reference curve (at $\eta=0.05$), and the right subplot shows the *new protocol* trace distance vs $\eta$ at fixed $\Omega=1.0$. Apply the same change in both the standalone Panel A export and the vectorized Figure 3+trace PDF builder, so PNG stitching and the vector PDF remain consistent.

**Steps**
1. Locate the existing Panel A plotting code under the notebook heading “### Trace distance between analytical old states”.
2. Keep the existing computation of `trace_results` as-is (it already produces `trace_results[eta]['old']` and `trace_results[eta]['new']` arrays vs `Omega_values`).
3. Replace the current “two panels = old vs new (both vs $\Omega$)” plotting section with a custom two-subplot layout:
   1. Left subplot (Panel A left):
      - Plot old protocol trace distance vs `Omega_values` for each `eta_val` in `eta_values` (same colors as today).
      - Overlay a single “reference line” for the new protocol at `eta_ref = 0.05` (use `trace_results[eta_ref]['new']` vs `Omega_values`) with a visually distinct style (e.g., black solid line, slightly thicker).
      - Keep x-axis log scale and label $\Omega\,[\nu]$.
      - Update legend strategy so it communicates both:
        - color → $\eta$ for the old protocol curves
        - plus one extra legend entry for the new-protocol reference curve
        (This can be done as a figure-level legend with custom `Line2D` handles, or an axis legend with a small second legend for the reference curve.)
   2. Right subplot (Panel A right):
      - Build $y(\eta)$ for the new protocol at fixed $\Omega=1.0$:
        - Compute `omega_ref = 1.0`.
        - Find the closest (or exact) index in `Omega_values`, e.g. `idx = int(np.argmin(np.abs(Omega_values - omega_ref)))`.
        - For each `eta_val`, take `trace_results[eta_val]['new'][idx]`.
      - Plot these as points/line vs `eta_values`.
      - Set x-axis label to $\eta$ (linear scale), keep y-axis shared with left plot.
      - Title this subplot like “New sequence ($\Omega=1.0$)” to make the conditioning explicit.
4. Ensure the panel label “A)” remains in the same place and that the exports (`figure_trace_distance.pdf` and `.png`) still write.
5. Apply the same plotting changes to the “Vectorized PDF version (for LaTeX)” section in the subsequent “combine” cell (the section that rebuilds the figure with `GridSpec` and re-plots the top-row trace-distance panels).
   - Update the top-row axes there to match the new left/right semantics (left: old vs $\Omega$ + new ref; right: new vs $\eta$ at $\Omega=1.0$).
   - Leave the bottom-row 3D panels (B/C) unchanged.
6. (Optional but recommended) Rename titles/labels in the top row so “Previous STP protocol”/“New sequence” no longer imply the previous two-panel split. For example:
   - Left title: “Previous STP protocol (ref: New, $\eta=0.05$)”
   - Right title: “New sequence at $\Omega=1.0$”

**Relevant files**
- trapped_ions/Fock_mixtures_sim.ipynb
  - Section “### Trace distance between analytical old states” — update the Plot block that currently creates `fig, axes = plt.subplots(1, 2, ...)` and loops over `panel_config`.
  - The following “combine” cell — update the “Vectorized PDF version (for LaTeX)” top-row plotting block that currently recreates the same two-panel “old vs new” plot.

**Verification**
1. Run the trace-distance cell and confirm:
   - Left subplot: old curves vs $\Omega$ (multi-$\eta$) + one new-protocol reference curve at $\eta=0.05$.
   - Right subplot: new-protocol points/line vs $\eta$ at $\Omega=1.0$.
2. Run the combine/vector-PDF cell and confirm `figure_3_plus_trace.pdf` shows the same updated Panel A top row.
3. Open `figure_trace_distance.png` and `figure_3_plus_trace.pdf` to confirm labels/legends are readable and no clipping occurs.

**Decisions**
- Panel A left: trace distance vs $\Omega$ for old protocol; overlay new-protocol reference curve at $\eta=0.05$.
- Panel A right: new-protocol trace distance vs $\eta$ at fixed $\Omega=1.0$ using the existing `eta_values` grid.
- No changes to the underlying simulation, only plotting/figure composition.
