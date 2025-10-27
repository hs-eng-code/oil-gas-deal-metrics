import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import tri as mtri
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (required for 3D projection)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_isorisk(deal_summaries, out_path="outputs/isorisk.png"):
    """Visualization: 2D isorisk contour (supports Step 13 outputs).

    Mapped responsibilities:
    - Step 13: Produce executive isorisk plots (2D contours) showing Deal size
        vs Var% colored by `DRS_100` with annotations. This is an executive
        visualization used after scoring and mapping.

    Inputs:
    - deal_summaries: list of dicts with keys 'size_mm', 'Var%', 'DRS_100', 'deal_id'
    - out_path: file path to save PNG
    """
    ensure_dir(os.path.dirname(out_path) or ".")
    xs = np.array([d["Var%"] for d in deal_summaries])
    ys = np.array([d["size_mm"] for d in deal_summaries])
    zs = np.array([d.get("DRS_100", 0.0) for d in deal_summaries])

    fig, ax = plt.subplots(figsize=(8, 6))
    # triangulated contour for DRS
    try:
        tcf = ax.tricontourf(xs, ys, zs, levels=12, cmap="RdYlBu_r", alpha=0.9)
        fig.colorbar(tcf, ax=ax, label="DRS (0-100)")
    except Exception:
        # fallback: use scatter color map
        sc = ax.scatter(xs, ys, c=zs, cmap="RdYlBu_r", s=40, edgecolor='k')
        fig.colorbar(sc, ax=ax, label="DRS (0-100)")

    # overlay points and labels for top/bottom
    ax.scatter(xs, ys, c="none", edgecolor="k", s=30)
    for d in deal_summaries:
        if d.get("DRS_100", 0) >= 90 or d.get("DRS_100", 0) <= 10:
            ax.text(d["Var%"], d["size_mm"], d["deal_id"], fontsize=8, ha="center", va="center")

    ax.set_xlabel("Var% (std/mean)")
    ax.set_ylabel("Deal size (P50 sum)")
    ax.set_title("Isorisk: Deal Size vs Variability (colored by DRS)")
    ax.grid(True, alpha=0.3)
    # explanatory caption: concise description of axes and color
    caption = (
        "Var% = portfolio std / mean of PV draws. "
        "Deal size = sum of per-well P50 PV (units). "
        "DRS_100 = standardized Deal Risk Score (0-100); higher => more attractive per model."
    )
    fig.text(0.01, 0.01, caption, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_deal_wrs_and_mcvare(wells_df, per_well_wrs_scaled, deal_summary, out_dir="outputs", top_n=10):
    """Per-deal diagnostic plot: WRS density + MCVaR waterfall (Step 13).

    This diagnostic visual is intended for deal-level technical review after
    step 8/9 calculations (MCVaR + per-well WRS). It overlays deal wells on the
    pooled WRS distribution and shows the top marginal CVaR contributors.

    Inputs:
    - wells_df: DataFrame with 'well_id' and other per-well columns
    - per_well_wrs_scaled: dict mapping well index -> array of scaled WRS draws (0-100)
    - deal_summary: dict from run_demo's deal_summaries for one deal (must contain 'mcv')
    - top_n: number of top MCVaR wells to show in waterfall
    """
    ensure_dir(out_dir)
    # flatten global WRS draws
    all_vals = np.concatenate([v for v in per_well_wrs_scaled.values()]) if len(per_well_wrs_scaled) > 0 else np.array([])

    # density histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1.2, 1]})
    ax = axes[0]
    if all_vals.size > 0:
        counts, bins = np.histogram(all_vals, bins=80, density=True)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        ax.fill_between(bin_centers, counts, alpha=0.6, color='#6baed6')
        ax.plot(bin_centers, counts, color='#08519c')
    ax.set_xlabel('WRS (0-100)')
    ax.set_ylabel('Density')
    ax.set_title('Global WRS density')

    # overlay deal wells
    deal_well_ids = list(deal_summary.get('mcv', {}).keys())
    # find their median WRS points if available
    marks = []
    for wid in deal_well_ids:
        # find index in wells_df
        matches = wells_df.index[wells_df['well_id'] == wid].tolist()
        if matches:
            idx = matches[0]
            draws = per_well_wrs_scaled.get(idx)
            if draws is not None and len(draws) > 0:
                med = np.median(draws)
                marks.append(med)
                ax.vlines(med, 0, ax.get_ylim()[1]*0.9, colors='r', alpha=0.6)

    # right panel: MCVaR waterfall
    ax2 = axes[1]
    mcv = deal_summary.get('mcv', {})
    if mcv:
        items = list(mcv.items())
        # items are well_id -> value (dollar MCVaR)
        items_sorted = sorted(items, key=lambda x: -x[1])
        top_items = items_sorted[:top_n]
        names = [it[0] for it in top_items]
        vals = [it[1] for it in top_items]
        ax2.barh(range(len(vals))[::-1], vals, color='#fb6a4a')
        ax2.set_yticks(range(len(vals))[::-1])
        ax2.set_yticklabels(names)
        ax2.set_xlabel('MCVaR ($)')
        ax2.set_title(f"Top {len(vals)} MCVaR contributors for {deal_summary.get('deal_id')}")
    else:
        ax2.text(0.5, 0.5, 'No MCVaR data', ha='center', va='center')

    # ensure space for caption at bottom and render caption visibly
    fig.subplots_adjust(bottom=0.18)
    # explanatory caption for per-deal plot (visible, not clipped)
    caption_pd = (
        "Left: global Well Risk Score (WRS) density across the universe (0-100). "
        "Red vertical lines mark deal wells' median WRS. "
        "Right: Top MCVaR contributors (dollar impact when removed at tail q)."
    )
    fig.text(0.02, 0.06, caption_pd, fontsize=9, wrap=True)
    out_file = os.path.join(out_dir, f"deal_{deal_summary.get('deal_id')}_wrs_mcvare.png")
    fig.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_file


def plot_isorisk_3d(deal_summaries, out_path="outputs/isorisk_3d.png", cmap="RdYlBu_r"):
    """3D isorisk surface or scatter (Step 13 visualization).

    Produces a 3D surface (or scatter fallback) mapping (Var%, Deal size) ->
    `DRS_100` for executive visuals. Used after scoring and scaling are completed.
    """
    # Note: this function attempts triangulation and falls back to scatter when
    # insufficient non-collinear points exist. It maps to the same conceptual
    # isorisk surface used in the 2D contour.
    ensure_dir(os.path.dirname(out_path) or ".")
    xs = np.array([d["Var%"] for d in deal_summaries])
    ys = np.array([d["size_mm"] for d in deal_summaries])
    zs = np.array([d.get("DRS_100", 0.0) for d in deal_summaries])

    # basic sanity: need at least 3 non-collinear points for a triangulation
    if len(xs) < 3:
        # fallback to scatter saved as a PNG
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(xs, ys, zs, c=zs, cmap=cmap, s=40)
        fig.colorbar(sc, ax=ax, pad=0.1, label="DRS (0-100)")
        ax.set_xlabel('Var% (std/mean)')
        ax.set_ylabel('Deal size (P50 sum)')
        ax.set_zlabel('DRS (0-100)')
        ax.set_title('Isorisk 3D (scatter fallback)')
        # caption to explain this fallback 3D scatter
        caption_fb = (
            "3D scatter fallback: not enough non-collinear points for a surface. "
            "Var% = portfolio std/mean. Deal size = P50 sum. DRS_100 (0-100)."
        )
        fig.text(0.01, 0.01, caption_fb, fontsize=8)
        plt.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    try:
        tri = mtri.Triangulation(xs, ys)
        if getattr(tri, 'triangles', None) is None or tri.triangles.size == 0:
            raise ValueError('degenerate triangulation')
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_trisurf(tri.x, tri.y, zs, triangles=tri.triangles, cmap=cmap, linewidth=0.2, antialiased=True, shade=True)
        ax.scatter(xs, ys, zs, c=zs, cmap=cmap, edgecolor='k', s=30)
        fig.colorbar(surf, ax=ax, pad=0.1, label='DRS (0-100)')
        ax.set_xlabel('Var% (std/mean)')
        ax.set_ylabel('Deal size (P50 sum)')
        ax.set_zlabel('DRS (0-100)')
        ax.set_title('Isorisk 3D Surface (Var% x Size -> DRS)')
        caption_tri = (
            'Triangulated surface maps (Var%, size) to DRS_100. '
            'Points are original deals; surface interpolates between them.'
        )
        fig.text(0.01, 0.01, caption_tri, fontsize=8)
        plt.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    except Exception:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(xs, ys, zs, c=zs, cmap=cmap, s=40)
        fig.colorbar(sc, ax=ax, pad=0.1, label='DRS (0-100)')
        ax.set_xlabel('Var% (std/mean)')
        ax.set_ylabel('Deal size (P50 sum)')
        ax.set_zlabel('DRS (0-100)')
        ax.set_title('Isorisk 3D (scatter fallback)')
        caption_fb = (
            '3D scatter fallback: triangulation failed (degenerate geometry).'
            ' Var% = portfolio std/mean. Deal size = P50 sum. DRS_100 (0-100).'
        )
        fig.text(0.01, 0.01, caption_fb, fontsize=8)
        plt.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path


def plot_isorisk_combined(deal_summaries, out_path="outputs/isorisk_combined.png", cmap="RdYlBu_r", use_log_size=True, grid_res=100):
    """Combined gridded 3D surface and 2D contour (Step 13 visualization).

    Builds a smoothed grid interpolation of `DRS_100` over (Var%, log1p(size)).
    Intended for slide visuals after scoring and mapping.
    """
    ensure_dir(os.path.dirname(out_path) or ".")
    xs = np.array([d["Var%"] for d in deal_summaries])
    ys_raw = np.array([d["size_mm"] for d in deal_summaries])
    zs = np.array([d.get("DRS_100", 0.0) for d in deal_summaries])

    # choose Y coordinate (log-scale recommended for size)
    if use_log_size:
        ys = np.log1p(ys_raw)
        y_label = 'log1p(Deal size)'
    else:
        ys = ys_raw
        y_label = 'Deal size (P50 sum)'

    # create grid bounds slightly expanded to avoid points on the border
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_pad = max(1e-6, 0.05 * (x_max - x_min)) if x_max > x_min else 0.1
    y_pad = max(1e-6, 0.05 * (y_max - y_min)) if y_max > y_min else 0.1
    xi = np.linspace(x_min - x_pad, x_max + x_pad, grid_res)
    yi = np.linspace(y_min - y_pad, y_max + y_pad, grid_res)
    XI, YI = np.meshgrid(xi, yi)

    # Try cubic interpolation first, fall back to linear then nearest
    methods = ["cubic", "linear", "nearest"]
    ZI = None
    for method in methods:
        try:
            ZI = griddata((xs, ys), zs, (XI, YI), method=method)
            # require that at least some finite values exist
            if np.isfinite(ZI).any():
                break
        except Exception:
            ZI = None

    # mask grid points outside convex hull (where interpolation yields NaN)
    mask = None
    if ZI is not None:
        mask = ~np.isfinite(ZI)

    # build figure with side-by-side subplots
    fig = plt.figure(figsize=(14, 6))
    # 3D surface on left
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    try:
        # For plot_surface we need Z as 2D array; apply small nan->nan handling
        if ZI is None:
            # fallback to scatter
            sc = ax3d.scatter(xs, ys, zs, c=zs, cmap=cmap, s=40)
        else:
            # replace masked values with nan so surface plotting skips them
            Zplot = np.array(ZI)
            # create a colormap-normalizer based on actual z range
            surf = ax3d.plot_surface(XI, YI, Zplot, cmap=cmap, linewidth=0, antialiased=True, rstride=1, cstride=1)
            # overlay data points
            ax3d.scatter(xs, ys, zs, c=zs, cmap=cmap, edgecolor='k', s=30)
    except Exception:
        sc = ax3d.scatter(xs, ys, zs, c=zs, cmap=cmap, s=40)

    ax3d.set_xlabel('Var% (std/mean)')
    ax3d.set_ylabel(y_label)
    ax3d.set_zlabel('DRS (0-100)')
    ax3d.set_title('Isorisk 3D Surface (smoothed grid)')

    # right: 2D contour projection from same grid
    ax2d = fig.add_subplot(1, 2, 2)
    if ZI is None:
        sc2 = ax2d.scatter(xs, ys, c=zs, cmap=cmap, s=60, edgecolor='k')
    else:
        # contourf expects finite values; mask NaNs
        Zshow = np.ma.array(ZI, mask=mask)
        cf = ax2d.contourf(XI, YI, Zshow, levels=15, cmap=cmap)
        # overlay data points
        ax2d.scatter(xs, ys, c='k', s=10)
        sc2 = cf

    ax2d.set_xlabel('Var% (std/mean)')
    ax2d.set_ylabel(y_label)
    ax2d.set_title('Isorisk (2D contour from same grid)')

    # shared colorbar
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    if ZI is None:
        # use scatter color mapping
        fig.colorbar(sc, cax=cax, label='DRS (0-100)')
    else:
        fig.colorbar(sc2, cax=cax, label='DRS (0-100)')

    plt.tight_layout(rect=[0, 0, 0.9, 1.0])
    # explanatory caption for combined figure
    interp_method = 'gridded interpolation (cubic->linear->nearest)'
    y_note = 'Y axis = log1p(Deal size)' if use_log_size else 'Y axis = Deal size (P50 sum)'
    caption_comb = (
        f"Surface = {interp_method}. {y_note}. "
        "Var% = portfolio std/mean. DRS_100 is scaled 0-100 (higher => more attractive). "
        "Areas outside convex hull are masked (no extrapolation)."
    )
    fig.text(0.01, 0.01, caption_comb, fontsize=8)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_drs_to_prec(deal_summaries, out_path="outputs/drs_to_prec.png", gamma=0.05, theta=50.0, p50=50.0, p_max=80.0):
    """Plot the logistic mapping from `DRS_100` to `P_rec` and overlay deals.

    This visualization directly supports Step 11 (mapping DRS_100 -> P_rec) and
    Step 13 (produce visuals). It draws the logistic curve and overlays deal points to help calibrate mapping parameters.
    """
    ensure_dir(os.path.dirname(out_path) or ".")
    import numpy as _np
    import matplotlib.pyplot as _plt

    # safety: accept either list-of-dicts or empty list
    if deal_summaries is None:
        deal_summaries = []

    # compute the logistic curve
    drs_vals = _np.linspace(0, 100, 501)
    x = gamma * (drs_vals - theta)
    s = 1.0 / (1.0 + _np.exp(-x))
    prec_vals = p50 + (p_max - p50) * s

    # prepare overlay points; if P_rec missing, compute via logistic_map for display
    xs = _np.array([_np.clip(float(d.get('DRS_100', 0.0)), 0.0, 100.0) for d in deal_summaries]) if len(deal_summaries) > 0 else _np.array([])
    ys = []
    for d in deal_summaries:
        if 'P_rec' in d and d['P_rec'] is not None:
            ys.append(float(d['P_rec']))
        else:
            # compute recommended percentile using same logistic formula for overlay consistency
            drs_val = float(d.get('DRS_100', 0.0))
            x0 = gamma * (drs_val - theta)
            s0 = 1.0 / (1.0 + _np.exp(-x0))
            ys.append(float(p50 + (p_max - p50) * s0))
    ys = _np.array(ys) if len(ys) > 0 else _np.array([])

    fig, ax = _plt.subplots(figsize=(8, 5))
    ax.plot(drs_vals, prec_vals, '-', color='#2b8cbe', lw=2, label='logistic map')
    ax.fill_between(drs_vals, prec_vals, p50, color='#a6bddb', alpha=0.15)

    # overlay deals if present
    if xs.size > 0 and ys.size > 0:
        sc = ax.scatter(xs, ys, c='r', s=36, edgecolor='k', alpha=0.9)
        # label extreme points for clarity
        for d in deal_summaries:
            drs_v = float(d.get('DRS_100', 0.0))
            if drs_v >= 90 or drs_v <= 10:
                ax.text(drs_v, float(d.get('P_rec', p50)), d.get('deal_id', ''), fontsize=8, ha='center', va='bottom')

    ax.set_xlabel('DRS_100 (0-100)')
    ax.set_ylabel('P_rec (underwriting percentile)')
    # sensible y-limits: just beyond p50/p_max but not too tight
    ymin = max(40.0, p50 - 10.0)
    ymax = max(95.0, p_max + 5.0)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-2, 102)
    ax.set_title('Mapping: DRS_100 -> Recommended underwriting percentile (P_rec)')
    ax.grid(alpha=0.3)
    ax.legend()

    # optional inset: histogram of DRS_100 values to show distribution
    if xs.size > 0:
        ax_inset = fig.add_axes([0.60, 0.58, 0.25, 0.25])
        ax_inset.hist(xs, bins=10, color='#2b8cbe', alpha=0.6)
        ax_inset.set_title('DRS distribution', fontsize=8)
        ax_inset.tick_params(axis='both', which='major', labelsize=7)

    caption = (
        f"Logistic map parameters: gamma={gamma}, theta={theta}, p50={p50}, p_max={p_max}.\n"
        "Points = deals plotted at (DRS_100, P_rec) where missing P_rec values are computed via the same logistic map for display.\n"
        "Mapping converts standardized DRS (0–100) to a recommended underwriting percentile; calibrate gamma/theta to match risk appetite."
    )
    fig.text(0.01, 0.01, caption, fontsize=8)
    _plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    _plt.close(fig)
    return out_path
