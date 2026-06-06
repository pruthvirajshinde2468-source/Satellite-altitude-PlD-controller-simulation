"""
Frequency-Domain Analysis and Performance Metrics

Provides:
  - Bode plot of the open-loop PID–plant transfer function
  - Phase margin and gain margin computation
  - Performance indices: IAE, ISE, ITAE, ITSE
  - Multi-tuning Bode comparison
  - Step-response rise-time extraction

Theory:
  Plant (continuous):  G(s) = 1 / (I·s² + b·s)
  PID  (continuous):   C(s) = Kp + Ki/s + Kd·s
  Open-loop:           L(s) = C(s)·G(s)
                             = (Kd·s² + Kp·s + Ki) / (I·s³ + b·s²)

Phase margin  PM = 180° + ∠L(jω_gc)   where |L(jω_gc)| = 1
Gain  margin  GM = -20 log₁₀|L(jω_pc)|  where ∠L(jω_pc) = -180°
"""

import numpy as np
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
def open_loop_response(omega: float, kp: float, ki: float, kd: float,
                       inertia: float, damping: float) -> complex:
    """Evaluate L(jω) = C(jω)·G(jω) at a single frequency."""
    s = 1j * omega
    numerator   = kd * s**2 + kp * s + ki
    denominator = inertia * s**3 + damping * s**2
    if abs(denominator) < 1e-30:
        return complex(1e15, 0.0)
    return numerator / denominator


# ──────────────────────────────────────────────────────────────────────────────
def compute_stability_margins(kp: float, ki: float, kd: float,
                               inertia: float, damping: float,
                               freq_range=(1e-4, 1e3),
                               n_points: int = 50_000) -> dict:
    """
    Compute phase margin (PM) and gain margin (GM).

    Returns a dict with:
        phase_margin_deg            – PM in degrees (>0 → stable)
        gain_margin_db              – GM in dB      (>0 → stable)
        gain_crossover_freq_rad_s   – ω where |L|=1
        phase_crossover_freq_rad_s  – ω where ∠L=-180°
        frequencies, magnitude, phase_deg  – Bode data arrays
    """
    freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), n_points)
    L = np.array([open_loop_response(w, kp, ki, kd, inertia, damping) for w in freqs])
    magnitude  = np.abs(L)
    phase_deg  = np.degrees(np.unwrap(np.angle(L)))

    # ── gain crossover  (|L| crosses 1 from above) ────────────────────
    sign_diff = np.diff(np.sign(magnitude - 1.0))
    gc_indices = np.where(sign_diff)[0]
    if len(gc_indices) > 0:
        gi        = gc_indices[0]
        gc_freq   = float(np.interp(0.0,
                                    [magnitude[gi] - 1.0, magnitude[gi+1] - 1.0],
                                    [freqs[gi], freqs[gi+1]]))
        phase_at_gc  = float(np.interp(gc_freq, freqs, phase_deg))
        phase_margin = 180.0 + phase_at_gc
    else:
        gc_freq = phase_margin = None

    # ── phase crossover  (∠L crosses -180°) ───────────────────────────
    pc_indices = np.where(np.diff(np.sign(phase_deg + 180.0)))[0]
    if len(pc_indices) > 0:
        pi_       = pc_indices[0]
        pc_freq   = float(np.interp(0.0,
                                    [phase_deg[pi_] + 180.0, phase_deg[pi_+1] + 180.0],
                                    [freqs[pi_], freqs[pi_+1]]))
        mag_at_pc    = float(np.interp(pc_freq, freqs, magnitude))
        gain_margin_db = float(-20.0 * np.log10(mag_at_pc)) if mag_at_pc > 0 else float('inf')
    else:
        pc_freq       = None
        gain_margin_db = float('inf')

    return {
        'phase_margin_deg':           phase_margin,
        'gain_margin_db':             gain_margin_db,
        'gain_crossover_freq_rad_s':  gc_freq,
        'phase_crossover_freq_rad_s': pc_freq,
        'frequencies':                freqs,
        'magnitude':                  magnitude,
        'phase_deg':                  phase_deg,
    }


# ──────────────────────────────────────────────────────────────────────────────
def compute_performance_indices(times: np.ndarray, errors: np.ndarray,
                                 dt: float) -> dict:
    """
    Standard integral performance indices.

    IAE  = ∫|e| dt          — penalises all errors equally
    ISE  = ∫e² dt           — penalises large errors more
    ITAE = ∫t·|e| dt        — penalises persistent errors (favoured for tuning)
    ITSE = ∫t·e² dt         — combined time and squared-error penalty
    """
    abs_e = np.abs(errors)
    return {
        'IAE':  float(np.trapezoid(abs_e,         times)),
        'ISE':  float(np.trapezoid(errors**2,      times)),
        'ITAE': float(np.trapezoid(times * abs_e,  times)),
        'ITSE': float(np.trapezoid(times * errors**2, times)),
    }


# ──────────────────────────────────────────────────────────────────────────────
def plot_bode(kp: float, ki: float, kd: float,
              inertia: float, damping: float,
              title: str = None,
              save_filename: str = None) -> dict:
    """
    Plot Bode magnitude + phase for L(s) and annotate stability margins.

    Returns the margins dict from compute_stability_margins().
    """
    margins = compute_stability_margins(kp, ki, kd, inertia, damping)
    freqs   = margins['frequencies']
    mag_db  = 20.0 * np.log10(np.maximum(margins['magnitude'], 1e-15))
    phase   = margins['phase_deg']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(title or f'Bode Plot  Kp={kp}, Ki={ki}, Kd={kd}', fontsize=13)

    # Magnitude
    ax1.semilogx(freqs, mag_db, 'b-', linewidth=2)
    ax1.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.set_ylabel('Magnitude (dB)', fontsize=11)
    ax1.grid(True, which='both', alpha=0.3)

    # Phase
    ax2.semilogx(freqs, phase, 'r-', linewidth=2)
    ax2.axhline(-180, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.set_xlabel('Frequency (rad/s)', fontsize=11)
    ax2.set_ylabel('Phase (deg)', fontsize=11)
    ax2.grid(True, which='both', alpha=0.3)

    # Annotate margins
    pm = margins['phase_margin_deg']
    gm = margins['gain_margin_db']
    gc = margins['gain_crossover_freq_rad_s']
    pc = margins['phase_crossover_freq_rad_s']

    if pm is not None and gc is not None:
        ax1.axvline(gc, color='g', linestyle=':', linewidth=1.2,
                    label=f'ω_gc  PM = {pm:.1f}°')
        ax2.axvline(gc, color='g', linestyle=':', linewidth=1.2)
        ax1.legend(fontsize=10)

    if pc is not None and not np.isinf(gm):
        ax2.axvline(pc, color='m', linestyle=':', linewidth=1.2,
                    label=f'ω_pc  GM = {gm:.1f} dB')
        ax2.legend(fontsize=10)

    plt.tight_layout()
    if save_filename:
        plt.savefig(save_filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_filename}")
    plt.show()

    return margins


# ──────────────────────────────────────────────────────────────────────────────
def compare_pid_bode(tunings: list, inertia: float, damping: float,
                     save_filename: str = None):
    """
    Overlay Bode plots for several PID tunings.

    Args:
        tunings: list of dicts, each with keys 'kp', 'ki', 'kd', and optional 'label'
    """
    freqs = np.logspace(-3, 2, 5000)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle('Bode Comparison — Multiple PID Tunings', fontsize=13)
    colors = plt.cm.tab10.colors

    for i, t in enumerate(tunings):
        kp, ki, kd = t['kp'], t['ki'], t['kd']
        L       = np.array([open_loop_response(w, kp, ki, kd, inertia, damping)
                             for w in freqs])
        mag_db  = 20.0 * np.log10(np.maximum(np.abs(L), 1e-15))
        phase   = np.degrees(np.unwrap(np.angle(L)))
        margins = compute_stability_margins(kp, ki, kd, inertia, damping)
        pm      = margins['phase_margin_deg']
        label   = t.get('label', f'Kp={kp} Ki={ki} Kd={kd}')
        if pm is not None:
            label += f'  PM={pm:.1f}deg'
        c = colors[i % len(colors)]
        ax1.semilogx(freqs, mag_db, color=c, linewidth=1.8, label=label)
        ax2.semilogx(freqs, phase,  color=c, linewidth=1.8)

    for ax, ref, ylabel in [(ax1, 0,    'Magnitude (dB)'),
                             (ax2, -180, 'Phase (deg)')]:
        ax.axhline(ref, color='k', linestyle='--', linewidth=0.8, alpha=0.4)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, which='both', alpha=0.3)

    ax1.legend(fontsize=9)
    ax2.set_xlabel('Frequency (rad/s)', fontsize=11)
    plt.tight_layout()
    if save_filename:
        plt.savefig(save_filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_filename}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
def print_stability_report(kp: float, ki: float, kd: float,
                            inertia: float, damping: float):
    """Print a formatted stability margin report to stdout."""
    m = compute_stability_margins(kp, ki, kd, inertia, damping)
    pm = m['phase_margin_deg']
    gm = m['gain_margin_db']

    print("\n" + "=" * 55)
    print("STABILITY MARGIN REPORT")
    print("=" * 55)
    print(f"  PID gains :  Kp={kp}  Ki={ki}  Kd={kd}")
    print(f"  Plant     :  I={inertia} kg.m^2  b={damping} N.m.s/rad")
    print("-" * 55)
    if pm is not None:
        status = "STABLE" if pm > 30 else ("MARGINAL" if pm > 0 else "UNSTABLE")
        print(f"  Phase Margin   : {pm:7.2f} deg  @ w={m['gain_crossover_freq_rad_s']:.4f} rad/s  [{status}]")
    else:
        print("  Phase Margin   : not found (gain never crosses 0 dB)")
    if not np.isinf(gm):
        status = "STABLE" if gm > 6 else ("MARGINAL" if gm > 0 else "UNSTABLE")
        print(f"  Gain  Margin   : {gm:7.2f} dB @ w={m['phase_crossover_freq_rad_s']:.4f} rad/s  [{status}]")
    else:
        print("  Gain  Margin   : inf  (phase never reaches -180 deg)")
    print("=" * 55)

    return m


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running frequency-domain analysis for default simulation parameters...")

    # Default simulation parameters
    INERTIA = 100.0
    DAMPING = 0.5
    KP, KI, KD = 0.8, 0.05, 1.2

    # Single-tuning report + Bode plot
    print_stability_report(KP, KI, KD, INERTIA, DAMPING)
    plot_bode(KP, KI, KD, INERTIA, DAMPING, save_filename="plot_bode.png")

    # Compare three tuning strategies
    compare_pid_bode([
        {'kp': 0.8,  'ki': 0.05, 'kd': 1.2, 'label': 'Baseline'},
        {'kp': 1.5,  'ki': 0.10, 'kd': 1.8, 'label': 'Aggressive'},
        {'kp': 0.4,  'ki': 0.02, 'kd': 0.6, 'label': 'Conservative'},
    ], inertia=INERTIA, damping=DAMPING, save_filename="plot_bode_comparison.png")
