"""src/spectrogram.py
Spectrogram generation for the Perch-Hoplite annotation tool.

Provides make_spectrogram_image() — a standalone function that generates
base64-encoded PNG spectrograms from audio arrays.

Spec types:
  "linear" — Linear-frequency STFT, 0–16 kHz. Best for orca/dolphin.
  "mel"    — Mel-scale log power, 10 Hz floor. Best for humpback.
  "perch"  — Exact Perch 2.0 frontend (what the model sees).
  "pcen"   — PCEN mel, makes quiet signals pop.
"""
import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_spectrogram_image(
    audio_array,
    sr: int,
    spec_type: str = "linear",
    highlight_start: float | None = None,
    highlight_end: float | None = None,
    colormap: str | None = None,
) -> str:
    """Return a base64-encoded PNG spectrogram.

    Parameters
    ----------
    audio_array : np.ndarray
        1-D float32 audio samples.
    sr : int
        Sample rate in Hz.
    spec_type : str
        One of "linear", "mel", "perch", "pcen".
    colormap : str | None
        Override the default colormap. Examples: "gray", "gray_r",
        "viridis", "inferno", "magma", "plasma", "cividis".
        None = use per-mode default (inferno/magma/viridis).
    highlight_start, highlight_end : float | None
        If both provided, draw yellow fiducial markers at these times (seconds)
        within the spectrogram — used by the 30-second context display to mark
        the 5-second clip location.

    Returns
    -------
    str
        data:image/png;base64,... string suitable for an <img src=...> tag.
    """
    import numpy as np

    # ── Compute time-frequency representation ────────────────────────────────
    if spec_type == "linear":
        from scipy.signal import spectrogram as _spec
        nperseg = min(512, len(audio_array) // 4)
        f, t, Sxx = _spec(
            audio_array, fs=sr,
            nperseg=nperseg, noverlap=nperseg * 3 // 4,
            scaling="density",
        )
        S_plot = 10 * np.log10(Sxx + 1e-10)
        f_max = min(16000, sr // 2)
        f_mask = f <= f_max
        f_plot = f[f_mask]
        S_plot = S_plot[f_mask, :]
        ylabel = "Hz"
        title_suffix = "Linear STFT"
        cmap = "inferno"

    elif spec_type in ("mel", "pcen"):
        import librosa
        n_fft = 512
        hop = n_fft // 4
        f_min = 10.0
        f_max = 16000.0
        n_mels = 128
        S = librosa.feature.melspectrogram(
            y=audio_array.astype(np.float32), sr=sr,
            n_fft=n_fft, hop_length=hop,
            n_mels=n_mels, fmin=f_min, fmax=f_max,
            power=2.0,
        )
        if spec_type == "pcen":
            S_plot = librosa.pcen(
                S * (2**31), sr=sr, hop_length=hop,
                gain=0.98, bias=2, power=0.5, time_constant=0.4,
                eps=1e-6,
            )
            title_suffix = "PCEN mel (10 Hz floor)"
            cmap = "magma"
        else:
            S_plot = librosa.power_to_db(S, ref=np.max)
            title_suffix = "Mel spectrogram (10 Hz floor)"
            cmap = "inferno"
        f_plot = librosa.mel_frequencies(n_mels=n_mels, fmin=f_min, fmax=f_max)
        t = librosa.frames_to_time(
            np.arange(S_plot.shape[1]), sr=sr, hop_length=hop)
        ylabel = "Hz (mel)"

    elif spec_type == "perch":
        # Exact Perch 2.0 frontend:
        # 128-band mel, 60 Hz–16 kHz, HTK scale, DC bin zeroed
        # 0.1 · log(max(mel_energy, 1e-5))
        import librosa
        n_fft  = 2048
        hop    = 320       # Perch uses 10ms hop at 32kHz
        n_mels = 128
        f_min  = 60.0
        f_max  = 16000.0
        S = librosa.feature.melspectrogram(
            y=audio_array.astype(np.float32), sr=sr,
            n_fft=n_fft, hop_length=hop,
            n_mels=n_mels, fmin=f_min, fmax=f_max,
            htk=True, power=1.0,    # power=1 → amplitude mel
        )
        S[0, :] = 0.0               # zero DC bin (Perch convention)
        S_plot  = 0.1 * np.log(np.maximum(S, 1e-5))
        f_plot  = librosa.mel_frequencies(
            n_mels=n_mels, fmin=f_min, fmax=f_max, htk=True)
        t = librosa.frames_to_time(
            np.arange(S_plot.shape[1]), sr=sr, hop_length=hop)
        ylabel = "Hz (Perch mel)"
        title_suffix = "Perch 2.0 frontend (what the model sees)"
        cmap = "viridis"

    else:
        raise ValueError(f"Unknown spec_type: {spec_type!r}. "
                         "Use 'linear', 'mel', 'perch', or 'pcen'.")

    # Percentile-based normalization — robust against saturating noise
    if spec_type == "pcen":
        vmax = float(np.percentile(S_plot, 99))
        vmin = float(np.percentile(S_plot,  1))
    else:
        vmax = float(np.percentile(S_plot, 99.5))
        vmin = vmax - 80.0   # 80dB range: shows quiet calls in noisy background

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        2, 1, figsize=(7, 3),
        gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.05},
    )
    fig.patch.set_facecolor("#111827")

    ax_spec = axes[0]
    # Apply colormap override if specified
    _cmap = colormap if colormap else cmap

    if spec_type in ("mel", "pcen", "perch"):
        # pcolormesh with mel bin edges — each patch spans exactly one mel
        # band boundary to the next, preserving the mel frequency spacing.
        # Y-axis is in Hz (mel-spaced) with ticks at perceptually meaningful
        # values. No log scale — the mel spacing itself provides the
        # perceptual compression. All frequency content is shown.
        f_edges = np.concatenate([
            [max(f_plot[0] - (f_plot[1] - f_plot[0]) / 2, f_plot[0])],
            (f_plot[:-1] + f_plot[1:]) / 2,
            [f_plot[-1] + (f_plot[-1] - f_plot[-2]) / 2],
        ])
        dt = t[1] - t[0] if len(t) > 1 else 1.0
        t_edges = np.concatenate([[t[0] - dt/2],
                                   (t[:-1] + t[1:]) / 2,
                                   [t[-1] + dt/2]])
        ax_spec.pcolormesh(
            t_edges, f_edges, S_plot,
            vmin=vmin, vmax=vmax,
            cmap=_cmap, shading="flat",
        )
        ax_spec.set_ylim(f_edges[0], f_edges[-1])
        ax_spec.set_xlim(t_edges[0], t_edges[-1])
        # Ticks at mel-meaningful Hz values
        tick_hz = [500, 1000, 2000, 4000, 8000, 16000]
        tick_hz = [h for h in tick_hz if f_edges[0] <= h <= f_edges[-1]]
        ax_spec.set_yticks(tick_hz)
        ax_spec.set_yticklabels(
            [f"{h//1000}k" if h >= 1000 else str(h) for h in tick_hz],
            color="#94a3b8", fontsize=7)
        ax_spec.grid(False)
        ax_spec.tick_params(axis="y", which="both", length=0)
    else:
        ax_spec.pcolormesh(
            t, f_plot, S_plot,
            vmin=vmin, vmax=vmax,
            cmap=_cmap, shading="gouraud",
        )
    # Yellow fiducial markers for the 5-second clip within a context window
    if highlight_start is not None and highlight_end is not None:
        ax_spec.axvspan(highlight_start, highlight_end,
                        alpha=0.25, color="#facc15", zorder=10)
        ax_spec.axvline(x=highlight_start, color="#facc15",
                        linewidth=2.5, linestyle="-", alpha=1.0, zorder=11)
        ax_spec.axvline(x=highlight_end,   color="#facc15",
                        linewidth=2.5, linestyle="-", alpha=1.0, zorder=11)
        y_bot, y_top = ax_spec.get_ylim()
        mid_x = (highlight_start + highlight_end) / 2
        ax_spec.text(mid_x, y_bot + (y_top - y_bot) * 0.03,
                     "◄ 5s ►", color="#facc15", fontsize=8,
                     ha="center", va="bottom", fontweight="bold",
                     zorder=12,
                     bbox=dict(boxstyle="round,pad=0.2",
                               facecolor="#0f172a", alpha=0.7,
                               edgecolor="none"))
    ax_spec.set_ylabel(ylabel, color="#94a3b8", fontsize=8)
    ax_spec.set_title(title_suffix, color="#64748b", fontsize=7, pad=2)
    ax_spec.tick_params(colors="#94a3b8", labelsize=7)
    ax_spec.set_facecolor("#111827")
    for spine in ax_spec.spines.values():
        spine.set_edgecolor("#334155")
    ax_spec.tick_params(bottom=False, labelbottom=False)

    ax_wave = axes[1]
    t_wave = np.linspace(0, len(audio_array) / sr, len(audio_array))
    ax_wave.plot(t_wave, audio_array, color="#38bdf8", linewidth=0.4)
    ax_wave.set_xlim([0, t_wave[-1]])
    ax_wave.set_xlabel("Time (s)", color="#94a3b8", fontsize=8)
    ax_wave.set_facecolor("#111827")
    ax_wave.tick_params(colors="#94a3b8", labelsize=7)
    for spine in ax_wave.spines.values():
        spine.set_edgecolor("#334155")
    ax_wave.set_yticks([])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight",
                facecolor="#111827")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()
