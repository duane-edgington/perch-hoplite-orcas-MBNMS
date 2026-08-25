"""src/audio.py
Audio loading, encoding, and 30-second context generation for the
Perch-Hoplite annotation tool.
"""
import base64
import io
import os

import numpy as np
import soundfile as sf

from src.spectrogram import make_spectrogram_image


def make_audio_b64(audio_array: np.ndarray, sr: int) -> str:
    """Return a base64-encoded WAV data URI for an HTML5 audio element."""
    buf = io.BytesIO()
    sf.write(buf, audio_array, sr, format="WAV")
    buf.seek(0)
    return "data:audio/wav;base64," + base64.b64encode(buf.read()).decode()


def load_30s_context(
    seg: dict,
    audio_base_dir: str,
    spectrogram_type: str = "linear",
    colormap: str | None = None,
) -> tuple:
    """Load a 30-second context window centered on the 5-second clip.

    Parameters
    ----------
    seg : dict
        Segment dict with keys: recording_id, offset_s, sample_rate.
    audio_base_dir : str
        Directory containing the source WAV files.
    spectrogram_type : str
        Spectrogram type for the 5-second clip display. The 30s context
        always uses mel (or the same type if not linear).

    Returns
    -------
    (spec_html: str, audio_tuple: (sr, np.ndarray) | None)
        spec_html is an HTML string with the 30s mel spectrogram.
        audio_tuple is (sample_rate, int16_array) for gr.Audio, or None on error.
    """
    fname = seg["recording_id"].split("/")[-1]
    wav_path = os.path.join(audio_base_dir, fname) if audio_base_dir else None
    if not wav_path or not os.path.exists(wav_path):
        err = (f"<div style='color:#ef4444;font-size:11px;padding:8px;'>"
               f"⚠ Audio file not found: {fname}</div>")
        return err, None
    try:
        sr_ctx = seg["sample_rate"]
        offset_s = seg["offset_s"]
        center_s = offset_s + 2.5
        file_info = sf.info(wav_path)
        file_dur  = file_info.duration
        ctx_start = max(0.0, center_s - 15.0)
        ctx_end   = min(file_dur, ctx_start + 30.0)
        ctx_start = max(0.0, ctx_end - 30.0)
        start_smp = int(ctx_start * sr_ctx)
        end_smp   = int(ctx_end   * sr_ctx)
        audio_ctx, _ = sf.read(wav_path, start=start_smp, stop=end_smp,
                                dtype="float32", always_2d=False)
        ctx_spec_type = "mel" if spectrogram_type == "linear" else spectrogram_type
        hl_start      = offset_s - ctx_start        # position in 30s window
        hl_end        = hl_start + 5.0
        spec_b64_ctx  = make_spectrogram_image(
            audio_ctx, sr_ctx,
            spec_type=ctx_spec_type,
            highlight_start=hl_start,
            highlight_end=hl_end,
            colormap=colormap,
        )
        actual_dur = ctx_end - ctx_start
        clip_note  = "" if actual_dur >= 29.9 else f" (clipped to {actual_dur:.0f}s)"
        spec_html = (
            f"<div style='background:#0f172a;border:1px solid #334155;"
            f"border-radius:8px;padding:10px;margin-top:4px;'>"
            f"<span style='color:#fbbf24;font-size:10px;font-family:monospace;'>"
            f"▶ 30s context &nbsp; {ctx_start:.1f}s – {ctx_end:.1f}s{clip_note}</span>"
            f"<br><span style='color:#64748b;font-size:9px;'>"
            f"5s clip at {offset_s:.1f}s – {offset_s+5:.1f}s within this window</span>"
            f"<img src='{spec_b64_ctx}' style='width:100%;margin-top:6px;"
            f"border-radius:4px;display:block;'/>"
            f"</div>"
        )
        # Peak-normalize for playback
        peak = np.abs(audio_ctx).max()
        if peak > 1e-8:
            audio_ctx_norm = audio_ctx / peak * 0.5
        else:
            audio_ctx_norm = audio_ctx
        audio_int16 = np.clip(
            audio_ctx_norm * 32767, -32768, 32767).astype(np.int16)
        return spec_html, (sr_ctx, audio_int16)
    except Exception as exc:
        return (f"<div style='color:#ef4444;font-size:11px;padding:8px;'>"
                f"⚠ Could not load context: {exc}</div>"), None
