"""src/review.py
Gradio-based interactive audio labeling GUI for the Perch-Hoplite pipeline.

Provides launch_labeling_gui() — displays audio segments with spectrograms,
30-second context windows, and label radio buttons. Labels are written to
the DB on every click (auto-save) and on explicit Save.
"""
import json
import logging
import os
import struct
import sqlite3
from pathlib import Path

from src.spectrogram import make_spectrogram_image
from src.audio import make_audio_b64, load_30s_context

log = logging.getLogger(__name__)

# Provenance base directory
PROVENANCE_BASE = "/mnt/PAM_Analysis/perch-hoplite/provenance"


# ---------------------------------------------------------------------------
# LabelType constants
# ---------------------------------------------------------------------------

class _LT:
    """LabelType constants — works across perch-hoplite versions."""
    try:
        from perch_hoplite.db import annotations as _a
        POSITIVE      = _a.LabelType.POSITIVE
        NEGATIVE      = _a.LabelType.NEGATIVE
        WEAK_NEGATIVE = _a.LabelType.WEAK_NEGATIVE
    except Exception:
        try:
            from perch_hoplite.db import interface as _b
            POSITIVE      = _b.LabelType.POSITIVE
            NEGATIVE      = _b.LabelType.NEGATIVE
            WEAK_NEGATIVE = _b.LabelType.WEAK_NEGATIVE
        except Exception:
            POSITIVE      = 1
            NEGATIVE      = 2
            WEAK_NEGATIVE = 3


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

def _provenance_path(subdir: str, stem: str) -> Path:
    p = Path(PROVENANCE_BASE) / subdir
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{stem}.json"


def save_label_provenance(
    session_id: str,
    db_dir: str,
    classifier_path,
    annotator_id: str,
    query_label: str,
    annotations: list,
) -> Path | None:
    """Write a labeling session provenance record."""
    import datetime as _dt
    record = {
        "session_id":       session_id,
        "timestamp":        _dt.datetime.now().isoformat(),
        "db_dir":           str(db_dir),
        "classifier":       str(classifier_path) if classifier_path else None,
        "annotator_id":     annotator_id,
        "query_label":      query_label,
        "annotation_count": len(annotations),
        "annotations":      annotations,
    }
    stem = f"labels_{session_id}"
    out  = _provenance_path("labels", stem)
    try:
        with open(out, "w") as f:
            json.dump(record, f, indent=2)
        log.info("Label provenance saved to %s", out)
        return out
    except Exception as exc:
        log.warning("Could not save label provenance: %s", exc)
        return None


# ---------------------------------------------------------------------------
# GUI CSS
# ---------------------------------------------------------------------------

_GRADIO_CSS = (
    "body { background: #0f172a; color: #e2e8f0; font-family: 'Courier New', monospace; }"
    ".gr-button-primary { background: #0ea5e9 !important; }"
    ".gr-button { border-radius: 6px !important; }"
    ".label-radio .wrap { display: flex; flex-direction: column; gap: 6px !important; }"
    ".label-radio .wrap label { border-radius: 8px; padding: 7px 14px;"
    "  font-weight: 700; font-size: 13px; cursor: pointer;"
    "  transition: box-shadow 0.15s; }"
    ".label-radio .wrap label:nth-child(1) { background:#15803d; color:#dcfce7; }"
    ".label-radio .wrap label:nth-child(2) { background:#b45309; color:#fef3c7; }"
    ".label-radio .wrap label:nth-child(3) { background:#1d4ed8; color:#dbeafe; }"
    ".label-radio .wrap label:nth-child(4) { background:#7e22ce; color:#f3e8ff; }"
    ".label-radio .wrap label:nth-child(5) { background:#0e7490; color:#cffafe; }"
    ".label-radio .wrap label:nth-child(6) { background:#c2410c; color:#ffedd5; }"
    ".label-radio .wrap label:last-child   { background:#374151; color:#d1d5db; }"
    ".label-radio .wrap label:nth-child(1):has(input:checked)"
    "  { background:#16a34a; box-shadow:0 0 0 3px #86efac; }"
    ".label-radio .wrap label:nth-child(2):has(input:checked)"
    "  { background:#d97706; box-shadow:0 0 0 3px #fcd34d; }"
    ".label-radio .wrap label:nth-child(3):has(input:checked)"
    "  { background:#2563eb; box-shadow:0 0 0 3px #93c5fd; }"
    ".label-radio .wrap label:nth-child(4):has(input:checked)"
    "  { background:#9333ea; box-shadow:0 0 0 3px #d8b4fe; }"
    ".label-radio .wrap label:nth-child(5):has(input:checked)"
    "  { background:#0891b2; box-shadow:0 0 0 3px #67e8f9; }"
    ".label-radio .wrap label:nth-child(6):has(input:checked)"
    "  { background:#ea580c; box-shadow:0 0 0 3px #fdba74; }"
    ".label-radio .wrap label:last-child:has(input:checked)"
    "  { background:#4b5563; box-shadow:0 0 0 3px #9ca3af; }"
)


# ---------------------------------------------------------------------------
# Main GUI function
# ---------------------------------------------------------------------------

def launch_labeling_gui(
    db,
    results_obj,
    audio_filepath_loader,
    sample_rate_hz: int,
    query_label: str,
    annotator_id: str,
    host: str,
    port: int,
    share: bool,
    db_dir: str = "",
    classifier_path=None,
    label_classes=None,
    detections_info=None,
    spectrogram_type: str = "linear",
    colormap: str | None = None,
    audio_base_dir: str = "",
) -> None:
    """Launch a Gradio web app for interactive audio labeling.

    Displays search results with spectrograms, 30-second context windows,
    and label radio buttons. Labels auto-save on click and on Save.

    Parameters
    ----------
    db : HopliteDBInterface
        Open Hoplite database.
    results_obj : SearchResults
        Search results with .search_results list of result objects.
    audio_filepath_loader : callable
        loader(recording_id, offset_s) -> (audio_array, sample_rate)
    sample_rate_hz : int
        Default sample rate if loader doesn't return one.
    query_label : str
        The label being searched for (shown in UI header).
    annotator_id : str
        Annotator name for provenance records.
    host : str
        Server host (e.g. "0.0.0.0").
    port : int
        Server port.
    share : bool
        Whether to create a public Gradio share link.
    db_dir : str
        DB directory path for provenance records.
    classifier_path : str | None
        Classifier path for provenance records.
    label_classes : list[str] | None
        Multi-class label choices. None = binary positive/negative.
    detections_info : dict | None
        Detection batch info for UI header (offset, total).
    spectrogram_type : str
        One of "linear", "mel", "perch", "pcen".
    audio_base_dir : str
        Override audio file directory.
    """
    import gradio as gr
    import datetime as _dt

    log.info("Building Gradio labeling interface...")

    session_id      = _dt.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{annotator_id}"
    _db_dir_cap     = db_dir
    _classifier_cap = classifier_path
    log.info("Labeling session ID: %s", session_id)

    # ── Load audio segments ───────────────────────────────────────────────────
    def _get_source(wid):
        from perch_hoplite.db import interface as iface
        try:
            window   = db.get_window(wid)
            recording = db.get_recording(window.recording_id)
            deployment = db.get_deployment(recording.deployment_id)
            class _Src:
                source_id    = recording.filename
                dataset      = deployment.project or ""
                offsets      = (window.offsets[0], window.offsets[1])
                recording_id = None
            return _Src()
        except Exception:
            # Fallback: use SQLite directly
            dbp = _sqlite_path_for(db)
            if dbp:
                con = sqlite3.connect(dbp)
                row = con.execute("""
                    SELECT r.filename, w.offsets
                    FROM windows w JOIN recordings r ON r.id = w.recording_id
                    WHERE w.id = ?
                """, (int(wid),)).fetchone()
                con.close()
                if row:
                    fname, off_blob = row
                    if isinstance(off_blob, (bytes, bytearray)) and len(off_blob) >= 16:
                        s, e = struct.unpack_from("<dd", off_blob)
                    else:
                        s, e = 0.0, 5.0
                    class _FallbackSrc:
                        source_id = fname
                        dataset   = ""
                        offsets   = (s, e)
                        recording_id = None
                    return _FallbackSrc()
            raise

    segments = []
    for r in results_obj.search_results:
        wid = r.window_id
        try:
            source = _get_source(wid)
            recording_id = getattr(source, "source_id", str(wid))
            offsets  = getattr(source, "offsets", (0.0, 5.0))
            offset_s = float(offsets[0]) if offsets else 0.0
            end_s    = float(offsets[1]) if offsets and len(offsets) > 1 else offset_s + 5.0
            audio, sr_actual = audio_filepath_loader(recording_id, offset_s)
        except Exception as exc:
            import traceback as _tb
            log.warning("Could not load audio for window %s: %s\n%s",
                        wid, exc, _tb.format_exc())
            continue
        segments.append({
            "window_id":    int(wid),
            "recording_id": recording_id,
            "offset_s":     offsets[0] if offsets else 0.0,
            "end_offset_s": offsets[1] if offsets else 5.0,
            "score":        r.sort_score,
            "audio":        audio,
            "sample_rate":  sr_actual or sample_rate_hz,
        })

    log.info("Loaded %d audio segments for labeling.", len(segments))

    # ── Segment card HTML ─────────────────────────────────────────────────────
    def _segment_card(seg, idx):
        wav_b64  = make_audio_b64(seg["audio"], seg["sample_rate"])
        spec_b64 = make_spectrogram_image(
            seg["audio"], seg["sample_rate"], spec_type=spectrogram_type,
            colormap=colormap)
        fname = seg["recording_id"].split("/")[-1]
        player_html = (
            f"<audio controls style='width:100%;margin-top:6px;height:40px;'"
            f" src='{wav_b64}'></audio>")
        return (
            f"<div style='background:#1e293b;border-radius:8px;padding:12px;"
            f"margin-bottom:8px;color:#e2e8f0;font-family:monospace;font-size:11px;'>"
            f"<b>#{idx+1}</b> &nbsp; <span style='color:#7dd3fc'>{fname}</span>"
            f" &nbsp; <span style='color:#94a3b8'>"
            f"{seg['offset_s']:.1f}s – {seg['end_offset_s']:.1f}s</span>"
            f" &nbsp; <span style='color:#fbbf24'>score={seg['score']:.3f}</span><br>"
            f"<img src='{spec_b64}' style='width:100%;margin-top:6px;"
            f"border-radius:4px;display:block;'/>"
            f"{player_html}"
            f"</div>"
        )

    # ── DB helpers ────────────────────────────────────────────────────────────
    def _sqlite_path_for(db_obj):
        p = None
        cfg = getattr(db_obj, "db_config", None)
        if cfg:
            p = str(getattr(cfg, "db_path", "") or "")
        if not p:
            for a in ("db_path", "_db_path", "path"):
                v = getattr(db_obj, a, None)
                if v:
                    p = str(v)
                    break
        if p and os.path.isdir(p):
            p = os.path.join(p, "hoplite.sqlite")
        return p

    _session_annotations: dict = {}

    def _write_label(wid, choice):
        if choice == "unlabeled":
            _session_annotations.pop(int(wid), None)
            return False
        lt = _LT.POSITIVE if (not label_classes or choice != "negative") else _LT.NEGATIVE
        dbp = _sqlite_path_for(db)
        con = sqlite3.connect(dbp)
        row = con.execute(
            "SELECT recording_id, offsets FROM windows WHERE id=?",
            (int(wid),)).fetchone()
        if row is None:
            con.close()
            raise KeyError(f"window {wid} not found")
        rec_id, off_blob = row
        if isinstance(off_blob, (bytes, bytearray)) and len(off_blob) >= 16:
            start_s, end_s = struct.unpack_from("<dd", off_blob)
        else:
            start_s, end_s = 0.0, 5.0
        off_enc    = struct.pack("<dd", start_s, end_s)
        prov       = f"gradio_gui:{annotator_id}"
        store_label = choice if label_classes else query_label
        con.execute("""
            DELETE FROM annotations WHERE recording_id=? AND offsets=?
        """, (rec_id, off_enc))
        con.execute("""
            INSERT INTO annotations (recording_id, offsets, label, label_type, provenance)
            VALUES (?, ?, ?, ?, ?)
        """, (rec_id, off_enc, store_label, int(lt), prov))
        con.commit()
        fname_row = con.execute(
            "SELECT filename FROM recordings WHERE id=?", (rec_id,)).fetchone()
        con.close()
        fname = fname_row[0] if fname_row else str(rec_id)
        score = next((s["score"] for s in segments if s["window_id"] == int(wid)), None)
        _session_annotations[int(wid)] = {
            "window_id":  int(wid),
            "filename":   fname,
            "offset_s":   round(start_s, 3),
            "end_s":      round(end_s, 3),
            "label":      choice,
            "label_type": int(lt),
            "score":      round(score, 4) if score is not None else None,
        }
        return True

    def _label_counts():
        try:
            dbp = _sqlite_path_for(db)
            con = sqlite3.connect(dbp)
            pos = dict(con.execute(
                "SELECT label, COUNT(*) FROM annotations WHERE label_type=? GROUP BY label",
                (_LT.POSITIVE,)).fetchall())
            neg = dict(con.execute(
                "SELECT label, COUNT(*) FROM annotations WHERE label_type=? GROUP BY label",
                (_LT.NEGATIVE,)).fetchall())
            con.close()
            return pos, neg
        except Exception as exc:
            return {}, {"error": str(exc)}

    # ── Label choices ─────────────────────────────────────────────────────────
    _default_classes = ["positive", "negative", "unlabeled"]
    if label_classes:
        _choices = [c for c in label_classes if c != "unlabeled"] + ["unlabeled"]
    else:
        _choices = _default_classes

    # ── Build Gradio interface ────────────────────────────────────────────────
    with gr.Blocks(
        title="Perch Hoplite — Audio Labeling",
        css=_GRADIO_CSS,
    ) as demo:
        _class_str = ", ".join(f"`{c}`" for c in _choices if c != "unlabeled")
        _det_info  = ""
        if detections_info:
            _offset   = detections_info.get("offset", 0)
            _total    = detections_info.get("total", len(segments))
            _batch_end = _offset + len(segments)
            _det_info = (
                f"**Detections:** showing {_offset+1}–{_batch_end} of {_total} "
                f"&nbsp;&nbsp; **Batch offset:** {_offset}  \n"
            )
        gr.Markdown(
            f"""
# 🐋 Perch Hoplite — Audio Labeling Interface
**Query label:** `{query_label}` &nbsp;&nbsp; **Annotator:** `{annotator_id}`  
**Results loaded:** {len(segments)}  
{_det_info}**Label classes:** {_class_str}  
Click a label for each segment, then **Save Labels to DB**.
"""
        )

        save_btn   = gr.Button("💾 Save Labels to DB", variant="primary")
        status_box = gr.Textbox(label="Status", interactive=False, lines=4)

        radio_components = []
        ctx_outputs      = []

        with gr.Column():
            for i, seg in enumerate(segments):
                with gr.Row():
                    with gr.Column(scale=4):
                        gr.HTML(_segment_card(seg, i))
                        log.info("Pre-computing 30s context for segment %d/%d...",
                                 i+1, len(segments))
                        _ctx_spec_html, _ctx_audio_data = load_30s_context(
                            seg, audio_base_dir, spectrogram_type, colormap=colormap)
                        ctx_spec  = gr.HTML(_ctx_spec_html, visible=True)
                        ctx_audio = gr.Audio(
                            value=_ctx_audio_data if _ctx_audio_data is not None else None,
                            label="30s context", visible=True,
                            show_label=False, interactive=False)
                        ctx_btn = gr.Button(
                            "↺ Reload 30s context", size="sm",
                            variant="secondary", elem_classes=["ctx-btn"],
                            visible=False)
                        ctx_outputs.append((ctx_btn, ctx_spec, ctx_audio, seg))
                    with gr.Column(scale=1):
                        radio = gr.Radio(
                            choices=_choices,
                            value="unlabeled",
                            label=f"Label #{i+1}  (wid={seg['window_id']})",
                            elem_classes=["label-radio"],
                        )
                        radio_components.append((seg["window_id"], radio))

        # Auto-save on radio click
        def _make_autosave(wid):
            def _autosave(choice):
                try:
                    saved = _write_label(wid, choice)
                    if saved:
                        log.info("Auto-saved: window %s -> %s", wid, choice)
                except Exception as exc:
                    log.warning("Auto-save failed for window %s: %s", wid, exc)
                return gr.update()
            return _autosave

        for wid, radio in radio_components:
            radio.change(
                fn=_make_autosave(wid),
                inputs=[radio],
                outputs=[],
            )

        # Reload 30s context buttons
        def _make_ctx_handler(s):
            def _handler():
                spec_html, audio_data = load_30s_context(
                    s, audio_base_dir, spectrogram_type)
                audio_update = (gr.Audio(value=audio_data, visible=True)
                                if audio_data is not None
                                else gr.Audio(visible=False))
                return gr.HTML(spec_html), audio_update
            return _handler

        for ctx_btn, ctx_spec, ctx_audio, seg in ctx_outputs:
            ctx_btn.click(
                fn=_make_ctx_handler(seg),
                inputs=[],
                outputs=[ctx_spec, ctx_audio],
            )

        def save_labels(*radio_values):
            saved = skipped = errors = 0
            for (wid, _), choice in zip(radio_components, radio_values):
                try:
                    if _write_label(wid, choice):
                        saved += 1
                    else:
                        skipped += 1
                except Exception as exc:
                    log.warning("Batch save failed for window %s: %s", wid, exc)
                    errors += 1
            pos, neg = _label_counts()
            err_str = f"  ({errors} errors)" if errors else ""
            msg = (
                f"Saved {saved} labels ({skipped} unlabeled skipped){err_str}.\n"
                f"DB totals — positive: {pos}  negative: {neg}\n"
                f"Labels are also auto-saved on each click — reload is safe."
            )
            log.info(msg)
            save_label_provenance(
                session_id=session_id,
                db_dir=_db_dir_cap,
                classifier_path=_classifier_cap,
                annotator_id=annotator_id,
                query_label=query_label,
                annotations=list(_session_annotations.values()),
            )
            return msg

        save_btn.click(
            fn=save_labels,
            inputs=[r for _, r in radio_components],
            outputs=status_box,
        )

    log.info("=" * 60)
    log.info("Launching Gradio labeling GUI")
    log.info("  Access at: http://%s:%d",
             host if host != "0.0.0.0" else "<server-ip>", port)
    log.info("  Press Ctrl+C to stop the server.")
    log.info("=" * 60)

    import gradio as _gr
    _launch_kwargs = dict(
        server_name=host,
        server_port=port,
        share=share,
        show_error=True,
        quiet=False,
    )
    if int(_gr.__version__.split(".")[0]) >= 5:
        _launch_kwargs["allowed_paths"] = [
            "/mnt/PAM_Analysis", "/mnt/PAM_Archive", "/home/duane", "/tmp"]
    demo.launch(**_launch_kwargs)
