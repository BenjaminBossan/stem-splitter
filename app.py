import logging
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

import demucs.separate


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


DEMUCS_MODELS = [
    "htdemucs",
    "htdemucs_ft",
    "htdemucs_6s",
    "hdemucs_mmi",
    "mdx",
    "mdx_extra",
]

MAX_STEMS = 6  # htdemucs_6s
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
DEFAULT_SR = 44100
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_GUIDE_MIDIS = list(range(36, 97, 3))  # reduce clutter with wider spacing


def _midi_to_name(midi: int) -> str:
    octave = (midi // 12) - 1
    return f"{NOTE_NAMES[midi % 12]}{octave}"


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
    )


def ffmpeg_has_rubberband() -> bool:
    p = _run(["ffmpeg", "-hide_banner", "-filters"])
    return p.returncode == 0 and " rubberband " in p.stdout


def _slugify(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s).strip("_")
    return s or "track"


def _ensure_audio_file(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File does not exist: {p}")
    if p.suffix.lower() not in AUDIO_EXTS:
        raise ValueError(
            f"Unsupported file type: {p.suffix}. Supported: {sorted(AUDIO_EXTS)}"
        )
    return p


def _cleanup_dir(path_like) -> None:
    if not path_like:
        return

    # Gradio v6 can sometimes hand you component objects if inputs are miswired.
    # Only accept actual filesystem-ish values.
    if not isinstance(path_like, (str, os.PathLike)):
        return

    p = Path(path_like)
    if p.exists() and p.is_dir():
        shutil.rmtree(p, ignore_errors=True)


def _ffmpeg_convert_to_wav(in_path: Path, out_path: Path) -> None:
    logger.info("Converting %s to wav at %s", in_path, out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    p = _run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(in_path),
            "-vn",
            "-ac",
            "2",
            "-ar",
            str(DEFAULT_SR),
            str(out_path),
        ]
    )
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg convert failed:\n{p.stderr.strip()}")
    logger.info("Finished converting %s to wav", in_path)


def _wav_to_mp3(wav_path: Path, mp3_path: Path, bitrate: str = "320k") -> None:
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    p = _run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(wav_path),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            bitrate,
            str(mp3_path),
        ]
    )
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg mp3 encode failed:\n{p.stderr.strip()}")


def _apply_rubberband(
    in_wav: Path, out_wav: Path, tempo: float, semitones: float
) -> None:
    pitch_ratio = float(2 ** (semitones / 12.0))
    logger.info(
        "Applying rubberband to %s with tempo=%s pitch_ratio=%s -> %s",
        in_wav,
        tempo,
        pitch_ratio,
        out_wav,
    )
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    af = f"rubberband=tempo={tempo}:pitch={pitch_ratio}"
    p = _run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(in_wav),
            "-af",
            af,
            str(out_wav),
        ]
    )
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg rubberband failed:\n{p.stderr.strip()}")
    logger.info("Finished rubberband processing for %s", in_wav)


def _mix_wavs(wav_paths: list[Path]) -> tuple[np.ndarray, int]:
    if not wav_paths:
        raise ValueError("No stems selected.")

    logger.info("Mixing %s wav files: %s", len(wav_paths), wav_paths)

    data_list: list[np.ndarray] = []
    sr: Optional[int] = None
    max_len = 0

    for wp in wav_paths:
        x, this_sr = sf.read(wp, always_2d=True)
        if sr is None:
            sr = int(this_sr)
        elif int(this_sr) != sr:
            raise ValueError(
                f"Sample-rate mismatch: {wp.name} is {this_sr} Hz, expected {sr} Hz"
            )
        x = x.astype(np.float32)
        data_list.append(x)
        max_len = max(max_len, x.shape[0])

    padded: list[np.ndarray] = []
    for x in data_list:
        if x.shape[0] < max_len:
            pad = np.zeros((max_len - x.shape[0], x.shape[1]), dtype=np.float32)
            x = np.concatenate([x, pad], axis=0)
        padded.append(x)

    mix = np.sum(padded, axis=0)
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 1.0:
        mix = mix / peak

    return mix, int(sr if sr is not None else DEFAULT_SR)


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr, subtype="PCM_16")


def _yt_dlp_download(url: str, outdir: Path) -> Path:
    logger.info("Starting download from %s", url)
    outdir.mkdir(parents=True, exist_ok=True)
    tmpl = str(outdir / "input.%(ext)s")

    # bestaudio is typically m4a/webm; we’ll convert to wav if needed.
    p = _run(
        [
            "yt-dlp",
            "-f",
            "bestaudio/best",
            "--no-playlist",
            "-o",
            tmpl,
            url,
        ]
    )
    if p.returncode != 0:
        raise RuntimeError(
            "yt-dlp failed. The link may be unsupported, geo/age restricted, or require cookies.\n\n"
            f"{p.stderr.strip()}"
        )

    # Find the downloaded file (newest file in outdir)
    files = [x for x in outdir.iterdir() if x.is_file()]
    if not files:
        raise RuntimeError("yt-dlp reported success but no file appeared.")
    downloaded = max(files, key=lambda x: x.stat().st_mtime)
    logger.info("Finished download to %s", downloaded)
    return downloaded


def _render_spectrogram(
    wav_path: Path, start_seconds: float, window_seconds: float
) -> tuple[Path, float]:
    info = sf.info(wav_path)
    sr = int(info.samplerate or DEFAULT_SR)
    total_frames = int(info.frames or 0)
    total_duration = total_frames / sr if total_frames else 0.0

    window_seconds = max(1.0, float(window_seconds))
    start_seconds = max(0.0, float(start_seconds))
    max_start = max(0.0, total_duration - window_seconds)
    start_seconds = min(start_seconds, max_start)

    start_frame = int(start_seconds * sr)
    end_frame = start_frame + int(window_seconds * sr)
    if total_frames:
        end_frame = min(end_frame, total_frames)

    segment, _ = sf.read(
        wav_path, start=start_frame, stop=end_frame, dtype="float32", always_2d=True
    )
    if segment.size == 0:
        raise ValueError("Selected segment is empty. Try resetting the segment start.")

    mono = segment.mean(axis=1)
    n_fft = 2048
    hop = 512
    window = np.hanning(n_fft)

    frames: list[np.ndarray] = []
    for i in range(0, max(1, mono.shape[0] - n_fft), hop):
        chunk = mono[i : i + n_fft]
        if chunk.shape[0] < n_fft:
            pad = np.zeros(n_fft - chunk.shape[0], dtype=chunk.dtype)
            chunk = np.concatenate([chunk, pad])
        chunk = chunk * window
        spec = np.fft.rfft(chunk)
        frames.append(np.abs(spec))

    if not frames:
        raise ValueError("Audio too short for spectrogram generation.")

    mag = np.maximum(np.stack(frames, axis=1), 1e-12)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    times = start_seconds + (np.arange(mag.shape[1]) * hop / sr)

    fig, ax = plt.subplots(figsize=(14, 10))
    mesh = ax.pcolormesh(times, freqs, 20 * np.log10(mag), shading="auto", cmap="magma")
    fontsize = 16
    ax.set_yscale("log")
    ax.set_xlabel("Time (s)", fontsize=fontsize)
    ax.set_ylabel("Frequency (Hz)", fontsize=fontsize)
    ax.set_title(f"Spectrogram: {wav_path.stem}", fontsize=fontsize)

    if len(times) > 0:
        x_min, x_max = float(times[0]), float(times[-1])
    else:
        x_min, x_max = start_seconds, start_seconds + window_seconds
    x_pad = max(0.5, (x_max - x_min) * 0.05)
    label_x = x_max + (x_pad * 0.3)

    for midi in NOTE_GUIDE_MIDIS:
        f = 440.0 * (2 ** ((midi - 69) / 12.0))
        if f <= freqs[-1]:
            color = "green"
            ax.axhline(
                f,
                color=color,
                alpha=0.75,
                linewidth=1.5,
                linestyle="--",
                zorder=2,
            )
            ax.text(
                label_x,
                f,
                _midi_to_name(midi),
                color=color,
                fontsize=fontsize,
                va="center",
                ha="left",
                bbox=dict(facecolor="black", alpha=0.3, edgecolor="none", pad=1.5),
            )

    fig.colorbar(mesh, ax=ax, label="Magnitude (dB)")
    ax.set_ylim(60, min(freqs[-1], 4000))
    ax.set_xlim(x_min, x_max + x_pad)
    fig.tight_layout()

    out_path = Path(tempfile.mkstemp(suffix="_spec.png", prefix="spectro_")[1])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path, start_seconds


@dataclass
class SepResult:
    workdir: Path
    model: str
    stems: dict[str, Path]


def _demucs_separate(
    audio_path: Path, model: str, device: str, two_stems: bool, shifts: int
) -> SepResult:
    logger.info(
        "Starting Demucs separation: model=%s device=%s two_stems=%s shifts=%s",
        model,
        device,
        two_stems,
        shifts,
    )
    workdir = Path(tempfile.mkdtemp(prefix="demucs_gradio_"))
    outdir = workdir / "separated"
    outdir.mkdir(parents=True, exist_ok=True)

    args = [
        "-n",
        model,
        "-o",
        str(outdir),
        "-d",
        device,
        "--float32",
    ]
    if shifts and int(shifts) > 0:
        args += ["--shifts", str(int(shifts))]
    if two_stems:
        args += ["--two-stems", "vocals"]
    args += [str(audio_path)]

    demucs.separate.main(args)

    model_dir = outdir / model
    candidates = (
        list(model_dir.glob("*")) if model_dir.exists() else list(outdir.glob("*/*"))
    )
    if not candidates:
        raise RuntimeError("Demucs produced no output directory.")
    track_dir = max(candidates, key=lambda p: p.stat().st_mtime)

    stems: dict[str, Path] = {wav.stem: wav for wav in track_dir.glob("*.wav")}
    if not stems:
        raise RuntimeError("Demucs produced no stem wav files.")
    logger.info("Finished Demucs separation with stems: %s", ", ".join(sorted(stems)))
    return SepResult(workdir=workdir, model=model, stems=stems)


def build_ui():
    has_rb = ffmpeg_has_rubberband()

    def _init_spectrogram_state():
        return 0.0

    def _empty_stem_outputs():
        name_vals = [""] * MAX_STEMS
        audio_vals = [None] * MAX_STEMS
        return name_vals, audio_vals

    with gr.Blocks(
        title="Demucs Stem Separator",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="cyan",
            neutral_hue="slate",
        ),
        css="""
        body {
            background: radial-gradient(circle at 12% 18%, rgba(91, 33, 182, 0.24), transparent 34%),
                        radial-gradient(circle at 88% 12%, rgba(34, 211, 238, 0.18), transparent 32%),
                        radial-gradient(circle at 18% 72%, rgba(59, 130, 246, 0.16), transparent 34%),
                        #0b1224;
        }
        .gradio-container {
            max-width: 1540px;
            width: min(1540px, 99vw);
            margin: 18px auto 26px;
            padding: 2px 8px 14px;
            background: transparent;
        }
        .app-header {
            background: linear-gradient(120deg, #5b21b6, #2563eb, #06b6d4);
            color: white;
            padding: 22px 26px;
            border-radius: 18px;
            box-shadow: 0 20px 50px rgba(45, 55, 72, 0.25);
        }
        .section-card {
            background: white;
            border-radius: 16px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 10px 30px rgba(17, 24, 39, 0.06);
            padding: 18px 18px 8px;
        }
        .section-card h3, .section-card h4 {
            margin-top: 0;
            margin-bottom: 10px;
        }
        .compact-row .gr-form .gr-block, .compact-row .gr-block {
            margin-bottom: 6px !important;
        }
        .button-row button {
            min-height: 52px;
            font-weight: 700;
        }
        .accent-text {
            color: #4338ca;
            font-weight: 600;
        }
        .spec-image img {
            border-radius: 12px;
        }
        """,
    ) as demo:
        state_workdir = gr.State(value=None)  # str
        state_input_audio = gr.State(value=None)  # str
        state_stems = gr.State(value=None)  # dict[str, str]

        gr.HTML(
            """
            <div class="app-header">
              <h1 style="margin: 0; font-size: 30px;">Demucs Stem Separator</h1>
              <p style="margin: 6px 0 0; font-size: 16px; opacity: 0.9;">Split songs into clean stems, audition them, and blend the pieces you need.</p>
            </div>
            """
        )

        with gr.Accordion("Quick tips", open=False):
            gr.Markdown(
                "- Choose **Upload file** or **From link** (YouTube etc.).\n"
                "- Link downloads require `yt-dlp` in the environment.\n"
                "- Note: Excessive use of yt-dlp may result in rate limits."
                "- Tempo/pitch controls require ffmpeg with the `rubberband` filter.\n"
                "- Separation can take time; GPU is recommended.",
            )

        with gr.Row(equal_height=True):
            with gr.Column(scale=6):
                with gr.Group(elem_classes=["section-card"]):
                    gr.Markdown("### 1) Choose your source")
                    with gr.Row():
                        source = gr.Radio(
                            choices=["Upload file", "From link"],
                            value="Upload file",
                            label="Input source",
                        )
                    with gr.Row():
                        inp_audio = gr.Audio(
                            label="Audio file",
                            type="filepath",
                            visible=True,
                        )
                        inp_url = gr.Textbox(
                            label="Audio link (YouTube etc.)",
                            placeholder="Paste a URL here",
                            visible=False,
                        )

                    gr.Markdown("### 2) Separation settings")
                    with gr.Row():
                        model = gr.Dropdown(
                            choices=DEMUCS_MODELS,
                            value="htdemucs",
                            label="Demucs model",
                            info="htdemucs is a strong default. htdemucs_6s provides extra stems (incl. guitar/piano) when supported.",
                        )

                    with gr.Row():
                        device = gr.Dropdown(
                            choices=["cuda", "cpu"],
                            value="cuda"
                            if _run(
                                [
                                    "python",
                                    "-c",
                                    "import torch; print(int(torch.cuda.is_available()))",
                                ]
                            ).stdout.strip()
                            == "1"
                            else "cpu",
                            label="Device",
                            info="Use cuda if available for speed.",
                        )
                        two_stems = gr.Checkbox(
                            label="Two-stems mode (vocals + no_vocals)",
                            value=False,
                            info="Uses Demucs' --two-stems=vocals. Helpful if you only want a quick vocal split.",
                        )
                        shifts = gr.Slider(
                            minimum=0,
                            maximum=10,
                            value=2,
                            step=1,
                            label="Shifts (quality vs speed)",
                            info=(
                                "Uses Demucs' time-shift trick. Higher usually improves quality but increases runtime roughly linearly. "
                                "0 disables it."
                            ),
                        )

                    with gr.Row(elem_classes=["button-row"]):
                        sep_btn = gr.Button("Separate", variant="primary")
                        reset_btn = gr.Button("Reset", variant="secondary")

                    gr.Markdown(
                        (
                            "**Note:** your ffmpeg does not appear to have the `rubberband` audio filter enabled, "
                            "so tempo/pitch controls will be ignored."
                        ),
                        visible=not has_rb,
                        elem_classes=["accent-text"],
                    )

            with gr.Column(scale=5):
                with gr.Group(elem_classes=["section-card"]):
                    gr.Markdown("### 3) Mixdown / Export")
                    with gr.Row():
                        export_fmt = gr.Radio(
                            choices=["wav", "mp3"], value="wav", label="Export format"
                        )
                        mp3_bitrate = gr.Dropdown(
                            choices=["192k", "256k", "320k"],
                            value="320k",
                            label="MP3 bitrate",
                        )

                    stem_selector = gr.CheckboxGroup(
                        label="Select stems to include in mixdown",
                        choices=[],
                        value=[],
                        info="The mixdown is a simple sum + peak normalization (no fancy loudness matching).",
                    )

                    with gr.Row():
                        tempo = gr.Slider(
                            minimum=0.5,
                            maximum=1.5,
                            value=1.0,
                            step=0.01,
                            label="Tempo (keep pitch)",
                            info="Requires ffmpeg rubberband. 1.0 = no change.",
                        )
                        semitones = gr.Slider(
                            minimum=-12.0,
                            maximum=12.0,
                            value=0.0,
                            step=0.1,
                            label="Pitch shift (keep tempo, semitones)",
                            info="Requires ffmpeg rubberband. 0.0 = no change.",
                        )

                    mix_btn = gr.Button("Create mixdown", variant="primary")
                    mix_audio = gr.Audio(
                        label="Mixdown preview", type="filepath", buttons=["download"]
                    )
                    export_file = gr.File(label="Download exported file")

        with gr.Group(elem_classes=["section-card"]):
            gr.Markdown("### Stems (preview)")
            stem_audio_outputs: list[gr.Audio] = []
            stem_name_outputs: list[gr.Textbox] = []
            for row_start in range(0, MAX_STEMS, 2):
                with gr.Row():
                    for j in range(row_start, min(row_start + 2, MAX_STEMS)):
                        with gr.Column():
                            name = gr.Textbox(
                                label=f"Stem {j + 1} name", interactive=False
                            )
                            audio = gr.Audio(
                                label=f"Stem {j + 1}",
                                type="filepath",
                                buttons=["download"],
                            )
                        stem_name_outputs.append(name)
                        stem_audio_outputs.append(audio)

        with gr.Group(elem_classes=["section-card"]):
            gr.Markdown("### Spectrogram viewer")
            with gr.Row():
                spec_stem = gr.Dropdown(
                    label="Stem to visualize",
                    choices=[],
                    value=None,
                    info="Select a separated stem to view its spectrogram.",
                )
                spec_window = gr.Slider(
                    minimum=2,
                    maximum=60,
                    value=10,
                    step=1,
                    label="Window length (seconds)",
                    info="To reduce load, only the current window is analyzed. Advance to move through the track.",
                )
            with gr.Row(elem_classes=["button-row"]):
                spec_show = gr.Button("Show spectrogram", variant="primary")
                spec_prev = gr.Button("Previous segment")
                spec_next = gr.Button("Next segment")
                spec_restart = gr.Button("Restart from beginning")
            with gr.Row():
                spec_offset_display = gr.Number(
                    label="Current segment start (s)",
                    value=0.0,
                    interactive=False,
                )
            spec_image = gr.Image(
                label="Spectrogram with note guides",
                type="filepath",
                height=640,
                interactive=False,
                elem_classes=["spec-image"],
            )
            spec_offset_state = gr.State(value=_init_spectrogram_state())

        def on_source_change(src: str):
            if src == "Upload file":
                return gr.update(visible=True), gr.update(visible=False, value="")
            else:
                return gr.update(visible=False, value=None), gr.update(visible=True)

        source.change(on_source_change, inputs=[source], outputs=[inp_audio, inp_url])

        def do_reset(prev_workdir: Optional[str]):
            _cleanup_dir(prev_workdir)
            name_vals, audio_vals = _empty_stem_outputs()
            return (
                "Upload file",
                None,
                "",
                "htdemucs",
                "cuda"
                if _run(
                    [
                        "python",
                        "-c",
                        "import torch; print(int(torch.cuda.is_available()))",
                    ]
                ).stdout.strip()
                == "1"
                else "cpu",
                False,
                None,
                None,
                *name_vals,
                *audio_vals,
                gr.update(choices=[], value=[]),
                gr.update(choices=[], value=None),
                "wav",
                "320k",
                1.0,
                0.0,
                None,
                None,
                _init_spectrogram_state(),
                None,
                0.0,
            )

        def do_separate(
            src: str,
            audio_fp: Optional[str],
            url: str,
            model_name: str,
            device_name: str,
            two_stems_flag: bool,
            shifts_val: int,
            prev_workdir: Optional[str],
        ):
            logger.info(
                "Starting separation with src=%s model=%s device=%s two_stems=%s shifts=%s",
                src,
                model_name,
                device_name,
                two_stems_flag,
                shifts_val,
            )
            _cleanup_dir(prev_workdir)

            if src == "Upload file":
                if not audio_fp:
                    raise gr.Error(
                        "Please upload an audio file, or switch to 'From link'."
                    )
                audio_path = _ensure_audio_file(audio_fp)
                workdir = Path(tempfile.mkdtemp(prefix="demucs_input_"))
                local_audio = Path(audio_path)  # already local
            else:
                if not url.strip():
                    raise gr.Error("Please paste a link, or switch to 'Upload file'.")
                workdir = Path(tempfile.mkdtemp(prefix="demucs_link_"))
                dl_dir = workdir / "download"
                downloaded = _yt_dlp_download(url.strip(), dl_dir)
                if downloaded.suffix.lower() in AUDIO_EXTS:
                    local_audio = downloaded
                else:
                    wav = workdir / "download" / "input.wav"
                    _ffmpeg_convert_to_wav(downloaded, wav)
                    local_audio = wav

            # Run demucs (creates its own workdir). We'll keep everything under that demucs workdir for cleanup.
            res = _demucs_separate(
                local_audio,
                model_name,
                device_name,
                bool(two_stems_flag),
                int(shifts_val),
            )

            logger.info(
                "Completed separation session at %s with stems: %s",
                workdir,
                ", ".join(sorted(res.stems)),
            )

            # Also include the download temp dir (if any) under res.workdir for single cleanup path:
            # move/link: simplest is to copy, but that’s wasteful. Instead, just cleanup both:
            # We’ll store a “root” dir that includes both by nesting the demucs dir into our workdir.
            # In practice, easiest: move demucs.workdir into our outer workdir, and return outer workdir.
            outer = workdir / "session"
            outer.mkdir(parents=True, exist_ok=True)
            new_demucs_dir = outer / "demucs"
            shutil.move(str(res.workdir), str(new_demucs_dir))

            # rewrite stem paths
            stems_sorted = []
            stems_dict: dict[str, str] = {}
            for k, v in sorted(res.stems.items(), key=lambda kv: kv[0]):
                nv = new_demucs_dir / v.relative_to(res.workdir)
                stems_sorted.append((k, nv))
                stems_dict[k] = str(nv)

            name_vals = [""] * MAX_STEMS
            audio_vals = [None] * MAX_STEMS
            for idx, (stem_name, wav_path) in enumerate(stems_sorted[:MAX_STEMS]):
                name_vals[idx] = stem_name
                audio_vals[idx] = str(wav_path)

            choices = [k for k, _ in stems_sorted]
            default_sel = choices[:]
            default_spec = choices[0] if choices else None

            return (
                str(workdir),  # <- single root dir to cleanup
                str(local_audio),
                stems_dict,
                *name_vals,
                *audio_vals,
                gr.update(choices=choices, value=default_sel),
                gr.update(choices=choices, value=default_spec),
                _init_spectrogram_state(),
                None,
                0.0,
            )

        def do_mixdown(
            selected_stems: list[str],
            stems_dict: dict[str, str],
            workdir: str,
            fmt: str,
            bitrate: str,
            tempo_val: float,
            semitones_val: float,
            src: str,
            audio_fp: Optional[str],
            url: str,
            input_audio: Optional[str],
        ):
            logger.info(
                "Starting mixdown: stems=%s fmt=%s tempo=%s semitones=%s",
                selected_stems,
                fmt,
                tempo_val,
                semitones_val,
            )
            wd = Path(workdir) if workdir else None
            next_input_audio = input_audio

            def _prepare_input(
                existing_workdir: Optional[Path],
            ) -> tuple[Path, Path]:
                if src == "Upload file":
                    if not audio_fp:
                        raise gr.Error(
                            "Please upload an audio file, or switch to 'From link'."
                        )
                    audio_path = _ensure_audio_file(audio_fp)
                    workdir_path = existing_workdir or Path(
                        tempfile.mkdtemp(prefix="mix_input_")
                    )
                    wav_path = workdir_path / "input" / "source.wav"
                    if audio_path.suffix.lower() == ".wav":
                        wav_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(audio_path, wav_path)
                    else:
                        _ffmpeg_convert_to_wav(audio_path, wav_path)
                    return workdir_path, wav_path

                if not url.strip():
                    raise gr.Error("Please paste a link, or switch to 'Upload file'.")

                workdir_path = existing_workdir or Path(
                    tempfile.mkdtemp(prefix="mix_link_")
                )
                dl_dir = workdir_path / "download"
                downloaded = _yt_dlp_download(url.strip(), dl_dir)
                wav_path = workdir_path / "input" / "source.wav"
                if downloaded.suffix.lower() == ".wav":
                    wav_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(downloaded, wav_path)
                else:
                    _ffmpeg_convert_to_wav(downloaded, wav_path)
                return workdir_path, wav_path

            if stems_dict:
                if not selected_stems:
                    raise gr.Error("Select at least one stem to mix.")

                if wd is None:
                    raise gr.Error("Missing session data. Please run separation again.")

                outdir = wd / "mix"
                outdir.mkdir(parents=True, exist_ok=True)

                stem_paths = [
                    Path(stems_dict[s]) for s in selected_stems if s in stems_dict
                ]
                mix, sr = _mix_wavs(stem_paths)

                base = "mix_" + _slugify("+".join(selected_stems))
                wav0 = outdir / f"{base}.wav"
                _write_wav(wav0, mix, sr)

            else:
                if input_audio and Path(input_audio).exists():
                    base_audio = Path(input_audio)
                    wd = (
                        wd or base_audio.parent.parent
                        if base_audio.parent.parent.exists()
                        else base_audio.parent
                    )
                else:
                    wd, base_audio = _prepare_input(wd)
                next_input_audio = str(base_audio)

                outdir = wd / "mix"
                outdir.mkdir(parents=True, exist_ok=True)

                base = _slugify(Path(base_audio).stem)
                wav0 = outdir / f"{base}.wav"
                if base_audio.suffix.lower() == ".wav":
                    shutil.copyfile(base_audio, wav0)
                else:
                    _ffmpeg_convert_to_wav(base_audio, wav0)

            final_wav = wav0
            if ffmpeg_has_rubberband() and (
                abs(float(tempo_val) - 1.0) > 1e-6 or abs(float(semitones_val)) > 1e-6
            ):
                wav1 = outdir / f"{base}_rb.wav"
                _apply_rubberband(wav0, wav1, float(tempo_val), float(semitones_val))
                final_wav = wav1

            if fmt == "wav":
                export_path = final_wav
            else:
                mp3_path = outdir / f"{base}.mp3"
                _wav_to_mp3(final_wav, mp3_path, bitrate=bitrate)
                export_path = mp3_path

            logger.info(
                "Mixdown complete: final_wav=%s export_path=%s", final_wav, export_path
            )

            return str(wd), next_input_audio, str(final_wav), str(export_path)

        def _spectro_common(
            stem_name: Optional[str],
            stems_dict: Optional[dict[str, str]],
            offset: float,
            window: float,
        ):
            if not stems_dict:
                raise gr.Error("No stems available yet. Run separation first.")
            if not stem_name:
                raise gr.Error("Choose a stem to visualize.")
            if stem_name not in stems_dict:
                raise gr.Error("Selected stem is not available.")

            img_path, used_offset = _render_spectrogram(
                Path(stems_dict[stem_name]), float(offset), float(window)
            )
            return str(img_path), float(used_offset), float(used_offset)

        def do_show_spectrogram(
            stem_name: Optional[str],
            stems_dict: Optional[dict[str, str]],
            offset: float,
            window: float,
        ):
            return _spectro_common(stem_name, stems_dict, offset, window)

        def do_next_spectrogram(
            stem_name: Optional[str],
            stems_dict: Optional[dict[str, str]],
            offset: float,
            window: float,
        ):
            return _spectro_common(stem_name, stems_dict, offset + window, window)

        def do_prev_spectrogram(
            stem_name: Optional[str],
            stems_dict: Optional[dict[str, str]],
            offset: float,
            window: float,
        ):
            return _spectro_common(
                stem_name, stems_dict, max(0.0, offset - window), window
            )

        def do_restart_spectrogram(
            stem_name: Optional[str],
            stems_dict: Optional[dict[str, str]],
            window: float,
        ):
            return _spectro_common(stem_name, stems_dict, 0.0, window)

        # Outputs for separation
        sep_outputs = [
            state_workdir,
            state_input_audio,
            state_stems,
            *stem_name_outputs,
            *stem_audio_outputs,
            stem_selector,
            spec_stem,
            spec_offset_state,
            spec_image,
            spec_offset_display,
        ]

        sep_btn.click(
            do_separate,
            inputs=[
                source,
                inp_audio,
                inp_url,
                model,
                device,
                two_stems,
                shifts,
                state_workdir,
            ],
            outputs=sep_outputs,
        )

        spec_show.click(
            do_show_spectrogram,
            inputs=[spec_stem, state_stems, spec_offset_state, spec_window],
            outputs=[spec_image, spec_offset_state, spec_offset_display],
        )
        spec_prev.click(
            do_prev_spectrogram,
            inputs=[spec_stem, state_stems, spec_offset_state, spec_window],
            outputs=[spec_image, spec_offset_state, spec_offset_display],
        )
        spec_next.click(
            do_next_spectrogram,
            inputs=[spec_stem, state_stems, spec_offset_state, spec_window],
            outputs=[spec_image, spec_offset_state, spec_offset_display],
        )
        spec_restart.click(
            do_restart_spectrogram,
            inputs=[spec_stem, state_stems, spec_window],
            outputs=[spec_image, spec_offset_state, spec_offset_display],
        )

        mix_btn.click(
            do_mixdown,
            inputs=[
                stem_selector,
                state_stems,
                state_workdir,
                export_fmt,
                mp3_bitrate,
                tempo,
                semitones,
                source,
                inp_audio,
                inp_url,
                state_input_audio,
            ],
            outputs=[state_workdir, state_input_audio, mix_audio, export_file],
        )

        # Reset outputs (explicitly list all components we want to clear)
        reset_btn.click(
            do_reset,
            inputs=[state_workdir],
            outputs=[
                source,
                inp_audio,
                inp_url,
                model,
                device,
                two_stems,
                state_workdir,
                state_input_audio,
                state_stems,
                *stem_name_outputs,
                *stem_audio_outputs,
                stem_selector,
                spec_stem,
                export_fmt,
                mp3_bitrate,
                tempo,
                semitones,
                mix_audio,
                export_file,
                spec_offset_state,
                spec_image,
                spec_offset_display,
            ],
        )

        # Best-effort cleanup on browser close (no inputs in your current Gradio usage)
        demo.unload(lambda: None)

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.queue(max_size=32)
    app.launch(theme=gr.themes.Citrus())
