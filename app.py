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
import numpy as np
import soundfile as sf

import demucs.separate


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


def _mix_wavs(wav_paths: list[Path]) -> tuple[np.ndarray, int]:
    if not wav_paths:
        raise ValueError("No stems selected.")

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
    return max(files, key=lambda x: x.stat().st_mtime)


@dataclass
class SepResult:
    workdir: Path
    model: str
    stems: dict[str, Path]


def _demucs_separate(
    audio_path: Path, model: str, device: str, two_stems: bool, shifts: int
) -> SepResult:
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

    return SepResult(workdir=workdir, model=model, stems=stems)


def build_ui():
    has_rb = ffmpeg_has_rubberband()

    def _empty_stem_outputs():
        name_vals = [""] * MAX_STEMS
        audio_vals = [None] * MAX_STEMS
        return name_vals, audio_vals

    with gr.Blocks(title="Demucs Stem Separator") as demo:
        state_workdir = gr.State(value=None)  # str
        state_stems = gr.State(value=None)  # dict[str, str]

        gr.Markdown(
            "### Demucs Stem Separator\n"
            "Separate a track into stems using Demucs, preview them, then mix selected stems into a new export."
        )

        with gr.Accordion("Help / Notes", open=False):
            gr.Markdown(
                "- Choose **Upload file** or **From link** (YouTube etc.).\n"
                "- Link downloads require `yt-dlp` in the environment.\n"
                "- Note: Excessive use of yt-dlp may result in rate limits."
                "- Tempo/pitch controls require ffmpeg with the `rubberband` filter.\n"
                "- Separation can take time; GPU is recommended."
            )

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

        with gr.Row():
            sep_btn = gr.Button("Separate", variant="primary")
            reset_btn = gr.Button("Reset", variant="secondary")

        gr.Markdown(
            (
                "**Note:** your ffmpeg does not appear to have the `rubberband` audio filter enabled, "
                "so tempo/pitch controls will be ignored."
            ),
            visible=not has_rb,
        )

        gr.Markdown("### Stems (preview)")
        stem_audio_outputs: list[gr.Audio] = []
        stem_name_outputs: list[gr.Textbox] = []
        for i in range(MAX_STEMS):
            with gr.Row():
                name = gr.Textbox(label=f"Stem {i + 1} name", interactive=False)
                audio = gr.Audio(
                    label=f"Stem {i + 1}", type="filepath", buttons=["download"]
                )
            stem_name_outputs.append(name)
            stem_audio_outputs.append(audio)

        stem_selector = gr.CheckboxGroup(
            label="Select stems to include in mixdown",
            choices=[],
            value=[],
            info="The mixdown is a simple sum + peak normalization (no fancy loudness matching).",
        )

        gr.Markdown("### Mixdown / Export")
        with gr.Row():
            export_fmt = gr.Radio(
                choices=["wav", "mp3"], value="wav", label="Export format"
            )
            mp3_bitrate = gr.Dropdown(
                choices=["192k", "256k", "320k"], value="320k", label="MP3 bitrate"
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
                "wav",
                "320k",
                1.0,
                0.0,
                None,
                None,
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

            return (
                str(workdir),  # <- single root dir to cleanup
                stems_dict,
                *name_vals,
                *audio_vals,
                gr.update(choices=choices, value=default_sel),
            )

        def do_mixdown(
            selected_stems: list[str],
            stems_dict: dict[str, str],
            workdir: str,
            fmt: str,
            bitrate: str,
            tempo_val: float,
            semitones_val: float,
        ):
            if not stems_dict:
                raise gr.Error("No stems available yet. Run separation first.")
            if not selected_stems:
                raise gr.Error("Select at least one stem to mix.")

            wd = Path(workdir)
            outdir = wd / "mix"
            outdir.mkdir(parents=True, exist_ok=True)

            stem_paths = [
                Path(stems_dict[s]) for s in selected_stems if s in stems_dict
            ]
            mix, sr = _mix_wavs(stem_paths)

            base = "mix_" + _slugify("+".join(selected_stems))
            wav0 = outdir / f"{base}.wav"
            _write_wav(wav0, mix, sr)

            final_wav = wav0
            if ffmpeg_has_rubberband() and (
                abs(float(tempo_val) - 1.0) > 1e-6 or abs(float(semitones_val)) > 1e-6
            ):
                wav1 = outdir / f"{base}_rb.wav"
                _apply_rubberband(wav0, wav1, float(tempo_val), float(semitones_val))
                final_wav = wav1

            if fmt == "wav":
                return str(final_wav), str(final_wav)
            else:
                mp3_path = outdir / f"{base}.mp3"
                _wav_to_mp3(final_wav, mp3_path, bitrate=bitrate)
                return str(mp3_path), str(mp3_path)

        # Outputs for separation
        sep_outputs = [
            state_workdir,
            state_stems,
            *stem_name_outputs,
            *stem_audio_outputs,
            stem_selector,
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
            ],
            outputs=[mix_audio, export_file],
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
                state_stems,
                *stem_name_outputs,
                *stem_audio_outputs,
                stem_selector,
                export_fmt,
                mp3_bitrate,
                tempo,
                semitones,
                mix_audio,
                export_file,
            ],
        )

        # Best-effort cleanup on browser close (no inputs in your current Gradio usage)
        demo.unload(lambda: None)

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.queue(max_size=32)
    app.launch()
