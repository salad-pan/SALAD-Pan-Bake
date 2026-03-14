from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import gradio as gr
import torch

from inference import InferenceRunner
from utils.config import prepare_custom_base_model, resolve_path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "inference.yaml"

BASE_MODEL_CHOICES = {
    "stable-diffusion-v1-5": "checkpoints/base/stable-diffusion-v1-5",
    "stable-diffusion-v2": "checkpoints/base/stable-diffusion-v2",
    "stable-diffusion-xl": "checkpoints/base/stable-diffusion-xl",
}
CUSTOM_BASE_KEY = "Custom upload"

PATH_PRESETS = {
    "Use config values": {
        "vae_path": "",
        "controlnet_model_name_or_path": "",
        "test_h5_path": "",
    },
    "Preset-A": {
        "vae_path": "checkpoints/preset_a/vae",
        "controlnet_model_name_or_path": "checkpoints/preset_a/controlnet",
        "test_h5_path": "data/preset_a/test.h5",
    },
    "Preset-B": {
        "vae_path": "checkpoints/preset_b/vae",
        "controlnet_model_name_or_path": "checkpoints/preset_b/controlnet",
        "test_h5_path": "data/preset_b/test.h5",
    },
    "Preset-C": {
        "vae_path": "checkpoints/preset_c/vae",
        "controlnet_model_name_or_path": "checkpoints/preset_c/controlnet",
        "test_h5_path": "data/preset_c/test.h5",
    },
}


_runner: InferenceRunner | None = None


def _to_text(value: Any) -> str:
    return "" if value is None else str(value)


def _normalize_path(path_value: str) -> str:
    path_value = _to_text(path_value).strip()
    if not path_value:
        return ""
    resolved = resolve_path(path_value, project_root=PROJECT_ROOT)
    return "" if resolved is None else str(resolved)


def _release_runner() -> None:
    global _runner

    if _runner is not None:
        try:
            _runner.close()
        except Exception:
            traceback.print_exc()
        finally:
            _runner = None

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _resolve_base_model_path(base_model_name: str, base_model_upload: Any) -> str:
    if base_model_name in BASE_MODEL_CHOICES:
        model_path = _normalize_path(BASE_MODEL_CHOICES[base_model_name])
        if not model_path:
            raise ValueError(f"Failed to resolve base model path for: {base_model_name}")
        return model_path

    if base_model_name == CUSTOM_BASE_KEY:
        if base_model_upload is None:
            raise ValueError("Custom upload was selected, but no file was provided.")
        upload_path = getattr(base_model_upload, "name", base_model_upload)
        return prepare_custom_base_model(upload_path, project_root=PROJECT_ROOT)

    raise ValueError(f"Unknown base model option: {base_model_name}")


def _merge_path_overrides(
    preset_name: str,
    vae_path: str,
    controlnet_path: str,
    test_h5_path: str,
) -> dict[str, str]:
    preset = PATH_PRESETS.get(preset_name, {})
    merged = {
        "vae_path": _normalize_path(preset.get("vae_path", "")),
        "controlnet_model_name_or_path": _normalize_path(
            preset.get("controlnet_model_name_or_path", "")
        ),
        "test_h5_path": _normalize_path(preset.get("test_h5_path", "")),
    }

    manual_vae_path = _normalize_path(vae_path)
    manual_controlnet_path = _normalize_path(controlnet_path)
    manual_test_h5_path = _normalize_path(test_h5_path)

    if manual_vae_path:
        merged["vae_path"] = manual_vae_path
    if manual_controlnet_path:
        merged["controlnet_model_name_or_path"] = manual_controlnet_path
    if manual_test_h5_path:
        merged["test_h5_path"] = manual_test_h5_path

    return merged


def _build_overrides(
    base_model_name: str,
    base_model_upload: Any,
    preset_name: str,
    vae_path: str,
    controlnet_path: str,
    test_h5_path: str,
    eval_mode: str,
    range_clip_max: float | None,
    resolution: int,
    num_steps: int,
    ctrl_scale: float,
    mixed_precision: str,
    seed: int | None,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "pretrained_model_name_or_path": _resolve_base_model_path(
            base_model_name,
            base_model_upload,
        ),
        "eval_mode": str(eval_mode),
        "resolution": int(resolution),
        "test_num_inference_steps": int(num_steps),
        "controlnet_conditioning_scale": float(ctrl_scale),
        "mixed_precision": str(mixed_precision),
    }

    if range_clip_max is not None:
        overrides["range_clip_max"] = float(range_clip_max)

    if seed is not None:
        overrides["seed"] = int(seed)

    overrides.update(
        _merge_path_overrides(
            preset_name=preset_name,
            vae_path=vae_path,
            controlnet_path=controlnet_path,
            test_h5_path=test_h5_path,
        )
    )

    return overrides


def ui_init(
    config_path: str,
    base_model_name: str,
    base_model_upload: Any,
    preset_name: str,
    vae_path: str,
    controlnet_path: str,
    test_h5_path: str,
    eval_mode: str,
    range_clip_max: float | None,
    resolution: int,
    num_steps: int,
    ctrl_scale: float,
    mixed_precision: str,
    seed: int | None,
    progress=gr.Progress(track_tqdm=True),
):
    global _runner

    _release_runner()

    try:
        progress(0.05, desc="Resolving configuration...")
        config_path = _normalize_path(config_path)
        if not config_path:
            raise ValueError("Config file path is empty.")

        progress(0.20, desc="Preparing runtime overrides...")
        overrides = _build_overrides(
            base_model_name=base_model_name,
            base_model_upload=base_model_upload,
            preset_name=preset_name,
            vae_path=vae_path,
            controlnet_path=controlnet_path,
            test_h5_path=test_h5_path,
            eval_mode=eval_mode,
            range_clip_max=range_clip_max,
            resolution=resolution,
            num_steps=num_steps,
            ctrl_scale=ctrl_scale,
            mixed_precision=mixed_precision,
            seed=seed,
        )

        progress(0.55, desc="Loading model and dataset...")
        _runner = InferenceRunner(config_path=config_path, overrides=overrides)

        progress(1.0, desc="Ready")
        return (
            gr.update(minimum=0, maximum=_runner.num_samples - 1, value=0, step=1, visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
        )
    except Exception:
        traceback.print_exc()
        progress(1.0, desc="Load failed")
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )


def ui_preview(idx: int):
    global _runner

    if _runner is None:
        return None, None, None, {"error": "Please load the model and dataset first."}

    try:
        lms_image, pan_image = _runner.preview_inputs(int(idx))
        return lms_image, pan_image, None, {}
    except Exception as exc:
        traceback.print_exc()
        return None, None, None, {"error": repr(exc)}


def ui_generate(idx: int):
    global _runner

    if _runner is None:
        yield None, {"error": "Please load the model and dataset first."}
        return

    try:
        for generated_image, metrics in _runner.infer_one_stream(int(idx)):
            yield generated_image, metrics
    except Exception as exc:
        traceback.print_exc()
        yield None, {"error": repr(exc)}


def _on_base_model_change(choice: str):
    return gr.update(visible=(choice == CUSTOM_BASE_KEY), value=None)


def _on_preset_change(preset_name: str):
    preset = PATH_PRESETS.get(
        preset_name,
        {
            "vae_path": "",
            "controlnet_model_name_or_path": "",
            "test_h5_path": "",
        },
    )
    return (
        gr.update(value=preset.get("vae_path", "")),
        gr.update(value=preset.get("controlnet_model_name_or_path", "")),
        gr.update(value=preset.get("test_h5_path", "")),
    )


def build_demo() -> gr.Blocks:
    custom_css = """
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }

    .hint-text {
        color: #6b7280;
        font-size: 0.92rem;
    }

    #img_lms, #img_pan, #img_gen {
        width: 100%;
        aspect-ratio: 1 / 1;
    }

    #img_lms img, #img_pan img, #img_gen img {
        width: 100%;
        height: 100%;
        object-fit: contain;
    }
    """

    with gr.Blocks(css=custom_css, title="SALAD-PAN Demo") as demo:
        gr.HTML(
            """
            <div class="main-header">
                <h1>SALAD-PAN Demo</h1>
                <p style="margin-top: 0.5rem; opacity: 0.9;">
                    Image fusion demo based on diffusion models
                </p>
            </div>
            """
        )

        gr.Markdown("## Configuration and inference settings")

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Base configuration")

                        config_path = gr.Textbox(
                            label="Config file path",
                            value=str(DEFAULT_CONFIG_PATH),
                            placeholder="Enter the full path to the config file.",
                        )

                        gr.Markdown("### Base model")

                        base_model_name = gr.Dropdown(
                            choices=list(BASE_MODEL_CHOICES.keys()) + [CUSTOM_BASE_KEY],
                            value="stable-diffusion-v1-5",
                            label="Base model",
                        )

                        base_model_upload = gr.File(
                            label="Upload custom base model",
                            type="filepath",
                            visible=False,
                            interactive=True,
                        )

                        base_model_name.change(
                            _on_base_model_change,
                            inputs=[base_model_name],
                            outputs=[base_model_upload],
                        )

                        gr.Markdown("### Runtime paths")

                        preset_name = gr.Dropdown(
                            choices=list(PATH_PRESETS.keys()),
                            value="Use config values",
                            label="Path preset",
                        )

                        vae_path = gr.Textbox(
                            label="VAE path",
                            value="",
                            placeholder="Optional. Leave empty to use the config or preset value.",
                        )

                        controlnet_path = gr.Textbox(
                            label="ControlNet path",
                            value="",
                            placeholder="Optional. Leave empty to use the config or preset value.",
                        )

                        test_h5_path = gr.Textbox(
                            label="Test H5 path",
                            value="",
                            placeholder="Optional. Leave empty to use the config or preset value.",
                        )

                        preset_name.change(
                            _on_preset_change,
                            inputs=[preset_name],
                            outputs=[vae_path, controlnet_path, test_h5_path],
                        )

                        load_button = gr.Button("Load model and dataset", variant="primary")

                    with gr.Column(scale=1):
                        gr.Markdown("### Inference settings")

                        with gr.Row():
                            eval_mode = gr.Radio(
                                choices=["reduced", "full"],
                                value="reduced",
                                label="Evaluation mode",
                            )
                            resolution = gr.Radio(
                                choices=[256, 512, 1024, 2048],
                                value=1024,
                                label="Output resolution",
                            )

                        with gr.Row():
                            range_clip_max = gr.Number(
                                value=2047.0,
                                label="Clip max value",
                            )
                            seed = gr.Number(
                                value=2025,
                                label="Random seed",
                                precision=0,
                            )
                            num_steps = gr.Slider(
                                minimum=20,
                                maximum=100,
                                value=50,
                                step=1,
                                label="Inference steps",
                            )

                        with gr.Row():
                            mixed_precision = gr.Radio(
                                choices=["none", "fp16", "bf16"],
                                value="fp16",
                                label="Mixed precision",
                            )
                            ctrl_scale = gr.Slider(
                                minimum=0.0,
                                maximum=2.0,
                                value=1.0,
                                step=0.1,
                                label="Control scale",
                            )

                        sample_header = gr.Markdown("### Sample selection", visible=False)

                        with gr.Row():
                            idx_slider = gr.Slider(
                                label="Sample index",
                                minimum=0,
                                maximum=0,
                                value=0,
                                step=1,
                                visible=False,
                                scale=3,
                            )
                            run_button = gr.Button(
                                "Run fusion and evaluation",
                                variant="primary",
                                visible=False,
                                scale=1,
                            )

        gr.Markdown("## Inference results")

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    out_lms = gr.Image(
                        label="LRMS input",
                        type="pil",
                        elem_id="img_lms",
                    )
                    out_pan = gr.Image(
                        label="PAN input",
                        type="pil",
                        elem_id="img_pan",
                    )
                    out_gen = gr.Image(
                        label="HRMS output",
                        type="pil",
                        elem_id="img_gen",
                    )

                gr.Markdown(
                    """
                    **Image meanings**
                    - Left: LRMS input
                    - Middle: PAN input
                    - Right: HRMS output
                    """
                )

        with gr.Row():
            with gr.Column(scale=2):
                out_metrics = gr.JSON(
                    label="Evaluation metrics",
                    show_label=True,
                )

                gr.Markdown(
                    """
                    **Metric notes**
                    - **PSNR**: higher is better
                    - **SSIM**: higher is better
                    - **SAM**: lower is better
                    - **Q4 / ERGAS / SCC / CC**: reference quality metrics
                    - **HQNR**: no-reference quality metric
                    """
                )

        load_button.click(
            ui_init,
            inputs=[
                config_path,
                base_model_name,
                base_model_upload,
                preset_name,
                vae_path,
                controlnet_path,
                test_h5_path,
                eval_mode,
                range_clip_max,
                resolution,
                num_steps,
                ctrl_scale,
                mixed_precision,
                seed,
            ],
            outputs=[idx_slider, run_button, sample_header],
        )

        idx_slider.change(
            ui_preview,
            inputs=[idx_slider],
            outputs=[out_lms, out_pan, out_gen, out_metrics],
        )

        run_button.click(
            ui_generate,
            inputs=[idx_slider],
            outputs=[out_gen, out_metrics],
        )

        demo.unload(_release_runner)

        gr.Markdown(
            """
            ---
            <div style="text-align: center; color: #6b7280; font-size: 0.875rem;">
                <p>
                    Tip: the first model load may take some time. A GPU can significantly improve inference speed.
                </p>
            </div>
            """
        )

    return demo


if __name__ == "__main__":
    app = build_demo()
    app.queue().launch()
