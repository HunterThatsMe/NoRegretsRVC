import argparse
import os
import shutil
import sys

now_dir = os.getcwd()
if now_dir not in sys.path:
    sys.path.append(now_dir)

CONFIG_PATH = os.path.join(now_dir, "assets", "config.json")
os.environ.setdefault("APPLIO_CONFIG_FILE", CONFIG_PATH)
CONFIG_TEMPLATE_PATH = os.path.join(now_dir, "assets", "config_template.json")
if not os.path.exists(CONFIG_PATH) and os.path.exists(CONFIG_TEMPLATE_PATH):
    shutil.copy(CONFIG_TEMPLATE_PATH, CONFIG_PATH)

try:
    from rvc.lib.platform import platform_config
    platform_config()
except Exception:
    pass

try:
    import rvc.lib.zluda  # noqa: F401
except Exception:
    pass

import gradio as gr

GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6

from assets.themes.NoRegretsRVC import NoRegretsRVCTheme
from tabs.train.train import train_tab
from tabs.download.download import download_tab
from tabs.inference.inference import inference_tab
from tabs.voice_blender.voice_blender import voice_blender_tab
from tabs.tts.tts import tts_tab
from tabs.extra.extra import extra_tab
from tabs.settings.settings import settings_tab

CUSTOM_CSS = """
.gradio-container { max-width: 1360px !important; margin: 0 auto !important; }
.tab-nav button.selected { color: #E51923 !important; border-color: #E51923 !important; }
.gr-box, .gr-panel, .gr-form { border-radius: 12px !important; }
footer { display: none !important; }
* { scrollbar-width: thin; scrollbar-color: #E51923 #100506; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: #100506; }
::-webkit-scrollbar-thumb { background: #E51923; border-radius: 999px; border: 2px solid #100506; }
::-webkit-scrollbar-thumb:hover { background: #ff2b35; }
"""


def _load_theme():
    try:
        return NoRegretsRVCTheme()
    except Exception:
        return gr.themes.Base()


def build_ui(theme=None) -> gr.Blocks:
    if theme is None:
        theme = _load_theme()
    blocks_kwargs = {"title": "NoRegretsRVC (Beta 0.0.0)"}
    if not GRADIO_6:
        blocks_kwargs["theme"] = theme
        blocks_kwargs["css"] = CUSTOM_CSS
    with gr.Blocks(**blocks_kwargs) as app:
        gr.Markdown("# NoRegretsRVC · Beta version 0.0.0")
        gr.Markdown(
            "RVC Voice Model Training, Inference & Kaggle Studio (32000 · HiFi-GAN · RMVPE) · "
            "Turbo Training Ready · [GitHub README](https://github.com/HunterThatsMe/NoRegretsRVC#readme)"
        )
        with gr.Tabs():
            with gr.TabItem("Training"):
                train_tab()
            with gr.TabItem("Models"):
                download_tab()
            with gr.TabItem("Samples"):
                inference_tab()
            with gr.TabItem("Voice Blender"):
                voice_blender_tab()
            with gr.TabItem("TTS"):
                tts_tab()
            with gr.TabItem("Extra"):
                extra_tab()
            with gr.TabItem("Settings"):
                settings_tab()
            with gr.TabItem("Free Samples"):
                gr.Markdown("### Free Voice Conversion Samples (3 Male · 3 Female)")
                gr.Markdown(
                    "Six dry 32 kHz mono speaking clips bundled with NoRegretsRVC. "
                    "Any clips placed in `assets/free_samples` or `assets/audios` appear automatically below and in the **Samples** dropdown."
                )
                with gr.Row():
                    fs_repo_box = gr.Textbox(
                        label="GitHub Repo URL (to fetch the 6 WAV samples into Kaggle)",
                        placeholder="https://github.com/HunterThatsMe/NoRegretsRVC",
                        value="https://github.com/HunterThatsMe/NoRegretsRVC",
                    )
                    fs_fetch_btn = gr.Button("Fetch 6 Free Samples from GitHub", variant="primary")
                fs_status = gr.Markdown("Ready. Pick a clip below or fetch from your GitHub repo.")
                fs_catalog = [
                    ("Male Speaking Test 1", "MaleSpeakingTest-1_931b5ffd.wav", "Male · Dry 32 kHz mono speaking clip 1"),
                    ("Male Speaking Test 2", "MaleSpeakingTest-2_c2e6e7fe.wav", "Male · Dry 32 kHz mono speaking clip 2"),
                    ("Male Speaking Test 3", "MaleSpeakingTest-3_d78e8233.wav", "Male · Dry 32 kHz mono speaking clip 3"),
                    ("Female Speaking Test 1", "FemaleSpeakingTest-1_bb3d674f.wav", "Female · Dry 32 kHz mono speaking clip 1"),
                    ("Female Speaking Test 2", "FemaleSpeakingTest-2_d1823132.wav", "Female · Dry 32 kHz mono speaking clip 2"),
                    ("Female Speaking Test 3", "FemaleSpeakingTest-3_458bd4fc.wav", "Female · Dry 32 kHz mono speaking clip 3"),
                ]
                with gr.Row():
                    fs_selector = gr.Dropdown(
                        label="Select Free Sample",
                        choices=[f"{title} ({fname})" for title, fname, _ in fs_catalog],
                        value=f"{fs_catalog[0][0]} ({fs_catalog[0][1]})",
                    )
                fs_player = gr.Audio(label="Sample Preview & Download", type="filepath", interactive=False)
                fs_copy_btn = gr.Button("Copy Available Free Samples into Samples Tab")

                def _resolve_sample_path(choice_label: str):
                    for title, fname, _ in fs_catalog:
                        if f"{title} ({fname})" == choice_label:
                            for d in ("assets/free_samples", "assets/audios"):
                                p = os.path.join(now_dir, d, fname)
                                if os.path.isfile(p):
                                    return p, f"Loaded **{title}** from `{d}/{fname}`."
                            return None, f"**{title}** (`{fname}`) is not in `assets/free_samples` yet. Paste your GitHub repo URL above and click Fetch."
                    return None, "Select a sample."

                def _fetch_from_github(repo_url_val: str):
                    import urllib.request
                    raw_repo = (repo_url_val or "").strip().rstrip("/")
                    if raw_repo.endswith(".git"):
                        raw_repo = raw_repo[:-4]
                    if "github.com/" not in raw_repo:
                        return "Enter a valid GitHub repo URL (https://github.com/user/repo).", None
                    slug = raw_repo.split("github.com/", 1)[1].strip("/")
                    parts = [p for p in slug.split("/") if p]
                    if len(parts) < 2:
                        return "Enter a full GitHub repo URL (https://github.com/user/repo).", None
                    owner_repo = f"{parts[0]}/{parts[1]}"
                    fs_dir = os.path.join(now_dir, "assets", "free_samples")
                    au_dir = os.path.join(now_dir, "assets", "audios")
                    os.makedirs(fs_dir, exist_ok=True)
                    os.makedirs(au_dir, exist_ok=True)
                    downloaded = 0
                    for _, fname, _ in fs_catalog:
                        for branch in ("main", "master"):
                            url = f"https://raw.githubusercontent.com/{owner_repo}/{branch}/public/assets/aistudio/{fname}"
                            try:
                                dst_fs = os.path.join(fs_dir, fname)
                                urllib.request.urlretrieve(url, dst_fs)
                                shutil.copy2(dst_fs, os.path.join(au_dir, fname))
                                downloaded += 1
                                break
                            except Exception:
                                continue
                    first_path = os.path.join(fs_dir, fs_catalog[0][1])
                    if downloaded == 0:
                        return f"Could not find `public/assets/aistudio/*.wav` on `{owner_repo}` (main/master).", None
                    return f"Fetched {downloaded}/6 free samples into `assets/free_samples` and `assets/audios`. Click Refresh in Samples if needed.", first_path if os.path.isfile(first_path) else None

                def _copy_to_audios():
                    fs_dir = os.path.join(now_dir, "assets", "free_samples")
                    au_dir = os.path.join(now_dir, "assets", "audios")
                    os.makedirs(au_dir, exist_ok=True)
                    copied = 0
                    for _, fname, _ in fs_catalog:
                        src = os.path.join(fs_dir, fname)
                        if os.path.isfile(src):
                            shutil.copy2(src, os.path.join(au_dir, fname))
                            copied += 1
                    return f"Copied {copied} free samples to `assets/audios`."

                fs_selector.change(_resolve_sample_path, inputs=[fs_selector], outputs=[fs_player, fs_status])
                fs_fetch_btn.click(_fetch_from_github, inputs=[fs_repo_box], outputs=[fs_status, fs_player])
                fs_copy_btn.click(_copy_to_audios, outputs=[fs_status])
    return app


def main():
    parser = argparse.ArgumentParser(description="NoRegretsRVC WebUI")
    parser.add_argument("--server-name", "--host", dest="server_name", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6969)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--client", action="store_true")
    args, _ = parser.parse_known_args()

    auth = None
    username = os.environ.get("APPLIO_AUTH_USERNAME", "").strip()
    password = os.environ.get("APPLIO_AUTH_PASSWORD", "").strip()
    if username and password:
        auth = (username, password)

    theme = _load_theme()
    app = build_ui(theme)
    launch_kwargs = {
        "server_name": args.server_name,
        "server_port": args.port,
        "share": args.share,
        "auth": auth,
        "inbrowser": args.client,
        "show_error": True,
        "allowed_paths": ["logs"],
    }
    if GRADIO_6:
        launch_kwargs["theme"] = theme
        launch_kwargs["css"] = CUSTOM_CSS
    app.queue(default_concurrency_limit=40).launch(**launch_kwargs)


if __name__ == "__main__":
    main()
