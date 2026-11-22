import { ACCENT_HEX, BRAND_NAME, REPO_README_URL, REPO_URL } from '../data/defaults.ts';
import type {
  ApplioTabId,
  DownloadModelItem,
  ForkConfig,
  VoiceModelProject,
} from '../types/studio.ts';
import {
  buildTrainPipelineCommands,
  inferMatchingDUrl,
  sanitizeModelName,
} from './rvcTuning.ts';

export interface GeneratedBundleFile {
  id: string;
  filename: string;
  zipPath: string;
  language: 'json' | 'python' | 'bash';
  description: string;
  content: string;
}

export function generateThemePython(): string {
  return `[theme]
base = "dark"
primaryColor = "${ACCENT_HEX}"
backgroundColor = "#0D0405"
secondaryBackgroundColor = "#160709"
textColor = "#FDF7F7"
font = "sans serif"

[server]
headless = true
enableCORS = false
enableXsrfProtection = false
maxUploadSize = 2048

[browser]
gatherUsageStats = false
`;
}

export function generateAppPython(
  config: ForkConfig,
  project: VoiceModelProject
): string {
  const safeDefaultName = (project.name || '').replace(/["\\]/g, '');
  const defaultSr = project.samplingRate || '32000';
  const defaultVocoder = project.vocoder || 'HiFi-GAN';
  const defaultBatch = project.batchSize || 2;
  const defaultSaveEvery = project.saveEveryEpoch || 5;
  const defaultTotalEpoch = project.totalEpoch || 200;
  const defaultRepo = (config.customRepoUrl.trim() || REPO_URL).replace(/["\\]/g, '');

  const enabledNav: string[] = [];
  if (config.activeTabs.train) enabledNav.push('Training');
  if (config.activeTabs.models) enabledNav.push('Models');
  if (config.activeTabs.samples) enabledNav.push('Samples');
  enabledNav.push('Free Samples');
  enabledNav.push('Diagnostics');
  if (config.activeTabs.voice_blender) enabledNav.push('Voice Blender');
  if (config.activeTabs.tts) enabledNav.push('TTS');
  if (config.activeTabs.extra) enabledNav.push('Extra');
  if (config.activeTabs.settings) enabledNav.push('Settings');
  if (enabledNav.length === 0) enabledNav.push('Training');

  return `from __future__ import annotations

import hmac
import html
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import streamlit as st

now_dir = os.getcwd()
if now_dir not in sys.path:
    sys.path.insert(0, now_dir)

CONFIG_PATH = os.path.join(now_dir, "assets", "config.json")
os.environ.setdefault("APPLIO_CONFIG_FILE", CONFIG_PATH)
CONFIG_TEMPLATE_PATH = os.path.join(now_dir, "assets", "config_template.json")
ACTIVE_TRAIN_META = os.path.join(now_dir, "assets", "nr_active_train.json")
LOG_CACHE_DIR = os.path.join(now_dir, "assets", "nr_logs")
os.makedirs(LOG_CACHE_DIR, exist_ok=True)

if not os.path.exists(CONFIG_PATH) and os.path.exists(CONFIG_TEMPLATE_PATH):
    shutil.copy(CONFIG_TEMPLATE_PATH, CONFIG_PATH)

def _load_config() -> dict[str, Any]:
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}

def _save_config(data: dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=4, ensure_ascii=False)
    except Exception:
        pass

def _ensure_defaults_on_boot() -> None:
    cfg = _load_config()
    changed = False
    if cfg.get("precision") not in ("fp32", "bf16") and cfg.get("nr_precision_initialized") is not True:
        cfg["precision"] = "fp32"
        cfg["nr_precision_initialized"] = True
        changed = True
    if cfg.get("discord_presence") is True and cfg.get("nr_discord_initialized") is not True:
        cfg["discord_presence"] = False
        cfg["nr_discord_initialized"] = True
        changed = True
    if changed:
        _save_config(cfg)

_ensure_defaults_on_boot()

try:
    from rvc.lib.platform import platform_config
    platform_config()
except Exception:
    pass

try:
    import rvc.lib.zluda  # noqa: F401
except Exception:
    pass

st.set_page_config(
    page_title="${BRAND_NAME} (Beta 0.0.0)",
    page_icon="NR",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --app-bg: #0D0405;
  --app-surface: #160709;
  --app-panel: #100506;
  --app-border: #47161D;
  --app-text: #FDF7F7;
  --app-muted: #AD8288;
  --app-sub: #DFBFC3;
  --app-accent: ${ACCENT_HEX};
}

html, body, [data-testid="stAppViewContainer"], .stApp {
  background-color: #0D0405 !important;
  color: #FDF7F7 !important;
  font-family: "Plus Jakarta Sans", ui-sans-serif, system-ui, sans-serif !important;
  -webkit-font-smoothing: antialiased;
}

header[data-testid="stHeader"],
#MainMenu,
footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
.stDeployButton {
  display: none !important;
  visibility: hidden !important;
  height: 0 !important;
}

* {
  scrollbar-width: thin;
  scrollbar-color: ${ACCENT_HEX} #100506;
}
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: #100506; }
::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, ${ACCENT_HEX} 0%, #b91c1c 100%);
  border-radius: 999px;
  border: 2px solid #100506;
}
::-webkit-scrollbar-thumb:hover { background: #FF2B35; }

.main .block-container,
[data-testid="stMainBlockContainer"] {
  max-width: 1280px !important;
  width: 100% !important;
  margin: 0 auto !important;
  padding: 18px 24px 48px !important;
}

/* Header Card matching AI Studio Preview */
.nr-header-card {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  background: #160709;
  border: 1px solid #47161D;
  border-radius: 14px;
  padding: 14px 20px;
  margin-bottom: 12px;
  width: 100%;
  box-sizing: border-box;
}
.nr-brand-left { display: flex; align-items: center; gap: 14px; }
.nr-logo-box {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 42px;
  height: 42px;
  border-radius: 10px;
  background: ${ACCENT_HEX};
  color: #FFFFFF;
  font-weight: 800;
  font-size: 16px;
  letter-spacing: -0.02em;
  flex-shrink: 0;
}
.nr-title-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.nr-title-row h1 {
  margin: 0 !important;
  padding: 0 !important;
  font-size: 22px !important;
  font-weight: 700 !important;
  color: #FDF7F7 !important;
  line-height: 1.2 !important;
}
.nr-version-badge {
  display: inline-flex;
  align-items: center;
  padding: 2px 9px;
  border-radius: 6px;
  background: #200B0E;
  border: 1px solid #47161D;
  color: #DFBFC3;
  font-size: 11px;
  font-weight: 600;
}
.nr-brand-sub { margin: 3px 0 0 !important; font-size: 12px !important; color: #AD8288 !important; }
.nr-status-box {
  background: #100506;
  border: 1px solid #47161D;
  border-radius: 10px;
  padding: 8px 14px;
  text-align: right;
}
.nr-status-title { margin: 0; font-size: 13px; font-weight: 600; color: #FDF7F7; }
.nr-status-meta { margin: 2px 0 0; font-size: 11px; color: #AD8288; font-family: "JetBrains Mono", monospace; }

/* Card & Container styling */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: #160709 !important;
  border: 1px solid #47161D !important;
  border-radius: 12px !important;
  padding: 18px 20px !important;
  margin-bottom: 14px !important;
}
[data-testid="stVerticalBlockBorderWrapper"] > div {
  gap: 10px !important;
}

.nr-nav-shell [data-testid="stVerticalBlockBorderWrapper"],
div.nr-nav-box {
  background: #160709 !important;
  border: 1px solid #47161D !important;
  border-radius: 12px !important;
  padding: 6px 8px !important;
  margin-bottom: 18px !important;
}

.applio-card-head { display: flex; flex-direction: column; gap: 3px; margin: 0 0 8px; }
.applio-card-heading {
  display: flex !important;
  align-items: center !important;
  margin: 0 !important;
  padding: 0 !important;
  font-size: 17px !important;
  font-weight: 700 !important;
  line-height: 1.3 !important;
  color: #FDF7F7 !important;
}
.applio-step {
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  width: 28px !important;
  height: 28px !important;
  min-width: 28px !important;
  border-radius: 8px !important;
  background: ${ACCENT_HEX} !important;
  color: #FFFFFF !important;
  font-size: 12px !important;
  font-weight: 700 !important;
  margin-right: 12px !important;
  flex-shrink: 0 !important;
}
.applio-card-desc {
  margin: 2px 0 0 40px !important;
  color: #AD8288 !important;
  font-size: 12px !important;
  line-height: 1.45 !important;
}
.nr-field-label {
  font-size: 13px;
  font-weight: 600;
  color: #FDF7F7;
  margin: 4px 0 1px;
}
.nr-field-help {
  font-size: 12px;
  color: #AD8288;
  margin: 0 0 6px;
  line-height: 1.4;
}
.nr-page-title {
  margin: 0 !important;
  padding: 0 !important;
  font-size: 26px !important;
  font-weight: 700 !important;
  color: #FDF7F7 !important;
  letter-spacing: -0.02em;
}
.nr-page-subtitle {
  margin: 4px 0 0 !important;
  font-size: 13px !important;
  color: #DFBFC3 !important;
}
.nr-quote-box {
  background: #160709;
  border: 1px solid rgba(71, 22, 29, 0.85);
  border-radius: 8px;
  padding: 10px 12px;
  font-size: 12px;
  color: #DFBFC3;
  line-height: 1.45;
  margin: 6px 0 10px;
}
.nr-mono-badge {
  display: inline-block;
  border: 1px solid #47161D;
  background: #160709;
  border-radius: 6px;
  padding: 2px 7px;
  font-family: "JetBrains Mono", monospace;
  font-size: 10px;
  color: #DFBFC3;
}
.nr-log-box {
  background: #100506;
  border: 1px solid #47161D;
  border-radius: 8px;
  padding: 10px 12px;
  font-family: "JetBrains Mono", monospace;
  font-size: 12px;
  color: #DFBFC3;
  white-space: pre-wrap;
  max-height: 220px;
  overflow-y: auto;
  line-height: 1.45;
}

/* Inputs, Dropdowns, Sliders, Buttons, Expanders, Tabs */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input,
[data-baseweb="select"] > div,
[data-baseweb="base-input"] {
  background-color: #100506 !important;
  border: 1px solid #47161D !important;
  border-radius: 8px !important;
  color: #FDF7F7 !important;
  box-shadow: none !important;
}
.stTextInput input:focus,
.stTextArea textarea:focus,
[data-baseweb="select"] > div:focus-within {
  border-color: ${ACCENT_HEX} !important;
  box-shadow: 0 0 0 2px rgba(229, 25, 35, 0.18) !important;
}
[data-baseweb="popover"] ul,
[data-baseweb="menu"] {
  background-color: #160709 !important;
  border: 1px solid #47161D !important;
}
[data-baseweb="menu"] li:hover {
  background-color: #2D0F14 !important;
}

.stButton > button,
.stDownloadButton > button,
.stLinkButton > a,
[data-testid="stPopover"] > button {
  border-radius: 8px !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  min-height: 38px !important;
  padding: 6px 12px !important;
  white-space: nowrap !important;
  transition: all 140ms ease !important;
}
.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {
  background-color: ${ACCENT_HEX} !important;
  border: 1px solid #9F1018 !important;
  color: #FFFFFF !important;
}
.stButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover {
  background-color: #FF2B35 !important;
  border-color: #FF2B35 !important;
}
.stButton > button[kind="secondary"],
.stDownloadButton > button[kind="secondary"],
.stLinkButton > a,
[data-testid="stPopover"] > button {
  background-color: #200B0E !important;
  border: 1px solid #47161D !important;
  color: #FDF7F7 !important;
}
.stButton > button[kind="secondary"]:hover,
.stDownloadButton > button[kind="secondary"]:hover,
.stLinkButton > a:hover,
[data-testid="stPopover"] > button:hover {
  border-color: ${ACCENT_HEX} !important;
  background-color: #2D0F14 !important;
  color: #FFFFFF !important;
}

/* Expanders (Advanced Settings) */
[data-testid="stExpander"] {
  background: #100506 !important;
  border: 1px solid #47161D !important;
  border-radius: 10px !important;
  overflow: hidden !important;
}
[data-testid="stExpander"] summary {
  color: #FDF7F7 !important;
  font-weight: 600 !important;
  font-size: 13px !important;
}

/* File Uploader compact dark style */
[data-testid="stFileUploaderDropzone"] {
  background: #100506 !important;
  border: 1px dashed #47161D !important;
  border-radius: 10px !important;
  padding: 12px 14px !important;
}

/* Segmented Tabs inside Settings / Samples / Extra */
.stTabs [data-baseweb="tab-list"] {
  background: #100506 !important;
  border: 1px solid #47161D !important;
  border-radius: 9px !important;
  padding: 4px !important;
  gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 6px !important;
  color: #DFBFC3 !important;
  padding: 6px 14px !important;
  font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
  background-color: ${ACCENT_HEX} !important;
  color: #FFFFFF !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
  display: none !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --- Helpers ---
FREE_SAMPLES_CATALOG = [
    ("Male Speaking 1", "MaleSpeakingTest-1_931b5ffd.wav", "Male", "Expressive Narration", "~9.5s", "If you need an expressive voice for comedic gaming, animation, weird science explainers, or just any other casual narration, this is the right voice."),
    ("Male Speaking 2", "MaleSpeakingTest-2_c2e6e7fe.wav", "Male", "Conversational Voiceover", "~6.5s", "Need a great AI voice? Look no further than this conversational clone trained by a professional voice actor."),
    ("Male Speaking 3", "MaleSpeakingTest-3_d78e8233.wav", "Male", "Casual Dialogue", "~6.6s", "Sure, I can get you in this week. I've got Thursday at 10 or Friday at 2:30. Which one works better for you?"),
    ("Female Speaking 1", "FemaleSpeakingTest-1_bb3d674f.wav", "Female", "Calm Professional", "~7.1s", "Hi there, I'm here to help with questions, support requests, and quick answers. All delivered with a calm, friendly, professional touch."),
    ("Female Speaking 2", "FemaleSpeakingTest-2_d1823132.wav", "Female", "Clear Narration", "~7.1s", "By repeating what students say, teachers can demonstrate that they are listening. By extending what students say..."),
    ("Female Speaking 3", "FemaleSpeakingTest-3_458bd4fc.wav", "Female", "Friendly Assistant", "~7.1s", "I can help with appointments, reminders, and general questions. If you need a person, I'll make sure you reach the right team."),
]

def _step_head(num: int, title: str, desc: str) -> None:
    st.markdown(
        f'<div class="applio-card-head">'
        f'<h2 class="applio-card-heading"><span class="applio-step">{num}</span>{html.escape(title)}</h2>'
        f'<p class="applio-card-desc">{html.escape(desc)}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

def _field_label(label: str, help_text: str = "") -> None:
    h = f'<div class="nr-field-help">{html.escape(help_text)}</div>' if help_text else ""
    st.markdown(f'<div class="nr-field-label">{html.escape(label)}</div>{h}', unsafe_allow_html=True)

def _log_box(text: str) -> None:
    safe = html.escape(text or "The output information will be displayed here.")
    st.markdown(f'<div class="nr-log-box">{safe}</div>', unsafe_allow_html=True)

def _get_gpu_summary() -> tuple[str, str]:
    try:
        from rvc.configs.config import get_gpu_info, get_number_of_gpus
        return str(get_number_of_gpus()), str(get_gpu_info())
    except Exception:
        try:
            import torch
            n = torch.cuda.device_count()
            if n > 0:
                ids = "-".join(str(i) for i in range(n))
                names = ", ".join(torch.cuda.get_device_name(i) for i in range(n))
                return ids, names
        except Exception:
            pass
    return "0", "CPU"

def _list_models() -> list[str]:
    logs_dir = Path(now_dir) / "logs"
    if not logs_dir.is_dir():
        return []
    skip = {"mute", "mute_spin", "reference", "zips", "downloads"}
    return sorted(p.name for p in logs_dir.iterdir() if p.is_dir() and p.name not in skip)

def _list_datasets() -> list[str]:
    found: list[str] = []
    for base in (Path("/kaggle/input"), Path(now_dir) / "assets" / "datasets"):
        if base.is_dir():
            for p in sorted(base.rglob("*")):
                if p.is_dir() and any(f.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg", ".m4a") for f in p.iterdir() if f.is_file()):
                    found.append(str(p))
    return found

def _list_pth_files(filter_q: str = "") -> list[str]:
    logs_dir = Path(now_dir) / "logs"
    if not logs_dir.is_dir():
        return []
    items = sorted(str(p.relative_to(now_dir)) for p in logs_dir.rglob("*.pth") if not p.name.startswith(("G_", "D_", "f0")))
    if filter_q.strip():
        q = filter_q.strip().lower()
        items = [x for x in items if q in x.lower()]
    return items

def _list_index_files(filter_q: str = "") -> list[str]:
    logs_dir = Path(now_dir) / "logs"
    if not logs_dir.is_dir():
        return []
    items = sorted(str(p.relative_to(now_dir)) for p in logs_dir.rglob("*.index") if "trained" not in p.name)
    if filter_q.strip():
        q = filter_q.strip().lower()
        items = [x for x in items if q in x.lower()]
    return items

def _match_index(pth_path: str, indexes: list[str]) -> str:
    if not pth_path or not indexes:
        return ""
    parent = str(Path(pth_path).parent)
    for idx in indexes:
        if str(Path(idx).parent) == parent:
            return idx
    stem = Path(pth_path).stem.split("_")[0].lower()
    for idx in indexes:
        if stem and stem in idx.lower():
            return idx
    return indexes[0] if indexes else ""

def _list_audio_inputs() -> list[str]:
    au_dir = Path(now_dir) / "assets" / "audios"
    if not au_dir.is_dir():
        return []
    return sorted(
        str(p.relative_to(now_dir))
        for p in au_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg", ".m4a") and not p.name.startswith("output")
    )

def _list_custom_pretrains(prefix: str) -> list[str]:
    cdir = Path(now_dir) / "rvc" / "models" / "pretraineds" / "custom"
    if not cdir.is_dir():
        return []
    items = sorted(str(p.relative_to(now_dir)) for p in cdir.glob("*.pth"))
    if prefix:
        matched = [x for x in items if prefix in Path(x).name.upper()]
        if matched:
            return matched
    return items

def _load_tts_voices() -> list[str]:
    vf = Path(now_dir) / "rvc" / "lib" / "tools" / "tts_voices.json"
    if vf.is_file():
        try:
            data = json.loads(vf.read_text(encoding="utf-8"))
            names = [v.get("ShortName", "") for v in data if v.get("ShortName")]
            if names:
                return names
        except Exception:
            pass
    return ["en-US-AndrewNeural", "en-US-EmmaNeural", "en-US-BrianNeural", "en-GB-SoniaNeural"]

def _run_core_sync(fn_name: str, kwargs: dict[str, Any], log_placeholder=None) -> tuple[int, str]:
    payload = json.dumps({"fn": fn_name, "kwargs": kwargs})
    py_code = (
        "import json, os, sys; "
        "d = json.loads(sys.argv[1]); "
        "import core; "
        "res = getattr(core, d['fn'])(**d['kwargs']); "
        "print('__NR_RESULT__:' + json.dumps(res, default=str))"
    )
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", py_code, payload],
        cwd=now_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    lines: list[str] = []
    result_str = ""
    assert proc.stdout is not None
    for raw in proc.stdout:
        ln = raw.rstrip("\\r\\n")
        if ln.startswith("__NR_RESULT__:"):
            result_str = ln.split("__NR_RESULT__:", 1)[1]
            continue
        lines.append(ln)
        print(ln, flush=True)
        if log_placeholder is not None and len(lines) % 3 == 0:
            log_placeholder.markdown(f'<div class="nr-log-box">{html.escape(chr(10).join(lines[-80:]))}</div>', unsafe_allow_html=True)
    rc = proc.wait()
    joined = "\\n".join(lines[-120:])
    if result_str:
        try:
            parsed = json.loads(result_str)
            if isinstance(parsed, list) and len(parsed) >= 1:
                joined = (joined + "\\n" + str(parsed[0])).strip()
            elif parsed:
                joined = (joined + "\\n" + str(parsed)).strip()
        except Exception:
            pass
    if log_placeholder is not None:
        log_placeholder.markdown(f'<div class="nr-log-box">{html.escape(joined or "Completed.")}</div>', unsafe_allow_html=True)
    return rc, joined or ("Completed." if rc == 0 else f"Process exited with code {rc}.")

def _start_training_bg(model_name: str, kwargs: dict[str, Any]) -> str:
    model_dir = Path(now_dir) / "logs" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_dir / "nr_train.log"
    payload = json.dumps({"fn": "run_train_script", "kwargs": kwargs})
    py_code = (
        "import json, sys; "
        "d = json.loads(sys.argv[1]); "
        "import core; "
        "res = getattr(core, d['fn'])(**d['kwargs']); "
        "print(res, flush=True)"
    )
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    log_fh = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", py_code, payload],
        cwd=now_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        start_new_session=True,
    )
    with open(ACTIVE_TRAIN_META, "w", encoding="utf-8") as mf:
        json.dump({"pid": proc.pid, "model_name": model_name, "log_path": str(log_path), "started": time.time()}, mf)

    def _pump() -> None:
        try:
            assert proc.stdout is not None
            for ln in proc.stdout:
                log_fh.write(ln)
                log_fh.flush()
                print(ln, end="", flush=True)
        finally:
            proc.wait()
            log_fh.close()

    threading.Thread(target=_pump, daemon=True).start()
    return str(log_path)

def _stop_training(model_name: str) -> str:
    killed = 0
    for meta_path in (ACTIVE_TRAIN_META, os.path.join(now_dir, "logs", model_name, "config.json")):
        if os.path.isfile(meta_path):
            try:
                data = json.loads(Path(meta_path).read_text(encoding="utf-8"))
                pids = []
                if isinstance(data.get("pid"), int):
                    pids.append(data["pid"])
                pids.extend(p for p in data.get("process_pids", []) if isinstance(p, int))
                for pid in pids:
                    try:
                        os.killpg(pid, signal.SIGTERM)
                        killed += 1
                    except Exception:
                        try:
                            os.kill(pid, signal.SIGTERM)
                            killed += 1
                        except Exception:
                            pass
            except Exception:
                pass
    try:
        from tabs.settings.sections.restart import stop_train
        stop_train(model_name)
    except Exception:
        pass
    return f"Stop signal sent to training ({killed} process group(s))."

def _read_train_log(model_name: str) -> str:
    candidates = []
    if model_name.strip():
        candidates.append(Path(now_dir) / "logs" / model_name.strip() / "nr_train.log")
    if os.path.isfile(ACTIVE_TRAIN_META):
        try:
            lp = json.loads(Path(ACTIVE_TRAIN_META).read_text(encoding="utf-8")).get("log_path", "")
            if lp:
                candidates.append(Path(lp))
        except Exception:
            pass
    for p in candidates:
        if p.is_file():
            txt = p.read_text(encoding="utf-8", errors="replace")
            return "\\n".join(txt.splitlines()[-80:])
    return ""


# --- Auth Gate (matches APPLIO_AUTH_USERNAME / APPLIO_AUTH_PASSWORD) ---
_auth_user = os.environ.get("APPLIO_AUTH_USERNAME", "").strip()
_auth_pass = os.environ.get("APPLIO_AUTH_PASSWORD", "").strip()
if _auth_pass and not st.session_state.get("nr_authed", False):
    _, col_mid, _ = st.columns([1, 1.4, 1])
    with col_mid:
        with st.container(border=True):
            st.markdown(
                '<div class="nr-header-card" style="margin-bottom:14px">'
                '<div class="nr-brand-left">'
                '<div class="nr-logo-box">NR</div>'
                '<div><div class="nr-title-row"><h1>${BRAND_NAME}</h1><span class="nr-version-badge">Beta version 0.0.0</span></div>'
                '<p class="nr-brand-sub">Sign in with the credentials shown in your Kaggle Live Monitor</p></div>'
                '</div></div>',
                unsafe_allow_html=True,
            )
            u_in = st.text_input("Username", value=_auth_user or "noregrets")
            p_in = st.text_input("Password", type="password", placeholder="Paste WebUI password from Kaggle output")
            if st.button("Sign In", type="primary", use_container_width=True):
                u_ok = hmac.compare_digest(u_in.strip(), _auth_user or "noregrets")
                p_ok = hmac.compare_digest(p_in.strip(), _auth_pass)
                if u_ok and p_ok:
                    st.session_state["nr_authed"] = True
                    st.rerun()
                else:
                    st.error("Invalid WebUI username or password.")
    st.stop()


# --- Session Defaults ---
NAV_TABS = ${JSON.stringify(enabledNav)}
if "nr_active_tab" not in st.session_state or st.session_state["nr_active_tab"] not in NAV_TABS:
    st.session_state["nr_active_tab"] = NAV_TABS[0]
st.session_state.setdefault("nr_model_name", "${safeDefaultName}")
st.session_state.setdefault("nr_sr", "${defaultSr}")
st.session_state.setdefault("nr_vocoder", "${defaultVocoder}")
st.session_state.setdefault("nr_batch_size", ${defaultBatch})
st.session_state.setdefault("nr_save_every", ${defaultSaveEvery})
st.session_state.setdefault("nr_total_epoch", ${defaultTotalEpoch})
st.session_state.setdefault("nr_silent_files", 0)
st.session_state.setdefault("nr_pre_log", "The output information will be displayed here.")
st.session_state.setdefault("nr_ext_log", "The output information will be displayed here.")
st.session_state.setdefault("nr_trn_log", "The output information will be displayed here.")
st.session_state.setdefault("nr_dl_log", "The output information will be displayed here.")
st.session_state.setdefault("nr_inf_log", "The output information will be displayed here.")
st.session_state.setdefault("nr_inf_out", "")
st.session_state.setdefault("nr_tts_log", "The output information will be displayed here.")
st.session_state.setdefault("nr_tts_out", "")
st.session_state.setdefault("nr_blend_log", "")
st.session_state.setdefault("nr_blend_out", "")
st.session_state.setdefault("nr_selected_sample_path", "")
st.session_state.setdefault("nr_free_gender", "All")

gpu_ids_default, gpu_desc_default = _get_gpu_summary()
gpu_count = max(1, len([g for g in gpu_ids_default.split("-") if g.strip()]))
active_model_display = st.session_state["nr_model_name"].strip() or "Untitled Model"
eff_batch = int(st.session_state["nr_batch_size"]) * gpu_count

# --- Top Brand Header Card ---
st.markdown(
    '<div class="nr-header-card">'
    '<div class="nr-brand-left">'
    '<div class="nr-logo-box">NR</div>'
    '<div>'
    '<div class="nr-title-row">'
    '<h1>${BRAND_NAME}</h1>'
    '<span class="nr-version-badge">Beta version 0.0.0</span>'
    '</div>'
    '<p class="nr-brand-sub">RVC Voice Model Training, Inference &amp; Kaggle Studio</p>'
    '</div>'
    '</div>'
    '<div class="nr-status-box">'
    f'<p class="nr-status-title">{html.escape(active_model_display)} ({html.escape(str(st.session_state["nr_sr"]))})</p>'
    f'<p class="nr-status-meta">{html.escape(str(st.session_state["nr_vocoder"]))} · Batch {st.session_state["nr_batch_size"]} ({eff_batch} eff.) · {st.session_state["nr_total_epoch"]} ep</p>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

# --- Horizontal Pill Navigation Bar ---
with st.container(border=True):
    nav_weights = [max(7, len(lbl) + 4) for lbl in NAV_TABS]
    nav_cols = st.columns(nav_weights, gap="small")
    for idx, tab_label in enumerate(NAV_TABS):
        is_active = st.session_state["nr_active_tab"] == tab_label
        with nav_cols[idx]:
            if st.button(
                tab_label,
                key=f"nr_nav_{idx}",
                type=("primary" if is_active else "secondary"),
                use_container_width=True,
            ):
                if not is_active:
                    st.session_state["nr_active_tab"] = tab_label
                    st.rerun()

active_tab = st.session_state["nr_active_tab"]
cfg_now = _load_config()
list_filter_on = bool(cfg_now.get("filter", False))


def _page_banner(title: str, subtitle: str, show_vram_info: bool = False) -> None:
    if show_vram_info:
        c_title, c_vram, c_readme = st.columns([6, 1.8, 1.7])
    else:
        c_title, c_readme = st.columns([7.8, 1.7])
        c_vram = None
    with c_title:
        st.markdown(
            f'<h2 class="nr-page-title">{html.escape(title)}</h2>'
            f'<p class="nr-page-subtitle">{html.escape(subtitle)}</p>',
            unsafe_allow_html=True,
        )
    if c_vram is not None:
        with c_vram:
            with st.popover("Model & VRAM Info", use_container_width=True):
                sr_val = st.session_state["nr_sr"]
                bs_val = int(st.session_state["nr_batch_size"])
                voc_val = st.session_state["nr_vocoder"]
                prec_val = cfg_now.get("precision", "fp32")
                est_vram = round(4.2 + (bs_val * 0.65) + (1.2 if sr_val == "48000" else 0.4) + (2.4 if voc_val == "RefineGAN" else 0.0), 1)
                st.markdown(f"**Detected GPUs:** \`{gpu_desc_default}\`")
                st.markdown(f"**Precision:** \`{prec_val.upper()}\` · **Effective Batch:** \`{bs_val * gpu_count}\`")
                st.markdown(f"**Estimated Peak VRAM:** \`{est_vram} GB / 15.0 GB T4\`")
    with c_readme:
        st.link_button("GitHub README ↗", "${REPO_README_URL}", use_container_width=True)


# ==============================================================================
# TAB 1: TRAINING
# ==============================================================================
if active_tab == "Training":
    _page_banner(
        "Training",
        "Set up your voice dataset, extract acoustic features, and run RVC training.",
        show_vram_info=True,
    )

    # [1] Model Settings
    with st.container(border=True):
        _step_head(1, "Model Settings", "Give your voice project a name, select target sample fidelity, and assign your synthesis vocoder.")
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            _field_label("Model Name", "Identifier used for saved checkpoints, index files, and log directories.")
            existing_models = _list_models()
            model_name_val = st.text_input(
                "Model Name",
                value=st.session_state["nr_model_name"],
                placeholder="Enter model name (e.g. VocalModel-32k)",
                label_visibility="collapsed",
                key="nr_model_name_input",
            )
            if existing_models:
                picked_existing = st.selectbox(
                    "Or resume existing model",
                    options=[""] + existing_models,
                    index=0,
                    format_func=lambda x: "Pick existing log folder..." if x == "" else x,
                    label_visibility="collapsed",
                    key="nr_pick_existing_model",
                )
                if picked_existing:
                    model_name_val = picked_existing
            st.session_state["nr_model_name"] = model_name_val
        with col2:
            _field_label("Sampling Rate", "Target audio resolution in Hz for preprocessed slices and model synthesis.")
            sr_opts = ["32000", "40000", "48000"]
            st.session_state["nr_sr"] = st.selectbox(
                "Sampling Rate",
                options=sr_opts,
                index=sr_opts.index(st.session_state["nr_sr"]) if st.session_state["nr_sr"] in sr_opts else 0,
                label_visibility="collapsed",
            )

        _field_label(
            "Vocoder",
            "Waveform synthesizer architecture: HiFi-GAN works across all RVC players; MRF HiFi-GAN boosts harmonic detail for Applio clients; RefineGAN maximizes clarity with extra VRAM demand.",
        )
        voc_opts = ["HiFi-GAN", "MRF HiFi-GAN", "RefineGAN"]
        st.session_state["nr_vocoder"] = st.selectbox(
            "Vocoder",
            options=voc_opts,
            index=voc_opts.index(st.session_state["nr_vocoder"]) if st.session_state["nr_vocoder"] in voc_opts else 0,
            label_visibility="collapsed",
        )

        with st.expander("Advanced Settings", expanded=False):
            a1, a2 = st.columns(2, gap="medium")
            with a1:
                cpu_cores = st.slider("CPU Cores", min_value=1, max_value=max(4, os.cpu_count() or 4), value=min(4, os.cpu_count() or 4))
            with a2:
                gpu_ids = st.text_input("GPU Number", value=gpu_ids_default, help="Separate GPU indices with - for multi-GPU (e.g. 0-1).")
                st.caption(f"Detected Hardware: {gpu_desc_default}")

    # [2] Prepare Dataset
    with st.container(border=True):
        _step_head(2, "Prepare Dataset", "Link your Kaggle dataset folder and split clean vocal segments for training.")
        _field_label("Dataset Path", "Mounted Kaggle input path containing your dry vocal recordings.")
        found_ds = _list_datasets()
        ds_path = st.text_input(
            "Dataset Path",
            value=st.session_state.get("nr_ds_path", found_ds[0] if found_ds else ""),
            placeholder="/kaggle/input/datasets/your-username/voice-dataset",
            label_visibility="collapsed",
        )
        st.session_state["nr_ds_path"] = ds_path
        if found_ds:
            ds_pick = st.selectbox(
                "Detected Datasets",
                options=[""] + found_ds,
                format_func=lambda x: "Pick discovered Kaggle/local dataset..." if x == "" else x,
                label_visibility="collapsed",
            )
            if ds_pick:
                ds_path = ds_pick
                st.session_state["nr_ds_path"] = ds_pick

        ds_row_l, ds_row_r = st.columns([3, 1.3], gap="small")
        with ds_row_l:
            dataset_creator = st.checkbox("Dataset Creator", value=False)
        with ds_row_r:
            if st.button("Refresh models and datasets", use_container_width=True):
                st.rerun()
        if dataset_creator:
            dc_name = st.text_input("Dataset Name", value="my-dataset", placeholder="Folder name under assets/datasets/")
            uploaded_audio = st.file_uploader(
                "Upload audio clips for your dataset",
                type=["wav", "mp3", "flac", "ogg", "m4a"],
                accept_multiple_files=True,
            )
            if uploaded_audio and st.button("Save Uploaded Audio to Dataset", type="secondary"):
                target_ds = Path(now_dir) / "assets" / "datasets" / (re.sub(r"[^a-zA-Z0-9_-]+", "_", dc_name) or "my-dataset")
                target_ds.mkdir(parents=True, exist_ok=True)
                for uf in uploaded_audio:
                    (target_ds / uf.name).write_bytes(uf.getvalue())
                st.session_state["nr_ds_path"] = str(target_ds)
                st.success(f"Saved {len(uploaded_audio)} clip(s) to {target_ds}")

        with st.expander("Advanced Settings", expanded=False):
            p1, p2, p3 = st.columns(3, gap="medium")
            with p1:
                cut_preprocess = st.selectbox("Audio Cutting", ["Automatic", "Simple", "Skip"], index=0)
            with p2:
                chunk_len = st.slider("Chunk Length (s)", 0.5, 5.0, 3.0, step=0.1)
            with p3:
                overlap_len = st.slider("Overlap Length (s)", 0.0, 0.4, 0.3, step=0.05)
            pe_col1, pe_col2 = st.columns(2, gap="medium")
            with pe_col1:
                process_effects = st.checkbox("Process Audio Effects / Noise Filter", value=False)
            with pe_col2:
                norm_mode = st.selectbox("Loudness Normalization Mode", ["post", "none", "none_strict"], index=0)

        _field_label("Preprocess output")
        pre_box = st.empty()
        pre_box.markdown(f'<div class="nr-log-box">{html.escape(st.session_state["nr_pre_log"])}</div>', unsafe_allow_html=True)
        if st.button("Preprocess Dataset", type="primary", use_container_width=True, key="btn_preprocess"):
            m_clean = st.session_state["nr_model_name"].strip()
            if not m_clean:
                st.warning("Enter a Model Name in Step 1 first.")
            elif not st.session_state["nr_ds_path"].strip():
                st.warning("Enter a Dataset Path first.")
            else:
                _, out_txt = _run_core_sync(
                    "run_preprocess_script",
                    {
                        "model_name": m_clean,
                        "dataset_path": st.session_state["nr_ds_path"].strip(),
                        "sample_rate": int(st.session_state["nr_sr"]),
                        "cpu_cores": int(cpu_cores),
                        "cut_preprocess": cut_preprocess,
                        "process_effects": bool(process_effects),
                        "chunk_len": float(chunk_len),
                        "overlap_len": float(overlap_len),
                        "normalization_mode": norm_mode if process_effects else "none",
                    },
                    pre_box,
                )
                st.session_state["nr_pre_log"] = out_txt

    # [3] Extract Features
    with st.container(border=True):
        _step_head(3, "Extract Features", "Compute pitch tracks and speaker embeddings from your sliced vocals.")
        e1, e2 = st.columns(2, gap="medium")
        with e1:
            _field_label("Pitch extraction algorithm", "F0 tracker used to capture vocal melody. RMVPE offers the sharpest accuracy across speech and singing.")
            f0_method = st.selectbox("F0 Method", ["rmvpe", "fcpe", "swift-f0", "crepe"], index=0, label_visibility="collapsed")
        with e2:
            _field_label("Embedder Model", "Speech encoder that extracts phonetic timbre representations from audio.")
            embedder_model = st.selectbox("Embedder Model", ["contentvec", "spin", "chinese-hubert-base"], index=0, label_visibility="collapsed")

        _field_label(
            "Silent training files",
            "Injects blank silent samples to help prevent breath artifacts during pauses. Leave at 0 if your dataset already includes natural quiet gaps.",
        )
        include_mutes = st.slider("Silent training files", 0, 10, int(st.session_state["nr_silent_files"]), step=1, label_visibility="collapsed")
        st.session_state["nr_silent_files"] = include_mutes

        _field_label("Feature extraction output")
        ext_box = st.empty()
        ext_box.markdown(f'<div class="nr-log-box">{html.escape(st.session_state["nr_ext_log"])}</div>', unsafe_allow_html=True)
        if st.button("Extract Features", type="primary", use_container_width=True, key="btn_extract"):
            m_clean = st.session_state["nr_model_name"].strip()
            if not m_clean:
                st.warning("Enter a Model Name in Step 1 first.")
            else:
                _, out_txt = _run_core_sync(
                    "run_extract_script",
                    {
                        "model_name": m_clean,
                        "f0_method": f0_method,
                        "cpu_cores": int(cpu_cores),
                        "gpu": gpu_ids.strip() or "0",
                        "sample_rate": int(st.session_state["nr_sr"]),
                        "embedder_model": embedder_model,
                        "embedder_model_custom": None,
                        "include_mutes": int(include_mutes),
                    },
                    ext_box,
                )
                st.session_state["nr_ext_log"] = out_txt

    # [4] Train Model
    with st.container(border=True):
        _step_head(4, "Train Model", "Dial in batch size, checkpoint frequency, and total training epochs before launching.")
        t1, t2, t3 = st.columns(3, gap="medium")
        with t1:
            _field_label("Batch Size", "Per-GPU slice batch size. Default is 2 (effective batch 4 across 2x T4).")
            st.session_state["nr_batch_size"] = st.slider("Batch Size", 1, 64, int(st.session_state["nr_batch_size"]), step=1, label_visibility="collapsed")
        with t2:
            _field_label("Save Every Epoch", "Interval of completed epochs between saved checkpoint snapshots.")
            st.session_state["nr_save_every"] = st.slider("Save Every Epoch", 1, 100, int(st.session_state["nr_save_every"]), step=1, label_visibility="collapsed")
        with t3:
            _field_label("Total Epoch", "Target number of full dataset passes before training stops automatically.")
            st.session_state["nr_total_epoch"] = st.slider("Total Epoch", 1, 10000, int(st.session_state["nr_total_epoch"]), step=1, label_visibility="collapsed")

        with st.expander("Advanced Settings", expanded=False):
            ac1, ac2 = st.columns(2, gap="medium")
            with ac1:
                save_only_latest = st.checkbox("Save Only Latest", value=True)
                save_every_weights = st.checkbox("Save Every Weights", value=True)
                pretrained = st.checkbox("Pretrained", value=True)
                fresh_cleanup = st.checkbox("Fresh Training (Cleanup)", value=False)
            with ac2:
                cache_gpu = st.checkbox("Cache Dataset in GPU", value=False)
                checkpointing = st.checkbox("Gradient Checkpointing", value=False)
                shutdown_check = st.checkbox("Shutdown After Finish", value=False)
                index_algo = st.selectbox("Index Algorithm", ["Auto", "Faiss", "KMeans"], index=0)

            custom_pretrained = st.checkbox("Custom Pretrained", value=False)
            g_pre_path = ""
            d_pre_path = ""
            if custom_pretrained:
                g_choices = _list_custom_pretrains("G")
                d_choices = _list_custom_pretrains("D")
                gp_col, dp_col = st.columns(2, gap="medium")
                with gp_col:
                    g_pre_path = st.text_input("G Pretrained Path (.pth)", value=g_choices[0] if g_choices else "", placeholder="rvc/models/pretraineds/custom/G.pth")
                with dp_col:
                    d_pre_path = st.text_input("D Pretrained Path (.pth)", value=d_choices[0] if d_choices else "", placeholder="rvc/models/pretraineds/custom/D.pth")

        terms_ok = st.checkbox(
            "I agree to the terms of use — Confirm that your training audio follows the usage guidelines for NoRegretsRVC.",
            value=True,
            key="nr_train_terms",
        )

        _field_label("Training output")
        live_tail = _read_train_log(st.session_state["nr_model_name"])
        if live_tail:
            st.session_state["nr_trn_log"] = live_tail
        _log_box(st.session_state["nr_trn_log"])

        b1, b2, b3, b4 = st.columns([2, 1.3, 1.5, 1.2], gap="small")
        with b1:
            if st.button("Start Training", type="primary", use_container_width=True):
                m_clean = st.session_state["nr_model_name"].strip()
                if not terms_ok:
                    st.warning("Confirm the terms of use checkbox first.")
                elif not m_clean:
                    st.warning("Enter a Model Name in Step 1 first.")
                else:
                    lp = _start_training_bg(
                        m_clean,
                        {
                            "model_name": m_clean,
                            "save_every_epoch": int(st.session_state["nr_save_every"]),
                            "save_only_latest": bool(save_only_latest),
                            "save_every_weights": bool(save_every_weights),
                            "total_epoch": int(st.session_state["nr_total_epoch"]),
                            "sample_rate": int(st.session_state["nr_sr"]),
                            "batch_size": int(st.session_state["nr_batch_size"]),
                            "gpu": gpu_ids.strip() or "0",
                            "pretrained": bool(pretrained),
                            "cleanup": bool(fresh_cleanup),
                            "index_algorithm": index_algo,
                            "cache_data_in_gpu": bool(cache_gpu),
                            "custom_pretrained": bool(custom_pretrained),
                            "g_pretrained_path": g_pre_path.strip() or None,
                            "d_pretrained_path": d_pre_path.strip() or None,
                            "vocoder": st.session_state["nr_vocoder"],
                            "checkpointing": bool(checkpointing),
                            "shutdown_check": bool(shutdown_check),
                        },
                    )
                    st.session_state["nr_trn_log"] = f"Started training '{m_clean}' -> streaming log to {lp} and Kaggle Live Monitor."
                    st.rerun()
        with b2:
            if st.button("Stop Training", type="secondary", use_container_width=True):
                st.session_state["nr_trn_log"] = _stop_training(st.session_state["nr_model_name"].strip())
                st.rerun()
        with b3:
            if st.button("Generate Index", type="secondary", use_container_width=True):
                m_clean = st.session_state["nr_model_name"].strip()
                if not m_clean:
                    st.warning("Enter a Model Name in Step 1 first.")
                else:
                    _, out_txt = _run_core_sync("run_index_script", {"model_name": m_clean, "index_algorithm": index_algo})
                    st.session_state["nr_trn_log"] = out_txt
                    st.rerun()
        with b4:
            if st.button("Refresh Log", type="secondary", use_container_width=True):
                st.rerun()

    # [5] Export Model
    with st.container(border=True):
        _step_head(5, "Export Model", "Select trained .pth and .index files to download from Kaggle.")
        pths = _list_pth_files()
        idxs = _list_index_files()
        ex1, ex2 = st.columns(2, gap="medium")
        with ex1:
            sel_pth = st.selectbox("Pth file", options=[""] + pths, format_func=lambda x: "Select exported .pth..." if x == "" else x)
            if sel_pth and os.path.isfile(os.path.join(now_dir, sel_pth)):
                st.download_button(
                    f"Download {Path(sel_pth).name}",
                    data=Path(os.path.join(now_dir, sel_pth)).read_bytes(),
                    file_name=Path(sel_pth).name,
                    mime="application/octet-stream",
                    type="primary",
                    use_container_width=True,
                )
        with ex2:
            sel_idx = st.selectbox("Index file", options=[""] + idxs, format_func=lambda x: "Select exported .index..." if x == "" else x)
            if sel_idx and os.path.isfile(os.path.join(now_dir, sel_idx)):
                st.download_button(
                    f"Download {Path(sel_idx).name}",
                    data=Path(os.path.join(now_dir, sel_idx)).read_bytes(),
                    file_name=Path(sel_idx).name,
                    mime="application/octet-stream",
                    type="primary",
                    use_container_width=True,
                )
        if st.button("Refresh trained models and indexes", use_container_width=True):
            st.rerun()


# ==============================================================================
# TAB 2: MODELS
# ==============================================================================
elif active_tab == "Models":
    _page_banner("Models", "Fetch remote voice archives, drop local weights, and manage pretrained bases for Kaggle runs.")

    with st.container(border=True):
        _step_head(1, "Download Model", "Paste a model link to download it into your library.")
        dl_link = st.text_input("Model Link", placeholder="Introduce the model link (HuggingFace / Drive / Pixeldrain / MediaFire)")
        dl_box = st.empty()
        dl_box.markdown(f'<div class="nr-log-box">{html.escape(st.session_state["nr_dl_log"])}</div>', unsafe_allow_html=True)
        if st.button("Download Model", type="primary", use_container_width=True):
            if not dl_link.strip():
                st.warning("Paste a model link first.")
            else:
                _, res = _run_core_sync("run_download_script", {"model_link": dl_link.strip()}, dl_box)
                st.session_state["nr_dl_log"] = res

    with st.container(border=True):
        _step_head(2, "Drop Files", "Drop .pth, .index, or .zip model archives to import them into logs/downloads.")
        dropped = st.file_uploader("Drop your .pth, .index, or .zip files", type=["pth", "index", "zip"], accept_multiple_files=True)
        if dropped and st.button("Save Dropped Model Files", type="primary"):
            dl_dir = Path(now_dir) / "logs" / "downloads"
            dl_dir.mkdir(parents=True, exist_ok=True)
            saved_names: list[str] = []
            for f in dropped:
                out_p = dl_dir / f.name
                out_p.write_bytes(f.getvalue())
                if f.name.lower().endswith(".zip"):
                    try:
                        with zipfile.ZipFile(out_p, "r") as zf:
                            for m in zf.namelist():
                                if not m.endswith("/") and Path(m).name and m.lower().endswith((".pth", ".index")):
                                    dest = dl_dir / Path(m).name
                                    dest.write_bytes(zf.read(m))
                                    saved_names.append(dest.name)
                    except Exception:
                        pass
                else:
                    saved_names.append(f.name)
            st.success(f"Imported {', '.join(saved_names) or 'files'} into logs/downloads/")

    with st.container(border=True):
        _step_head(3, "Download Pretrained Models", "Install custom G/D pretrains or Hugging Face pretrain repositories into rvc/models/pretraineds/custom.")
        use_custom_pre = st.checkbox("Download from Custom URL / Hugging Face Repo / ZIP", value=True)
        if use_custom_pre:
            cu1, cu2 = st.columns(2, gap="medium")
            with cu1:
                g_url = st.text_input("Repository, pretrained ZIP or G .pth link", placeholder="https://huggingface.co/author/pretrained")
            with cu2:
                d_url = st.text_input("Matching D .pth link (optional)", placeholder="Leave empty for automatic discovery")
            if st.button("Download Pretrained Pair", type="primary", use_container_width=True):
                cdir = Path(now_dir) / "rvc" / "models" / "pretraineds" / "custom"
                cdir.mkdir(parents=True, exist_ok=True)
                downloaded: list[str] = []
                urls = [u.strip() for u in (g_url, d_url) if u.strip()]
                if g_url.strip() and not d_url.strip() and "G_" in g_url:
                    urls.append(g_url.strip().replace("G_", "D_"))
                for u in urls:
                    try:
                        parsed = urllib.parse.urlsplit(u)
                        if "huggingface.co/" in u.lower() and "/resolve/" not in u and not parsed.path.lower().endswith((".pth", ".zip")):
                            parts = [p for p in parsed.path.strip("/").split("/") if p]
                            if len(parts) >= 2:
                                owner_repo = f"{parts[0]}/{parts[1]}"
                                tree = json.loads(urllib.request.urlopen(f"https://huggingface.co/api/models/{owner_repo}/tree/main", timeout=20).read().decode("utf-8"))
                                for node in tree:
                                    fp = node.get("path", "")
                                    if fp.lower().endswith(".pth"):
                                        fn = Path(fp).name
                                        urllib.request.urlretrieve(f"https://huggingface.co/{owner_repo}/resolve/main/{urllib.parse.quote(fp)}?download=true", cdir / fn)
                                        downloaded.append(fn)
                        elif parsed.path.lower().endswith(".zip"):
                            ztmp = cdir / "tmp_pretrains.zip"
                            urllib.request.urlretrieve(u, ztmp)
                            with zipfile.ZipFile(ztmp, "r") as zf:
                                for mem in zf.namelist():
                                    if not mem.endswith("/") and Path(mem).name and mem.lower().endswith(".pth"):
                                        outp = cdir / Path(mem).name
                                        outp.write_bytes(zf.read(mem))
                                        downloaded.append(outp.name)
                            ztmp.unlink(missing_ok=True)
                        else:
                            fn = urllib.parse.unquote(parsed.path.split("/")[-1]) or "custom_pretrained.pth"
                            urllib.request.urlretrieve(u, cdir / fn)
                            downloaded.append(fn)
                    except Exception as exc:
                        st.error(f"Failed {u}: {exc}")
                if downloaded:
                    st.success(f"Installed into rvc/models/pretraineds/custom/: {', '.join(downloaded)}")
        else:
            if st.button("Download Default Applio Prerequisites", type="primary", use_container_width=True):
                _, msg = _run_core_sync("run_prerequisites_script", {"pretraineds_hifigan": True, "models": True, "exe": True})
                st.info(msg)


# ==============================================================================
# TAB 3: SAMPLES (INFERENCE)
# ==============================================================================
elif active_tab == "Samples":
    _page_banner("Samples", "Run single or batch RVC voice conversion inference with full pitch, index, and audio controls.")

    filter_q = ""
    if list_filter_on:
        filter_q = st.text_input("Filter models & indexes", placeholder="Type to filter...")

    pth_choices = _list_pth_files(filter_q)
    idx_choices = _list_index_files(filter_q)

    with st.container(border=True):
        _step_head(1, "Voice Model", "Select the .pth voice model and .index feature file for conversion.")
        m1, m2 = st.columns(2, gap="medium")
        with m1:
            _field_label("Voice Model", "Select the trained .pth checkpoint.")
            sel_voice = st.selectbox("Voice Model", options=[""] + pth_choices, format_func=lambda x: "Select voice model..." if x == "" else x, label_visibility="collapsed")
        with m2:
            _field_label("Index File", "Automatically matched when available.")
            auto_idx = _match_index(sel_voice, idx_choices)
            idx_default = (idx_choices.index(auto_idx) + 1) if auto_idx in idx_choices else 0
            sel_index = st.selectbox("Index File", options=[""] + idx_choices, index=idx_default, format_func=lambda x: "None / Optional..." if x == "" else x, label_visibility="collapsed")
        if st.button("Refresh models and indexes", use_container_width=True):
            st.rerun()

    inf_mode = st.tabs(["Single", "Batch"])
    with inf_mode[0]:
        with st.container(border=True):
            _step_head(2, "Audio Input", "Upload an audio recording or pick an existing/free sample clip from assets/audios.")
            au_choices = _list_audio_inputs()
            pre_selected = st.session_state.get("nr_selected_sample_path", "")
            au_idx = (au_choices.index(pre_selected) + 1) if pre_selected in au_choices else (1 if au_choices else 0)
            chosen_au = st.selectbox("Audio path", options=[""] + au_choices, index=au_idx, format_func=lambda x: "Pick audio file in assets/audios..." if x == "" else x)
            up_au = st.file_uploader("Or upload audio clip", type=["wav", "mp3", "flac", "ogg", "m4a"], key="inf_up_au")
            if up_au is not None:
                au_dir = Path(now_dir) / "assets" / "audios"
                au_dir.mkdir(parents=True, exist_ok=True)
                dest_au = au_dir / up_au.name
                dest_au.write_bytes(up_au.getvalue())
                chosen_au = str(dest_au.relative_to(now_dir))
                st.session_state["nr_selected_sample_path"] = chosen_au
            if chosen_au and os.path.isfile(os.path.join(now_dir, chosen_au)):
                st.audio(os.path.join(now_dir, chosen_au))

        with st.container(border=True):
            _step_head(3, "Conversion Parameters", "Tune pitch (-32..+32), index influence, volume envelope, and consonant protection.")
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                pitch = st.slider("Pitch", -32, 32, 0, step=1)
                vol_env = st.slider("Volume Envelope", 0.0, 1.0, 1.0, step=0.05)
            with c2:
                idx_rate = st.slider("Search Feature Ratio (Index Rate)", 0.0, 1.0, 0.75, step=0.05)
                protect = st.slider("Protect Voiceless Consonants", 0.0, 0.5, 0.5, step=0.05)

            with st.expander("Models, Pitch Tuning & Output Settings", expanded=False):
                pa1, pa2, pa3 = st.columns(3, gap="medium")
                with pa1:
                    f0_inf = st.selectbox("Pitch Extractor", ["rmvpe", "fcpe", "swift-f0", "crepe"], index=0)
                    sid_inf = st.number_input("Speaker ID", min_value=0, max_value=200, value=0, step=1)
                with pa2:
                    emb_inf = st.selectbox("Embedder Model", ["contentvec", "spin", "chinese-hubert-base"], index=0)
                    exp_fmt = st.selectbox("Export Format", ["WAV", "MP3", "FLAC", "OGG"], index=0)
                with pa3:
                    autotune = st.checkbox("F0 Autotune", value=False)
                    proposed_pitch = st.checkbox("Proposed Pitch", value=False)
                    clean_mode = st.selectbox("Audio Cleanup", ["off", "mild", "balanced", "aggressive"], index=0)
                    out_inf_path = st.text_input("Output Path", value=os.path.join(now_dir, "assets", "audios", "output.wav"))

            terms_inf = st.checkbox(
                "I agree to the terms of use — Confirm that your conversion audio follows the usage guidelines for NoRegretsRVC.",
                value=True,
                key="nr_inf_terms",
            )
            inf_box = st.empty()
            inf_box.markdown(f'<div class="nr-log-box">{html.escape(st.session_state["nr_inf_log"])}</div>', unsafe_allow_html=True)
            if st.button("Convert", type="primary", use_container_width=True, key="btn_inf_single"):
                if not terms_inf:
                    st.warning("Confirm the terms of use checkbox first.")
                elif not sel_voice or not chosen_au:
                    st.warning("Select both a Voice Model and an input audio clip.")
                else:
                    rc, msg = _run_core_sync(
                        "run_infer_script",
                        {
                            "pitch": int(pitch),
                            "index_rate": float(idx_rate),
                            "volume_envelope": float(vol_env),
                            "protect": float(protect),
                            "f0_method": f0_inf,
                            "input_path": os.path.join(now_dir, chosen_au),
                            "output_path": out_inf_path.strip(),
                            "pth_path": os.path.join(now_dir, sel_voice),
                            "index_path": os.path.join(now_dir, sel_index) if sel_index else "",
                            "f0_autotune": bool(autotune),
                            "f0_autotune_strength": 1.0,
                            "proposed_pitch": bool(proposed_pitch),
                            "proposed_pitch_threshold": 155.0,
                            "export_format": exp_fmt,
                            "embedder_model": emb_inf,
                            "sid": int(sid_inf),
                            "cleanup_mode": clean_mode,
                            "cleanup_strength": 12,
                        },
                        inf_box,
                    )
                    st.session_state["nr_inf_log"] = msg
                    expected = str(Path(out_inf_path.strip()).with_suffix(f".{exp_fmt.lower()}"))
                    if rc == 0 and os.path.isfile(expected):
                        st.session_state["nr_inf_out"] = expected
            if st.session_state["nr_inf_out"] and os.path.isfile(st.session_state["nr_inf_out"]):
                st.audio(st.session_state["nr_inf_out"])
                out_p = Path(st.session_state["nr_inf_out"])
                st.download_button(f"Download {out_p.name}", data=out_p.read_bytes(), file_name=out_p.name, type="secondary")

    with inf_mode[1]:
        with st.container(border=True):
            _step_head(2, "Batch Folder Conversion", "Convert every audio file inside a folder at once.")
            b_in = st.text_input("Input Folder", value=os.path.join(now_dir, "assets", "audios"))
            b_out = st.text_input("Output Folder", value=os.path.join(now_dir, "assets", "audios", "batch_out"))
            b_pitch = st.slider("Batch Pitch", -32, 32, 0, step=1)
            b_terms = st.checkbox("I agree to the terms of use — Confirm that your conversion audio follows the usage guidelines for NoRegretsRVC.", value=True, key="nr_inf_batch_terms")
            if st.button("Convert Batch", type="primary", use_container_width=True):
                if not b_terms or not sel_voice:
                    st.warning("Select a Voice Model and confirm terms first.")
                else:
                    os.makedirs(b_out.strip(), exist_ok=True)
                    _, msg = _run_core_sync(
                        "run_batch_infer_script",
                        {
                            "pitch": int(b_pitch),
                            "index_rate": 0.75,
                            "volume_envelope": 1.0,
                            "protect": 0.5,
                            "f0_method": "rmvpe",
                            "input_folder": b_in.strip(),
                            "output_folder": b_out.strip(),
                            "pth_path": os.path.join(now_dir, sel_voice),
                            "index_path": os.path.join(now_dir, sel_index) if sel_index else "",
                            "f0_autotune": False,
                            "f0_autotune_strength": 1.0,
                            "proposed_pitch": False,
                            "proposed_pitch_threshold": 155.0,
                            "export_format": "WAV",
                            "embedder_model": "contentvec",
                        },
                    )
                    st.info(msg)


# ==============================================================================
# TAB 4: FREE SAMPLES
# ==============================================================================
elif active_tab == "Free Samples":
    _page_banner("Free Samples", "Six dry 32 kHz mono speaking clips (3 male, 3 female) ready to preview, download, or use in Samples.")

    with st.container(border=True):
        fc1, fc2, fc3 = st.columns([3, 1.4, 1.5], gap="small")
        with fc1:
            fs_repo = st.text_input("GitHub Repo URL", value="${defaultRepo}", label_visibility="collapsed")
        with fc2:
            if st.button("Fetch 6 from GitHub", type="primary", use_container_width=True):
                fs_dir = Path(now_dir) / "assets" / "free_samples"
                au_dir = Path(now_dir) / "assets" / "audios"
                fs_dir.mkdir(parents=True, exist_ok=True)
                au_dir.mkdir(parents=True, exist_ok=True)
                slug = fs_repo.strip().rstrip("/").removesuffix(".git").split("github.com/")[-1]
                parts = [p for p in slug.split("/") if p]
                ok_cnt = 0
                if len(parts) >= 2:
                    owner_repo = f"{parts[0]}/{parts[1]}"
                    for _, fname, *_ in FREE_SAMPLES_CATALOG:
                        for br in ("main", "master"):
                            try:
                                urllib.request.urlretrieve(f"https://raw.githubusercontent.com/{owner_repo}/{br}/public/assets/aistudio/{fname}", fs_dir / fname)
                                shutil.copy2(fs_dir / fname, au_dir / fname)
                                ok_cnt += 1
                                break
                            except Exception:
                                continue
                st.success(f"Fetched {ok_cnt}/6 free samples into assets/free_samples/ & assets/audios/.")
        with fc3:
            st.session_state["nr_free_gender"] = st.selectbox("Filter", ["All", "Male", "Female"], label_visibility="collapsed")

    for group_name, gender_key in [("Male Speaking Samples (3)", "Male"), ("Female Speaking Samples (3)", "Female")]:
        if st.session_state["nr_free_gender"] not in ("All", gender_key):
            continue
        group_items = [x for x in FREE_SAMPLES_CATALOG if x[2] == gender_key]
        with st.container(border=True):
            st.markdown(f"**{group_name}**")
            cols = st.columns(3, gap="medium")
            for idx, (title, fname, _, style, dur, quote) in enumerate(group_items):
                with cols[idx]:
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;align-items:center">'
                        f'<div><strong style="font-size:14px;color:#FDF7F7">{html.escape(title)}</strong><br/>'
                        f'<span style="font-size:11px;color:#AD8288">{html.escape(style)} · {html.escape(dur)}</span></div>'
                        f'<span class="nr-mono-badge">Mono 32k WAV</span></div>'
                        f'<div class="nr-quote-box">&ldquo;{html.escape(quote)}&rdquo;</div>',
                        unsafe_allow_html=True,
                    )
                    wav_path = None
                    for d in ("assets/free_samples", "assets/audios"):
                        cand = Path(now_dir) / d / fname
                        if cand.is_file():
                            wav_path = cand
                            break
                    if wav_path is not None:
                        st.audio(str(wav_path))
                        ub1, ub2 = st.columns(2, gap="small")
                        with ub1:
                            if st.button("Use in Samples", key=f"use_fs_{fname}", type="primary", use_container_width=True):
                                au_dst = Path(now_dir) / "assets" / "audios" / fname
                                au_dst.parent.mkdir(parents=True, exist_ok=True)
                                if wav_path != au_dst:
                                    shutil.copy2(wav_path, au_dst)
                                st.session_state["nr_selected_sample_path"] = f"assets/audios/{fname}"
                                st.session_state["nr_active_tab"] = "Samples"
                                st.rerun()
                        with ub2:
                            st.download_button("Download", data=wav_path.read_bytes(), file_name=fname, mime="audio/wav", key=f"dl_fs_{fname}", use_container_width=True)
                    else:
                        st.caption("Click Fetch 6 from GitHub above to download this WAV.")


# ==============================================================================
# TAB 5: DIAGNOSTICS
# ==============================================================================
elif active_tab == "Diagnostics":
    _page_banner("Diagnostics", "Inspect training loss curves, sweet-spot checkpoints, and launch TensorBoard.")

    with st.container(border=True):
        _step_head(1, "Live Training Loss Analyzer", "Parses logs/<model>/nr_train.log or pasted Kaggle console output.")
        raw_log = _read_train_log(st.session_state["nr_model_name"])
        pasted_log = st.text_area("Training Log Output", value=raw_log, height=160, placeholder="Paste training lines containing epoch=X | step=Y | lowest_value=...")
        points: list[dict[str, Any]] = []
        for ln in pasted_log.splitlines():
            m = re.search(r"epoch=(\\d+)\\s*\\|\\s*step=(\\d+).*?lowest_value=([0-9.]+)", ln)
            if m:
                points.append({"Epoch": int(m.group(1)), "Step": int(m.group(2)), "Lowest Loss": float(m.group(3))})
        if points:
            best_pt = min(points, key=lambda x: x["Lowest Loss"])
            d1, d2, d3 = st.columns(3, gap="medium")
            d1.metric("Latest Epoch", points[-1]["Epoch"])
            d2.metric("Latest Step", points[-1]["Step"])
            d3.metric("Lowest Loss", f"{best_pt['Lowest Loss']:.3f} (ep {best_pt['Epoch']})")
            chart_rows = [{"Epoch": p["Epoch"], "Lowest Loss": p["Lowest Loss"]} for p in points]
            st.line_chart(chart_rows, x="Epoch", y="Lowest Loss", color="${ACCENT_HEX}")
        else:
            st.caption("No completed epoch loss points detected yet. Start training or paste training stdout above.")
        if st.button("Launch TensorBoard (Port 25565)", type="secondary"):
            threading.Thread(target=lambda: _run_core_sync("run_tensorboard_script", {}), daemon=True).start()
            st.info("TensorBoard launch triggered.")


# ==============================================================================
# TAB 6: VOICE BLENDER
# ==============================================================================
elif active_tab == "Voice Blender":
    _page_banner("Voice Blender", "Merge two compatible RVC .pth checkpoints into a single hybrid voice weights file.")

    pths = _list_pth_files()
    with st.container(border=True):
        _step_head(1, "Blend Checkpoints", "Choose Model A, Model B, blend ratio, and output name.")
        vb1, vb2 = st.columns(2, gap="medium")
        with vb1:
            pth_a = st.selectbox("Model A (.pth)", options=[""] + pths, format_func=lambda x: "Select Model A..." if x == "" else x)
        with vb2:
            pth_b = st.selectbox("Model B (.pth)", options=[""] + pths, format_func=lambda x: "Select Model B..." if x == "" else x)
        vb3, vb4 = st.columns(2, gap="medium")
        with vb3:
            blend_name = st.text_input("Blended Output Name", value="BlendedVoice")
        with vb4:
            blend_ratio = st.slider("Blend Ratio (Model A weight)", 0.0, 1.0, 0.5, step=0.05)

        if st.button("Blend Models", type="primary", use_container_width=True):
            if not pth_a or not pth_b or not blend_name.strip():
                st.warning("Select both Model A and Model B and enter an output name.")
            else:
                _, msg = _run_core_sync(
                    "run_model_blender_script",
                    {
                        "model_name": blend_name.strip(),
                        "pth_path_1": os.path.join(now_dir, pth_a),
                        "pth_path_2": os.path.join(now_dir, pth_b),
                        "ratio": float(blend_ratio),
                    },
                )
                st.session_state["nr_blend_log"] = msg
                out_candidate = Path(now_dir) / "logs" / f"{blend_name.strip()}.pth"
                if out_candidate.is_file():
                    st.session_state["nr_blend_out"] = str(out_candidate)
        if st.session_state["nr_blend_log"]:
            _log_box(st.session_state["nr_blend_log"])
        if st.session_state["nr_blend_out"] and os.path.isfile(st.session_state["nr_blend_out"]):
            bp = Path(st.session_state["nr_blend_out"])
            st.download_button(f"Download {bp.name}", data=bp.read_bytes(), file_name=bp.name, type="secondary")


# ==============================================================================
# TAB 7: TTS
# ==============================================================================
elif active_tab == "TTS":
    _page_banner("TTS", "Synthesize spoken dialogue with Edge-TTS and route it through your RVC voice model.")

    pths = _list_pth_files()
    idxs = _list_index_files()
    voices = _load_tts_voices()

    with st.container(border=True):
        _step_head(1, "Voice Model & Speech Input", "Select your RVC voice model, Edge-TTS speaker, and script.")
        t1, t2 = st.columns(2, gap="medium")
        with t1:
            tts_pth = st.selectbox("Voice Model", options=[""] + pths, format_func=lambda x: "Optional: Edge-TTS only..." if x == "" else x)
        with t2:
            tts_idx = st.selectbox("Index File", options=[""] + idxs, format_func=lambda x: "Optional index..." if x == "" else x)
        tv1, tv2 = st.columns(2, gap="medium")
        with tv1:
            tts_voice = st.selectbox("TTS Voice", options=voices, index=0)
        with tv2:
            tts_rate = st.slider("TTS Speed", -100, 100, 0, step=1)
        tts_text = st.text_area("Text to Speak", value="", placeholder="Enter text to synthesize...")

    with st.container(border=True):
        _step_head(2, "Conversion Parameters & Generate", "Tune pitch, index rate, volume envelope, and generate audio.")
        tp1, tp2 = st.columns(2, gap="medium")
        with tp1:
            tts_pitch = st.slider("Pitch", -32, 32, 0, step=1, key="tts_pitch")
            tts_vol = st.slider("Volume Envelope", 0.0, 1.0, 1.0, step=0.05, key="tts_vol")
        with tp2:
            tts_idx_r = st.slider("Index Rate", 0.0, 1.0, 0.75, step=0.05, key="tts_idx_r")
            tts_prot = st.slider("Protect", 0.0, 0.5, 0.5, step=0.05, key="tts_prot")

        tts_terms = st.checkbox("I agree to the terms of use — Confirm that your conversion audio follows the usage guidelines for NoRegretsRVC.", value=True, key="nr_tts_terms")
        tts_box = st.empty()
        tts_box.markdown(f'<div class="nr-log-box">{html.escape(st.session_state["nr_tts_log"])}</div>', unsafe_allow_html=True)
        if st.button("Convert TTS", type="primary", use_container_width=True):
            if not tts_terms or not tts_text.strip():
                st.warning("Enter text to speak and confirm terms first.")
            else:
                out_tts = os.path.join(now_dir, "assets", "audios", "tts_output.wav")
                out_rvc = os.path.join(now_dir, "assets", "audios", "tts_rvc_output.wav")
                rc, msg = _run_core_sync(
                    "run_tts_script",
                    {
                        "tts_file": "",
                        "tts_text": tts_text.strip(),
                        "tts_voice": tts_voice,
                        "tts_rate": int(tts_rate),
                        "pitch": int(tts_pitch),
                        "index_rate": float(tts_idx_r),
                        "volume_envelope": float(tts_vol),
                        "protect": float(tts_prot),
                        "f0_method": "rmvpe",
                        "output_tts_path": out_tts,
                        "output_rvc_path": out_rvc,
                        "pth_path": os.path.join(now_dir, tts_pth) if tts_pth else "",
                        "index_path": os.path.join(now_dir, tts_idx) if tts_idx else "",
                        "f0_autotune": False,
                        "f0_autotune_strength": 1.0,
                        "proposed_pitch": False,
                        "proposed_pitch_threshold": 155.0,
                        "export_format": "WAV",
                        "embedder_model": "contentvec",
                    },
                    tts_box,
                )
                st.session_state["nr_tts_log"] = msg
                target_audio = out_rvc if (tts_pth and os.path.isfile(out_rvc)) else out_tts
                if rc == 0 and os.path.isfile(target_audio):
                    st.session_state["nr_tts_out"] = target_audio
        if st.session_state["nr_tts_out"] and os.path.isfile(st.session_state["nr_tts_out"]):
            st.audio(st.session_state["nr_tts_out"])
            tp = Path(st.session_state["nr_tts_out"])
            st.download_button(f"Download {tp.name}", data=tp.read_bytes(), file_name=tp.name, type="secondary")


# ==============================================================================
# TAB 8: EXTRA
# ==============================================================================
elif active_tab == "Extra":
    _page_banner("Extra", "Inspect model checkpoint metadata, analyze audio tracks, and export ONNX weights.")
    ex_tabs = st.tabs(["Model information", "Audio Analyzer", "ONNX Export"])
    pths = _list_pth_files()

    with ex_tabs[0]:
        with st.container(border=True):
            _step_head(1, "Model Information", "Inspect epoch, sampling rate, architecture, and author metadata inside a .pth file.")
            info_pth = st.selectbox("Checkpoint (.pth)", options=[""] + pths, format_func=lambda x: "Select .pth..." if x == "" else x)
            if st.button("View Model Information", type="primary") and info_pth:
                _, res = _run_core_sync("run_model_information_script", {"pth_path": os.path.join(now_dir, info_pth)})
                _log_box(res)

    with ex_tabs[1]:
        with st.container(border=True):
            _step_head(2, "Audio Analyzer", "Inspect waveform duration, sample rate, and spectrum plot.")
            au_list = _list_audio_inputs()
            an_au = st.selectbox("Audio File", options=[""] + au_list, format_func=lambda x: "Select audio..." if x == "" else x)
            if st.button("Analyze Audio", type="primary") and an_au:
                plot_dst = os.path.join(now_dir, "logs", "audio_analysis.png")
                _, res = _run_core_sync("run_audio_analyzer_script", {"input_path": os.path.join(now_dir, an_au), "save_plot_path": plot_dst})
                _log_box(res)
                if os.path.isfile(plot_dst):
                    st.image(plot_dst)

    with ex_tabs[2]:
        with st.container(border=True):
            _step_head(3, "ONNX Export", "Export a .pth checkpoint into an optimized .onnx file.")
            ox_pth = st.selectbox("Source .pth", options=[""] + pths, format_func=lambda x: "Select .pth..." if x == "" else x, key="ox_pth")
            if st.button("Export to ONNX", type="primary") and ox_pth:
                src = os.path.join(now_dir, ox_pth)
                _, res = _run_core_sync("run_model_onnx_export_script", {"pth_path": src, "output_path": src.replace(".pth", ".onnx")})
                _log_box(res)


# ==============================================================================
# TAB 9: SETTINGS
# ==============================================================================
elif active_tab == "Settings":
    _page_banner("Settings", "Customize engine preferences, training precision, and language defaults.")
    st_tabs = st.tabs(["General", "Training"])

    with st_tabs[0]:
        with st.container(border=True):
            _step_head(1, "List Filtering", "Display a search bar to quickly filter long checkpoint and index dropdowns.")
            new_filter = st.checkbox("Enable model/index list filter", value=bool(cfg_now.get("filter", False)))
            if new_filter != bool(cfg_now.get("filter", False)):
                cfg_now["filter"] = bool(new_filter)
                _save_config(cfg_now)
                st.success("Filter setting saved.")

        with st.container(border=True):
            _step_head(2, "Pitch Correction & Discord Integration", "Configure RMVPE octave correction and Discord Rich Presence.")
            rmvpe_cfg = cfg_now.get("rmvpe_high_register", {}) if isinstance(cfg_now.get("rmvpe_high_register"), dict) else {}
            rmvpe_on = st.checkbox("RMVPE high-register correction", value=bool(rmvpe_cfg.get("enabled", False)))
            disc_on = st.checkbox("Enable Discord Rich Presence", value=bool(cfg_now.get("discord_presence", False)))
            if st.button("Save General Preferences", type="primary"):
                cfg_now["rmvpe_high_register"] = {**rmvpe_cfg, "enabled": bool(rmvpe_on)}
                cfg_now["discord_presence"] = bool(disc_on)
                cfg_now["nr_discord_initialized"] = True
                _save_config(cfg_now)
                st.success("General preferences saved.")

        with st.container(border=True):
            _step_head(3, "Language & System", "Pick interface language or check version status.")
            lang_cfg = cfg_now.get("lang", {}) if isinstance(cfg_now.get("lang"), dict) else {}
            langs = ["en_US", "es_ES", "pt_BR", "fr_FR", "it_IT", "nl_NL", "ja_JP", "zh_CN", "ar_SA"]
            cur_lang = lang_cfg.get("selected_lang", "en_US")
            sel_lang = st.selectbox("Language", langs, index=langs.index(cur_lang) if cur_lang in langs else 0)
            st.caption("${BRAND_NAME} Beta version 0.0.0")
            if st.button("Save Language", type="secondary"):
                cfg_now["lang"] = {"override": True, "selected_lang": sel_lang}
                _save_config(cfg_now)
                st.success("Language saved.")

    with st_tabs[1]:
        with st.container(border=True):
            _step_head(1, "Model Author", "Set the name written into exported model metadata headers.")
            cur_author = str(cfg_now.get("model_author", "None"))
            new_author = st.text_input("Model Author Name", value=cur_author)
            if st.button("Set name", type="primary"):
                cfg_now["model_author"] = new_author.strip() or "None"
                _save_config(cfg_now)
                st.success(f"Model author set to {cfg_now['model_author']}.")

        with st.container(border=True):
            _step_head(2, "Training Precision", "Choose numeric precision for training and inference (defaults to FP32 for Kaggle T4 safety).")
            prec_opts = ["fp32", "fp16", "bf16"]
            cur_prec = str(cfg_now.get("precision", "fp32"))
            new_prec = st.selectbox("Precision", options=prec_opts, index=prec_opts.index(cur_prec) if cur_prec in prec_opts else 0)
            if st.button("Update precision", type="primary"):
                cfg_now["precision"] = new_prec
                cfg_now["nr_precision_initialized"] = True
                _save_config(cfg_now)
                st.success(f"Precision set to {new_prec}.")
`;
}

export function generateTurboTrainerPython(): string {
  return `from __future__ import annotations

import atexit
from concurrent.futures import Future, ThreadPoolExecutor
import inspect
import io
import os
from pathlib import Path
import threading
from typing import Any

_APPLIED = False
_CACHE_LOCK = threading.Lock()
_ARRAY_CACHE: dict[str, Any] = {}
_CACHE_BYTES = 0
_CACHE_CHECKS = 0
_PENDING_SAVES: dict[str, Future[None]] = {}
_SAVE_LOCK = threading.Lock()
_SAVE_POOL: ThreadPoolExecutor | None = None

_DATASET_SUBDIRS = ("sliced_audios", "extracted", "f0", "f0_voiced")
_MAX_SINGLE_FILE_BYTES = 16 * 1024 * 1024
_PER_PROCESS_RAM_CAP_BYTES = 2 * 1024 * 1024 * 1024
_MIN_SYSTEM_FREE_RAM_BYTES = 8 * 1024 * 1024 * 1024


def _read_mem_available_bytes() -> int:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb * 1024
    except Exception:
        pass
    return 0


def _has_safe_ram(incoming_bytes: int) -> bool:
    global _CACHE_CHECKS
    if _CACHE_BYTES + incoming_bytes > _PER_PROCESS_RAM_CAP_BYTES:
        return False
    _CACHE_CHECKS += 1
    if _CACHE_CHECKS == 1 or _CACHE_CHECKS % 24 == 0:
        avail = _read_mem_available_bytes()
        if avail > 0 and avail - incoming_bytes < _MIN_SYSTEM_FREE_RAM_BYTES:
            return False
    return True


def _is_cacheable_dataset_path(path: Any) -> str | None:
    if not isinstance(path, (str, os.PathLike)):
        return None
    raw = os.fspath(path)
    norm = raw.replace("\\\\", "/")
    if "/logs/" not in norm and not norm.startswith("logs/"):
        return None
    if not any(f"/{part}/" in norm or norm.startswith(f"{part}/") for part in _DATASET_SUBDIRS):
        return None
    try:
        if os.path.getsize(raw) > _MAX_SINGLE_FILE_BYTES:
            return None
    except OSError:
        return None
    return raw


def _apply_cuda_tuning() -> None:
    import torch

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True


def _patch_dataloader() -> None:
    import torch.utils.data

    orig_init = torch.utils.data.DataLoader.__init__
    if getattr(orig_init, "_noregrets_patched", False):
        return

    cpu_cap = max(2, min(4, os.cpu_count() or 2))

    def fast_init(self: Any, *args: Any, **kwargs: Any) -> Any:
        num_workers = kwargs.get("num_workers", 0)
        if "num_workers" not in kwargs and len(args) >= 6:
            num_workers = args[5]
        if isinstance(num_workers, int) and num_workers > 0:
            if "num_workers" in kwargs and num_workers > cpu_cap:
                kwargs["num_workers"] = cpu_cap
            kwargs.setdefault("pin_memory", True)
            kwargs.setdefault("persistent_workers", True)
            kwargs.setdefault("prefetch_factor", 4)
        try:
            return orig_init(self, *args, **kwargs)
        except (TypeError, ValueError):
            kwargs.pop("persistent_workers", None)
            kwargs.pop("prefetch_factor", None)
            kwargs.pop("pin_memory", None)
            return orig_init(self, *args, **kwargs)

    fast_init._noregrets_patched = True  # type: ignore[attr-defined]
    torch.utils.data.DataLoader.__init__ = fast_init  # type: ignore[method-assign]


def _patch_optimizer() -> None:
    import torch

    orig_adamw_init = torch.optim.AdamW.__init__
    if not getattr(orig_adamw_init, "_noregrets_patched", False):
        supports_fused = "fused" in inspect.signature(orig_adamw_init).parameters

        def fast_adamw_init(self: Any, params: Any, *args: Any, **kwargs: Any) -> Any:
            _apply_cuda_tuning()
            if supports_fused and torch.cuda.is_available() and "fused" not in kwargs:
                if not isinstance(params, (list, tuple)):
                    params = list(params)
                try:
                    return orig_adamw_init(self, params, *args, fused=True, **kwargs)
                except Exception:
                    pass
            return orig_adamw_init(self, params, *args, **kwargs)

        fast_adamw_init._noregrets_patched = True  # type: ignore[attr-defined]
        torch.optim.AdamW.__init__ = fast_adamw_init  # type: ignore[method-assign]

    orig_zero_grad = torch.optim.Optimizer.zero_grad
    if not getattr(orig_zero_grad, "_noregrets_patched", False):
        def fast_zero_grad(self: Any, set_to_none: bool = True) -> Any:
            return orig_zero_grad(self, set_to_none=set_to_none)

        fast_zero_grad._noregrets_patched = True  # type: ignore[attr-defined]
        torch.optim.Optimizer.zero_grad = fast_zero_grad  # type: ignore[method-assign]


def _patch_ddp() -> None:
    import torch.nn.parallel

    ddp_cls = torch.nn.parallel.DistributedDataParallel
    orig_ddp_init = ddp_cls.__init__
    if getattr(orig_ddp_init, "_noregrets_patched", False):
        return

    def fast_ddp_init(self: Any, *args: Any, **kwargs: Any) -> Any:
        _apply_cuda_tuning()
        kwargs.setdefault("gradient_as_bucket_view", True)
        try:
            return orig_ddp_init(self, *args, **kwargs)
        except TypeError:
            kwargs.pop("gradient_as_bucket_view", None)
            return orig_ddp_init(self, *args, **kwargs)

    fast_ddp_init._noregrets_patched = True  # type: ignore[attr-defined]
    ddp_cls.__init__ = fast_ddp_init  # type: ignore[method-assign]


def _patch_tensorboard_images() -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        return

    def _skip_image(*args: Any, **kwargs: Any) -> None:
        return None

    for method in ("add_image", "add_images", "add_figure"):
        if hasattr(SummaryWriter, method):
            setattr(SummaryWriter, method, _skip_image)


def _patch_ram_cache() -> None:
    import numpy as np

    orig_np_load = np.load
    if not getattr(orig_np_load, "_noregrets_patched", False):
        def fast_np_load(file: Any, *args: Any, **kwargs: Any) -> Any:
            global _CACHE_BYTES
            if args or kwargs:
                return orig_np_load(file, *args, **kwargs)
            key = _is_cacheable_dataset_path(file)
            if key is None:
                return orig_np_load(file)
            cache_key = f"np::{key}"
            with _CACHE_LOCK:
                cached = _ARRAY_CACHE.get(cache_key)
            if cached is not None:
                return cached.copy()
            arr = orig_np_load(file)
            nbytes = int(getattr(arr, "nbytes", 0))
            if nbytes > 0 and isinstance(arr, np.ndarray):
                with _CACHE_LOCK:
                    if cache_key not in _ARRAY_CACHE and _has_safe_ram(nbytes):
                        _ARRAY_CACHE[cache_key] = arr.copy()
                        _CACHE_BYTES += nbytes
            return arr

        fast_np_load._noregrets_patched = True  # type: ignore[attr-defined]
        np.load = fast_np_load  # type: ignore[assignment]

    try:
        import scipy.io.wavfile as wavfile

        orig_wav_read = wavfile.read
        if not getattr(orig_wav_read, "_noregrets_patched", False):
            def fast_wav_read(filename: Any, *args: Any, **kwargs: Any) -> Any:
                global _CACHE_BYTES
                if args or kwargs:
                    return orig_wav_read(filename, *args, **kwargs)
                key = _is_cacheable_dataset_path(filename)
                if key is None:
                    return orig_wav_read(filename)
                cache_key = f"wav::{key}"
                with _CACHE_LOCK:
                    cached = _ARRAY_CACHE.get(cache_key)
                if cached is not None:
                    sr, data = cached
                    return sr, data.copy()
                sr, data = orig_wav_read(filename)
                nbytes = int(getattr(data, "nbytes", 0))
                if nbytes > 0:
                    with _CACHE_LOCK:
                        if cache_key not in _ARRAY_CACHE and _has_safe_ram(nbytes):
                            _ARRAY_CACHE[cache_key] = (sr, data.copy())
                            _CACHE_BYTES += nbytes
                return sr, data

            fast_wav_read._noregrets_patched = True  # type: ignore[attr-defined]
            wavfile.read = fast_wav_read  # type: ignore[assignment]
    except Exception:
        pass

    try:
        import soundfile as sf

        orig_sf_read = sf.read
        if not getattr(orig_sf_read, "_noregrets_patched", False):
            def fast_sf_read(file: Any, *args: Any, **kwargs: Any) -> Any:
                global _CACHE_BYTES
                if args or kwargs:
                    return orig_sf_read(file, *args, **kwargs)
                key = _is_cacheable_dataset_path(file)
                if key is None:
                    return orig_sf_read(file)
                cache_key = f"sf::{key}"
                with _CACHE_LOCK:
                    cached = _ARRAY_CACHE.get(cache_key)
                if cached is not None:
                    data, sr = cached
                    return data.copy(), sr
                data, sr = orig_sf_read(file)
                nbytes = int(getattr(data, "nbytes", 0))
                if nbytes > 0:
                    with _CACHE_LOCK:
                        if cache_key not in _ARRAY_CACHE and _has_safe_ram(nbytes):
                            _ARRAY_CACHE[cache_key] = (data.copy(), sr)
                            _CACHE_BYTES += nbytes
                return data, sr

            fast_sf_read._noregrets_patched = True  # type: ignore[attr-defined]
            sf.read = fast_sf_read  # type: ignore[assignment]
    except Exception:
        pass


def _write_atomic(target: Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(payload)
        os.replace(tmp, target)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise


def _patch_async_save() -> None:
    global _SAVE_POOL
    import torch

    orig_save = torch.save
    orig_load = torch.load
    if getattr(orig_save, "_noregrets_patched", False):
        return

    _SAVE_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nr-ckpt")
    atexit.register(lambda: _SAVE_POOL.shutdown(wait=True) if _SAVE_POOL else None)

    def fast_save(obj: Any, f: Any, *args: Any, **kwargs: Any) -> Any:
        if (
            _SAVE_POOL is not None
            and isinstance(f, (str, os.PathLike))
            and not args
            and not kwargs
        ):
            raw = os.fspath(f)
            norm = raw.replace("\\\\", "/")
            if norm.endswith(".pth") and ("/logs/" in norm or norm.startswith("logs/")):
                buf = io.BytesIO()
                orig_save(obj, buf)
                target = Path(raw).resolve()
                fut = _SAVE_POOL.submit(_write_atomic, target, buf.getvalue())
                with _SAVE_LOCK:
                    _PENDING_SAVES[str(target)] = fut
                return None
        return orig_save(obj, f, *args, **kwargs)

    def safe_load(f: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(f, (str, os.PathLike)):
            key = str(Path(os.fspath(f)).resolve())
            with _SAVE_LOCK:
                fut = _PENDING_SAVES.get(key)
            if fut is not None:
                fut.result()
        return orig_load(f, *args, **kwargs)

    fast_save._noregrets_patched = True  # type: ignore[attr-defined]
    torch.save = fast_save  # type: ignore[assignment]
    torch.load = safe_load  # type: ignore[assignment]


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True
    for step in (
        _apply_cuda_tuning,
        _patch_dataloader,
        _patch_optimizer,
        _patch_ddp,
        _patch_tensorboard_images,
        _patch_ram_cache,
        _patch_async_save,
    ):
        try:
            step()
        except Exception:
            pass
`;
}

export function generateKaggleNotebookJson(
  config: ForkConfig,
  project: VoiceModelProject,
  downloads: DownloadModelItem[]
): string {
  const repoTarget = config.customRepoUrl.trim() || 'https://github.com/IAHispano/Applio.git';
  const themeB64 = toBase64Utf8(generateThemePython());
  const appB64 = toBase64Utf8(generateAppPython(config, project));
  const turboEnabled = config.turboTraining ?? true;
  const turboB64 = toBase64Utf8(generateTurboTrainerPython());

  const installLines: string[] = [
    `import base64`,
    `import ctypes.util`,
    `import hashlib`,
    `import os`,
    `import shutil`,
    `import subprocess`,
    `import sys`,
    `import urllib.request`,
    `from pathlib import Path`,
    `from IPython.display import clear_output`,
    ``,
    `repo_dir = Path("/kaggle/working/program_ml")`,
    `repo_url = "${repoTarget}"`,
    ``,
    `subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)`,
    `if repo_dir.exists() and (repo_dir / ".git").is_dir():`,
    `    subprocess.run(["git", "-C", str(repo_dir), "pull", "--ff-only"], check=False)`,
    `else:`,
    `    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)], check=True)`,
    ``,
    `if shutil.which("fuser") is None or ctypes.util.find_library("portaudio") is None:`,
    `    subprocess.run(["apt", "update", "-y"], check=True)`,
    `    subprocess.run(["apt", "install", "-y", "portaudio19-dev", "psmisc"], check=True)`,
    ``,
    `uv_bin = shutil.which("uv") or str(Path(sys.executable).with_name("uv"))`,
  ];

  if (config.useUvVenv) {
    installLines.push(
      `venv_python = Path("/kaggle/.venv/bin/python")`,
      `if not venv_python.exists():`,
      `    subprocess.run([uv_bin, "venv", "/kaggle/.venv", "--python", "3.12", "--seed"], check=True)`,
      `py_exec = str(venv_python)`,
      `subprocess.run([uv_bin, "pip", "install", "-q", "-r", str(repo_dir / "requirements.txt"), "--extra-index-url", "https://download.pytorch.org/whl/cu128", "--index-strategy", "unsafe-best-match", "--python", py_exec], check=True)`,
      `subprocess.run([uv_bin, "pip", "install", "-q", "streamlit>=1.40.0", "matplotlib-inline", "websockets", "--python", py_exec], check=True)`
    );
  } else {
    installLines.push(
      `py_exec = sys.executable`,
      `subprocess.run([uv_bin, "pip", "install", "-q", "-r", str(repo_dir / "requirements.txt"), "--extra-index-url", "https://download.pytorch.org/whl/cu128", "--index-strategy", "unsafe-best-match", "--system"], check=True)`,
      `subprocess.run([uv_bin, "pip", "install", "-q", "streamlit>=1.40.0", "matplotlib-inline", "websockets", "--system"], check=True)`
    );
  }

  if (config.verifyOnnxGpu) {
    installLines.push(
      `subprocess.run([uv_bin, "pip", "install", "-q", "onnxruntime-gpu", "--python", py_exec] if ${config.useUvVenv ? 'True' : 'False'} else [uv_bin, "pip", "install", "-q", "onnxruntime-gpu", "--system"], check=False)`,
      `subprocess.run([py_exec, "-c", "import onnxruntime as ort; print('ONNX Providers:', ort.get_available_providers())"], check=False)`
    );
  }

  installLines.push(
    ``,
    `# Reset previously patched config so re-runs stay clean`,
    `if (repo_dir / ".git").is_dir():`,
    `    subprocess.run(["git", "-C", str(repo_dir), "checkout", "--", "assets/config_template.json"], check=False)`,
    ``,
    `# Enforce ${BRAND_NAME} engine defaults (fp32 precision, Discord presence off)`,
    `import json as _cfg_json`,
    `for _cfg_rel in ("assets/config_template.json", "assets/config.json"):`,
    `    _cfg_p = repo_dir / _cfg_rel`,
    `    if _cfg_p.is_file():`,
    `        try:`,
    `            _cfg_d = _cfg_json.loads(_cfg_p.read_text(encoding="utf-8"))`,
    `            _cfg_d["precision"] = "fp32"`,
    `            _cfg_d["discord_presence"] = False`,
    `            _cfg_p.write_text(_cfg_json.dumps(_cfg_d, indent=4, ensure_ascii=False), encoding="utf-8")`,
    `        except Exception:`,
    `            pass`,
    ``,
    `# Write ${BRAND_NAME} Streamlit WebUI and locked crimson theme`,
    `(repo_dir / ".streamlit").mkdir(parents=True, exist_ok=True)`,
    `(repo_dir / ".streamlit" / "config.toml").write_bytes(base64.b64decode("${themeB64}"))`,
    `(repo_dir / "streamlit_app.py").write_bytes(base64.b64decode("${appB64}"))`,
    ...(turboEnabled
      ? [
          `(repo_dir / "noregrets_turbo.py").write_bytes(base64.b64decode("${turboB64}"))`,
          `train_py = repo_dir / "rvc" / "train" / "train.py"`,
          `if train_py.is_file():`,
          `    _src = train_py.read_text(encoding="utf-8", errors="replace")`,
          `    if "noregrets_turbo.apply()" not in _src:`,
          `        _lines = _src.splitlines(keepends=True)`,
          `        _ins = 0`,
          `        for _i, _ln in enumerate(_lines[:40]):`,
          `            if _ln.startswith("from __future__ import"):`,
          `                _ins = _i + 1`,
          `        _hook = "import os as _nr_os, sys as _nr_sys\\n_nr_sys.path.insert(0, _nr_os.getcwd())\\ntry:\\n    import noregrets_turbo\\n    noregrets_turbo.apply()\\nexcept Exception:\\n    pass\\n"`,
          `        _lines.insert(_ins, _hook)`,
          `        train_py.write_text("".join(_lines), encoding="utf-8")`,
        ]
      : []),
    ``,
    `# Best-effort prefetch of the 6 Free Samples WAVs from ${REPO_URL}`,
    `fs_dir = repo_dir / "assets" / "free_samples"`,
    `au_dir = repo_dir / "assets" / "audios"`,
    `fs_dir.mkdir(parents=True, exist_ok=True)`,
    `au_dir.mkdir(parents=True, exist_ok=True)`,
    `for _wav_name in (`,
    `    "MaleSpeakingTest-1_931b5ffd.wav",`,
    `    "MaleSpeakingTest-2_c2e6e7fe.wav",`,
    `    "MaleSpeakingTest-3_d78e8233.wav",`,
    `    "FemaleSpeakingTest-1_bb3d674f.wav",`,
    `    "FemaleSpeakingTest-2_d1823132.wav",`,
    `    "FemaleSpeakingTest-3_458bd4fc.wav",`,
    `):`,
    `    _dst_fs = fs_dir / _wav_name`,
    `    if not _dst_fs.is_file():`,
    `        try:`,
    `            urllib.request.urlretrieve(f"https://raw.githubusercontent.com/HunterThatsMe/NoRegretsRVC/main/public/assets/aistudio/{_wav_name}", _dst_fs)`,
    `            shutil.copy2(_dst_fs, au_dir / _wav_name)`,
    `        except Exception:`,
    `            pass`,
    ``,
    `os.chdir(repo_dir)`,
    `prereq_cmd = [py_exec, "core.py", "prerequisites", "--models", "--exe", "--pretraineds-hifigan"]`,
    `subprocess.run(prereq_cmd, check=True)`
  );

  const bootItems: { name: string; url: string; category: string }[] = downloads.map((d) => ({
    name: d.name,
    url: d.url,
    category: d.category,
  }));
  if (project.customPretrained) {
    const autoD = !project.customPretrainedD.trim()
      ? inferMatchingDUrl(project.customPretrainedG)
      : '';
    for (const rawUrl of [project.customPretrainedG, project.customPretrainedD || autoD]) {
      const clean = (rawUrl || '').trim();
      if (/^https?:\/\//i.test(clean) && !bootItems.some((b) => b.url === clean)) {
        bootItems.push({
          name: clean.split('/').pop()?.split('?')[0] || 'custom_pretrained.pth',
          url: clean,
          category: 'pretrained',
        });
      }
    }
  }

  if (bootItems.length > 0) {
    installLines.push(
      ``,
      `# Pre-boot Model & Custom Pretrains Downloader (HF repo auto-discovery, ZIP & G/D .pth)`,
      `import json as _json`,
      `import urllib.parse`,
      `import zipfile as _zipfile`,
      `boot_downloads = ${JSON.stringify(bootItems, null, 4)}`,
      `dl_dir = repo_dir / "logs" / "downloads"`,
      `custom_pretrain_dir = repo_dir / "rvc" / "models" / "pretraineds" / "custom"`,
      `dl_dir.mkdir(parents=True, exist_ok=True)`,
      `custom_pretrain_dir.mkdir(parents=True, exist_ok=True)`,
      `def _download_hf_repo_pretrains(_repo_url, _dst_dir):`,
      `    _parts = [p for p in urllib.parse.urlsplit(_repo_url).path.strip("/").split("/") if p]`,
      `    if len(_parts) < 2:`,
      `        return`,
      `    _owner_repo = f"{_parts[0]}/{_parts[1]}"`,
      `    _api = f"https://huggingface.co/api/models/{_owner_repo}/tree/main"`,
      `    with urllib.request.urlopen(_api, timeout=30) as _resp:`,
      `        _tree = _json.loads(_resp.read().decode("utf-8"))`,
      `    for _node in _tree:`,
      `        _fpath = _node.get("path", "")`,
      `        if _fpath.lower().endswith(".pth"):`,
      `            _fname = Path(_fpath).name`,
      `            _out = _dst_dir / _fname`,
      `            if not _out.is_file():`,
      `                _file_url = f"https://huggingface.co/{_owner_repo}/resolve/main/{urllib.parse.quote(_fpath)}?download=true"`,
      `                print("Downloading discovered pretrain", _fname, "->", _out)`,
      `                urllib.request.urlretrieve(_file_url, _out)`,
      `for entry in boot_downloads:`,
      `    _url = entry["url"].strip()`,
      `    _cat = entry.get("category", "voice")`,
      `    _target_dir = custom_pretrain_dir if _cat == "pretrained" else dl_dir`,
      `    _parsed_path = urllib.parse.urlsplit(_url).path`,
      `    _raw_name = urllib.parse.unquote(_parsed_path.split("/")[-1]) or entry["name"]`,
      `    try:`,
      `        if _cat == "pretrained" and "huggingface.co/" in _url.lower() and "/resolve/" not in _url and not _parsed_path.lower().endswith((".pth", ".zip")):`,
      `            _download_hf_repo_pretrains(_url, _target_dir)`,
      `        elif _cat == "pretrained" and _parsed_path.lower().endswith(".zip"):`,
      `            _zip_tmp = _target_dir / (_raw_name if _raw_name.endswith(".zip") else "custom_pretrains.zip")`,
      `            print("Downloading pretrained ZIP", _url, "->", _zip_tmp)`,
      `            urllib.request.urlretrieve(_url, _zip_tmp)`,
      `            with _zipfile.ZipFile(_zip_tmp, "r") as _zf:`,
      `                for _member in _zf.namelist():`,
      `                    if not _member.endswith("/") and Path(_member).name and _member.lower().endswith(".pth"):`,
      `                        _out_pth = _target_dir / Path(_member).name`,
      `                        _out_pth.write_bytes(_zf.read(_member))`,
      `                        print("Extracted", _out_pth.name)`,
      `            _zip_tmp.unlink(missing_ok=True)`,
      `        else:`,
      `            _dst = _target_dir / _raw_name`,
      `            if not _dst.is_file():`,
      `                print("Downloading", entry["name"], "->", _dst)`,
      `                urllib.request.urlretrieve(_url, _dst)`,
      `    except Exception as _dl_err:`,
      `        print("Skipping unreachable download", _url, ":", _dl_err)`
    );
  }

  installLines.push(
    ``,
    `cloudflared_path = repo_dir.parent / "cloudflared"`,
    `cloudflared_url = "https://github.com/cloudflare/cloudflared/releases/download/2026.9.0/cloudflared-linux-amd64"`,
    `cloudflared_sha256 = "53b7a7a5420d188758d24341294acb0d1bca54296548ac05e38811a694ac6134"`,
    `if not (cloudflared_path.is_file() and hashlib.sha256(cloudflared_path.read_bytes()).hexdigest() == cloudflared_sha256):`,
    `    tmp = cloudflared_path.with_suffix(".download")`,
    `    with urllib.request.urlopen(cloudflared_url, timeout=120) as resp, open(tmp, "wb") as dst:`,
    `        shutil.copyfileobj(resp, dst)`,
    `    if hashlib.sha256(tmp.read_bytes()).hexdigest() != cloudflared_sha256:`,
    `        tmp.unlink(missing_ok=True)`,
    `        raise RuntimeError("cloudflared checksum verification failed.")`,
    `    os.replace(tmp, cloudflared_path)`,
    `cloudflared_path.chmod(0o755)`,
    `clear_output()`,
    `print("${BRAND_NAME} Streamlit setup complete! Model: ${sanitizeModelName(project.name)} (${project.samplingRate} · ${project.vocoder})")`
  );

  const pyExecutableExpr = config.useUvVenv ? '"/kaggle/.venv/bin/python"' : 'sys.executable';

  const startLines: string[] = [
    `import os`,
    `import re`,
    `import secrets`,
    `import subprocess`,
    `import sys`,
    `import time`,
    `from collections import deque`,
    `from pathlib import Path`,
    `from IPython.display import clear_output`,
    ``,
    `Tunnel = "${config.tunnelMode}" #@param ["Gradio + Cloudflare", "Horizon"]`,
    `horizon_id = "${config.horizonId}" #@param {type:"string"}`,
    ``,
    `repo_dir = Path("/kaggle/working/program_ml")`,
    `cloudflared_path = repo_dir.parent / "cloudflared"`,
    `py_exec = ${pyExecutableExpr}`,
    ``,
    `app_user = "noregrets"`,
    `app_pass = ${config.randomAuthEnabled ? 'secrets.token_urlsafe(24)' : '""'}`,
    ``,
    `cf_pattern = r"https://[a-zA-Z0-9-]+\\.trycloudflare\\.com"`,
    `hz_pattern = r"https://[a-zA-Z0-9.-]+\\.hrzn\\.run"`,
    `tunnel_procs = []`,
    `tunnel_logs = []`,
    `app_proc = None`,
    ``,
    `def start_tunnel(cmd, name, pattern):`,
    `    out_path = repo_dir / f"{name}.log"`,
    `    f = open(out_path, "w", encoding="utf-8")`,
    `    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)`,
    `    tunnel_procs.append(proc)`,
    `    tunnel_logs.append(f)`,
    `    deadline = time.time() + 60`,
    `    while time.time() < deadline:`,
    `        txt = out_path.read_text(encoding="utf-8", errors="replace") if out_path.exists() else ""`,
    `        m = re.search(pattern, txt)`,
    `        if m:`,
    `            return m.group(0)`,
    `        time.sleep(1)`,
    `    raise RuntimeError(f"Tunnel {name} failed to emit URL")`,
    ``,
    `os.chdir(repo_dir)`,
    `subprocess.run(["fuser", "-k", "6969/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)`,
    ``,
    `if Tunnel == "Horizon":`,
    `    if not horizon_id.strip():`,
    `        raise ValueError("Enter your horizon_id before running Horizon tunnel.")`,
    `    subprocess.run(["npm", "install", "-g", "@hrzn/cli"], check=True)`,
    `    subprocess.run(["hrzn", "login", horizon_id], check=True)`,
    ``,
    `# --- ${BRAND_NAME} Kaggle Live Monitor (spaced rows + mid-epoch + Speed|Step|Epoch table) ---`,
    `BOLD = "\\033[1m"`,
    `RESET = "\\033[0m"`,
    `ANSI_RE = re.compile(r"\\x1B\\[[0-?]*[ -/]*[@-~]")`,
    `EPOCH_RE = re.compile(r"epoch=(\\d+)\\s*\\|\\s*step=(\\d+).*?training_speed=(\\d+:\\d{2}:\\d{2})(?:.*?lowest_value=([0-9.]+))?")`,
    `TQDM_RE = re.compile(r"(\\d+)%\\|[^|]*\\|\\s*(\\d+)/(\\d+)\\s*\\[([^,\\]]+),\\s*([0-9.]+\\s*(?:it/s|s/it))")`,
    `LOCAL_URL_RE = re.compile(r"(?:Running on local URL|Local URL|Network URL):\\s*(https?://\\S+)")`,
    `NOISE_RE = re.compile(r"(FutureWarning|UserWarning|DeprecationWarning|RuntimeWarning|pkg_resources|warnings\\.warn|oneDNN|tensorflow/core|computation_placer|numa_node|Successfully opened dynamic library|weights_only|HTTP Request:|Checking out|libpng warning|ALSA|jack server|You can now view your Streamlit)", re.IGNORECASE)`,
    `MILESTONE_RE = re.compile(r"(Starting preprocess|Preprocess completed|preprocessed successfully|Starting pitch extraction|Pitch extraction completed|Starting embedding extraction|Embedding extraction completed|Saved index file|Starting training|Saving checkpoint|Saving model|Training finished|Training completed|Index file.*generated|Using .* vocoder|Using .* for training)", re.IGNORECASE)`,
    `ERROR_START_RE = re.compile(r"(Traceback \\(most recent call last\\)|RuntimeError:|ValueError:|FileNotFoundError:|OSError:|ModuleNotFoundError:|ImportError:|CUDA out of memory|OutOfMemoryError|Exception:|\\bERROR:)")`,
    ``,
    `session_links = []`,
    `milestones = deque(maxlen=6)`,
    `errors = deque(maxlen=25)`,
    `epoch_rows = deque(maxlen=15)`,
    `seen_epochs = set()`,
    `live_mid_epoch = ""`,
    `last_tq_render = 0.0`,
    `tb_lines_seen = 0`,
    ``,
    `def upsert_link(label, value):`,
    `    for idx, (k, _) in enumerate(session_links):`,
    `        if k.strip() == label.strip():`,
    `            session_links[idx] = (label, value)`,
    `            return`,
    `    session_links.append((label, value))`,
    ``,
    `def render_monitor(status="Running"):`,
    `    w_speed = max(9, max((len(r[0]) for r in epoch_rows), default=0))`,
    `    w_step = max(8, max((len(r[1]) for r in epoch_rows), default=0))`,
    `    w_epoch = max(6, max((len(r[2]) for r in epoch_rows), default=0))`,
    `    lines = [`,
    `        f"{BOLD}${BRAND_NAME} Live Monitor (Beta 0.0.0){RESET}  [{status}]",`,
    `        "=" * 64,`,
    `        "",`,
    `    ]`,
    `    if app_pass:`,
    `        lines.append(f"WebUI Login      : {app_user} / {app_pass}")`,
    `        lines.append("")`,
    `    for label, val in session_links:`,
    `        if val:`,
    `            lines.append(f"{label:<17}: {val}")`,
    `            lines.append("")`,
    `    lines.append("-" * 64)`,
    `    lines.append("")`,
    `    if milestones:`,
    `        for m in milestones:`,
    `            lines.append(f"[Stage] {m}")`,
    `        lines.append("")`,
    `    if live_mid_epoch:`,
    `        lines.append(f"{BOLD}Live Epoch:{RESET} {live_mid_epoch}")`,
    `        lines.append("")`,
    `    if epoch_rows:`,
    `        lines.append(f"{'Speed':<{w_speed}} | {'Step':<{w_step}} | {BOLD}{'Epoch':<{w_epoch}}{RESET} | Lowest Loss")`,
    `        lines.append(f"{'-' * w_speed}-+-{'-' * w_step}-+-{'-' * w_epoch}-+------------")`,
    `        for spd, stp, ep, low in epoch_rows:`,
    `            lines.append(f"{spd:<{w_speed}} | {stp:<{w_step}} | {BOLD}{ep:<{w_epoch}}{RESET} | {low}")`,
    `    elif not errors:`,
    `        lines.append("Waiting for training epochs...")`,
    `    if errors:`,
    `        lines.append("")`,
    `        lines.append(f"{BOLD}Errors:{RESET}")`,
    `        for err in errors:`,
    `            lines.append(f"  {err}")`,
    `    clear_output(wait=True)`,
    `    print("\\n".join(lines), flush=True)`,
    ``,
    `try:`,
    `    env = os.environ.copy()`,
    `    if app_pass:`,
    `        env["APPLIO_AUTH_USERNAME"] = app_user`,
    `        env["APPLIO_AUTH_PASSWORD"] = app_pass`,
    `    env.update({`,
    `        "PYTHONWARNINGS": "ignore",`,
    `        "PYTHONUNBUFFERED": "1",`,
    `        "TF_CPP_MIN_LOG_LEVEL": "3",`,
    `        "TORCH_CPP_LOG_LEVEL": "ERROR",`,
    `        "TOKENIZERS_PARALLELISM": "false",`,
    `    })`,
    ``,
    `    if Tunnel == "Gradio + Cloudflare":`,
    `        cf_url = start_tunnel([str(cloudflared_path), "tunnel", "--no-autoupdate", "--protocol", "http2", "--url", "http://127.0.0.1:6969"], "webui-cf", cf_pattern)`,
    `        upsert_link("${BRAND_NAME} URL", cf_url)`,
    `        upsert_link("Cloudflare URL", cf_url)`,
    `    else:`,
    `        web_url = start_tunnel(["hrzn", "tunnel", "http://localhost:6969"], "webui-hrzn", hz_pattern)`,
    `        upsert_link("${BRAND_NAME} URL", web_url)`,
    ``,
    `    app_cmd = [`,
    `        py_exec, "-u", "-m", "streamlit", "run", "streamlit_app.py",`,
    `        "--server.address", "0.0.0.0",`,
    `        "--server.port", "6969",`,
    `        "--server.headless", "true",`,
    `        "--server.enableCORS", "false",`,
    `        "--server.enableXsrfProtection", "false",`,
    `    ]`,
    `    render_monitor("Starting ${BRAND_NAME} WebUI")`,
    `    app_proc = subprocess.Popen(app_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)`,
    `    assert app_proc.stdout is not None`,
    `    for raw_line in app_proc.stdout:`,
    `        line = ANSI_RE.sub("", raw_line).replace("\\r", "").strip()`,
    `        if not line:`,
    `            continue`,
    `        loc_m = LOCAL_URL_RE.search(line)`,
    `        if loc_m:`,
    `            upsert_link("Local UI URL", loc_m.group(1))`,
    `            render_monitor("Ready")`,
    `            continue`,
    `        ep_m = EPOCH_RE.search(line)`,
    `        if ep_m:`,
    `            ep_num, stp_num, spd_str = ep_m.group(1), ep_m.group(2), ep_m.group(3)`,
    `            low_str = ep_m.group(4) or "-"`,
    `            key = (ep_num, stp_num)`,
    `            if key not in seen_epochs:`,
    `                seen_epochs.add(key)`,
    `                epoch_rows.append((spd_str, stp_num, ep_num, low_str))`,
    `                live_mid_epoch = ""`,
    `                render_monitor("Training")`,
    `            continue`,
    `        tq_m = TQDM_RE.search(line)`,
    `        if tq_m:`,
    `            now_ts = time.time()`,
    `            if now_ts - last_tq_render >= 2.0 or tq_m.group(1) == "100":`,
    `                last_tq_render = now_ts`,
    `                live_mid_epoch = f"{tq_m.group(1)}% ({tq_m.group(2)}/{tq_m.group(3)}) · {tq_m.group(4)} · {tq_m.group(5)}"`,
    `                render_monitor("Training")`,
    `            continue`,
    `        if NOISE_RE.search(line) or "%|" in line or "it/s]" in line or "s/it]" in line:`,
    `            continue`,
    `        if tb_lines_seen > 0:`,
    `            errors.append(line)`,
    `            tb_lines_seen = 0 if re.match(r"^[A-Za-z_]*(Error|Exception):", line) else (tb_lines_seen - 1)`,
    `            render_monitor("Error")`,
    `            continue`,
    `        if ERROR_START_RE.search(line):`,
    `            errors.append(line)`,
    `            if "Traceback" in line:`,
    `                tb_lines_seen = 25`,
    `            render_monitor("Error")`,
    `            continue`,
    `        if MILESTONE_RE.search(line):`,
    `            if not milestones or milestones[-1] != line:`,
    `                milestones.append(line)`,
    `                render_monitor("Running")`,
    `    app_proc.wait()`,
    `finally:`,
    `    for p in reversed(tunnel_procs):`,
    `        if p.poll() is None: p.terminate()`,
    `    if app_proc is not None and app_proc.poll() is None:`,
    `        app_proc.terminate()`,
    `    for f in tunnel_logs:`,
    `        f.close()`,
    `    subprocess.run(["fuser", "-k", "6969/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)`,
  ];

  const notebook = {
    nbformat: 4,
    nbformat_minor: 4,
    metadata: {
      kernelspec: {
        display_name: 'Python 3',
        language: 'python',
        name: 'python3',
      },
      language_info: {
        name: 'python',
        version: '3.12.0',
      },
      kaggle: {
        accelerator: 'nvidiaTeslaT4',
        isGpuEnabled: true,
        isInternetEnabled: true,
        language: 'python',
        sourceType: 'notebook',
      },
    },
    cells: [
      {
        cell_type: 'markdown',
        metadata: {},
        source: [
          `## **${BRAND_NAME}**\n`,
          `Custom RVC voice model training environment built on Applio + Streamlit (${project.samplingRate} · ${project.vocoder} · ${project.pitchExtractor.toUpperCase()}).\n\n`,
          `- **Dataset Path**: \`${project.datasetPath || '/kaggle/input/datasets/your-username/dataset'}\`\n`,
          `- **Included Tabs**: ${Object.entries(config.activeTabs).filter(([, v]) => v).map(([k]) => k).join(', ')}\n`,
          `- **Tunnel & Auth**: ${config.tunnelMode} with ${config.randomAuthEnabled ? 'auto-generated 24-byte WebUI password' : 'standard access'}\n`,
        ],
      },
      {
        cell_type: 'markdown',
        metadata: {},
        source: ['## 1. Install Dependencies & Apply NoRegretsRVC Streamlit WebUI\n'],
      },
      {
        cell_type: 'code',
        execution_count: null,
        metadata: { trusted: true },
        outputs: [],
        source: installLines.map((l) => l + '\n'),
      },
      {
        cell_type: 'markdown',
        metadata: {},
        source: [
          '## 2. Launch Streamlit WebUI & Live Monitor\n',
        ],
      },
      {
        cell_type: 'code',
        execution_count: null,
        metadata: { trusted: true },
        outputs: [],
        source: startLines.map((l) => l + '\n'),
      },
    ],
  };

  return JSON.stringify(notebook, null, 2);
}

function toBase64Utf8(str: string): string {
  if (typeof Buffer !== 'undefined') {
    return Buffer.from(str, 'utf8').toString('base64');
  }
  const bytes = new TextEncoder().encode(str);
  let bin = '';
  for (let i = 0; i < bytes.length; i++) {
    bin += String.fromCharCode(bytes[i]);
  }
  return btoa(bin);
}

export function generateForkBundle(
  config: ForkConfig,
  project: VoiceModelProject,
  downloads: DownloadModelItem[]
): GeneratedBundleFile[] {
  return [
    {
      id: 'notebook',
      filename: `${BRAND_NAME}_Kaggle.ipynb`,
      zipPath: `assets/${BRAND_NAME}_Kaggle.ipynb`,
      language: 'json',
      description:
        'Kaggle-ready notebook launching the crimson NoRegretsRVC Streamlit WebUI with SHA-256 verified tunnels, randomized password, and dual-T4 Turbo setup.',
      content: generateKaggleNotebookJson(config, project, downloads),
    },
    {
      id: 'app-py',
      filename: 'streamlit_app.py',
      zipPath: 'streamlit_app.py',
      language: 'python',
      description:
        'NoRegretsRVC Streamlit WebUI registering Training, Models, Samples, Free Samples, Diagnostics, Voice Blender, TTS, Extra, and Settings.',
      content: generateAppPython(config, project),
    },
    {
      id: 'theme-py',
      filename: 'config.toml',
      zipPath: '.streamlit/config.toml',
      language: 'json',
      description:
        'Fixed crimson-red Streamlit theme configuration built specifically for NoRegretsRVC.',
      content: generateThemePython(),
    },
    {
      id: 'turbo-py',
      filename: 'noregrets_turbo.py',
      zipPath: 'noregrets_turbo.py',
      language: 'python',
      description:
        'Lossless Kaggle training speedup patch (adaptive RAM cache, fused AdamW, DataLoader tuning, cuDNN benchmark, DDP bucket view, async .pth writes).',
      content: generateTurboTrainerPython(),
    },
    {
      id: 'commands-sh',
      filename: 'run_commands.sh',
      zipPath: 'run_commands.sh',
      language: 'bash',
      description: `Generated Kaggle shell commands for "${sanitizeModelName(project.name)}" (${project.samplingRate} · ${project.vocoder}).`,
      content: `#!/usr/bin/env bash\nset -euo pipefail\n\n${buildTrainPipelineCommands(project)}\n`,
    },
  ];
}
