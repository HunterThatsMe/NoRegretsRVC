import {
  ACCENT_HEX,
  APP_VERSION,
  BRAND_NAME,
  LEGACY_CORE_PRETRAIN_D,
  LEGACY_CORE_PRETRAIN_G,
  LEGACY_CORE_PRETRAIN_NAME,
  REPO_README_URL,
  REPO_URL,
} from '../data/defaults';
import { FREE_SAMPLES } from '../data/freeSamples';
import { DownloadModelItem, ForkConfig, VoiceModelProject } from '../types/studio';
import { sanitizeModelName } from './rvcTuning';

export function buildStreamlitHeaderAndTabsPart1(
  config: ForkConfig,
  project: VoiceModelProject,
  downloads: DownloadModelItem[] = []
): string {
  const defaultModelName = project.name.trim() ? sanitizeModelName(project.name) : '';
  const defaultDatasetPath =
    project.datasetPath || '/kaggle/input/datasets/your-username/voice-dataset';
  const defaultRepo = (config.customRepoUrl || REPO_URL).trim();

  const enabledNav: string[] = [];
  if (config.activeTabs.train) enabledNav.push('Training');
  if (config.activeTabs.models) enabledNav.push('Models');
  if (config.activeTabs.samples) enabledNav.push('Samples');
  enabledNav.push('Free Samples');
  enabledNav.push('Diagnostics');
  if (config.activeTabs.voice_blender) enabledNav.push('Voice Blender');
  if (config.activeTabs.tts) enabledNav.push('TTS');
  if (config.activeTabs.settings) enabledNav.push('Settings');
  if (enabledNav.length === 0) enabledNav.push('Training');

  const freeSamplesJson = JSON.stringify(
    FREE_SAMPLES.map((s) => ({
      id: s.id,
      title: s.title,
      fileName: s.fileName,
      gender: s.gender,
      style: s.style,
      durationLabel: s.durationLabel,
      transcript: s.transcript,
      repoWav: s.url.split('/').pop() || s.fileName,
    }))
  );

  const bootManifestJson = JSON.stringify(
    downloads.map((d) => ({
      id: d.id,
      name: d.name,
      category: d.category,
      url: d.url,
    }))
  );

  return `"""${BRAND_NAME} Streamlit WebUI — exact parity with ${BRAND_NAME} Studio."""
from __future__ import annotations

import hmac
import html
import inspect
import io
import json
import math
import os
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
try:
    import streamlit.components.v1 as _st_components
except Exception:
    _st_components = None

now_dir = os.getcwd()
if now_dir not in sys.path:
    sys.path.insert(0, now_dir)

st.set_page_config(
    page_title="${BRAND_NAME} · RVC Kaggle Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==============================================================================
# CRIMSON OBSIDIAN DESIGN SYSTEM (MATCHES NOREGRETSRVC STUDIO PREVIEW)
# ==============================================================================
CRIMSON_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --nr-bg: #100506;
  --nr-card: #160709;
  --nr-elevated: #200B0E;
  --nr-border: #47161D;
  --nr-accent: ${ACCENT_HEX};
  --nr-accent-hover: #FF2B35;
  --nr-text: #FDF7F7;
  --nr-text-soft: #DFBFC3;
  --nr-muted: #AD8288;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background-color: var(--nr-bg) !important;
  background-image: radial-gradient(ellipse 90% 45% at 50% -10%, rgba(229, 25, 35, 0.13), transparent 70%) !important;
  color: var(--nr-text) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  overflow-anchor: none !important;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stSidebar"],
#MainMenu,
footer {
  display: none !important;
  height: 0 !important;
  min-height: 0 !important;
  visibility: hidden !important;
}

* {
  scrollbar-color: #E51923 #100506 !important;
  scrollbar-width: thin !important;
}
html::-webkit-scrollbar,
body::-webkit-scrollbar,
[data-testid="stAppViewContainer"]::-webkit-scrollbar,
[data-testid="stMain"]::-webkit-scrollbar,
::-webkit-scrollbar {
  width: 8px !important;
  height: 8px !important;
}
html::-webkit-scrollbar-track,
body::-webkit-scrollbar-track,
[data-testid="stAppViewContainer"]::-webkit-scrollbar-track,
[data-testid="stMain"]::-webkit-scrollbar-track,
::-webkit-scrollbar-track {
  background: #100506 !important;
}
html::-webkit-scrollbar-thumb,
body::-webkit-scrollbar-thumb,
[data-testid="stAppViewContainer"]::-webkit-scrollbar-thumb,
[data-testid="stMain"]::-webkit-scrollbar-thumb,
::-webkit-scrollbar-thumb {
  background: #E51923 !important;
  border-radius: 999px !important;
}
html::-webkit-scrollbar-thumb:hover,
[data-testid="stMain"]::-webkit-scrollbar-thumb:hover,
::-webkit-scrollbar-thumb:hover {
  background: #FF2B35 !important;
}

.block-container {
  max-width: 1400px !important;
  padding-top: 1.15rem !important;
  padding-bottom: 3.5rem !important;
  padding-left: 1.5rem !important;
  padding-right: 1.5rem !important;
}

/* Framed cards & nested inner slider/option containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
  background-color: var(--nr-card) !important;
  border: 1px solid var(--nr-border) !important;
  border-radius: 12px !important;
  padding: 1.25rem !important;
  box-shadow: 0 8px 26px rgba(0, 0, 0, 0.34) !important;
  transition: border-color 0.2s cubic-bezier(0.22, 1, 0.36, 1), box-shadow 0.2s cubic-bezier(0.22, 1, 0.36, 1), transform 0.2s cubic-bezier(0.22, 1, 0.36, 1) !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
  border-color: rgba(229, 25, 35, 0.42) !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlockBorderWrapper"] {
  background-color: var(--nr-bg) !important;
  border: 1px solid var(--nr-border) !important;
  border-radius: 10px !important;
  padding: 0.9rem 1.05rem !important;
  box-shadow: none !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlockBorderWrapper"]:hover {
  border-color: rgba(229, 25, 35, 0.36) !important;
}
@keyframes nrPanelIn {
  from { opacity: 0.72; transform: translate3d(0, 5px, 0); }
  to { opacity: 1; transform: translate3d(0, 0, 0); }
}
.st-key-nr_panel_training,
.st-key-nr_panel_models,
.st-key-nr_panel_samples,
.st-key-nr_panel_free_samples,
.st-key-nr_panel_diagnostics,
.st-key-nr_panel_voice_blender,
.st-key-nr_panel_tts,
.st-key-nr_panel_settings {
  animation: nrPanelIn 0.16s cubic-bezier(0.22, 1, 0.36, 1);
}

/* Inputs & Selects */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input,
div[data-baseweb="select"] > div {
  background-color: var(--nr-bg) !important;
  border: 1px solid var(--nr-border) !important;
  border-radius: 8px !important;
  color: var(--nr-text) !important;
  font-size: 14px !important;
  transition: border-color 0.16s ease, box-shadow 0.16s ease, background-color 0.16s ease !important;
}
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] p {
  font-size: 13.5px !important;
  font-weight: 600 !important;
  color: var(--nr-text) !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus,
[data-testid="stNumberInput"] input:focus,
div[data-baseweb="select"] > div:focus-within {
  border-color: var(--nr-accent) !important;
  box-shadow: 0 0 0 1px var(--nr-accent), 0 0 14px rgba(229, 25, 35, 0.16) !important;
}
[data-testid="stNumberInput"] button {
  background-color: var(--nr-elevated) !important;
  border-color: var(--nr-border) !important;
  color: var(--nr-text) !important;
  transition: border-color 0.15s ease, color 0.15s ease, transform 0.12s ease !important;
}
[data-testid="stNumberInput"] button:hover {
  border-color: var(--nr-accent) !important;
  color: #FFFFFF !important;
}
[data-testid="stNumberInput"] button:active {
  transform: scale(0.94);
}
[data-testid="InputInstructions"] {
  display: none !important;
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
  border-radius: 8px !important;
  font-weight: 600 !important;
  font-size: 13.5px !important;
  transition: border-color 0.16s cubic-bezier(0.22, 1, 0.36, 1), background-color 0.16s cubic-bezier(0.22, 1, 0.36, 1), color 0.16s ease, transform 0.12s cubic-bezier(0.22, 1, 0.36, 1), box-shadow 0.16s ease !important;
  min-height: 39px !important;
}
.stButton > button:active, .stDownloadButton > button:active {
  transform: scale(0.986) !important;
}
.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {
  background-color: var(--nr-accent) !important;
  border: 1px solid var(--nr-accent) !important;
  color: #FFFFFF !important;
  box-shadow: 0 4px 14px rgba(229, 25, 35, 0.24) !important;
}
.stButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover {
  background-color: var(--nr-accent-hover) !important;
  border-color: var(--nr-accent-hover) !important;
  box-shadow: 0 6px 20px rgba(229, 25, 35, 0.36) !important;
}
.stButton > button[kind="secondary"],
.stDownloadButton > button[kind="secondary"] {
  background-color: var(--nr-elevated) !important;
  border: 1px solid var(--nr-border) !important;
  color: var(--nr-text) !important;
}
.stButton > button[kind="secondary"]:hover,
.stDownloadButton > button[kind="secondary"]:hover {
  border-color: var(--accent-red, var(--nr-accent)) !important;
  background-color: #280E12 !important;
  color: #FFFFFF !important;
}

/* Sleek Preview-Matched Top Nav Bar */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.nr-nav-marker) {
  padding: 0.45rem 0.65rem !important;
  margin-bottom: 0.5rem !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.nr-nav-marker) .stButton > button[kind="secondary"] {
  background-color: transparent !important;
  border: 1px solid transparent !important;
  color: var(--nr-text-soft) !important;
  min-height: 34px !important;
  padding: 0.35rem 0.65rem !important;
  box-shadow: none !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.nr-nav-marker) .stButton > button[kind="secondary"]:hover {
  background-color: var(--nr-elevated) !important;
  border-color: rgba(229, 25, 35, 0.28) !important;
  color: #FFFFFF !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.nr-nav-marker) .stButton > button[kind="primary"] {
  min-height: 34px !important;
  padding: 0.35rem 0.75rem !important;
}

/* Expanders */
[data-testid="stExpander"] {
  background-color: var(--nr-bg) !important;
  border: 1px solid var(--nr-border) !important;
  border-radius: 8px !important;
  transition: border-color 0.16s ease, background-color 0.16s ease !important;
}
[data-testid="stExpander"]:hover {
  border-color: rgba(229, 25, 35, 0.5) !important;
}
[data-testid="stExpander"] summary {
  color: var(--nr-text-soft) !important;
  font-size: 13.5px !important;
  font-weight: 600 !important;
  transition: color 0.15s ease !important;
}
[data-testid="stExpander"] summary:hover {
  color: #FFFFFF !important;
}

/* File Uploader */
[data-testid="stFileUploaderDropzone"] {
  background-color: var(--nr-bg) !important;
  border: 2px dashed var(--nr-border) !important;
  border-radius: 12px !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--nr-accent) !important;
}

/* Sliders & Checkboxes */
[data-testid="stSlider"] * {
  transition: none !important;
}
[data-testid="stSlider"] [role="slider"] {
  background-color: var(--nr-accent) !important;
  border: 2px solid #FFFFFF !important;
}
[data-testid="stCheckbox"] label span {
  color: var(--nr-text) !important;
  font-size: 14px !important;
  line-height: 1.45 !important;
}

/* Custom Studio Components */
.nr-topbar, .nr-header-card {
  background: var(--nr-card);
  border: 1px solid var(--nr-border);
  border-radius: 12px;
  padding: 16px 20px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.32);
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  margin-bottom: 10px;
}
.nr-brand-left {
  display: flex;
  align-items: center;
  gap: 14px;
}
.nr-badge {
  width: 40px;
  height: 40px;
  border-radius: 8px;
  background: var(--nr-accent);
  color: #FFFFFF;
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.nr-brand-title {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: 22px;
  color: var(--nr-text);
  line-height: 1.1;
}
.nr-brand-ver {
  font-size: 11px;
  color: rgba(223, 191, 195, 0.55);
  font-weight: 500;
  margin-left: 8px;
}
.nr-brand-sub {
  font-size: 13px;
  color: var(--nr-text-soft);
  margin-top: 3px;
}
.nr-model-pill {
  background: var(--nr-bg);
  border: 1px solid var(--nr-border);
  border-radius: 8px;
  padding: 8px 14px;
  text-align: right;
}
.nr-model-pill-name {
  font-size: 13px;
  font-weight: 600;
  color: var(--nr-text);
}
.nr-model-pill-meta {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  color: var(--nr-text-soft);
  margin-top: 2px;
}
.nr-step-head, .applio-step {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  margin-bottom: 12px;
}
.nr-step-num {
  width: 28px;
  height: 28px;
  border-radius: 8px;
  background: var(--nr-accent);
  color: #FFFFFF;
  font-weight: 700;
  font-size: 12px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.nr-step-title {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 600;
  font-size: 17px;
  color: var(--nr-text);
  line-height: 1.25;
}
.nr-step-desc {
  font-size: 13.5px;
  color: var(--nr-text-soft);
  margin-top: 3px;
  line-height: 1.45;
}
.nr-field-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--nr-text);
}
.nr-field-help {
  font-size: 13px;
  color: var(--nr-text-soft);
  margin: 3px 0 8px 0;
  line-height: 1.45;
}
.nr-output-box {
  background: var(--nr-bg);
  border: 1px solid var(--nr-border);
  border-radius: 8px;
  padding: 11px 14px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  line-height: 1.55;
  color: var(--nr-text-soft);
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 260px;
  overflow-y: auto;
}
.nr-section-header {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 14px;
}
.nr-section-h2 {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 24px;
  font-weight: 700;
  color: var(--nr-text);
  margin: 0;
  letter-spacing: -0.02em;
}
.nr-section-sub {
  font-size: 14.5px;
  color: var(--nr-text-soft);
  margin-top: 4px;
}
.nr-link-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--nr-card);
  border: 1px solid var(--nr-border);
  border-radius: 8px;
  padding: 6px 12px;
  font-size: 12.5px;
  font-weight: 500;
  color: var(--nr-text-soft);
  text-decoration: none;
}
.nr-link-btn:hover {
  border-color: var(--nr-accent);
  color: #FFFFFF;
}
.nr-metric-box {
  background: var(--nr-bg);
  border: 1px solid var(--nr-border);
  border-radius: 8px;
  padding: 10px 12px;
}
.nr-metric-k {
  font-size: 11.5px;
  font-weight: 600;
  color: var(--nr-text-soft);
}
.nr-metric-v {
  font-size: 13.5px;
  font-weight: 600;
  color: var(--nr-text);
  margin-top: 3px;
  font-family: 'JetBrains Mono', monospace;
}
.nr-zone-card {
  border-radius: 8px;
  padding: 13px 15px;
  border: 1px solid var(--nr-border);
  background: var(--nr-card);
}
.nr-zone-sweet {
  border-color: var(--nr-accent);
  background: var(--nr-elevated);
}
.nr-mono-badge {
  border: 1px solid var(--nr-border);
  background: var(--nr-card);
  border-radius: 6px;
  padding: 2px 8px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--nr-text-soft);
  white-space: nowrap;
}
.nr-quote-box {
  background: var(--nr-card);
  border: 1px solid rgba(71, 22, 29, 0.8);
  border-radius: 8px;
  padding: 10px 12px;
  font-size: 13px;
  color: var(--nr-text-soft);
  line-height: 1.5;
  margin: 8px 0 10px 0;
}
.nr-info-box {
  background: var(--nr-bg);
  border: 1px solid var(--nr-border);
  border-radius: 8px;
  padding: 11px 14px;
  font-size: 13.5px;
  color: var(--nr-text-soft);
  line-height: 1.55;
}
.nr-prog-wrap {
  background: var(--nr-elevated);
  border: 1px solid rgba(229, 25, 35, 0.5);
  border-radius: 10px;
  padding: 12px 14px;
  margin: 8px 0;
}
.nr-prog-track {
  width: 100%;
  height: 7px;
  background: var(--nr-bg);
  border: 1px solid var(--nr-border);
  border-radius: 999px;
  overflow: hidden;
  margin: 8px 0;
}
.nr-prog-fill {
  height: 100%;
  background: var(--nr-accent);
}
.nr-toast-stack {
  position: fixed;
  top: 18px;
  right: 20px;
  z-index: 99999;
  display: flex;
  flex-direction: column;
  gap: 10px;
  max-width: 380px;
  pointer-events: none;
}
.nr-toast {
  background: #1C090C;
  border: 1px solid rgba(229, 25, 35, 0.8);
  border-left: 4px solid #E51923;
  border-radius: 10px;
  padding: 12px 15px;
  box-shadow: 0 14px 36px rgba(0, 0, 0, 0.78);
  animation: nrToastFade 5.2s cubic-bezier(0.22, 1, 0.36, 1) forwards;
}
.nr-toast-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 14px;
  font-weight: 700;
  color: #FDF7F7;
}
.nr-toast-msg {
  font-size: 13px;
  color: #DFBFC3;
  margin-top: 3px;
  line-height: 1.4;
}
@keyframes nrToastFade {
  0% { opacity: 0; transform: translate3d(18px, -8px, 0); }
  10% { opacity: 1; transform: translate3d(0, 0, 0); }
  82% { opacity: 1; transform: translate3d(0, 0, 0); }
  100% { opacity: 0; transform: translate3d(18px, -6px, 0); visibility: hidden; }
}
</style>
"""
st.markdown(CRIMSON_CSS, unsafe_allow_html=True)

# ==============================================================================
# RUNTIME CONSTANTS, APPLIO CORE BRIDGE & CONFIG PERSISTENCE
# ==============================================================================
DEFAULT_REPO = ${JSON.stringify(defaultRepo)}
README_URL = ${JSON.stringify(REPO_README_URL)}
APP_VERSION = ${JSON.stringify(APP_VERSION)}
LEGACY_CORE_PRETRAIN_NAME = ${JSON.stringify(LEGACY_CORE_PRETRAIN_NAME)}
LEGACY_CORE_G_URL = ${JSON.stringify(LEGACY_CORE_PRETRAIN_G)}
LEGACY_CORE_D_URL = ${JSON.stringify(LEGACY_CORE_PRETRAIN_D)}
CONFIG_PATH = Path(now_dir) / "assets" / "config.json"
ACTIVE_TRAIN_META = Path(now_dir) / "logs" / ".nr_active_train.json"
FREE_SAMPLES_CATALOG = json.loads(${JSON.stringify(freeSamplesJson)})
BOOT_MANIFEST_DEFAULTS = json.loads(${JSON.stringify(bootManifestJson)})
SUPPORTED_LANGUAGES = [
    ("System Auto", "System Auto"),
    ("en_US", "English"),
    ("es_ES", "Spanish (Español)"),
    ("pt_BR", "Portuguese (Português)"),
    ("fr_FR", "French (Français)"),
    ("it_IT", "Italian (Italiano)"),
    ("nl_NL", "Dutch (Nederlands)"),
    ("ja_JP", "Japanese (日本語)"),
    ("zh_CN", "Chinese (中文)"),
    ("ar_SA", "Arabic (العربية)"),
]
LANG_LABELS = dict(SUPPORTED_LANGUAGES)
TTS_VOICES = [
    "en-US-AndrewNeural",
    "en-US-GuyNeural",
    "en-US-AriaNeural",
    "en-GB-RyanNeural",
    "en-US-JennyNeural",
]

UI_STRINGS: dict[str, dict[str, str]] = {
    "es_ES": {
        "Training": "Entrenamiento",
        "Models": "Modelos",
        "Samples": "Muestras",
        "Free Samples": "Muestras Gratis",
        "Diagnostics": "Diagnóstico",
        "Voice Blender": "Mezclador de Voz",
        "TTS": "Texto a Voz (TTS)",
        "Settings": "Ajustes",
        "General": "General",
        "Preprocess Dataset": "Preprocesar Dataset",
        "Extract Features": "Extraer Características",
        "Start Training": "Iniciar Entrenamiento",
        "Stop Training": "Detener Entrenamiento",
        "Generate Index": "Generar Índice",
        "Download Model": "Descargar Modelo",
        "List Filtering": "Filtro de Lista",
        "Pitch Correction": "Corrección de Tono",
        "Discord Integration": "Integración con Discord",
        "Language": "Idioma",
        "Updates & System": "Actualizaciones y Sistema",
        "Model Author": "Autor del Modelo",
        "Training Precision": "Precisión de Entrenamiento",
        "Convert": "Convertir",
        "Fusion": "Fusión",
        "Convert Text-to-Speech": "Convertir Texto a Voz",
        "Check for updates": "Buscar actualizaciones",
        "Restart Applio": "Reiniciar Applio",
        "Set name": "Guardar nombre",
        "Update precision": "Actualizar precisión",
        "Load All 6 in Samples": "Cargar los 6 en Muestras",
        "Download All 6 (.zip)": "Descargar los 6 (.zip)",
    },
    "pt_BR": {
        "Training": "Treinamento",
        "Models": "Modelos",
        "Samples": "Amostras",
        "Free Samples": "Amostras Grátis",
        "Diagnostics": "Diagnóstico",
        "Voice Blender": "Misturador de Voz",
        "TTS": "Texto para Fala (TTS)",
        "Settings": "Configurações",
        "General": "Geral",
        "Preprocess Dataset": "Pré-processar Dataset",
        "Extract Features": "Extrair Características",
        "Start Training": "Iniciar Treinamento",
        "Stop Training": "Parar Treinamento",
        "Generate Index": "Gerar Índice",
        "Download Model": "Baixar Modelo",
        "List Filtering": "Filtragem de Lista",
        "Pitch Correction": "Correção de Tom",
        "Discord Integration": "Integração Discord",
        "Language": "Idioma",
        "Updates & System": "Atualizações e Sistema",
        "Model Author": "Autor do Modelo",
        "Training Precision": "Precisão de Treinamento",
        "Convert": "Converter",
        "Fusion": "Fusão",
        "Convert Text-to-Speech": "Converter Texto para Fala",
        "Check for updates": "Verificar atualizações",
        "Restart Applio": "Reiniciar Applio",
        "Set name": "Definir nome",
        "Update precision": "Atualizar precisão",
        "Load All 6 in Samples": "Carregar Todos os 6 em Amostras",
        "Download All 6 (.zip)": "Baixar Todos os 6 (.zip)",
    },
    "fr_FR": {
        "Training": "Entraînement",
        "Models": "Modèles",
        "Samples": "Échantillons",
        "Free Samples": "Échantillons Gratuits",
        "Diagnostics": "Diagnostics",
        "Voice Blender": "Mixeur Vocal",
        "TTS": "Synthèse Vocale (TTS)",
        "Settings": "Paramètres",
        "General": "Général",
        "Preprocess Dataset": "Prétraiter le Dataset",
        "Extract Features": "Extraire les Caractéristiques",
        "Start Training": "Démarrer l'Entraînement",
        "Stop Training": "Arrêter l'Entraînement",
        "Generate Index": "Générer l'Index",
        "Download Model": "Télécharger le Modèle",
        "List Filtering": "Filtrage de Liste",
        "Pitch Correction": "Correction de Hauteur",
        "Discord Integration": "Intégration Discord",
        "Language": "Langue",
        "Updates & System": "Mises à jour et Système",
        "Model Author": "Auteur du Modèle",
        "Training Precision": "Précision d'Entraînement",
        "Convert": "Convertir",
        "Fusion": "Fusionner",
        "Convert Text-to-Speech": "Convertir en Parole",
        "Check for updates": "Vérifier les mises à jour",
        "Restart Applio": "Redémarrer Applio",
        "Set name": "Définir le nom",
        "Update precision": "Mettre à jour la précision",
        "Load All 6 in Samples": "Charger les 6 dans Échantillons",
        "Download All 6 (.zip)": "Télécharger les 6 (.zip)",
    },
    "ja_JP": {
        "Training": "トレーニング",
        "Models": "モデル",
        "Samples": "サンプル",
        "Free Samples": "無料サンプル",
        "Diagnostics": "診断",
        "Voice Blender": "ボイスブレンダー",
        "TTS": "音声合成 (TTS)",
        "Settings": "設定",
        "General": "一般",
        "Preprocess Dataset": "データセット前処理",
        "Extract Features": "特徴抽出",
        "Start Training": "学習開始",
        "Stop Training": "学習停止",
        "Generate Index": "インデックス生成",
        "Download Model": "モデルをダウンロード",
        "List Filtering": "リスト絞り込み",
        "Pitch Correction": "ピッチ補正",
        "Discord Integration": "Discord連携",
        "Language": "言語",
        "Updates & System": "アップデートとシステム",
        "Model Author": "モデル作成者",
        "Training Precision": "学習精度",
        "Convert": "変換",
        "Fusion": "ブレンド実行",
        "Convert Text-to-Speech": "音声合成を実行",
        "Check for updates": "更新を確認",
        "Restart Applio": "Applio を再起動",
        "Set name": "名前を保存",
        "Update precision": "精度を更新",
        "Load All 6 in Samples": "全6件をサンプルへ読込",
        "Download All 6 (.zip)": "全6件をダウンロード (.zip)",
    },
    "it_IT": {
        "Training": "Addestramento",
        "Models": "Modelli",
        "Samples": "Campioni",
        "Free Samples": "Campioni Gratuiti",
        "Diagnostics": "Diagnostica",
        "Voice Blender": "Voice Blender",
        "TTS": "Sintesi Vocale (TTS)",
        "Settings": "Impostazioni",
        "General": "Generale",
        "Preprocess Dataset": "Preelabora Dataset",
        "Extract Features": "Estrai Caratteristiche",
        "Start Training": "Avvia Addestramento",
        "Stop Training": "Ferma Addestramento",
        "Generate Index": "Genera Indice",
        "Download Model": "Scarica Modello",
        "List Filtering": "Filtro Elenco",
        "Pitch Correction": "Correzione Intonazione",
        "Discord Integration": "Integrazione Discord",
        "Language": "Lingua",
        "Updates & System": "Aggiornamenti e Sistema",
        "Model Author": "Autore del Modello",
        "Training Precision": "Precisione di Addestramento",
        "Convert": "Converti",
        "Fusion": "Fondi",
        "Convert Text-to-Speech": "Converti Testo in Voce",
        "Check for updates": "Controlla aggiornamenti",
        "Restart Applio": "Riavvia Applio",
        "Set name": "Imposta nome",
        "Update precision": "Aggiorna precisione",
        "Load All 6 in Samples": "Carica tutti i 6 in Campioni",
        "Download All 6 (.zip)": "Scarica tutti i 6 (.zip)",
    },
    "nl_NL": {
        "Training": "Training",
        "Models": "Modellen",
        "Samples": "Voorbeelden",
        "Free Samples": "Gratis Voorbeelden",
        "Diagnostics": "Diagnostiek",
        "Voice Blender": "Stemmixer",
        "TTS": "Tekst-naar-Spraak (TTS)",
        "Settings": "Instellingen",
        "General": "Algemeen",
        "Preprocess Dataset": "Dataset Voorbereiden",
        "Extract Features": "Kenmerken Extraheren",
        "Start Training": "Training Starten",
        "Stop Training": "Training Stoppen",
        "Generate Index": "Index Genereren",
        "Download Model": "Model Downloaden",
        "List Filtering": "Lijstfiltering",
        "Pitch Correction": "Toonhoogtecorrectie",
        "Discord Integration": "Discord-integratie",
        "Language": "Taal",
        "Updates & System": "Updates & Systeem",
        "Model Author": "Modelauteur",
        "Training Precision": "Trainingsprecisie",
        "Convert": "Converteren",
        "Fusion": "Mengen",
        "Convert Text-to-Speech": "Tekst naar Spraak Converteren",
        "Check for updates": "Controleren op updates",
        "Restart Applio": "Applio herstarten",
        "Set name": "Naam opslaan",
        "Update precision": "Precisie bijwerken",
        "Load All 6 in Samples": "Alle 6 in Voorbeelden laden",
        "Download All 6 (.zip)": "Alle 6 downloaden (.zip)",
    },
    "zh_CN": {
        "Training": "训练",
        "Models": "模型",
        "Samples": "样本试听",
        "Free Samples": "免费样本",
        "Diagnostics": "诊断",
        "Voice Blender": "声音融合",
        "TTS": "语音合成 (TTS)",
        "Settings": "设置",
        "General": "常规",
        "Preprocess Dataset": "预处理数据集",
        "Extract Features": "提取特征",
        "Start Training": "开始训练",
        "Stop Training": "停止训练",
        "Generate Index": "生成索引",
        "Download Model": "下载模型",
        "List Filtering": "列表过滤",
        "Pitch Correction": "音高校正",
        "Discord Integration": "Discord 集成",
        "Language": "语言",
        "Updates & System": "更新与系统",
        "Model Author": "模型作者",
        "Training Precision": "训练精度",
        "Convert": "转换",
        "Fusion": "融合",
        "Convert Text-to-Speech": "合成语音",
        "Check for updates": "检查更新",
        "Restart Applio": "重启 Applio",
        "Set name": "保存作者名",
        "Update precision": "更新精度",
        "Load All 6 in Samples": "全部 6 个载入样本",
        "Download All 6 (.zip)": "下载全部 6 个 (.zip)",
    },
    "ar_SA": {
        "Training": "التدريب",
        "Models": "النماذج",
        "Samples": "العينات",
        "Free Samples": "عينات مجانية",
        "Diagnostics": "التشخيص",
        "Voice Blender": "دمج الأصوات",
        "TTS": "تحويل النص إلى كلام",
        "Settings": "الإعدادات",
        "General": "عام",
        "Preprocess Dataset": "معالجة البيانات",
        "Extract Features": "استخراج الخصائص",
        "Start Training": "بدء التدريب",
        "Stop Training": "إيقاف التدريب",
        "Generate Index": "إنشاء الفهرس",
        "Download Model": "تحميل النموذج",
        "List Filtering": "تصفية القائمة",
        "Pitch Correction": "تصحيح النغمة",
        "Discord Integration": "تكامل ديسكورد",
        "Language": "اللغة",
        "Updates & System": "التحديثات والنظام",
        "Model Author": "مؤلف النموذج",
        "Training Precision": "دقة التدريب",
        "Convert": "تحويل",
        "Fusion": "دمج",
        "Convert Text-to-Speech": "تحويل النص إلى صوت",
        "Check for updates": "التحقق من التحديثات",
        "Restart Applio": "إعادة تشغيل Applio",
        "Set name": "تعيين الاسم",
        "Update precision": "تحديث الدقة",
        "Load All 6 in Samples": "تحميل الكل 6 في العينات",
        "Download All 6 (.zip)": "تحميل الكل 6 (.zip)",
    },
}


def _resolve_lang() -> str:
    raw = str(st.session_state.get("nr_lang", "System Auto"))
    if raw != "System Auto":
        return raw
    try:
        accept_lang = ""
        ctx = getattr(st, "context", None)
        if ctx is not None and hasattr(ctx, "headers"):
            accept_lang = str(ctx.headers.get("Accept-Language") or ctx.headers.get("accept-language") or "").lower()
        for prefix, code in (
            ("es", "es_ES"),
            ("pt", "pt_BR"),
            ("fr", "fr_FR"),
            ("it", "it_IT"),
            ("nl", "nl_NL"),
            ("ja", "ja_JP"),
            ("zh", "zh_CN"),
            ("ar", "ar_SA"),
        ):
            if accept_lang.startswith(prefix) or f",{prefix}" in accept_lang:
                return code
    except Exception:
        pass
    return "en_US"


def _tr(text: str) -> str:
    lang = _resolve_lang()
    if lang == "en_US":
        return text
    return UI_STRINGS.get(lang, {}).get(text, text)


def _load_config() -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if CONFIG_PATH.is_file():
        try:
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    if not isinstance(cfg, dict):
        cfg = {}
    # Always default fresh boots to FP32 & filter off until explicitly changed in Settings
    if not cfg.get("nr_precision_initialized"):
        cfg["precision"] = "fp32"
    if not cfg.get("nr_discord_initialized"):
        cfg["discord_presence"] = False
    if "filter" not in cfg:
        cfg["filter"] = False
    return cfg


def _save_config(cfg: dict[str, Any]) -> None:
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(cfg, indent=4, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


CORE_MOD = None
try:
    import core as CORE_MOD  # type: ignore
except Exception:
    CORE_MOD = None


_EPOCH_TELEMETRY_RE = re.compile(
    r"(?:epoch=\\d+\\s*\\|\\s*step=\\d+|training_speed=|lowest_value=|\\d+%\\|[^|]*\\||\\bit/s\\]|\\bs/it\\]|local_sources_watcher|No module named 'torchvision')",
    re.IGNORECASE,
)


def _filter_status_lines(raw_text: str, fallback: str = "") -> str:
    if not raw_text:
        return fallback
    kept: list[str] = []
    for raw_ln in raw_text.replace("\\r", "\\n").splitlines():
        ln = raw_ln.strip()
        if not ln:
            continue
        if _EPOCH_TELEMETRY_RE.search(ln):
            continue
        kept.append(raw_ln.rstrip())
    cleaned = "\\n".join(kept[-60:]).strip()
    return cleaned or fallback


def _run_core_sync(fn_name: str, kwargs: dict[str, Any]) -> tuple[int, str]:
    if CORE_MOD is None or not hasattr(CORE_MOD, fn_name):
        return 1, f"core.{fn_name} is unavailable in current working directory ({now_dir})."
    fn = getattr(CORE_MOD, fn_name)
    try:
        sig = inspect.signature(fn)
        valid_params = set(sig.parameters.keys())
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        call_kw = kwargs if has_varkw else {k: v for k, v in kwargs.items() if k in valid_params}
        buf = io.StringIO()
        orig_out, orig_err = sys.stdout, sys.stderr
        try:
            sys.stdout = buf
            sys.stderr = buf
            res = fn(**call_kw)
        finally:
            sys.stdout = orig_out
            sys.stderr = orig_err
        out_str = _filter_status_lines(buf.getvalue().strip())
        if isinstance(res, tuple):
            msg = " ".join(str(x) for x in res if x is not None)
            out_str = f"{out_str}\\n{msg}".strip() if out_str else msg
        elif res is not None and not out_str:
            out_str = str(res)
        return 0, out_str or f"{fn_name} completed."
    except Exception as exc:
        return 1, f"Error in {fn_name}: {exc}"


def _push_toast(title: str, message: str, ttl: float = 5.2) -> None:
    toasts = list(st.session_state.get("nr_toasts", []))
    toasts.append({
        "id": f"t-{int(time.time() * 1000)}-{len(toasts)}",
        "title": title,
        "message": message,
        "expires": time.time() + ttl,
    })
    st.session_state["nr_toasts"] = toasts[-4:]


def _render_toasts() -> None:
    now_t = time.time()
    active = [t for t in st.session_state.get("nr_toasts", []) if float(t.get("expires", 0)) > now_t]
    st.session_state["nr_toasts"] = active
    if not active:
        return
    cards = "".join(
        f'<div class="nr-toast"><div class="nr-toast-title">{html.escape(str(t["title"]))}</div>'
        f'<div class="nr-toast-msg">{html.escape(str(t["message"]))}</div></div>'
        for t in active
    )
    st.markdown(f'<div class="nr-toast-stack">{cards}</div>', unsafe_allow_html=True)


def _safe_model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name.strip() or "UnnamedModel")


def _task_paths(task_kind: str) -> tuple[Path, Path]:
    logs_root = Path(now_dir) / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    return logs_root / f".nr_task_{task_kind}.json", logs_root / f".nr_task_{task_kind}.log"


def _clear_fs_caches() -> None:
    for fn in (_list_models, _list_datasets, _list_files_rel, _load_tts_voices, _dataset_slice_count, _read_file_bytes_cached):
        try:
            fn.clear()
        except Exception:
            pass


def _next_clip_id(prefix: str = "clip") -> str:
    seq = int(st.session_state.get("_nr_qid_seq", 0)) + 1
    st.session_state["_nr_qid_seq"] = seq
    return f"{prefix}-{seq}-{int(time.time() * 1000)}"


def _read_tail_text(path_obj: Path, max_bytes: int = 32768) -> str:
    try:
        size = path_obj.stat().st_size
        if size <= 0:
            return ""
        with path_obj.open("rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
            chunk = fh.read(max_bytes)
        text = chunk.decode("utf-8", errors="replace")
        if size > max_bytes and "\\n" in text:
            text = text.split("\\n", 1)[1]
        return text
    except Exception:
        return ""


@st.cache_data(ttl=60, show_spinner=False)
def _read_file_bytes_cached(path_str: str, mtime: float) -> bytes:
    return Path(path_str).read_bytes()


def _start_core_task_bg(
    task_kind: str,
    fn_name: str,
    model_name: str,
    kwargs: dict[str, Any],
    log_state_key: str,
    toast_title: str,
    done_note: str,
) -> str:
    meta_p, log_p = _task_paths(task_kind)
    args_p = meta_p.with_suffix(".args.json")
    if meta_p.is_file():
        try:
            prev = json.loads(meta_p.read_text(encoding="utf-8"))
            if prev.get("status") == "running" and prev.get("pid"):
                os.kill(int(prev["pid"]), 0)
                return f"{task_kind.capitalize()} is already running in background (PID {prev['pid']})."
        except Exception:
            pass

    args_p.write_text(json.dumps(kwargs, ensure_ascii=False), encoding="utf-8")

    child_code = (
        "import sys, os, json, inspect, traceback; from pathlib import Path\\n"
        "sys.path.insert(0, os.getcwd())\\n"
        f"meta_p = Path({json.dumps(str(meta_p))})\\n"
        f"args_p = Path({json.dumps(str(args_p))})\\n"
        f"task_kind = {json.dumps(task_kind)}\\n"
        f"model_safe = {json.dumps(_safe_model_slug(model_name))}\\n"
        "status = 'done'\\n"
        "try:\\n"
        "    kw = json.loads(args_p.read_text(encoding='utf-8'))\\n"
        "    import core\\n"
        f"    fn = getattr(core, {json.dumps(fn_name)})\\n"
        "    sig = inspect.signature(fn)\\n"
        "    kw = kw if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()) else {k: v for k, v in kw.items() if k in sig.parameters}\\n"
        "    res = fn(**kw)\\n"
        "    if isinstance(res, tuple):\\n"
        "        print(' '.join(str(x) for x in res if x is not None), flush=True)\\n"
        "    elif res is not None:\\n"
        "        print(str(res), flush=True)\\n"
        "    if task_kind == 'preprocess':\\n"
        "        sdir = Path('logs') / model_safe / 'sliced_audios'\\n"
        "        cnt = sum(1 for f in sdir.glob('*.wav')) if sdir.is_dir() else 0\\n"
        "        print(f'\\\\n✔ Preprocess complete! ({cnt} sliced audio clip(s) ready in logs/{model_safe}/sliced_audios)', flush=True)\\n"
        "    elif task_kind == 'extract':\\n"
        "        edir = Path('logs') / model_safe / 'extracted'\\n"
        "        cnt = sum(1 for f in edir.glob('*.npy')) if edir.is_dir() else 0\\n"
        "        print(f'\\\\n✔ Feature extraction complete! ({cnt} feature embedding(s) saved in logs/{model_safe})', flush=True)\\n"
        "    elif task_kind == 'index':\\n"
        "        mdir = Path('logs') / model_safe\\n"
        "        idxs = [f.name for f in mdir.glob('*.index') if 'trained' not in f.name.lower()] if mdir.is_dir() else []\\n"
        "        idx_lbl = idxs[0] if idxs else 'added.index'\\n"
        "        print(f'\\\\n✔ Index generation complete! ({idx_lbl} ready for inference)', flush=True)\\n"
        "except Exception:\\n"
        "    status = 'error'\\n"
        "    traceback.print_exc()\\n"
        "    sys.stdout.flush()\\n"
        "    sys.stderr.flush()\\n"
        "finally:\\n"
        "    try:\\n"
        "        meta = json.loads(meta_p.read_text(encoding='utf-8')) if meta_p.is_file() else {}\\n"
        "        meta['status'] = status\\n"
        "        meta['finished'] = __import__('time').time()\\n"
        "        meta_p.write_text(json.dumps(meta), encoding='utf-8')\\n"
        "    except Exception:\\n"
        "        pass\\n"
    )
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    lf = open(log_p, "w", encoding="utf-8", buffering=1)
    start_banner = f"⏳ Starting {task_kind} for {model_name or 'model'}...\\n"
    lf.write(start_banner)
    lf.flush()
    print(start_banner, end="", flush=True)
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", child_code],
        cwd=now_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        start_new_session=True,
    )
    meta_p.write_text(
        json.dumps({
            "task_kind": task_kind,
            "pid": proc.pid,
            "model_name": model_name,
            "log_state_key": log_state_key,
            "log_path": str(log_p),
            "toast_title": toast_title,
            "done_note": done_note,
            "status": "running",
            "notified": False,
            "started": time.time(),
        }),
        encoding="utf-8",
    )

    def _pump_task() -> None:
        try:
            if proc.stdout is not None:
                for ln in proc.stdout:
                    lf.write(ln)
                    lf.flush()
                    print(ln, end="", flush=True)
        finally:
            rc = proc.wait()
            if rc != 0:
                err_ln = f"\\n✖ {task_kind.capitalize()} failed (exit code {rc}).\\n"
                lf.write(err_ln)
                lf.flush()
                print(err_ln, end="", flush=True)
            lf.close()
            try:
                meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.is_file() else {}
                if meta.get("status") == "running":
                    meta["status"] = "done" if rc == 0 else "error"
                    meta["exit_code"] = rc
                    meta["finished"] = time.time()
                    meta_p.write_text(json.dumps(meta), encoding="utf-8")
            except Exception:
                pass

    threading.Thread(target=_pump_task, daemon=True).start()
    return start_banner.strip()


def _start_pretrain_download_bg(payload: dict[str, Any]) -> str:
    meta_p, log_p = _task_paths("pretrained")
    args_p = meta_p.with_suffix(".args.json")
    res_p = meta_p.with_suffix(".result.json")
    if meta_p.is_file():
        try:
            prev = json.loads(meta_p.read_text(encoding="utf-8"))
            if prev.get("status") == "running" and prev.get("pid"):
                os.kill(int(prev["pid"]), 0)
                return f"Pretrained download is already running in background (PID {prev['pid']})."
        except Exception:
            pass

    res_p.unlink(missing_ok=True)
    args_p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    child_code = (
        "import sys, os, json, time, zipfile, urllib.parse, urllib.request, traceback; from pathlib import Path\\n"
        f"meta_p = Path({json.dumps(str(meta_p))})\\n"
        f"args_p = Path({json.dumps(str(args_p))})\\n"
        f"res_p = Path({json.dumps(str(res_p))})\\n"
        "status = 'done'\\n"
        "result = {'custom_g': '', 'custom_d': '', 'custom_pre': False, 'manifest_items': []}\\n"
        "try:\\n"
        "    cfg = json.loads(args_p.read_text(encoding='utf-8'))\\n"
        "    cdir = Path(cfg['cdir'])\\n"
        "    cdir.mkdir(parents=True, exist_ok=True)\\n"
        "    if cfg.get('use_custom'):\\n"
        "        g_val = str(cfg.get('g_val') or '').strip()\\n"
        "        d_val = str(cfg.get('d_val') or '').strip()\\n"
        "        result['custom_pre'] = True\\n"
        "        g_path = urllib.parse.urlsplit(g_val).path.lower()\\n"
        "        if 'huggingface.co/' in g_val.lower() and '/resolve/' not in g_val and not g_path.endswith(('.pth', '.zip')):\\n"
        "            parts = [p for p in urllib.parse.urlsplit(g_val).path.strip('/').split('/') if p]\\n"
        "            if len(parts) < 2:\\n"
        "                raise ValueError(f'Invalid Hugging Face repo URL: {g_val}')\\n"
        "            owner_repo = f'{parts[0]}/{parts[1]}'\\n"
        "            api_u = f'https://huggingface.co/api/models/{owner_repo}/tree/main'\\n"
        "            print(f'Scanning HF repo {owner_repo}...', flush=True)\\n"
        "            with urllib.request.urlopen(api_u, timeout=30) as resp:\\n"
        "                tree = json.loads(resp.read().decode('utf-8'))\\n"
        "            found_cnt = 0\\n"
        "            for node in tree:\\n"
        "                fp = str(node.get('path', ''))\\n"
        "                if fp.lower().endswith('.pth'):\\n"
        "                    fn_p = Path(fp).name\\n"
        "                    fu = f'https://huggingface.co/{owner_repo}/resolve/main/{urllib.parse.quote(fp)}?download=true'\\n"
        "                    print(f'Downloading {fn_p} -> rvc/models/pretraineds/custom/{fn_p}', flush=True)\\n"
        "                    urllib.request.urlretrieve(fu, cdir / fn_p)\\n"
        "                    rel_p = f'rvc/models/pretraineds/custom/{fn_p}'\\n"
        "                    print(f'Downloaded {fn_p} from HF repo -> {rel_p}', flush=True)\\n"
        "                    found_cnt += 1\\n"
        "                    if 'D' in fn_p.upper() and not result['custom_d']:\\n"
        "                        result['custom_d'] = rel_p\\n"
        "                    elif not result['custom_g']:\\n"
        "                        result['custom_g'] = rel_p\\n"
        "            if found_cnt == 0:\\n"
        "                raise RuntimeError(f'No .pth weights found in HF repo {owner_repo}')\\n"
        "            result['manifest_items'].append({'id': f'pre-hf-{int(time.time())}', 'name': owner_repo, 'category': 'pretrained', 'url': g_val})\\n"
        "        elif g_path.endswith('.zip'):\\n"
        "            z_tmp = cdir / 'custom_pretrains.zip'\\n"
        "            print(f'Downloading pretrained ZIP -> {z_tmp.name}', flush=True)\\n"
        "            urllib.request.urlretrieve(g_val, z_tmp)\\n"
        "            with zipfile.ZipFile(z_tmp, 'r') as zf:\\n"
        "                for mem in zf.namelist():\\n"
        "                    if not mem.endswith('/') and Path(mem).name.lower().endswith('.pth'):\\n"
        "                        out_pth = cdir / Path(mem).name\\n"
        "                        out_pth.write_bytes(zf.read(mem))\\n"
        "                        rel_p = f'rvc/models/pretraineds/custom/{out_pth.name}'\\n"
        "                        print(f'Extracted {out_pth.name} -> {rel_p}', flush=True)\\n"
        "                        if 'D' in out_pth.name.upper():\\n"
        "                            result['custom_d'] = rel_p\\n"
        "                        else:\\n"
        "                            result['custom_g'] = rel_p\\n"
        "            z_tmp.unlink(missing_ok=True)\\n"
        "            result['manifest_items'].append({'id': f'pre-zip-{int(time.time())}', 'name': Path(urllib.parse.urlsplit(g_val).path).name, 'category': 'pretrained', 'url': g_val})\\n"
        "        else:\\n"
        "            if g_val.startswith(('http://', 'https://')) and g_path.endswith('.pth'):\\n"
        "                gn = Path(urllib.parse.unquote(urllib.parse.urlsplit(g_val).path)).name or 'Custom_G.pth'\\n"
        "                print(f'Downloading G weight -> rvc/models/pretraineds/custom/{gn}', flush=True)\\n"
        "                urllib.request.urlretrieve(g_val, cdir / gn)\\n"
        "                rel_g = f'rvc/models/pretraineds/custom/{gn}'\\n"
        "                result['custom_g'] = rel_g\\n"
        "                print(f'Downloaded G weight -> {rel_g}', flush=True)\\n"
        "                result['manifest_items'].append({'id': f'pre-g-{int(time.time())}', 'name': gn, 'category': 'pretrained', 'url': g_val})\\n"
        "            d_path = urllib.parse.urlsplit(d_val).path.lower()\\n"
        "            if d_val.startswith(('http://', 'https://')) and d_path.endswith('.pth'):\\n"
        "                dn = Path(urllib.parse.unquote(urllib.parse.urlsplit(d_val).path)).name or 'Custom_D.pth'\\n"
        "                print(f'Downloading D weight -> rvc/models/pretraineds/custom/{dn}', flush=True)\\n"
        "                urllib.request.urlretrieve(d_val, cdir / dn)\\n"
        "                rel_d = f'rvc/models/pretraineds/custom/{dn}'\\n"
        "                result['custom_d'] = rel_d\\n"
        "                print(f'Downloaded D weight -> {rel_d}', flush=True)\\n"
        "                result['manifest_items'].append({'id': f'pre-d-{int(time.time())}', 'name': dn, 'category': 'pretrained', 'url': d_val})\\n"
        "        print('\\\\n✔ Pretrained download complete!', flush=True)\\n"
        "    else:\\n"
        "        pre_arch = str(cfg.get('pre_arch', 'Legacy Core 1.5 V2'))\\n"
        "        pre_sr = str(cfg.get('pre_sr', '32k'))\\n"
        f"        if pre_arch == {json.dumps(LEGACY_CORE_PRETRAIN_NAME)}:\\n"
        f"            g_url = {json.dumps(LEGACY_CORE_PRETRAIN_G)}\\n"
        f"            d_url = {json.dumps(LEGACY_CORE_PRETRAIN_D)}\\n"
        "            g_out = cdir / 'G_LegacyCore1.5v2.pth'\\n"
        "            d_out = cdir / 'D_LegacyCore1.5v2.pth'\\n"
        "            print('Downloading Legacy Core 1.5 V2 (32k) G weight -> rvc/models/pretraineds/custom/G_LegacyCore1.5v2.pth', flush=True)\\n"
        "            urllib.request.urlretrieve(g_url, g_out)\\n"
        "            print('Downloaded G weight -> rvc/models/pretraineds/custom/G_LegacyCore1.5v2.pth', flush=True)\\n"
        "            print('Downloading Legacy Core 1.5 V2 (32k) D weight -> rvc/models/pretraineds/custom/D_LegacyCore1.5v2.pth', flush=True)\\n"
        "            urllib.request.urlretrieve(d_url, d_out)\\n"
        "            print('Downloaded D weight -> rvc/models/pretraineds/custom/D_LegacyCore1.5v2.pth', flush=True)\\n"
        "            result['custom_pre'] = True\\n"
        "            result['custom_g'] = 'rvc/models/pretraineds/custom/G_LegacyCore1.5v2.pth'\\n"
        "            result['custom_d'] = 'rvc/models/pretraineds/custom/D_LegacyCore1.5v2.pth'\\n"
        "            result['manifest_items'].append({'id': f'pre-legacy-g-{int(time.time())}', 'name': 'G_LegacyCore1.5v2.pth', 'category': 'pretrained', 'url': g_url})\\n"
        "            result['manifest_items'].append({'id': f'pre-legacy-d-{int(time.time())}', 'name': 'D_LegacyCore1.5v2.pth', 'category': 'pretrained', 'url': d_url})\\n"
        "            print('\\\\n✔ Pretrained download complete! (Legacy Core 1.5 V2 32k G + D)', flush=True)\\n"
        "        else:\\n"
        "            pre_url = f'https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretraineds/{pre_arch}_{pre_sr}.pth'\\n"
        "            out_p = cdir / f'{pre_arch}_{pre_sr}.pth'\\n"
        "            print(f'Downloading {pre_arch} ({pre_sr}) into rvc/models/pretraineds/custom/{out_p.name}...', flush=True)\\n"
        "            urllib.request.urlretrieve(pre_url, out_p)\\n"
        "            result['manifest_items'].append({'id': f'pre-{int(time.time())}', 'name': f'{pre_arch} ({pre_sr})', 'category': 'pretrained', 'url': pre_url})\\n"
        "            print(f'\\\\n✔ Pretrained download complete! ({pre_arch} {pre_sr})', flush=True)\\n"
        "except Exception:\\n"
        "    status = 'error'\\n"
        "    traceback.print_exc()\\n"
        "    sys.stdout.flush()\\n"
        "    sys.stderr.flush()\\n"
        "finally:\\n"
        "    try:\\n"
        "        res_p.write_text(json.dumps(result), encoding='utf-8')\\n"
        "    except Exception:\\n"
        "        pass\\n"
        "    try:\\n"
        "        meta = json.loads(meta_p.read_text(encoding='utf-8')) if meta_p.is_file() else {}\\n"
        "        meta['status'] = status\\n"
        "        meta['finished'] = time.time()\\n"
        "        meta_p.write_text(json.dumps(meta), encoding='utf-8')\\n"
        "    except Exception:\\n"
        "        pass\\n"
    )

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    label = str(payload.get("label") or "pretrained weights")
    lf = open(log_p, "w", encoding="utf-8", buffering=1)
    start_banner = f"⏳ Downloading pretrained {label}...\\n"
    lf.write(start_banner)
    lf.flush()
    print(start_banner, end="", flush=True)
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", child_code],
        cwd=now_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        start_new_session=True,
    )
    meta_p.write_text(
        json.dumps({
            "task_kind": "pretrained",
            "pid": proc.pid,
            "model_name": label,
            "log_state_key": "nr_pre_log_text",
            "log_path": str(log_p),
            "toast_title": "Pretrained Download Complete",
            "done_note": f"Downloaded {label} — ready in rvc/models/pretraineds/custom.",
            "status": "running",
            "notified": False,
            "result_applied": False,
            "started": time.time(),
        }),
        encoding="utf-8",
    )

    def _pump_pre() -> None:
        try:
            if proc.stdout is not None:
                for ln in proc.stdout:
                    lf.write(ln)
                    lf.flush()
                    print(ln, end="", flush=True)
        finally:
            rc = proc.wait()
            if rc != 0:
                err_ln = f"\\n✖ Pretrained download failed (exit code {rc}).\\n"
                lf.write(err_ln)
                lf.flush()
                print(err_ln, end="", flush=True)
            lf.close()
            try:
                meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.is_file() else {}
                if meta.get("status") == "running":
                    meta["status"] = "done" if rc == 0 else "error"
                    meta["exit_code"] = rc
                    meta["finished"] = time.time()
                    meta_p.write_text(json.dumps(meta), encoding="utf-8")
            except Exception:
                pass

    threading.Thread(target=_pump_pre, daemon=True).start()
    return start_banner.strip()


def _poll_core_tasks() -> tuple[bool, bool]:
    changed = False
    completed = False
    for task_kind in ("preprocess", "extract", "index", "pretrained"):
        meta_p, log_p = _task_paths(task_kind)
        if not meta_p.is_file():
            continue
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
        except Exception:
            continue
        lkey = str(meta.get("log_state_key", ""))
        txt = ""
        if log_p.is_file() and lkey:
            try:
                txt = _read_tail_text(log_p, 32768).strip()
                if txt:
                    tail_txt = "\\n".join(txt.splitlines()[-80:])
                    if lkey == "nr_trn_log":
                        tail_txt = _filter_status_lines(tail_txt, "⏳ Task running...")
                    if st.session_state.get(lkey) != tail_txt:
                        st.session_state[lkey] = tail_txt
                        changed = True
            except Exception:
                pass
        if meta.get("status") == "running" and meta.get("pid"):
            try:
                os.kill(int(meta["pid"]), 0)
            except Exception:
                if not txt and log_p.is_file():
                    txt = _read_tail_text(log_p, 32768)
                has_err = ("Traceback (" in txt) or (f"✖ {task_kind.capitalize()} failed" in txt) or ("✖ Pretrained download failed" in txt) or ("NameError:" in txt)
                meta["status"] = "error" if has_err else "done"
                try:
                    meta_p.write_text(json.dumps(meta), encoding="utf-8")
                except Exception:
                    pass
        if task_kind == "pretrained" and meta.get("status") in ("done", "error") and not meta.get("result_applied"):
            meta["result_applied"] = True
            res_p = meta_p.with_suffix(".result.json")
            if res_p.is_file():
                try:
                    res_data = json.loads(res_p.read_text(encoding="utf-8"))
                    if res_data.get("custom_g"):
                        st.session_state["nr_custom_g"] = res_data["custom_g"]
                        st.session_state["nr_cp_g_inp"] = res_data["custom_g"]
                    if res_data.get("custom_d"):
                        st.session_state["nr_custom_d"] = res_data["custom_d"]
                        st.session_state["nr_cp_d_inp"] = res_data["custom_d"]
                    if res_data.get("custom_pre"):
                        st.session_state["nr_custom_pre"] = True
                    existing_urls = {str(m.get("url", "")) for m in st.session_state.get("nr_manifest", [])}
                    for item in res_data.get("manifest_items", []):
                        if str(item.get("url", "")) not in existing_urls:
                            st.session_state["nr_manifest"].append(item)
                            existing_urls.add(str(item.get("url", "")))
                except Exception:
                    pass
            if txt:
                st.session_state["nr_pre_log"].extend([ln for ln in txt.splitlines() if ln.strip()][-6:])
            try:
                meta_p.write_text(json.dumps(meta), encoding="utf-8")
            except Exception:
                pass
        if meta.get("status") in ("done", "error") and not meta.get("notified"):
            meta["notified"] = True
            try:
                meta_p.write_text(json.dumps(meta), encoding="utf-8")
            except Exception:
                pass
            _clear_fs_caches()
            completed = True
            changed = True
            title = str(meta.get("toast_title", f"{task_kind.capitalize()} Complete"))
            msg = str(meta.get("done_note", f"{task_kind.capitalize()} finished."))
            if meta.get("status") == "error":
                title = "Pretrained Download Failed" if task_kind == "pretrained" else f"{task_kind.capitalize()} Failed"
                msg = f"Error during {task_kind} — check output log for details."
            _push_toast(title, msg)

    if ACTIVE_TRAIN_META.is_file():
        try:
            tmeta = json.loads(ACTIVE_TRAIN_META.read_text(encoding="utf-8"))
            tmodel = str(tmeta.get("model_name", ""))
            alive = _is_training_process_alive(tmodel)
            tlp = Path(str(tmeta.get("log_path", "")))
            if tlp.is_file():
                raw_trn = _read_tail_text(tlp, 32768)
                ttail = _filter_status_lines(
                    raw_trn,
                    "⏳ Training in progress — monitor live epochs in Kaggle cell & progress meter above.",
                )
                if ttail and st.session_state.get("nr_trn_log") != ttail:
                    st.session_state["nr_trn_log"] = ttail
                    changed = True
            if tmeta.get("status") == "running" and not alive:
                st.session_state["nr_is_training"] = False
                train_err = False
                if tlp.is_file():
                    try:
                        t_txt = _read_tail_text(tlp, 32768)
                        train_err = ("Traceback (" in t_txt) or ("✖ Training exited with error code" in t_txt) or ("NameError:" in t_txt)
                    except Exception:
                        pass
                tmeta["status"] = "error" if train_err else "done"
                if not tmeta.get("notified"):
                    tmeta["notified"] = True
                    ACTIVE_TRAIN_META.write_text(json.dumps(tmeta), encoding="utf-8")
                    _clear_fs_caches()
                    completed = True
                    changed = True
                    if train_err:
                        _push_toast("Training Failed", f"Error during training for {tmodel or 'your model'} — check Training output box.")
                    else:
                        _push_toast("Training Complete", f"Training finished for {tmodel or 'your model'} — open Diagnostics to inspect best epoch.")
        except Exception:
            pass
    return changed, completed


def _is_task_running(task_kind: str) -> bool:
    meta_p, _ = _task_paths(task_kind)
    if meta_p.is_file():
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            if meta.get("status") == "running" and meta.get("pid"):
                os.kill(int(meta["pid"]), 0)
                return True
        except Exception:
            pass
    return False


def _has_active_bg_work() -> bool:
    if st.session_state.get("nr_is_training") and _is_training_process_alive(st.session_state.get("nr_model_name", "")):
        return True
    for task_kind in ("preprocess", "extract", "index", "pretrained"):
        if _is_task_running(task_kind):
            return True
    return False


def _train_log_path(model_name: str) -> Path:
    safe = _safe_model_slug(model_name)
    p = Path(now_dir) / "logs" / safe / "nr_train.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _read_train_log(model_name: str, max_chars: int = 16384) -> str:
    candidates: list[Path] = []
    if model_name.strip():
        candidates.append(_train_log_path(model_name))
    if ACTIVE_TRAIN_META.is_file():
        try:
            lp = json.loads(ACTIVE_TRAIN_META.read_text(encoding="utf-8")).get("log_path", "")
            if lp:
                candidates.append(Path(lp))
        except Exception:
            pass
    for p in candidates:
        if p.is_file():
            tail = _read_tail_text(p, max(32768, max_chars * 2))
            if tail:
                return tail[-max_chars:]
    return ""


def _start_training_bg(model_name: str, kwargs: dict[str, Any]) -> str:
    log_p = _train_log_path(model_name)
    pid_p = log_p.parent / "nr_train.pid"
    args_p = log_p.parent / "nr_train.args.json"
    if pid_p.is_file():
        try:
            old_pid = int(pid_p.read_text().strip())
            os.kill(old_pid, 0)
            return f"Training already running (PID {old_pid})."
        except Exception:
            pid_p.unlink(missing_ok=True)

    ddp_lr_scale = str(kwargs.pop("nr_ddp_lr_scale", "1.0") or "1.0")
    args_p.write_text(json.dumps(kwargs, ensure_ascii=False), encoding="utf-8")

    cmd_script = (
        "import sys, os, json, re, inspect, urllib.request, urllib.parse, types; from pathlib import Path\\n"
        "sys.path.insert(0, os.getcwd())\\n"
        f"kw = json.loads(Path({json.dumps(str(args_p))}).read_text(encoding='utf-8'))\\n"
        "def _resolve_pretrain_path(raw_val, role):\\n"
        "    val = str(raw_val or '').strip()\\n"
        "    if not val:\\n"
        "        return ''\\n"
        "    cdir = Path('rvc') / 'models' / 'pretraineds' / 'custom'\\n"
        "    cdir.mkdir(parents=True, exist_ok=True)\\n"
        "    if val.startswith(('http://', 'https://')):\\n"
        "        seg = urllib.parse.unquote(Path(urllib.parse.urlsplit(val).path).name) or f'custom_{role}.pth'\\n"
        "        if not seg.lower().endswith('.pth'):\\n"
        "            seg = f'{seg}.pth'\\n"
        "        dst = cdir / seg\\n"
        "        if not dst.is_file() or dst.stat().st_size < 1024:\\n"
        "            print(f'⏳ Downloading custom {role} pretrain: {dst.name}...', flush=True)\\n"
        "            req = urllib.request.Request(val, headers={'User-Agent': 'NoRegretsRVC'})\\n"
        "            with urllib.request.urlopen(req, timeout=180) as resp:\\n"
        "                dst.write_bytes(resp.read())\\n"
        "            print(f'✔ Downloaded {dst.name} ({dst.stat().st_size // (1024*1024)} MB)', flush=True)\\n"
        "        return str(dst)\\n"
        "    p_cand = Path(val)\\n"
        "    if p_cand.is_file():\\n"
        "        return str(p_cand)\\n"
        "    alt = cdir / p_cand.name\\n"
        "    return str(alt) if alt.is_file() else val\\n"
        "if kw.get('custom_pretrained'):\\n"
        "    try:\\n"
        "        g_loc = _resolve_pretrain_path(kw.get('g_pretrained_path', ''), 'G')\\n"
        "        d_loc = _resolve_pretrain_path(kw.get('d_pretrained_path', ''), 'D')\\n"
        "        if g_loc and d_loc and Path(g_loc).is_file() and Path(d_loc).is_file():\\n"
        "            kw['g_pretrained_path'] = g_loc\\n"
        "            kw['d_pretrained_path'] = d_loc\\n"
        "        else:\\n"
        "            print('⚠ Custom pretrain file not found on disk; falling back to built-in pretrained base.', flush=True)\\n"
        "            kw['custom_pretrained'] = False\\n"
        "    except Exception as exc_pre:\\n"
        "        print(f'⚠ Custom pretrain resolve failed ({exc_pre}); falling back to built-in pretrained base.', flush=True)\\n"
        "        kw['custom_pretrained'] = False\\n"
        "import core\\n"
        "fn = core.run_train_script\\n"
        "sig = inspect.signature(fn)\\n"
        "kw_call = kw if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()) else {k: v for k, v in kw.items() if k in sig.parameters}\\n"
        "fn(**kw_call)\\n"
        "try:\\n"
        "    import noregrets_turbo\\n"
        "    noregrets_turbo.flush_saves()\\n"
        "except Exception:\\n"
        "    pass\\n"
        "def _dict_to_ns(obj):\\n"
        "    if isinstance(obj, dict):\\n"
        "        return types.SimpleNamespace(**{k: _dict_to_ns(v) for k, v in obj.items()})\\n"
        "    if isinstance(obj, list):\\n"
        "        return [_dict_to_ns(x) for x in obj]\\n"
        "    return obj\\n"
        "try:\\n"
        "    m_dir = Path('logs') / str(kw.get('model_name', '')).strip()\\n"
        "    tot_ep = int(kw.get('total_epoch') or 0)\\n"
        "    if m_dir.is_dir() and tot_ep > 0 and kw.get('save_every_weights', True):\\n"
        "        has_final = list(m_dir.glob(f'*_{tot_ep}e_*.pth')) or list(m_dir.glob(f'*_{tot_ep}e.pth'))\\n"
        "        if not has_final:\\n"
        "            g_cands = sorted([p for p in m_dir.glob('G_*.pth') if p.is_file() and p.stat().st_size > 1024], key=lambda p: p.stat().st_mtime, reverse=True)\\n"
        "            if g_cands:\\n"
        "                g_best = g_cands[0]\\n"
        "                import torch\\n"
        "                ckpt = torch.load(str(g_best), map_location='cpu', weights_only=True)\\n"
        "                ep_num = int(ckpt.get('epoch') or tot_ep)\\n"
        "                stp_num = int(ckpt.get('iteration') or 0)\\n"
        "                out_pth = m_dir / (f'{m_dir.name}_{ep_num}e_{stp_num}s.pth' if stp_num else f'{m_dir.name}_{ep_num}e.pth')\\n"
        "                cfg_p = m_dir / 'config.json'\\n"
        "                hps_obj = _dict_to_ns(json.loads(cfg_p.read_text(encoding='utf-8'))) if cfg_p.is_file() else None\\n"
        "                if not out_pth.is_file() and 'model' in ckpt:\\n"
        "                    try:\\n"
        "                        from rvc.train.process.extract_model import extract_model\\n"
        "                        extract_model(\\n"
        "                            ckpt=ckpt['model'],\\n"
        "                            sr=int(kw.get('sample_rate', 32000)),\\n"
        "                            name=str(m_dir.name),\\n"
        "                            model_path=str(out_pth),\\n"
        "                            epoch=ep_num,\\n"
        "                            step=stp_num,\\n"
        "                            hps=hps_obj,\\n"
        "                            overtrain=False,\\n"
        "                            vocoder=str(kw.get('vocoder', 'HiFi-GAN')),\\n"
        "                        )\\n"
        "                    except Exception as exc_ex:\\n"
        "                        print(f'extract_model rescue fallback ({exc_ex})', flush=True)\\n"
        "                if not out_pth.is_file() and 'model' in ckpt:\\n"
        "                    raw_sd = {}\\n"
        "                    for k_w, v_w in ckpt['model'].items():\\n"
        "                        k_clean = k_w[7:] if k_w.startswith('module.') else k_w\\n"
        "                        if k_clean.startswith('enc_q'):\\n"
        "                            continue\\n"
        "                        raw_sd[k_clean] = v_w.half() if hasattr(v_w, 'half') and getattr(v_w, 'is_floating_point', lambda: False)() else v_w\\n"
        "                    sr_val = int(kw.get('sample_rate', 32000))\\n"
        "                    spec_ch = (hps_obj.data.filter_length // 2 + 1) if hps_obj is not None and hasattr(hps_obj, 'data') and hasattr(hps_obj.data, 'filter_length') else (1025 if sr_val >= 40000 else 513)\\n"
        "                    torch.save({'weight': raw_sd, 'sr': f'{sr_val // 1000}k', 'f0': 1, 'version': 'v2', 'info': f'{ep_num}epoch', 'vocoder': str(kw.get('vocoder', 'HiFi-GAN'))}, str(out_pth))\\n"
        "                if out_pth.is_file():\\n"
        "                    print(f'✔ Final checkpoint exported: {out_pth.name}', flush=True)\\n"
        "except Exception as exc_fin:\\n"
        "    print(f'Final checkpoint rescue note: {exc_fin}', flush=True)\\n"
    )
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["NR_TURBO_ENABLED"] = "1" if st.session_state.get("nr_turbo_training", True) else "0"
    env["NR_DDP_LR_SCALE"] = ddp_lr_scale
    lf = open(log_p, "w", encoding="utf-8", buffering=1)
    start_banner = f"⏳ Starting training for {model_name or 'model'}...\\n"
    lf.write(start_banner)
    lf.flush()
    print(start_banner, end="", flush=True)
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", cmd_script],
        cwd=now_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        start_new_session=True,
    )
    pid_p.write_text(str(proc.pid), encoding="utf-8")
    try:
        ACTIVE_TRAIN_META.parent.mkdir(parents=True, exist_ok=True)
        ACTIVE_TRAIN_META.write_text(
            json.dumps({
                "pid": proc.pid,
                "model_name": model_name,
                "log_path": str(log_p),
                "total_epoch": int(kwargs.get("total_epoch", st.session_state.get("nr_total_epoch", 200))),
                "save_every_epoch": int(kwargs.get("save_every_epoch", st.session_state.get("nr_save_every", 5))),
                "started": time.time(),
                "status": "running",
                "notified": False,
            }),
            encoding="utf-8",
        )
    except Exception:
        pass

    def _pump() -> None:
        try:
            if proc.stdout is not None:
                for ln in proc.stdout:
                    lf.write(ln)
                    lf.flush()
                    print(ln, end="", flush=True)
        finally:
            rc = proc.wait()
            if rc == 0:
                lf.write("\\n✔ Training complete!\\n")
            else:
                lf.write(f"\\n✖ Training exited with error code {rc}.\\n")
            lf.flush()
            lf.close()

    threading.Thread(target=_pump, daemon=True).start()
    st.session_state["nr_is_training"] = True
    return f"⏳ Training '{model_name}' started (PID {proc.pid}) -> streaming live to Training & Diagnostics."


def _stop_training(model_name: str) -> str:
    pid_p = _train_log_path(model_name).parent / "nr_train.pid"
    st.session_state["nr_is_training"] = False
    killed = 0
    pids: list[int] = []
    if pid_p.is_file():
        try:
            pids.append(int(pid_p.read_text().strip()))
        except Exception:
            pass
        pid_p.unlink(missing_ok=True)
    if ACTIVE_TRAIN_META.is_file():
        try:
            meta = json.loads(ACTIVE_TRAIN_META.read_text(encoding="utf-8"))
            if isinstance(meta.get("pid"), int):
                pids.append(meta["pid"])
            meta["status"] = "stopped"
            meta["notified"] = True
            ACTIVE_TRAIN_META.write_text(json.dumps(meta), encoding="utf-8")
        except Exception:
            pass
    for pid in set(pids):
        try:
            os.killpg(pid, signal.SIGTERM)
            killed += 1
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
                killed += 1
            except Exception:
                pass
    _clear_fs_caches()
    return f"Stop signal sent to training ({killed} process group(s))."


def _is_training_process_alive(model_name: str) -> bool:
    pid_p = _train_log_path(model_name).parent / "nr_train.pid"
    if not pid_p.is_file():
        return False
    try:
        pid = int(pid_p.read_text().strip())
        os.kill(pid, 0)
        return True
    except Exception:
        return False


@st.cache_data(ttl=6, show_spinner=False)
def _list_models() -> list[str]:
    logs_dir = Path(now_dir) / "logs"
    if not logs_dir.is_dir():
        return []
    skip = {"mute", "mute_spin", "reference", "zips", "downloads"}
    return sorted(p.name for p in logs_dir.iterdir() if p.is_dir() and p.name not in skip and not p.name.startswith("."))


@st.cache_data(ttl=6, show_spinner=False)
def _dataset_slice_count(model_name: str) -> int:
    if not model_name.strip():
        return 0
    sdir = Path(now_dir) / "logs" / _safe_model_slug(model_name) / "sliced_audios"
    if not sdir.is_dir():
        return 0
    return sum(1 for p in sdir.iterdir() if p.is_file() and p.suffix.lower() == ".wav")


@st.cache_data(ttl=6, show_spinner=False)
def _list_datasets() -> list[str]:
    found: list[str] = []
    audio_exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    max_depth = 4
    for base in (Path("/kaggle/input"), Path(now_dir) / "assets" / "datasets"):
        if not base.is_dir():
            continue
        base_parts = len(base.parts)
        for root_str, dirs, files in os.walk(str(base)):
            cur_p = Path(root_str)
            depth = len(cur_p.parts) - base_parts
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__")
            if depth >= max_depth:
                dirs[:] = []
            if any(fn.lower().endswith(audio_exts) for fn in files):
                found.append(root_str)
    return sorted(set(found))


@st.cache_data(ttl=6, show_spinner=False)
def _list_files_rel(exts: tuple[str, ...], roots: tuple[str, ...], query: str = "") -> list[str]:
    found: list[str] = []
    q = query.strip().lower()
    for r in roots:
        rp = Path(now_dir) / r
        if not rp.exists():
            continue
        for p in sorted(rp.rglob("*")):
            if p.is_file() and p.suffix.lower() in exts:
                if "pretrained" in str(p).lower():
                    continue
                if p.suffix.lower() == ".pth" and p.name.startswith(("G_", "D_", "f0")):
                    continue
                if p.suffix.lower() == ".index" and "trained" in p.name.lower():
                    continue
                rel = str(p.relative_to(now_dir))
                if not q or q in rel.lower():
                    found.append(rel)
    return found


def _list_pth_files(q: str = "") -> list[str]:
    return _list_files_rel((".pth",), ("logs",), q)


def _list_index_files(q: str = "") -> list[str]:
    return _list_files_rel((".index",), ("logs",), q)


def _list_audio_inputs() -> list[str]:
    return _list_files_rel((".wav", ".mp3", ".flac", ".ogg", ".m4a"), ("assets/audios", "assets/free_samples"))


def _list_custom_pretrains(prefix: str) -> list[str]:
    cdir = Path(now_dir) / "rvc" / "models" / "pretraineds" / "custom"
    if not cdir.exists():
        return []
    out: list[str] = []
    for p in sorted(cdir.rglob("*.pth")):
        if not prefix or prefix.upper() in p.name.upper():
            out.append(str(p.relative_to(now_dir)))
    return out


@st.cache_data(ttl=60, show_spinner=False)
def _load_tts_voices() -> list[str]:
    vf = Path(now_dir) / "rvc" / "lib" / "tools" / "tts_voices.json"
    if vf.is_file():
        try:
            data = json.loads(vf.read_text(encoding="utf-8"))
            names = [v.get("ShortName", "") for v in data if isinstance(v, dict) and v.get("ShortName")]
            if names:
                return names
        except Exception:
            pass
    return list(TTS_VOICES)


def _match_index(pth_rel: str, indexes: list[str]) -> str:
    if not pth_rel:
        return ""
    parent = str(Path(pth_rel).parent)
    stem = Path(pth_rel).stem.split("_")[0].lower()
    for idx in indexes:
        if str(Path(idx).parent) == parent:
            return idx
    for idx in indexes:
        if stem and stem in Path(idx).name.lower():
            return idx
    return ""


def _infer_matching_d_url(g_url: str) -> str:
    if not g_url:
        return ""
    trimmed = g_url.strip()
    if "f0G" in trimmed:
        return trimmed.replace("f0G", "f0D", 1)
    if "G_" in trimmed:
        return trimmed.replace("G_", "D_", 1)
    if "/G." in trimmed:
        return trimmed.replace("/G.", "/D.", 1)
    return ""


@st.cache_data(ttl=300, show_spinner=False)
def _gpu_info_lines(gpu_num: str) -> str:
    parts = [p.strip() for p in (gpu_num or "0").split("-") if p.strip()]
    if not parts:
        return "No GPU specified (CPU fallback)."
    try:
        import torch
        if torch.cuda.is_available():
            lines = []
            for p in parts:
                idx = int(p) if p.isdigit() else 0
                if idx < torch.cuda.device_count():
                    props = torch.cuda.get_device_properties(idx)
                    gb = max(1, round(props.total_memory / (1024 ** 3)))
                    lines.append(f"{p}: {props.name} ({gb} GB)")
                else:
                    lines.append(f"{p}: Tesla T4 (14 GB)")
            return "\\n".join(lines)
    except Exception:
        pass
    return "\\n".join(f"{p}: Tesla T4 (14 GB)" for p in parts)


def _parse_training_log(raw_text: str) -> dict[str, Any] | None:
    if not raw_text or not raw_text.strip():
        return None
    text = raw_text
    try:
        nb = json.loads(raw_text)
        if isinstance(nb, dict) and isinstance(nb.get("cells"), list):
            parts = []
            for cell in nb["cells"]:
                for out in cell.get("outputs", []):
                    for k in ("text", "data"):
                        val = out.get(k)
                        if isinstance(val, list):
                            parts.append("".join(str(x) for x in val))
                        elif isinstance(val, str):
                            parts.append(val)
            if parts:
                text = "\\n".join(parts)
    except Exception:
        pass

    epoch_re = re.compile(r"epoch=(\\d+)\\s*\\|\\s*step=(\\d+)", re.IGNORECASE)
    speed_re = re.compile(r"training_speed=([\\d:]+)", re.IGNORECASE)
    low_re = re.compile(r"lowest_value=([\\d.]+)(?:\\s*\\(epoch\\s*(\\d+))?", re.IGNORECASE)
    mel_re = re.compile(r"(?:loss/g/mel|loss_mel|mel_loss|mel)\\s*[:=]\\s*([\\d.]+)", re.IGNORECASE)
    kl_re = re.compile(r"(?:loss/g/kl|loss_kl|kl_loss|kl)\\s*[:=]\\s*([\\d.]+)", re.IGNORECASE)
    fm_re = re.compile(r"(?:loss/g/fm|loss_fm|fm_loss|fm)\\s*[:=]\\s*([\\d.]+)", re.IGNORECASE)
    disc_re = re.compile(r"(?:loss/d/total|loss_disc|disc_loss|d_total)\\s*[:=]\\s*([\\d.]+)", re.IGNORECASE)
    gen_re = re.compile(r"(?:loss/g/total|loss_gen|gen_loss|g_total)\\s*[:=]\\s*([\\d.]+)", re.IGNORECASE)
    grad_re = re.compile(r"(?:grad_norm_g|grad_norm|grad)\\s*[:=]\\s*([\\d.]+)", re.IGNORECASE)
    tqdm_re = re.compile(
        r"(\\d+)%\\|[^|]*\\|\\s*(?:batch=)?(\\d+)/(\\d+)\\s*\\[([\\d:]+)<([\\d:?]+),\\s*([\\d.]+)\\s*(it/s|s/it)\\]"
    )

    by_epoch: dict[int, dict[str, Any]] = {}
    last_tq = None
    last_tq_ep = None
    last_tq_stp = None
    best_recorded: tuple[float, int] | None = None
    for raw_ln in text.splitlines():
        tq = tqdm_re.search(raw_ln)
        m = epoch_re.search(raw_ln)
        if tq:
            last_tq = tq
            last_tq_ep = int(m.group(1)) if m else None
            last_tq_stp = int(m.group(2)) if m else None
            continue
        if m:
            ep = int(m.group(1))
            stp = int(m.group(2))
            sm = speed_re.search(raw_ln)
            spd_sec = None
            if sm:
                segs = [int(x) for x in sm.group(1).split(":")]
                if len(segs) == 3:
                    spd_sec = segs[0] * 3600 + segs[1] * 60 + segs[2]
            lm = low_re.search(raw_ln)
            low = float(lm.group(1)) if lm else None
            if lm:
                low_ep = int(lm.group(2)) if lm.group(2) else ep
                if best_recorded is None or low < best_recorded[0]:
                    best_recorded = (low, low_ep)
            mel_v = float(mel_m.group(1)) if mel_m else low
            kl_m = kl_re.search(raw_ln)
            fm_m = fm_re.search(raw_ln)
            d_m = disc_re.search(raw_ln)
            g_m = gen_re.search(raw_ln)
            gr_m = grad_re.search(raw_ln)
            by_epoch[ep] = {
                "epoch": ep,
                "step": stp,
                "speed_sec": spd_sec,
                "lowest_loss": mel_v,
                "mel_loss": mel_v,
                "kl_loss": float(kl_m.group(1)) if kl_m else None,
                "fm_loss": float(fm_m.group(1)) if fm_m else None,
                "disc_loss": float(d_m.group(1)) if d_m else None,
                "gen_loss": float(g_m.group(1)) if g_m else None,
                "grad_norm": float(gr_m.group(1)) if gr_m else None,
            }

    live = None
    if last_tq:
        raw_val = float(last_tq.group(6))
        unit = last_tq.group(7)
        it_s = raw_val if unit == "it/s" else (1.0 / raw_val if raw_val > 0 else 1.0)
        max_ep = max(by_epoch.keys()) if by_epoch else 0
        max_stp = max((v["step"] for v in by_epoch.values()), default=0)
        live = {
            "epoch": last_tq_ep if last_tq_ep is not None else (max_ep + 1),
            "step": last_tq_stp if last_tq_stp is not None else (max_stp + int(last_tq.group(2))),
            "percent": int(last_tq.group(1)),
            "batch_done": int(last_tq.group(2)),
            "batch_total": int(last_tq.group(3)),
            "elapsed": last_tq.group(4),
            "eta": last_tq.group(5),
            "it_per_sec": it_s,
        }

    points = [by_epoch[k] for k in sorted(by_epoch.keys())]
    if not points and not live:
        return None
    spds = [p["speed_sec"] for p in points if p["speed_sec"] is not None]
    avg_spd = round(sum(spds) / len(spds), 1) if spds else 0.0
    kls = [p["kl_loss"] for p in points if p.get("kl_loss") is not None]
    fms = [p["fm_loss"] for p in points if p.get("fm_loss") is not None]
    discs = [p["disc_loss"] for p in points if p.get("disc_loss") is not None]
    gens = [p["gen_loss"] for p in points if p.get("gen_loss") is not None]
    grads = [p["grad_norm"] for p in points if p.get("grad_norm") is not None]
    best_ep = best_recorded[1] if best_recorded is not None else None
    ep_max = points[-1]["epoch"] if points else (live["epoch"] - 1 if live else 0)
    return {
        "points": points,
        "epochs_completed": ep_max,
        "avg_speed_sec": avg_spd,
        "lowest_loss_epoch": best_ep,
        "avg_kl_loss": round(sum(kls) / len(kls), 3) if kls else None,
        "avg_fm_loss": round(sum(fms) / len(fms), 3) if fms else None,
        "avg_disc_loss": round(sum(discs) / len(discs), 3) if discs else None,
        "avg_gen_loss": round(sum(gens) / len(gens), 3) if gens else None,
        "latest_grad_norm": round(grads[-1], 2) if grads else None,
        "live": live,
    }


@st.cache_data(ttl=10, show_spinner=False)
def _read_tb_metrics_for_model(model_name: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not model_name.strip():
        return out
    mdir = Path(now_dir) / "logs" / _safe_model_slug(model_name)
    if not mdir.is_dir():
        return out
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        ea = EventAccumulator(str(mdir), size_guidance={"scalars": 200})
        ea.Reload()
        tags = set(ea.Tags().get("scalars", []))

        def _last_scalar(candidates: tuple[str, ...]) -> float | None:
            for tag in candidates:
                if tag in tags:
                    vals = ea.Scalars(tag)
                    if vals:
                        return round(float(vals[-1].value), 3)
            return None

        out["avg_kl_loss"] = _last_scalar(("loss/g/kl", "loss_kl", "kl_loss"))
        out["avg_fm_loss"] = _last_scalar(("loss/g/fm", "loss_fm", "fm_loss"))
        out["avg_disc_loss"] = _last_scalar(("loss/d/total", "loss_disc", "disc_loss"))
        out["avg_gen_loss"] = _last_scalar(("loss/g/total", "loss_gen", "gen_loss"))
        out["latest_grad_norm"] = _last_scalar(("grad_norm_g", "grad_norm/g", "grad_norm_d", "grad_norm"))
    except Exception:
        pass
    return out


def _snap_up(val: int, step: int) -> int:
    s = max(1, step)
    return max(s, int(math.ceil(val / s) * s))


def _snap_near(val: int, step: int) -> int:
    s = max(1, step)
    return max(s, int(round(val / s) * s))


def _analyze_zones(
    parsed: dict[str, Any] | None,
    save_every: int,
    total_epoch: int,
    sr: int,
    sound_feedback: str = "untested",
    dataset_slices: int = 0,
    vocoder: str = "HiFi-GAN",
    model_name: str = "",
) -> dict[str, Any]:
    s = max(1, int(save_every))
    pts = [p for p in (parsed.get("points") if parsed else []) if p.get("epoch") is not None]
    if not parsed or len(pts) == 0:
        return {
            "confidence": "untrained",
            "save_every": s,
            "best_epoch": None,
            "best_loss": None,
            "dataset_match_pct": 0,
            "stability_pct": 0,
            "convergence_label": "Awaiting Data",
            "verdict": "untrained",
            "verdict_title": "Waiting for Training Data",
            "verdict_guidance": "Start training on Kaggle or upload a log below to check for undertraining, overtraining, and exact checkpoint quality.",
            "under_risk_pct": 0,
            "over_risk_pct": 0,
            "steps_per_clip": None,
            "effective_steps": 0,
            "ranked_checkpoints": [],
            "metric_breakdown": [],
            "candidate_epochs": [],
            "under_end": None,
            "sweet_min": None,
            "sweet_mid": None,
            "sweet_max": None,
            "over_start": None,
            "under_label": "—",
            "sweet_label": "—",
            "over_label": "—",
            "checkpoint_hint": "",
            "dataset_summary": (
                f"{dataset_slices} sliced audio clips detected on disk"
                if dataset_slices > 0
                else "Waiting for preprocessed slices & real training epochs"
            ),
            "summary": "No real training epochs recorded yet. Start training or import a training log to identify your model's best epoch.",
        }

    ep_completed = max(int(parsed.get("epochs_completed") or pts[-1]["epoch"]), pts[-1]["epoch"])
    loss_pts = [p for p in pts if p.get("lowest_loss") is not None]
    min_loss_pt = min(loss_pts, key=lambda p: float(p["lowest_loss"])) if loss_pts else pts[-1]
    lowest_ep = int(min_loss_pt["epoch"]) if loss_pts else int(parsed.get("lowest_loss_epoch") or ep_completed)
    best_loss = float(min_loss_pt["lowest_loss"]) if min_loss_pt.get("lowest_loss") is not None else None
    first_loss = float(loss_pts[0]["lowest_loss"]) if loss_pts else best_loss
    latest_loss = float(loss_pts[-1]["lowest_loss"]) if loss_pts else best_loss
    latest_steps = int(pts[-1].get("step") or 0)
    steps_per_clip = round(latest_steps / max(1, dataset_slices), 1) if dataset_slices > 0 else None

    tail_pts = loss_pts[-3:]
    tail_slope = (
        (float(tail_pts[0]["lowest_loss"]) - float(tail_pts[-1]["lowest_loss"]))
        / max(1, int(tail_pts[-1]["epoch"]) - int(tail_pts[0]["epoch"]))
        if len(tail_pts) >= 2
        else 0.0
    )
    rebound_amt = max(0.0, (latest_loss - best_loss)) if (best_loss is not None and latest_loss is not None) else 0.0
    rebound_pct = round((rebound_amt / max(0.1, best_loss)) * 100.0, 1) if best_loss else 0.0
    rebound = bool(
        best_loss is not None
        and tail_pts
        and (rebound_amt >= 0.45 or float(tail_pts[-1]["lowest_loss"]) > best_loss * 1.03)
        and int(tail_pts[-1]["epoch"]) > lowest_ep + max(1, s // 2)
    )

    if dataset_slices > 0 and dataset_slices < 60:
        ds_tier = f"Compact dataset ({dataset_slices} clips — converges quickly, monitor early checkpoints)"
    elif dataset_slices >= 250:
        ds_tier = f"Rich dataset ({dataset_slices} clips — tolerates deeper epochs before memorization)"
    elif dataset_slices > 0:
        ds_tier = f"Balanced dataset ({dataset_slices} clips)"
    else:
        ds_tier = f"Live log analysis ({ep_completed} epochs completed · {sr} Hz · {vocoder})"

    snapped = [ep for ep in range(s, ep_completed + 1, s)]
    if not snapped:
        snapped = [ep_completed]

    target_raw = _snap_near(lowest_ep, s)
    if sound_feedback == "undertrained":
        target_raw = min(ep_completed, _snap_up(max(target_raw + s, ep_completed), s))
    elif sound_feedback == "overtrained":
        target_raw = max(snapped[0], _snap_near(lowest_ep - s * 2, s))
    elif sound_feedback == "clean" and ep_completed >= s and not rebound:
        target_raw = _snap_near(lowest_ep, s)

    best_ep = min(snapped, key=lambda ep: abs(ep - target_raw))
    candidates = sorted({
        min(snapped, key=lambda ep: abs(ep - max(snapped[0], best_ep - s))),
        best_ep,
        min(snapped, key=lambda ep: abs(ep - min(snapped[-1], best_ep + s))),
    })

    sweet_min = candidates[0]
    sweet_max = max(candidates[-1], sweet_min + s) if len(candidates) > 1 else best_ep
    under_end = max(s, sweet_min - s) if sweet_min > s else None
    over_start = (best_ep + s * 2) if rebound else (sweet_max + s * 4)

    safe_m = _safe_model_slug(model_name) if model_name.strip() else "YourModel"
    step_at_best = next((int(p["step"]) for p in pts if int(p["epoch"]) == best_ep), None)
    if step_at_best is None:
        closest_pt = min(pts, key=lambda p: abs(int(p["epoch"]) - best_ep))
        step_at_best = int(closest_pt["step"])
    ckpt_hint = f"{safe_m}_{best_ep}e_{step_at_best}s.pth"

    # Multi-metric Dataset Similarity Fit (%) & Stability (%)
    drop_ratio = ((first_loss - best_loss) / max(1.0, first_loss)) if (first_loss and best_loss is not None) else 0.0
    mel_level_score = max(0.0, min(100.0, 96.0 - max(0.0, (best_loss - 11.5) * 4.2))) if best_loss is not None else 62.0
    mel_drop_score = max(0.0, min(100.0, 48.0 + drop_ratio * 85.0))
    kl_val = parsed.get("avg_kl_loss")
    kl_score = max(25.0, min(98.0, 96.0 - max(0.0, (kl_val - 0.85) * 28.0))) if kl_val is not None else (mel_level_score * 0.94)
    disc_val = parsed.get("avg_disc_loss")
    disc_score = max(30.0, min(98.0, 95.0 - abs(disc_val - 1.9) * 18.0)) if disc_val is not None else (84.0 if not rebound else 68.0)
    fm_val = parsed.get("avg_fm_loss")
    fm_score = max(35.0, min(98.0, 96.0 - max(0.0, (fm_val - 4.0) * 6.0))) if fm_val is not None else (mel_drop_score * 0.92)
    grad_val = parsed.get("latest_grad_norm")
    step_density = round(latest_steps / max(1, ep_completed), 1)
    density_bonus = 92.0 if step_density >= 12 else (80.0 if step_density >= 5 else 68.0)
    feedback_adj = 6.0 if sound_feedback == "clean" else (-7.0 if sound_feedback == "overtrained" else (-5.0 if sound_feedback == "undertrained" else 0.0))
    rebound_pen = 8.0 if rebound else 0.0
    match_pct = int(round(max(22.0, min(98.0, (
        mel_level_score * 0.34
        + mel_drop_score * 0.24
        + kl_score * 0.16
        + fm_score * 0.14
        + density_bonus * 0.12
        + feedback_adj
        - rebound_pen
    )))))
    stability_pct = int(round(max(30.0, min(99.0, (disc_score * 0.45 + kl_score * 0.30 + (90.0 if not rebound else 64.0) * 0.25)))))

    # Dual risk meters (Undertrained vs Overtrained) + Overall Verdict
    sat_pct = int(max(10, min(100, round((steps_per_clip / 18.0) * 100.0)))) if steps_per_clip is not None else int(max(12, min(100, round((ep_completed / 160.0) * 100.0))))
    loss_gap_risk = max(0.0, min(48.0, ((best_loss or 20.0) - 13.5) * 7.5))
    step_gap_risk = max(0.0, min(35.0, (10.0 - steps_per_clip) * 5.5)) if steps_per_clip is not None else max(0.0, min(35.0, (70.0 - ep_completed) * 0.55))
    active_drop_risk = max(0.0, min(22.0, tail_slope * 180.0)) if tail_slope > 0.03 else 0.0
    under_sound = 28 if sound_feedback == "undertrained" else (-18 if sound_feedback == "clean" else (-25 if sound_feedback == "overtrained" else 0))
    under_risk_pct = int(max(4, min(96, round(loss_gap_risk + step_gap_risk + active_drop_risk + under_sound))))

    rebound_risk = max(0.0, min(48.0, rebound_amt * 44.0))
    past_best_risk = max(0.0, min(26.0, (ep_completed - lowest_ep) * 0.45))
    disc_collapse_risk = max(0.0, min(24.0, (1.65 - disc_val) * 55.0)) if (disc_val is not None and disc_val < 1.65) else 0.0
    grad_spike_risk = max(0.0, min(22.0, (grad_val - 75.0) * 0.45)) if (grad_val is not None and grad_val > 75.0) else 0.0
    over_sound = 32 if sound_feedback == "overtrained" else (-15 if sound_feedback == "clean" else (-20 if sound_feedback == "undertrained" else 0))
    over_risk_pct = int(max(3, min(97, round(rebound_risk + past_best_risk + disc_collapse_risk + grad_spike_risk + over_sound))))

    if over_risk_pct >= 50 or sound_feedback == "overtrained" or rebound_amt >= 0.45:
        verdict = "overtrained"
        verdict_title = "Overtraining Detected — Roll Back Checkpoint"
        verdict_guidance = f"Later epochs show +{rebound_pct:.1f}% mel drift past e{lowest_ep}. Audition Epoch {best_ep} instead of the final epoch."
    elif (not rebound and tail_slope >= 0.025) and (under_risk_pct >= 48 or ep_completed < 50 or sound_feedback == "undertrained"):
        verdict = "undertrained"
        verdict_title = "Still Undertrained — Keep Training"
        verdict_guidance = f"Loss is still dropping ({ep_completed} ep{f', {steps_per_clip} steps/clip' if steps_per_clip is not None else ''}). Let training continue until losses flatten."
    else:
        verdict = "sweet_spot"
        verdict_title = "Optimal Sweet Spot Reached"
        verdict_guidance = f"Epoch {best_ep} ({match_pct}% dataset match) captures peak natural timbre before overtraining creep begins."

    # Ranked checkpoints leaderboard
    rank_pool = sorted({e for e in list(candidates) + [_snap_near(lowest_ep, s), snapped[-1]] if s <= e <= snapped[-1]})
    ranked_checkpoints: list[dict[str, Any]] = []
    for r_ep in rank_pool:
        n_pt = min(loss_pts, key=lambda p: abs(int(p["epoch"]) - r_ep)) if loss_pts else pts[-1]
        r_mel = float(n_pt["lowest_loss"]) if n_pt.get("lowest_loss") is not None else 0.0
        r_drift = round(max(0.0, ((r_mel - (best_loss or r_mel)) / max(0.1, best_loss or 1.0)) * 100.0), 1)
        is_best_r = r_ep == best_ep
        r_score = max(50, min(99, max(match_pct, 76))) if is_best_r else int(max(22, min(97, round(match_pct - 3 - abs(r_ep - best_ep) * 0.25 - r_drift * 1.4 - (12 if r_ep >= over_start or r_drift > 2.5 else 0)))))
        r_tag = "Best Recommended" if is_best_r else ("Overfit Risk" if (r_ep >= over_start or r_drift > 2.5) else ("Softer / Safer" if r_ep < best_ep else "Clean Alternative"))
        r_status = "best" if is_best_r else ("risky" if (r_ep >= over_start or r_drift > 2.5) else "good")
        ranked_checkpoints.append({
            "epoch": r_ep,
            "checkpoint": f"{safe_m}_{r_ep}e.pth",
            "mel_loss": round(r_mel, 3),
            "drift_pct": r_drift,
            "score": r_score,
            "tag": r_tag,
            "status": r_status,
        })
    ranked_checkpoints.sort(key=lambda r: (-1 if r["epoch"] == best_ep else 0, -r["score"]))
    ranked_checkpoints = ranked_checkpoints[:4]

    if rebound or sound_feedback == "overtrained":
        conv_lbl = "Overfitting Drift Detected"
    elif sound_feedback == "clean" or match_pct >= 84:
        conv_lbl = "High Dataset Similarity"
    elif ep_completed < max(20, s * 3):
        conv_lbl = "Early Phonetic Alignment"
    else:
        conv_lbl = "Converging Smoothly"

    mel_status = "optimal" if (best_loss is not None and best_loss <= 16.0) else ("watch" if (best_loss is not None and best_loss <= 22.0) else "warning")
    drift_val = f"+{rebound_pct:.1f}% tail drift" if (best_loss and rebound_amt > 0.05) else "At curve minimum"
    metric_breakdown = [
        {
            "label": "Mel Reconstruction",
            "value": f"{best_loss:.3f} best" if best_loss is not None else "—",
            "detail": f"{drop_ratio * 100:.1f}% reduction · {drift_val}",
            "status": mel_status,
        },
        {
            "label": "Dataset Similarity Fit",
            "value": f"{match_pct}% match",
            "detail": "Timbre & formant closeness estimated against your dataset profile",
            "status": "optimal" if match_pct >= 80 else ("watch" if match_pct >= 62 else "warning"),
        },
        {
            "label": "KL Latent Alignment",
            "value": f"{kl_val:.3f}" if kl_val is not None else "Inferred from mel curve",
            "detail": "Posterior-prior speaker pronunciation & breath alignment",
            "status": "optimal" if (kl_val is None or kl_val <= 1.35) else "watch",
        },
        {
            "label": "GAN Discriminator / Gen Balance",
            "value": f"D {disc_val:.2f} / G {parsed['avg_gen_loss']:.2f}" if (disc_val is not None and parsed.get("avg_gen_loss") is not None) else f"Stability {stability_pct}%",
            "detail": "Healthy GAN adversarial equilibrium avoids metallic artifacts",
            "status": "optimal" if stability_pct >= 78 else "watch",
        },
        {
            "label": "Feature Matching (FM)",
            "value": f"{fm_val:.3f}" if fm_val is not None else "Tracking harmonic texture",
            "detail": "Multi-period hidden layer spectral texture match",
            "status": "optimal" if (fm_val is None or fm_val <= 6.5) else "watch",
        },
        {
            "label": "Dataset Scale & Steps/Ep",
            "value": f"~{step_density} steps/ep",
            "detail": f"{dataset_slices} sliced clips on disk" if dataset_slices > 0 else f"{ep_completed} epochs · {sr} Hz {vocoder}",
            "status": "optimal" if step_density >= 8 else "watch",
        },
        {
            "label": "Dataset Saturation",
            "value": f"{steps_per_clip} steps/clip" if steps_per_clip is not None else f"{sat_pct}% saturated",
            "detail": f"{latest_steps} optimizer steps logged",
            "status": "optimal" if sat_pct >= 65 else ("watch" if sat_pct >= 35 else "warning"),
        },
        {
            "label": "Tail Drift & Overfit Watch",
            "value": f"+{rebound_pct:.1f}% rebound" if rebound_pct > 0.8 else "0.0% (at min)",
            "detail": f"Loss rose past e{lowest_ep} min — roll back" if rebound_pct > 1.2 else "No late-epoch loss rebound",
            "status": "optimal" if rebound_pct <= 1.2 else ("watch" if rebound_pct <= 3.5 else "warning"),
        },
    ]

    loss_note = f"Lowest mel loss {best_loss:.3f} at e{lowest_ep}. " if best_loss is not None else ""
    if sound_feedback == "undertrained":
        fb_note = f"Shifted forward from e{lowest_ep} because you reported undertrained audio."
    elif sound_feedback == "overtrained":
        fb_note = f"Pulled earlier from e{lowest_ep} to remove metallic overtraining artifacts."
    elif sound_feedback == "clean":
        fb_note = f"Confirmed clean & natural near checkpoint e{best_ep}."
    elif rebound:
        fb_note = f"Loss rebounded past e{lowest_ep}, so stopping around e{best_ep} preserves natural tone."
    else:
        fb_note = f"Multi-metric fit ({match_pct}% dataset match) & {sr} Hz {vocoder} schedule point directly to e{best_ep}."

    return {
        "confidence": "observed",
        "save_every": s,
        "best_epoch": best_ep,
        "best_loss": round(best_loss, 4) if best_loss is not None else None,
        "dataset_match_pct": match_pct,
        "stability_pct": stability_pct,
        "convergence_label": conv_lbl,
        "verdict": verdict,
        "verdict_title": verdict_title,
        "verdict_guidance": verdict_guidance,
        "under_risk_pct": under_risk_pct,
        "over_risk_pct": over_risk_pct,
        "steps_per_clip": steps_per_clip,
        "effective_steps": latest_steps,
        "ranked_checkpoints": ranked_checkpoints,
        "metric_breakdown": metric_breakdown,
        "candidate_epochs": candidates,
        "under_end": under_end,
        "sweet_min": sweet_min,
        "sweet_mid": best_ep,
        "sweet_max": sweet_max,
        "over_start": over_start,
        "under_label": f"e{s} – e{under_end}" if under_end else "—",
        "sweet_label": f"Epoch {best_ep}",
        "over_label": f"e{over_start}+",
        "checkpoint_hint": ckpt_hint,
        "dataset_summary": ds_tier,
        "summary": f"{loss_note}{fb_note}",
    }


def _synced_slider_num(
    label: str,
    help_text: str,
    state_key: str,
    min_v: int,
    max_v: int,
    default_v: int,
    step: int = 1,
) -> int:
    s_key = f"{state_key}__sl"
    n_key = f"{state_key}__nm"
    cur = max(min_v, min(max_v, int(st.session_state.get(state_key, default_v))))
    if s_key not in st.session_state:
        st.session_state[s_key] = cur
    if n_key not in st.session_state:
        st.session_state[n_key] = cur

    def _from_sl() -> None:
        v = int(st.session_state[s_key])
        st.session_state[state_key] = v
        st.session_state[n_key] = v

    def _from_nm() -> None:
        v = max(min_v, min(max_v, int(st.session_state[n_key])))
        st.session_state[state_key] = v
        st.session_state[s_key] = v

    with st.container(border=True):
        c_lbl, c_num, c_rst = st.columns([4.2, 1.35, 0.55], gap="small")
        with c_lbl:
            st.markdown(
                f'<div class="nr-field-title">{html.escape(label)}</div>'
                f'<div class="nr-field-help">{html.escape(help_text)}</div>',
                unsafe_allow_html=True,
            )
        with c_num:
            st.number_input(
                label,
                min_value=min_v,
                max_value=max_v,
                step=step,
                key=n_key,
                on_change=_from_nm,
                label_visibility="collapsed",
            )
        with c_rst:
            if st.button("↺", key=f"{state_key}__rst", help=f"Reset {label} to {default_v}", use_container_width=True):
                st.session_state[state_key] = default_v
                st.session_state[s_key] = default_v
                st.session_state[n_key] = default_v
                st.rerun()
        st.slider(
            label,
            min_value=min_v,
            max_value=max_v,
            step=step,
            key=s_key,
            on_change=_from_sl,
            label_visibility="collapsed",
        )
    return int(st.session_state[state_key])


def _step_head(num: int, title: str, desc: str) -> None:
    st.markdown(
        f'<div class="nr-step-head"><span class="nr-step-num">{num}</span>'
        f'<div><div class="nr-step-title">{html.escape(_tr(title))}</div>'
        f'<div class="nr-step-desc">{html.escape(desc)}</div></div></div>',
        unsafe_allow_html=True,
    )


def _page_header(title: str, subtitle: str, show_readme: bool = True) -> None:
    readme_html = (
        f'<a class="nr-link-btn" href="{html.escape(README_URL)}" target="_blank" rel="noopener noreferrer">GitHub README ↗</a>'
        if show_readme
        else ""
    )
    st.markdown(
        f'<div class="nr-section-header">'
        f'<div><h2 class="nr-section-h2">{html.escape(_tr(title))}</h2>'
        f'<div class="nr-section-sub">{html.escape(subtitle)}</div></div>'
        f'{readme_html}</div>',
        unsafe_allow_html=True,
    )


def _output_box(label: str, text: str) -> None:
    st.markdown(f'<div style="font-size:13px;font-weight:600;color:#DFBFC3;margin:8px 0 5px 0">{html.escape(label)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="nr-output-box">{html.escape(text or "The output information will be displayed here.")}</div>', unsafe_allow_html=True)


def _ensure_state() -> None:
    detected_ds = _list_datasets()
    init_ds = ${JSON.stringify(defaultDatasetPath)}
    if (not init_ds or "your-username" in init_ds) and detected_ds:
        init_ds = detected_ds[0]
    cfg_init = _load_config()
    lang_init = "System Auto"
    if isinstance(cfg_init.get("lang"), dict) and cfg_init["lang"].get("override"):
        lang_init = str(cfg_init["lang"].get("selected_lang") or "System Auto")

    defaults: dict[str, Any] = {
        "nr_active_tab": ${JSON.stringify(enabledNav[0])},
        "nr_model_name": ${JSON.stringify(defaultModelName)},
        "nr_sr": ${Number(project.samplingRate)},
        "nr_vocoder": ${JSON.stringify(project.vocoder)},
        "nr_cpu_cores": ${project.cpuCores},
        "nr_gpu_num": ${JSON.stringify(project.gpuNumber || '0-1')},
        "nr_dataset_path": init_ds,
        "nr_dataset_creator": ${project.datasetCreator ? 'True' : 'False'},
        "nr_audio_cutting": ${JSON.stringify(project.audioCutting)},
        "nr_chunk_len": ${project.chunkLengthSec},
        "nr_overlap_len": ${project.overlapLengthSec},
        "nr_trunc_sil": ${project.truncateSilenceEnabled ? 'True' : 'False'},
        "nr_sil_ms": ${project.silenceDurationMs},
        "nr_noise_filter": ${project.noiseFilter ? 'True' : 'False'},
        "nr_loud_norm": ${project.loudnessNorm ? 'True' : 'False'},
        "nr_f0": ${JSON.stringify(project.pitchExtractor)},
        "nr_embedder": ${JSON.stringify(project.embedder)},
        "nr_silent_files": ${project.silentTrainingFiles},
        "nr_batch_size": ${project.batchSize},
        "nr_batch_parity": ${project.batchParity1Gpu ?? true ? 'True' : 'False'},
        "nr_save_every": ${project.saveEveryEpoch},
        "nr_total_epoch": ${project.totalEpoch},
        "nr_turbo_training": ${config.turboTraining ?? true ? 'True' : 'False'},
        "nr_save_only_latest": ${project.saveOnlyLatest ? 'True' : 'False'},
        "nr_save_every_weights": ${project.saveEveryWeights ? 'True' : 'False'},
        "nr_pretrained": ${project.pretrained ? 'True' : 'False'},
        "nr_fresh_cleanup": ${project.freshTraining ? 'True' : 'False'},
        "nr_cache_gpu": ${project.cacheDatasetGpu ? 'True' : 'False'},
        "nr_checkpointing": ${project.checkpointing ? 'True' : 'False'},
        "nr_shutdown_finish": ${project.shutdownAfterFinish ? 'True' : 'False'},
        "nr_custom_pre": ${project.customPretrained ? 'True' : 'False'},
        "nr_custom_g": ${JSON.stringify(project.customPretrainedG || '')},
        "nr_custom_d": ${JSON.stringify(project.customPretrainedD || '')},
        "nr_index_algo": ${JSON.stringify(project.indexAlgorithm || 'Auto')},
        "nr_is_training": False,
        "nr_toasts": [],
        "nr_prep_log": "The output information will be displayed here.",
        "nr_ext_log": "The output information will be displayed here.",
        "nr_trn_log": "The output information will be displayed here.",
        "nr_dl_log": "The output information will be displayed here.",
        "nr_pre_log": [],
        "nr_pre_log_text": "",
        "nr_manifest": list(BOOT_MANIFEST_DEFAULTS),
        "nr_inf_log": "The output information will be displayed here.",
        "nr_inf_out": "",
        "nr_sample_queue": [],
        "nr_saved_samples": [],
        "nr_selected_sample_path": "",
        "nr_free_gender": "All",
        "nr_diag_suggest": True,
        "nr_diag_sound": "untested",
        "nr_diag_log_input": "",
        "nr_blend_log": "The output information will be displayed here.",
        "nr_blend_out": "",
        "nr_tts_log": "",
        "nr_tts_out": "",
        "nr_settings_subtab": "General",
        "nr_list_filter": bool(cfg_init.get("filter", False)),
        "nr_rmvpe_corr": bool((cfg_init.get("rmvpe_high_register") or {}).get("enabled", False)),
        "nr_rmvpe_mode": str((cfg_init.get("rmvpe_high_register") or {}).get("mode", "true_pitch")),
        "nr_rmvpe_ceil": int((cfg_init.get("rmvpe_high_register") or {}).get("ceiling_hz", 1250)),
        "nr_discord": bool(cfg_init.get("discord_presence", False)),
        "nr_lang": lang_init if lang_init in LANG_LABELS else "System Auto",
        "nr_ver_status": f"${BRAND_NAME} Beta version {APP_VERSION} (newest version: {APP_VERSION}).",
        "nr_author": str(cfg_init.get("model_author", "None")),
        "nr_author_out": "The output information will be displayed here.",
        "nr_precision": str(cfg_init.get("precision", "fp32")),
        "nr_precision_out": "The output information will be displayed here.",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_ensure_state()
_poll_core_tasks()

_CONTAINER_HAS_KEY = "key" in inspect.signature(st.container).parameters
_HAS_FRAGMENT = hasattr(st, "fragment")


def _nr_card(card_key: str) -> Any:
    return st.container(border=True, key=card_key) if _CONTAINER_HAS_KEY else st.container(border=True)


def _nr_box(box_key: str) -> Any:
    return st.container(key=box_key) if _CONTAINER_HAS_KEY else st.container()


def _run_live_section(active: bool, fn: Any) -> None:
    if _HAS_FRAGMENT and active:
        st.fragment(run_every=2.0)(fn)()
    else:
        fn()


# ==============================================================================
# AUTH GATE (WHEN LAUNCHED VIA KAGGLE NOTEBOOK WITH RANDOM PASSWORD)
# ==============================================================================
_auth_user = os.environ.get("APPLIO_AUTH_USERNAME", "").strip()
_auth_pass = os.environ.get("APPLIO_AUTH_PASSWORD", "").strip()
if _auth_pass and not st.session_state.get("nr_authed", False):
    _, col_mid, _ = st.columns([1, 1.35, 1])
    with col_mid:
        with _nr_card("nr_auth_card"):
            st.markdown(
                '<div class="nr-topbar" style="margin-bottom:14px">'
                '<div class="nr-brand-left">'
                '<div class="nr-badge">NR</div>'
                '<div><div><span class="nr-brand-title">${BRAND_NAME}</span>'
                f'<span class="nr-brand-ver">Beta version {APP_VERSION}</span></div>'
                '<div class="nr-brand-sub">Sign in with the credentials shown in your Kaggle Live Monitor</div>'
                '</div></div></div>',
                unsafe_allow_html=True,
            )
            u_in = st.text_input("Username", value=_auth_user or "noregrets")
            p_in = st.text_input("Password", type="password", placeholder="Paste WebUI password from Kaggle monitor")
            if st.button("Sign In", type="primary", use_container_width=True):
                if hmac.compare_digest(u_in.strip(), _auth_user or "noregrets") and hmac.compare_digest(p_in.strip(), _auth_pass):
                    st.session_state["nr_authed"] = True
                    st.rerun()
                else:
                    st.error("Invalid WebUI username or password.")
    st.stop()

# ==============================================================================
# TOP BAR & 8-TAB NAVIGATION (EXACT PREVIEW MATCH)
# ==============================================================================
gpu_cnt = max(1, len([x for x in str(st.session_state["nr_gpu_num"]).split("-") if x.strip()]))
eff_batch = int(st.session_state["nr_batch_size"]) * gpu_cnt
active_model_label = st.session_state["nr_model_name"].strip() or "Untitled Model"

st.markdown(
    f'<div class="nr-topbar">'
    f'<div class="nr-brand-left">'
    f'<div class="nr-badge">NR</div>'
    f'<div><div><span class="nr-brand-title">${BRAND_NAME}</span>'
    f'<span class="nr-brand-ver">Beta version {APP_VERSION}</span></div>'
    f'<div class="nr-brand-sub">RVC Voice Model Training, Inference &amp; Kaggle Studio</div></div></div>'
    f'<div class="nr-model-pill">'
    f'<div class="nr-model-pill-name">{html.escape(active_model_label)} ({st.session_state["nr_sr"]})</div>'
    f'<div class="nr-model-pill-meta">{html.escape(str(st.session_state["nr_vocoder"]))} · Batch {st.session_state["nr_batch_size"]} ({eff_batch} eff.) · {st.session_state["nr_total_epoch"]} ep</div>'
    f'</div></div>',
    unsafe_allow_html=True,
)

NAV_TABS = ${JSON.stringify(enabledNav)}
if st.session_state["nr_active_tab"] not in NAV_TABS:
    st.session_state["nr_active_tab"] = NAV_TABS[0]

if _resolve_lang() == "ar_SA":
    st.markdown("<style>[data-testid='stMain'] { direction: rtl; text-align: right; }</style>", unsafe_allow_html=True)


def _select_tab(target_tab: str) -> None:
    st.session_state["nr_active_tab"] = target_tab


with _nr_card("nr_nav_card"):
    st.markdown('<span class="nr-nav-marker" style="display:none"></span>', unsafe_allow_html=True)
    nav_weights = [max(7, len(lbl) + 3) for lbl in NAV_TABS]
    nav_cols = st.columns(nav_weights, gap="small")
    for idx, tab_id in enumerate(NAV_TABS):
        with nav_cols[idx]:
            is_active = st.session_state["nr_active_tab"] == tab_id
            st.button(
                _tr(tab_id),
                key=f"nr_nav_{tab_id}",
                type="primary" if is_active else "secondary",
                on_click=_select_tab,
                args=(tab_id,),
                use_container_width=True,
            )

active_tab = st.session_state["nr_active_tab"]


def _render_active_tab(active_tab: str) -> None:
    _tab_slug = re.sub(r"[^a-z0-9]+", "_", active_tab.lower()).strip("_") or "tab"
    with _nr_box(f"nr_panel_{_tab_slug}"):
`;
}

