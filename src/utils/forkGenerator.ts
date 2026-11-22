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
  return `from __future__ import annotations
from typing import Iterable
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes


class NoRegretsRVCTheme(Base):
    """Locked deep crimson-red Gradio theme for ${BRAND_NAME}."""

    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.neutral,
        secondary_hue: colors.Color | str = colors.neutral,
        neutral_hue: colors.Color | str = colors.neutral,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_lg,
        text_size: sizes.Size | str = sizes.text_md,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Plus Jakarta Sans"),
            "ui-sans-serif",
            "system-ui",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("JetBrains Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        self.name = "${BRAND_NAME}"
        super().set(
            body_background_fill="#0D0405",
            body_background_fill_dark="#0D0405",
            background_fill_primary="#160709",
            background_fill_primary_dark="#160709",
            background_fill_secondary="#200B0E",
            background_fill_secondary_dark="#200B0E",
            block_background_fill="#160709",
            block_background_fill_dark="#160709",
            block_border_color="#47161D",
            block_border_color_dark="#47161D",
            block_border_width="1px",
            block_border_width_dark="1px",
            block_label_background_fill="#200B0E",
            block_label_background_fill_dark="#200B0E",
            block_label_text_color="#DFBFC3",
            block_label_text_color_dark="#DFBFC3",
            block_title_text_color="#FDF7F7",
            block_title_text_color_dark="#FDF7F7",
            body_text_color="#FDF7F7",
            body_text_color_dark="#FDF7F7",
            body_text_color_subdued="#AD8288",
            body_text_color_subdued_dark="#AD8288",
            input_background_fill="#100506",
            input_background_fill_dark="#100506",
            input_border_color="#47161D",
            input_border_color_dark="#47161D",
            input_border_color_focus="${ACCENT_HEX}",
            input_border_color_focus_dark="${ACCENT_HEX}",
            button_primary_background_fill="${ACCENT_HEX}",
            button_primary_background_fill_dark="${ACCENT_HEX}",
            button_primary_background_fill_hover="#FF2B35",
            button_primary_background_fill_hover_dark="#FF2B35",
            button_primary_text_color="#FFFFFF",
            button_primary_text_color_dark="#FFFFFF",
            button_primary_border_color="#9F1018",
            button_primary_border_color_dark="#9F1018",
            button_secondary_background_fill="#200B0E",
            button_secondary_background_fill_dark="#200B0E",
            button_secondary_background_fill_hover="#331116",
            button_secondary_background_fill_hover_dark="#331116",
            button_secondary_text_color="#FDF7F7",
            button_secondary_text_color_dark="#FDF7F7",
            button_secondary_border_color="#47161D",
            button_secondary_border_color_dark="#47161D",
            slider_color="${ACCENT_HEX}",
            slider_color_dark="${ACCENT_HEX}",
            checkbox_background_color_selected="${ACCENT_HEX}",
            checkbox_background_color_selected_dark="${ACCENT_HEX}",
            checkbox_border_color_focus="${ACCENT_HEX}",
            checkbox_border_color_focus_dark="${ACCENT_HEX}",
            color_accent="${ACCENT_HEX}",
            color_accent_soft="#5E0E14",
        )
`;
}

export function generateAppPython(
  config: ForkConfig,
  project: VoiceModelProject
): string {
  const tabDefs: Record<
    ApplioTabId,
    { importLine: string; tabTitle: string; fnCall: string; subtitle: string }
  > = {
    train: {
      importLine: 'from tabs.train.train import train_tab',
      tabTitle: 'Training',
      fnCall: 'train_tab()',
      subtitle: 'Set up your voice dataset, extract acoustic features, and run RVC training.',
    },
    models: {
      importLine: 'from tabs.download.download import download_tab',
      tabTitle: 'Models',
      fnCall: 'download_tab()',
      subtitle: 'Fetch remote voice archives, drop local weights, and manage custom pretrains.',
    },
    samples: {
      importLine: 'from tabs.inference.inference import inference_tab',
      tabTitle: 'Samples',
      fnCall: 'inference_tab()',
      subtitle: 'Run single or batch RVC voice conversion inference with full pitch and index control.',
    },
    voice_blender: {
      importLine: 'from tabs.voice_blender.voice_blender import voice_blender_tab',
      tabTitle: 'Voice Blender',
      fnCall: 'voice_blender_tab()',
      subtitle: 'Merge two compatible RVC checkpoints into a single hybrid voice weights file.',
    },
    tts: {
      importLine: 'from tabs.tts.tts import tts_tab',
      tabTitle: 'TTS',
      fnCall: 'tts_tab()',
      subtitle: 'Synthesize spoken dialogue with Edge-TTS and route it through your RVC model.',
    },
    tensorboard: {
      importLine: 'from tabs.tensorboard.tensorboard import tensorboard_tab',
      tabTitle: 'TensorBoard',
      fnCall: 'tensorboard_tab()',
      subtitle: 'Inspect live training loss curves and TensorBoard metrics.',
    },
    extra: {
      importLine: 'from tabs.extra.extra import extra_tab',
      tabTitle: 'Extra',
      fnCall: 'extra_tab()',
      subtitle: 'Inspect checkpoints, analyze audio files, and extract F0 pitch curves.',
    },
    settings: {
      importLine: 'from tabs.settings.settings import settings_tab',
      tabTitle: 'Settings',
      fnCall: 'settings_tab()',
      subtitle: 'Customize engine preferences, training precision, and language defaults.',
    },
  };

  const enabledTabs = (Object.keys(tabDefs) as ApplioTabId[]).filter(
    (id) => config.activeTabs[id]
  );
  const imports = enabledTabs.map((id) => tabDefs[id].importLine).join('\n');

  const freeSamplesCardBody = `            with gr.Column(scale=0, min_width=0, elem_classes="applio-card"):
                gr.HTML(
                    '<div class="applio-card-head">'
                    '<h2 class="applio-card-heading"><span class="applio-step">1</span>Free Voice Samples</h2>'
                    '<p class="applio-card-desc">Download and audition six dry 32 kHz mono speaking clips directly inside Kaggle.</p>'
                    '</div>',
                    elem_classes="applio-step-header",
                )
                fs_repo_box = gr.Textbox(
                    label="GitHub Repo URL (to fetch the 6 WAV samples into Kaggle)",
                    placeholder="${REPO_URL}",
                    value="${config.customRepoUrl.trim() || REPO_URL}",
                )
                with gr.Row():
                    fs_fetch_btn = gr.Button("Fetch 6 Free Samples from GitHub", variant="primary")
                    fs_copy_btn = gr.Button("Copy Available Free Samples into Samples Tab")
                fs_status = gr.Markdown("Ready. Pick a clip below or fetch from your GitHub repo.")
                fs_catalog = [
                    ("Male Speaking Test 1", "MaleSpeakingTest-1_931b5ffd.wav", "Male · Dry 32 kHz mono speaking clip 1"),
                    ("Male Speaking Test 2", "MaleSpeakingTest-2_c2e6e7fe.wav", "Male · Dry 32 kHz mono speaking clip 2"),
                    ("Male Speaking Test 3", "MaleSpeakingTest-3_d78e8233.wav", "Male · Dry 32 kHz mono speaking clip 3"),
                    ("Female Speaking Test 1", "FemaleSpeakingTest-1_bb3d674f.wav", "Female · Dry 32 kHz mono speaking clip 1"),
                    ("Female Speaking Test 2", "FemaleSpeakingTest-2_d1823132.wav", "Female · Dry 32 kHz mono speaking clip 2"),
                    ("Female Speaking Test 3", "FemaleSpeakingTest-3_458bd4fc.wav", "Female · Dry 32 kHz mono speaking clip 3"),
                ]
                fs_selector = gr.Dropdown(
                    label="Select Free Sample",
                    choices=[f"{title} ({fname})" for title, fname, _ in fs_catalog],
                    value=f"{fs_catalog[0][0]} ({fs_catalog[0][1]})",
                )
                fs_player = gr.Audio(label="Sample Preview & Download", type="filepath", interactive=False)

                def _resolve_sample_path(choice_label: str):
                    for title, fname, _ in fs_catalog:
                        if f"{title} ({fname})" == choice_label:
                            for d in ("assets/free_samples", "assets/audios"):
                                p = os.path.join(now_dir, d, fname)
                                if os.path.isfile(p):
                                    return p, f"Loaded **{title}** from \`{d}/{fname}\`."
                            return None, f"**{title}** (\`{fname}\`) is not in \`assets/free_samples\` yet. Paste your GitHub repo URL above and click Fetch."
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
                        return f"Could not find \`public/assets/aistudio/*.wav\` on \`{owner_repo}\` (main/master).", None
                    return f"Fetched {downloaded}/6 free samples into \`assets/free_samples\` and \`assets/audios\`. Click Refresh in Samples if needed.", first_path if os.path.isfile(first_path) else None

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
                    return f"Copied {copied} free samples to \`assets/audios\`."

                fs_selector.change(_resolve_sample_path, inputs=[fs_selector], outputs=[fs_player, fs_status])
                fs_fetch_btn.click(_fetch_from_github, inputs=[fs_repo_box], outputs=[fs_status, fs_player])
                fs_copy_btn.click(_copy_to_audios, outputs=[fs_status])`;

  interface PageSpec {
    key: string;
    title: string;
    subtitle: string;
    body: string;
  }
  const orderedPages: PageSpec[] = [];
  for (const id of enabledTabs) {
    const def = tabDefs[id];
    const body =
      id === 'settings' || id === 'extra'
        ? `            with gr.Tabs(elem_classes="nr-subtabs"):\n                ${def.fnCall}`
        : `            ${def.fnCall}`;
    orderedPages.push({
      key: id,
      title: def.tabTitle,
      subtitle: def.subtitle,
      body,
    });
    if (id === 'samples') {
      orderedPages.push({
        key: 'free_samples',
        title: 'Free Samples',
        subtitle: 'Six dry 32 kHz mono speaking clips (3 male, 3 female) ready for testing in Samples.',
        body: freeSamplesCardBody,
      });
    }
  }
  if (!enabledTabs.includes('samples')) {
    orderedPages.push({
      key: 'free_samples',
      title: 'Free Samples',
      subtitle: 'Six dry 32 kHz mono speaking clips (3 male, 3 female) ready for testing in Samples.',
      body: freeSamplesCardBody,
    });
  }

  const navBtnInits = orderedPages
    .map(
      (p, idx) =>
        `            nav_btns.append(gr.Button("${p.title}", variant="${idx === 0 ? 'primary' : 'secondary'}", size="sm", elem_classes="nr-nav-btn"))`
    )
    .join('\n');

  const pageColBlocks = orderedPages
    .map(
      (p, idx) =>
        `        with gr.Column(visible=${idx === 0 ? 'True' : 'False'}, elem_classes="nr-page-col") as _page_${idx}:\n` +
        `            gr.HTML(\n` +
        `                '<div class="nr-page-heading">'\n` +
        `                '<div><h2>${p.title}</h2><p>${p.subtitle}</p></div>'\n` +
        `                '<a class="nr-readme-pill" href="${REPO_README_URL}" target="_blank" rel="noopener noreferrer">GitHub README ↗</a>'\n` +
        `                '</div>'\n` +
        `            )\n` +
        `${p.body}\n` +
        `        page_cols.append(_page_${idx})`
    )
    .join('\n');

  const safeHeaderName = (project.name.trim() || 'Untitled Model').replace(/[<>&"']/g, '');
  const effBatch = project.batchSize * Math.max(1, project.gpuCount || 2);

  return `import argparse
import json
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
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as _cf:
            _cfg_data = json.load(_cf)
        if not isinstance(_cfg_data, dict):
            _cfg_data = {}
        _changed = False
        if _cfg_data.get("precision") not in ("fp32", "bf16") and _cfg_data.get("nr_precision_initialized") is not True:
            _cfg_data["precision"] = "fp32"
            _cfg_data["nr_precision_initialized"] = True
            _changed = True
        if _cfg_data.get("discord_presence") is True and _cfg_data.get("nr_discord_initialized") is not True:
            _cfg_data["discord_presence"] = False
            _cfg_data["nr_discord_initialized"] = True
            _changed = True
        if _changed:
            with open(CONFIG_PATH, "w", encoding="utf-8") as _cf:
                json.dump(_cfg_data, _cf, indent=4, ensure_ascii=False)
    except Exception:
        pass

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
${imports}

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root, body, .gradio-container {
  --app-bg: #0D0405;
  --app-surface: #160709;
  --app-panel: #100506;
  --app-border: #47161D;
  --app-text: #FDF7F7;
  --app-muted: #AD8288;
  --app-heading: #FDF7F7;
  --font: "Plus Jakarta Sans", ui-sans-serif, system-ui, sans-serif !important;
  --font-mono: "JetBrains Mono", ui-monospace, monospace !important;
  --body-background-fill: #0D0405 !important;
  --body-background-fill-dark: #0D0405 !important;
  --background-fill-primary: #160709 !important;
  --background-fill-secondary: #200B0E !important;
  --block-background-fill: #160709 !important;
  --block-border-color: #47161D !important;
  --block-border-width: 1px !important;
  --block-radius: 12px !important;
  --block-label-background-fill: transparent !important;
  --block-label-border-width: 0px !important;
  --block-label-padding: 0px !important;
  --block-label-text-color: #FDF7F7 !important;
  --block-title-background-fill: transparent !important;
  --block-title-border-width: 0px !important;
  --block-title-padding: 0px !important;
  --block-title-text-color: #FDF7F7 !important;
  --block-info-text-color: #AD8288 !important;
  --border-color-primary: #47161D !important;
  --panel-background-fill: #160709 !important;
  --panel-border-color: #47161D !important;
  --input-background-fill: #100506 !important;
  --input-background-fill-focus: #100506 !important;
  --input-border-color: #47161D !important;
  --input-border-color-focus: ${ACCENT_HEX} !important;
  --input-radius: 8px !important;
  --input-placeholder-color: #8A5E64 !important;
  --form-gap-width: 14px !important;
  --button-primary-background-fill: ${ACCENT_HEX} !important;
  --button-primary-background-fill-hover: #FF2B35 !important;
  --button-primary-text-color: #FFFFFF !important;
  --button-primary-border-color: #9F1018 !important;
  --button-secondary-background-fill: #200B0E !important;
  --button-secondary-background-fill-hover: #331116 !important;
  --button-secondary-border-color: #47161D !important;
  --button-secondary-text-color: #FDF7F7 !important;
  --button-large-radius: 8px !important;
  --button-small-radius: 8px !important;
  --checkbox-background-color: #100506 !important;
  --checkbox-background-color-selected: ${ACCENT_HEX} !important;
  --checkbox-border-color: #6E232D !important;
  --checkbox-border-color-selected: ${ACCENT_HEX} !important;
  --checkbox-label-background-fill: #100506 !important;
  --checkbox-label-background-fill-selected: rgba(229, 25, 35, 0.14) !important;
  --checkbox-label-text-color-selected: #FFFFFF !important;
  --slider-color: ${ACCENT_HEX} !important;
  --color-accent: ${ACCENT_HEX} !important;
  color-scheme: dark;
  background: #0D0405 !important;
  color: #FDF7F7 !important;
  font-family: var(--font) !important;
  -webkit-font-smoothing: antialiased;
}

/* Lock full width on every page so Models / Samples / Voice Blender / TTS never shrink */
.gradio-container {
  width: 100% !important;
  max-width: 1280px !important;
  margin: 0 auto !important;
  padding: 20px 24px 40px !important;
  box-sizing: border-box !important;
}
.gradio-container > .main,
.gradio-container > .main > .wrap,
.gradio-container > .main > .wrap > .contain,
.gradio-container .contain,
.nr-page-col,
.applio-card {
  width: 100% !important;
  max-width: 100% !important;
  box-sizing: border-box !important;
}
footer { display: none !important; }

/* Custom Red Side Scrollbar */
* { scrollbar-width: thin; scrollbar-color: ${ACCENT_HEX} #100506; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: #100506; }
::-webkit-scrollbar-thumb { background: linear-gradient(180deg, ${ACCENT_HEX} 0%, #b91c1c 100%); border-radius: 999px; border: 2px solid #100506; }
::-webkit-scrollbar-thumb:hover { background: #FF2B35; }

/* NoRegretsRVC Studio Top Header Card */
.nr-header-wrap { width: 100% !important; padding: 0 !important; border: 0 !important; background: transparent !important; margin-bottom: 12px !important; box-shadow: none !important; }
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
.nr-title-row h1 { margin: 0 !important; font-size: 20px !important; font-weight: 700 !important; color: #FDF7F7 !important; line-height: 1.2 !important; }
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
.nr-status-meta { margin: 2px 0 0; font-size: 11px; color: #AD8288; font-family: var(--font-mono); }

/* Fixed Pill Navigation Bar (never collapses tabs into ... overflow) */
#nr-nav-bar {
  width: 100% !important;
  background: #160709 !important;
  border: 1px solid #47161D !important;
  border-radius: 12px !important;
  padding: 6px !important;
  gap: 6px !important;
  margin-bottom: 18px !important;
  display: flex !important;
  flex-wrap: wrap !important;
  align-items: center !important;
  box-sizing: border-box !important;
}
#nr-nav-bar > button,
#nr-nav-bar .nr-nav-btn {
  flex: 0 0 auto !important;
  width: auto !important;
  min-width: 0 !important;
  min-height: 36px !important;
  padding: 7px 14px !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  border-radius: 8px !important;
  border: 0 !important;
  box-shadow: none !important;
  margin: 0 !important;
}
#nr-nav-bar > button.secondary,
#nr-nav-bar .nr-nav-btn.secondary {
  background: transparent !important;
  color: #DFBFC3 !important;
}
#nr-nav-bar > button.secondary:hover,
#nr-nav-bar .nr-nav-btn.secondary:hover {
  background: #200B0E !important;
  color: #FFFFFF !important;
}
#nr-nav-bar > button.primary,
#nr-nav-bar .nr-nav-btn.primary {
  background: ${ACCENT_HEX} !important;
  color: #FFFFFF !important;
}

.nr-page-col {
  padding: 0 !important;
  gap: 16px !important;
  border: 0 !important;
  background: transparent !important;
}
.nr-page-col > .row,
.nr-page-col > .column,
.applio-card > .row,
.applio-card > .column {
  flex: 0 0 auto !important;
  width: 100% !important;
}

/* Page Section Heading inside each page */
.nr-page-heading {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  width: 100%;
  margin: 0 0 4px !important;
  padding: 0 2px;
}
.nr-page-heading h2 { margin: 0 !important; font-size: 24px !important; font-weight: 700 !important; color: #FDF7F7 !important; letter-spacing: -0.02em; }
.nr-page-heading p { margin: 4px 0 0 !important; font-size: 13px !important; color: #DFBFC3 !important; }
.nr-readme-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: #160709;
  border: 1px solid #47161D;
  border-radius: 8px;
  padding: 7px 12px;
  font-size: 12px;
  font-weight: 500;
  color: #DFBFC3 !important;
  text-decoration: none !important;
  white-space: nowrap;
}
.nr-readme-pill:hover { border-color: ${ACCENT_HEX}; color: #FFFFFF !important; }

/* Applio Step Cards & Crimson Number Badges */
.applio-card {
  background: #160709 !important;
  border: 1px solid #47161D !important;
  border-radius: 12px !important;
  padding: 20px !important;
  gap: 16px !important;
  margin-bottom: 16px !important;
  box-shadow: none !important;
}
.applio-step-header, .applio-step-header .html-container {
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
}
.applio-card-head { display: flex; flex-direction: column; gap: 4px; margin: 0 0 4px; }
.applio-card-heading {
  display: flex !important;
  align-items: center !important;
  margin: 0 !important;
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

/* Proper Row Flex Sizing inside Cards (fixes cramped Model Name & overflowing sliders) */
.applio-card .row {
  width: 100% !important;
  gap: 16px !important;
  align-items: stretch !important;
  flex-wrap: wrap !important;
  overflow: visible !important;
}
.applio-card .row > *,
.applio-card .row > .form,
.applio-card .row > .column {
  flex: 1 1 0% !important;
  min-width: min(200px, 100%) !important;
  max-width: 100% !important;
  box-sizing: border-box !important;
}
.applio-card .row > .form {
  display: flex !important;
  flex-direction: row !important;
  flex-wrap: wrap !important;
  gap: 16px !important;
  width: 100% !important;
}
.applio-card .row > .form > .block {
  flex: 1 1 0% !important;
  min-width: min(200px, 100%) !important;
  max-width: 100% !important;
}
.applio-card .row > button {
  flex: 1 1 0% !important;
  min-width: 140px !important;
}

/* Strip Gradio outer form/block boxes so only clean inset inputs have borders */
.gradio-container .form {
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  gap: 14px !important;
  overflow: visible !important;
}
.applio-card .block,
.applio-card .form > .block {
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 !important;
  overflow: visible !important;
}
.applio-card .group,
.applio-card .gr-group {
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 !important;
}
.gradio-container .block > label > span:first-child,
.gradio-container label[data-testid="block-label"] {
  background: transparent !important;
  border: 0 !important;
  padding: 0 0 4px !important;
  color: #FDF7F7 !important;
  font-size: 13px !important;
  font-weight: 600 !important;
}
.gradio-container .block span[data-testid="block-info"],
.gradio-container .info {
  color: #AD8288 !important;
  font-size: 12px !important;
  line-height: 1.4 !important;
}

/* Single crisp border on Gradio Dropdown/Textbox .wrap (no triple nested borders!) */
.gradio-container .wrap {
  background: #100506 !important;
  border: 1px solid #47161D !important;
  border-radius: 8px !important;
  box-shadow: none !important;
}
.gradio-container .wrap:focus-within {
  border-color: ${ACCENT_HEX} !important;
  box-shadow: 0 0 0 2px rgba(229, 25, 35, 0.18) !important;
}
.gradio-container .wrap .wrap-inner,
.gradio-container .wrap .secondary-wrap,
.gradio-container .wrap input,
.gradio-container .wrap textarea {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  color: #FDF7F7 !important;
}
.gradio-container input[type="text"]:not(.wrap input),
.gradio-container textarea:not(.wrap textarea),
.gradio-container select {
  background: #100506 !important;
  border: 1px solid #47161D !important;
  border-radius: 8px !important;
  color: #FDF7F7 !important;
}
.gradio-container input[type="text"]:not(.wrap input):focus,
.gradio-container textarea:not(.wrap textarea):focus {
  border-color: ${ACCENT_HEX} !important;
  box-shadow: 0 0 0 2px rgba(229, 25, 35, 0.18) !important;
}
.gradio-container ul.options {
  background: #160709 !important;
  border: 1px solid #47161D !important;
  border-radius: 8px !important;
  box-shadow: 0 12px 28px rgba(0, 0, 0, 0.55) !important;
}
.gradio-container ul.options li.item:hover,
.gradio-container ul.options li.item.active {
  background: #2D0F14 !important;
  color: #FFFFFF !important;
}

/* Settings, Extra & Card Segmented Sub-Tabs */
.nr-subtabs > .tab-wrapper,
.nr-subtabs > .tab-nav,
.nr-subtabs > [role="tablist"],
.applio-card .tab-wrapper,
.applio-card .tab-nav,
.applio-card [role="tablist"],
.applio-card .tab-container[role="tablist"] {
  background: #100506 !important;
  border: 1px solid #47161D !important;
  border-radius: 9px !important;
  padding: 4px !important;
  gap: 4px !important;
}
.nr-subtabs > .tab-wrapper button,
.nr-subtabs > .tab-nav button,
.nr-subtabs > [role="tablist"] button,
.applio-card .tab-wrapper button,
.applio-card .tab-nav button,
.applio-card [role="tablist"] button,
.applio-card .tab-container[role="tablist"] button {
  border: 0 !important;
  border-radius: 6px !important;
  color: #DFBFC3 !important;
  background: transparent !important;
}
.nr-subtabs button.selected,
.nr-subtabs button[aria-selected="true"],
.applio-card .tab-wrapper button.selected,
.applio-card .tab-wrapper button[aria-selected="true"],
.applio-card .tab-nav button.selected,
.applio-card [role="tablist"] button.selected,
.applio-card [role="tablist"] button[aria-selected="true"],
.applio-card .tab-container[role="tablist"] button.selected,
.applio-card .tab-container[role="tablist"] button[aria-selected="true"] {
  background: ${ACCENT_HEX} !important;
  color: #FFFFFF !important;
}
.nr-subtabs button::after,
.applio-card [role="tablist"] button::after,
.applio-card .tab-container[role="tablist"] button::after { display: none !important; }
.nr-subtabs > .tabitem { border: 0 !important; padding: 14px 0 0 !important; background: transparent !important; }

/* Custom Checkboxes, Radios, Sliders, Compact File Dropzones & Accordions (zero red horizontal scrollbar!) */
.gradio-container input[type="checkbox"] {
  appearance: none !important;
  -webkit-appearance: none !important;
  flex: 0 0 18px !important;
  width: 18px !important;
  min-width: 18px !important;
  max-width: 18px !important;
  height: 18px !important;
  border-radius: 5px !important;
  border: 1px solid #6E232D !important;
  background: #100506 !important;
  cursor: pointer !important;
}
.gradio-container input[type="checkbox"]:checked {
  background-color: ${ACCENT_HEX} !important;
  border-color: ${ACCENT_HEX} !important;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 12 12'%3E%3Cpath d='m2 6.2 2.6 2.6 5.4-5.6' fill='none' stroke='%23ffffff' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E") !important;
  background-size: 12px 12px !important;
  background-position: center !important;
  background-repeat: no-repeat !important;
}
.gradio-container input[type="radio"] {
  appearance: none !important;
  flex: 0 0 16px !important;
  width: 16px !important;
  min-width: 16px !important;
  max-width: 16px !important;
  height: 16px !important;
  border-radius: 50% !important;
  border: 1px solid #6E232D !important;
  background: #100506 !important;
}
.gradio-container input[type="radio"]:checked {
  border-color: ${ACCENT_HEX} !important;
  background: radial-gradient(circle, #FFFFFF 0 3px, ${ACCENT_HEX} 3.5px) !important;
}
.gradio-container .tab-like-container {
  background: #100506 !important;
  border: 1px solid #47161D !important;
  border-radius: 8px !important;
}
.gradio-container .tab-like-container input[type="number"] {
  background: transparent !important;
  border: 0 !important;
  color: #FDF7F7 !important;
}
.gradio-container .tab-like-container .reset-button {
  border-left: 1px solid #47161D !important;
  color: #AD8288 !important;
}
.gradio-container input[type="range"] {
  width: 100% !important;
  min-width: 0 !important;
  max-width: 100% !important;
  accent-color: ${ACCENT_HEX} !important;
}
.gr-accordion,
.accordion {
  background: #100506 !important;
  border: 1px solid #47161D !important;
  border-radius: 10px !important;
  padding: 10px 14px !important;
  width: 100% !important;
  box-sizing: border-box !important;
  overflow: hidden !important;
}
.gr-accordion > button.label-wrap,
.accordion > button.label-wrap,
.gr-accordion > div,
.accordion > div,
[data-testid="accordion-content"] {
  max-width: 100% !important;
  overflow-x: hidden !important;
  scrollbar-width: none !important;
}
.gr-accordion > button.label-wrap::-webkit-scrollbar,
.accordion > button.label-wrap::-webkit-scrollbar,
.gr-accordion > div::-webkit-scrollbar,
.accordion > div::-webkit-scrollbar {
  display: none !important;
  height: 0 !important;
}
.file-preview,
[data-testid="file"] .wrap,
.upload-container {
  min-height: 76px !important;
  background: #100506 !important;
  border: 1px dashed #47161D !important;
  border-radius: 10px !important;
}
[data-testid="file"] button {
  min-height: 68px !important;
  padding: 12px !important;
}
button.primary {
  background: ${ACCENT_HEX} !important;
  border-color: #9F1018 !important;
  color: #FFFFFF !important;
  font-weight: 600 !important;
  border-radius: 8px !important;
}
button.primary:hover { background: #FF2B35 !important; }
button.secondary {
  background: #200B0E !important;
  border-color: #47161D !important;
  color: #FDF7F7 !important;
  border-radius: 8px !important;
}
button.secondary:hover { border-color: ${ACCENT_HEX} !important; background: #2D0F14 !important; }
"""


def _load_theme():
    try:
        return NoRegretsRVCTheme()
    except Exception:
        return gr.themes.Base()


def build_ui(theme=None) -> gr.Blocks:
    if theme is None:
        theme = _load_theme()
    blocks_kwargs = {"title": "${BRAND_NAME} (Beta 0.0.0)"}
    if not GRADIO_6:
        blocks_kwargs["theme"] = theme
        blocks_kwargs["css"] = CUSTOM_CSS
    with gr.Blocks(**blocks_kwargs) as app:
        gr.HTML(
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
            '<p class="nr-status-title">${safeHeaderName} (${project.samplingRate})</p>'
            '<p class="nr-status-meta">${project.vocoder} · Batch ${project.batchSize} (${effBatch} eff.) · ${project.totalEpoch} ep</p>'
            '</div>'
            '</div>',
            elem_classes="nr-header-wrap",
        )
        nav_btns = []
        page_cols = []
        with gr.Row(elem_id="nr-nav-bar"):
${navBtnInits}
${pageColBlocks}

        def _make_nav_handler(target_idx: int):
            def _switch():
                vis_updates = [gr.update(visible=(i == target_idx)) for i in range(len(page_cols))]
                btn_updates = [gr.update(variant=("primary" if i == target_idx else "secondary")) for i in range(len(nav_btns))]
                return (*vis_updates, *btn_updates)
            return _switch

        for _idx, _btn in enumerate(nav_btns):
            _btn.click(
                fn=_make_nav_handler(_idx),
                inputs=[],
                outputs=[*page_cols, *nav_btns],
                queue=False,
            )
    return app


def main():
    parser = argparse.ArgumentParser(description="${BRAND_NAME} WebUI")
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
        "allowed_paths": ["logs", "assets"],
    }
    if GRADIO_6:
        launch_kwargs["theme"] = theme
        launch_kwargs["css"] = CUSTOM_CSS
    app.queue(default_concurrency_limit=40).launch(**launch_kwargs)


if __name__ == "__main__":
    main()
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
    global _CACHE_BYTES

    try:
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
                with _CACHE_LOCK:
                    cached = _ARRAY_CACHE.get(key)
                if cached is not None:
                    return cached.copy()
                arr = orig_np_load(file)
                nbytes = int(getattr(arr, "nbytes", 0))
                if nbytes > 0:
                    with _CACHE_LOCK:
                        if key not in _ARRAY_CACHE and _has_safe_ram(nbytes):
                            _ARRAY_CACHE[key] = arr.copy()
                            _CACHE_BYTES += nbytes
                return arr

            fast_np_load._noregrets_patched = True  # type: ignore[attr-defined]
            np.load = fast_np_load  # type: ignore[assignment]
    except Exception:
        pass

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
      `subprocess.run([uv_bin, "pip", "install", "-q", "matplotlib-inline", "websockets", "--python", py_exec], check=True)`
    );
  } else {
    installLines.push(
      `py_exec = sys.executable`,
      `subprocess.run([uv_bin, "pip", "install", "-q", "-r", str(repo_dir / "requirements.txt"), "--extra-index-url", "https://download.pytorch.org/whl/cu128", "--index-strategy", "unsafe-best-match", "--system"], check=True)`,
      `subprocess.run([uv_bin, "pip", "install", "-q", "matplotlib-inline", "websockets", "--system"], check=True)`
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
    `# Reset previously patched Applio UI files & config so re-runs stay clean`,
    `if (repo_dir / ".git").is_dir():`,
    `    subprocess.run(["git", "-C", str(repo_dir), "checkout", "--", "tabs", "assets/app_links.py", "assets/app_shell.py", "assets/config_template.json"], check=False)`,
    ``,
    `# Enforce NoRegretsRVC engine defaults (fp32 precision, Discord presence off)`,
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
    `# Write ${BRAND_NAME} custom WebUI, locked red theme, and UI overrides`,
    `(repo_dir / "assets" / "themes").mkdir(parents=True, exist_ok=True)`,
    `(repo_dir / "assets" / "themes" / "NoRegretsRVC.py").write_bytes(base64.b64decode("${themeB64}"))`,
    `(repo_dir / "app.py").write_bytes(base64.b64decode("${appB64}"))`,
    ``,
    `# Patch Applio tabs to match ${BRAND_NAME} defaults (-32..+32 pitch, hide Preset Settings, dual-T4 defaults)`,
    `import re as _re`,
    `inf_tab_py = repo_dir / "tabs" / "inference" / "inference.py"`,
    `if inf_tab_py.is_file():`,
    `    _inf = inf_tab_py.read_text(encoding="utf-8", errors="replace").replace("\\r\\n", "\\n")`,
    `    _inf = _inf.replace("minimum=-24,\\n                    maximum=24,", "minimum=-32,\\n                    maximum=32,")`,
    `    _inf = _inf.replace("minimum=-24, maximum=24,", "minimum=-32, maximum=32,")`,
    `    _inf = _inf.replace('with gr.Accordion(i18n("Preset Settings"), open=False):', 'with gr.Accordion(i18n("Preset Settings"), open=False, visible=False):')`,
    `    _inf = _re.sub(`,
    `        r'(terms_checkbox(?:_batch)?\\s*=\\s*gr\\.Checkbox\\([\\s\\S]*?info=i18n\\(\\s*)"[^"]*"(\\s*\\),\\s*)value=False,',`,
    `        r'\\g<1>"Confirm that your conversion audio follows the usage guidelines for NoRegretsRVC."\\g<2>value=True,',`,
    `        _inf,`,
    `    )`,
    `    inf_tab_py.write_text(_inf, encoding="utf-8")`,
    ``,
    `tts_tab_py = repo_dir / "tabs" / "tts" / "tts.py"`,
    `if tts_tab_py.is_file():`,
    `    _tts = tts_tab_py.read_text(encoding="utf-8", errors="replace").replace("\\r\\n", "\\n")`,
    `    _tts = _re.sub(`,
    `        r'(terms_checkbox\\s*=\\s*gr\\.Checkbox\\([\\s\\S]*?info=i18n\\(\\s*)"[^"]*"(\\s*\\),\\s*)value=False,',`,
    `        r'\\g<1>"Confirm that your conversion audio follows the usage guidelines for NoRegretsRVC."\\g<2>value=True,',`,
    `        _tts,`,
    `        count=1,`,
    `    )`,
    `    tts_tab_py.write_text(_tts, encoding="utf-8")`,
    ``,
    `train_tab_py = repo_dir / "tabs" / "train" / "train.py"`,
    `if train_tab_py.is_file():`,
    `    _tr = train_tab_py.read_text(encoding="utf-8", errors="replace").replace("\\r\\n", "\\n")`,
    `    _tr = _tr.replace('choices=["32000", "40000", "48000"],\\n                    value="40000",', 'choices=["32000", "40000", "48000"],\\n                    value="32000",')`,
    `    _tr = _tr.replace("batch_size = gr.Slider(\\n                1,\\n                64,\\n                4,", "batch_size = gr.Slider(\\n                1,\\n                64,\\n                2,")`,
    `    _tr = _tr.replace("save_every_epoch = gr.Slider(\\n                1,\\n                100,\\n                10,", "save_every_epoch = gr.Slider(\\n                1,\\n                100,\\n                5,")`,
    `    _tr = _tr.replace('value="my-project",', 'value="",')`,
    `    _tr = _re.sub(`,
    `        r'(include_mutes\\s*=\\s*gr\\.Slider\\(\\s*0,\\s*10,\\s*)2,(?=[\\s\\S]*?value=)(?P<body>[\\s\\S]*?)value=True,',`,
    `        r'\\g<1>0,\\g<body>value=0,',`,
    `        _tr,`,
    `        count=1,`,
    `    )`,
    `    _tr = _re.sub(`,
    `        r'(terms_checkbox\\s*=\\s*gr\\.Checkbox\\([\\s\\S]*?info=i18n\\(\\s*)"[^"]*"(\\s*\\),\\s*)value=False,',`,
    `        r'\\g<1>"Confirm that your training audio follows the usage guidelines for NoRegretsRVC."\\g<2>value=True,',`,
    `        _tr,`,
    `        count=1,`,
    `    )`,
    `    if 'if os.getenv("COLAB_RELEASE_TAG"):' not in _tr:`,
    `        _tr = _re.sub(`,
    `            r'([ \\t]*)gr\\.Markdown\\(\\s*i18n\\(\\s*"The \\'Upload\\' button packages the model[^"]*"\\s*\\)\\s*\\)',`,
    `            r'\\1if os.getenv("COLAB_RELEASE_TAG"):\\n\\1    gr.Markdown(i18n("The \\'Upload\\' button packages the model into a .zip and saves it to the ApplioExported folder in your Google Drive."))',`,
    `            _tr,`,
    `            count=1,`,
    `        )`,
    `    _ms_pat = _re.compile(`,
    `        r'(?P<head>with step_card\\(1, "Model Settings"[^\\n]*\\n)'`,
    `        r'        with gr\\.Row\\(\\):\\n'`,
    `        r'            with gr\\.Column\\(\\):\\n'`,
    `        r'(?P<mname>                model_name = gr\\.Dropdown\\([\\s\\S]*?\\n                \\)\\n)'`,
    `        r'(?P<arch>                architecture = gr\\.Dropdown\\([\\s\\S]*?\\n                \\)\\n)'`,
    `        r'            with gr\\.Column\\(\\):\\n'`,
    `        r'(?P<srate>                sampling_rate = gr\\.Dropdown\\([\\s\\S]*?\\n                \\)\\n)'`,
    `        r'(?P<voc>                vocoder = gr\\.Dropdown\\([\\s\\S]*?\\n                \\)\\n)',`,
    `    )`,
    `    def _reindent_step(_blk, _from="                ", _to="            "):`,
    `        return "\\n".join((_to + _ln[len(_from):]) if _ln.startswith(_from) else _ln for _ln in _blk.splitlines()) + "\\n"`,
    `    _ms_m = _ms_pat.search(_tr)`,
    `    if _ms_m:`,
    `        _new_ms = (`,
    `            _ms_m.group("head")`,
    `            + "        with gr.Row():\\n"`,
    `            + _reindent_step(_ms_m.group("mname"), "                ", "            ")`,
    `            + _reindent_step(_ms_m.group("srate"), "                ", "            ")`,
    `            + _reindent_step(_ms_m.group("voc"), "                ", "        ")`,
    `            + _reindent_step(_ms_m.group("arch"), "                ", "        ")`,
    `        )`,
    `        _tr = _tr[:_ms_m.start()] + _new_ms + _tr[_ms_m.end():]`,
    `    train_tab_py.write_text(_tr, encoding="utf-8")`,
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
    `app_links_py = repo_dir / "assets" / "app_links.py"`,
    `if app_links_py.is_file():`,
    `    _al = app_links_py.read_text(encoding="utf-8", errors="replace")`,
    `    _al = _al.replace('return urljoin(ECOSYSTEM["links"]["docs"], ECOSYSTEM["guides"][guide])', 'return "${REPO_README_URL}"')`,
    `    _al = _al.replace('"User guide"', '"GitHub README"')`,
    `    app_links_py.write_text(_al, encoding="utf-8")`,
    ``,
    `app_shell_py = repo_dir / "assets" / "app_shell.py"`,
    `if app_shell_py.is_file():`,
    `    _as = app_shell_py.read_text(encoding="utf-8", errors="replace")`,
    `    _as = _as.replace('translate("User guide")', '"GitHub README"')`,
    `    app_shell_py.write_text(_as, encoding="utf-8")`,
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
      `                    if _member.lower().endswith(".pth"):`,
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
    `print("${BRAND_NAME} setup complete! Model: ${sanitizeModelName(project.name)} (${project.samplingRate} · ${project.vocoder})")`
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
    `# --- ${BRAND_NAME} Kaggle Live Monitor (pinned URLs/creds + mid-epoch + Speed|Step|Epoch table) ---`,
    `BOLD = "\\033[1m"`,
    `RESET = "\\033[0m"`,
    `ANSI_RE = re.compile(r"\\x1B\\[[0-?]*[ -/]*[@-~]")`,
    `EPOCH_RE = re.compile(r"epoch=(\\d+)\\s*\\|\\s*step=(\\d+).*?training_speed=(\\d+:\\d{2}:\\d{2})(?:.*?lowest_value=([0-9.]+))?")`,
    `TQDM_RE = re.compile(r"(\\d+)%\\|[^|]*\\|\\s*(\\d+)/(\\d+)\\s*\\[([^,\\]]+),\\s*([0-9.]+\\s*(?:it/s|s/it))")`,
    `PUBLIC_URL_RE = re.compile(r"Running on public URL:\\s*(https?://\\S+)")`,
    `LOCAL_URL_RE = re.compile(r"Running on local URL:\\s*(https?://\\S+)")`,
    `NOISE_RE = re.compile(r"(FutureWarning|UserWarning|DeprecationWarning|RuntimeWarning|pkg_resources|warnings\\.warn|oneDNN|tensorflow/core|computation_placer|numa_node|Successfully opened dynamic library|weights_only|gradio\\.analytics|HTTP Request:|Checking out|libpng warning|ALSA|jack server|To create a public link)", re.IGNORECASE)`,
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
    `    session_links.insert(0, (label, value))`,
    ``,
    `def render_monitor(status="Running"):`,
    `    w_speed = max(9, max((len(r[0]) for r in epoch_rows), default=0))`,
    `    w_step = max(8, max((len(r[1]) for r in epoch_rows), default=0))`,
    `    w_epoch = max(6, max((len(r[2]) for r in epoch_rows), default=0))`,
    `    lines = [`,
    `        f"{BOLD}${BRAND_NAME} Live Monitor (Beta 0.0.0){RESET}  [{status}]",`,
    `        "=" * 64,`,
    `    ]`,
    `    if app_pass:`,
    `        lines.append(f"WebUI Login      : {app_user} / {app_pass}")`,
    `    for label, val in session_links:`,
    `        if val:`,
    `            lines.append(f"{label:<17}: {val}")`,
    `    lines.append("-" * 64)`,
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
    `        upsert_link("Cloudflare URL", cf_url)`,
    `    else:`,
    `        web_url = start_tunnel(["hrzn", "tunnel", "http://localhost:6969"], "webui-hrzn", hz_pattern)`,
    `        upsert_link("${BRAND_NAME} URL", web_url)`,
    ``,
    `    app_cmd = [py_exec, "-u", "app.py", "--server-name", "0.0.0.0", "--port", "6969"]`,
    `    if Tunnel == "Gradio + Cloudflare":`,
    `        app_cmd.append("--share")`,
    `    render_monitor("Starting ${BRAND_NAME} WebUI")`,
    `    app_proc = subprocess.Popen(app_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)`,
    `    assert app_proc.stdout is not None`,
    `    for raw_line in app_proc.stdout:`,
    `        line = ANSI_RE.sub("", raw_line).replace("\\r", "").strip()`,
    `        if not line:`,
    `            continue`,
    `        pub_m = PUBLIC_URL_RE.search(line)`,
    `        if pub_m:`,
    `            upsert_link("${BRAND_NAME} URL", pub_m.group(1))`,
    `            render_monitor("Ready")`,
    `            continue`,
    `        loc_m = LOCAL_URL_RE.search(line)`,
    `        if loc_m and not any(k.strip() == "${BRAND_NAME} URL" for k, _ in session_links):`,
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
          `Custom RVC voice model training environment built on Applio (${project.samplingRate} · ${project.vocoder} · ${project.pitchExtractor.toUpperCase()}).\n\n`,
          `- **Dataset Path**: \`${project.datasetPath || '/kaggle/input/datasets/your-username/dataset'}\`\n`,
          `- **Included Tabs**: ${Object.entries(config.activeTabs).filter(([, v]) => v).map(([k]) => k).join(', ')}\n`,
          `- **Tunnel & Auth**: ${config.tunnelMode} with ${config.randomAuthEnabled ? 'auto-generated 24-byte WebUI password' : 'standard access'}\n`,
        ],
      },
      {
        cell_type: 'markdown',
        metadata: {},
        source: ['## 1. Install Dependencies & Apply NoRegretsRVC Patches\n'],
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
          '## 2. Launch WebUI\n',
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
        'Kaggle-ready notebook preconfigured with SHA-256 verified tunnels, randomized WebUI password, and dual-T4 Turbo setup.',
      content: generateKaggleNotebookJson(config, project, downloads),
    },
    {
      id: 'app-py',
      filename: 'app.py',
      zipPath: 'app.py',
      language: 'python',
      description:
        'NoRegretsRVC launcher registering Training, Models, Samples, Voice Blender, TTS, Extra, and Settings.',
      content: generateAppPython(config, project),
    },
    {
      id: 'theme-py',
      filename: `${BRAND_NAME}.py`,
      zipPath: `assets/themes/${BRAND_NAME}.py`,
      language: 'python',
      description:
        'Fixed crimson-red Gradio styling sheet built specifically for NoRegretsRVC.',
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
