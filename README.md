# NoRegretsRVC

**NoRegretsRVC** is an RVC v2 voice-model training studio and configuration platform built around Applio-based Retrieval-based Voice Conversion workflows. It combines model training configuration, dataset preparation, voice conversion, model management, diagnostics, and Kaggle deployment tools in a unified interface.

<text>Current Version: **Beta 0.0 (0.0.0)**</text>

## Project Scope

This repository contains the NoRegretsRVC web interface, configuration utilities, pre-packaged notebook and archive generators, and Kaggle deployment generator. It prepares training and conversion settings and generates configurable workflows for an Applio-based Streamlit runtime.

The repository does **not** bundle the complete Applio training runtime or pretrained model weights. Some workspaces configure settings or display project data, while actual training and conversion depend on the corresponding Applio runtime and generated Kaggle workflow.

---

## Contents

- [Features](#features)
  - [1. Training Workspace](#1-training-workspace)
  - [2. Dataset Preprocessing](#2-dataset-preprocessing)
  - [3. Pitch and Feature Extraction](#3-pitch-and-feature-extraction)
  - [4. Samples and Voice Conversion](#4-samples-and-voice-conversion)
  - [5. Advanced Voice Matching and Audio Settings](#5-advanced-voice-matching-and-audio-settings)
  - [6. Models Manager](#6-models-manager)
  - [7. Voice Blender](#7-voice-blender)
  - [8. TTS Studio](#8-tts-studio)
  - [9. Free Samples Studio](#9-free-samples-studio)
  - [10. Diagnostics and Training Analysis](#10-diagnostics-and-training-analysis)
  - [11. Application Settings](#11-application-settings)
- [Kaggle Integration and Exclusive Features](#kaggle-integration-and-exclusive-features)
  - [12. One-Click Kaggle Notebook Launch](#12-one-click-kaggle-notebook-launch)
  - [13. Pre-Packaged Notebook, WebUI, and Bundle Exporter](#13-pre-packaged-notebook-webui-and-bundle-exporter)
  - [14. Kaggle Live Monitor and Training Progress](#14-kaggle-live-monitor-and-training-progress)
  - [15. Turbo Training Optimizations](#15-turbo-training-optimizations)
  - [16. Browser-Side DSP Audio Cleanup](#16-browser-side-dsp-audio-cleanup)
  - [17. Live Dual-T4 VRAM Estimator and Kaggle Survival Predictor](#17-live-dual-t4-vram-estimator-and-kaggle-survival-predictor)
  - [18. Cloud Hardening, Preflight Guards, and Seamless Auto-Update](#18-cloud-hardening-preflight-guards-and-seamless-auto-update)
  - [19. Integrated GitHub Documentation](#19-integrated-github-documentation)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Environment Variables](#environment-variables)
- [Upstream Project and Acknowledgements](#upstream-project-and-acknowledgements)
- [Version Information](#version-information)
- [License](#license)

---

# Features

## 1. Training Workspace

A four-step interactive workflow in the web studio for configuring and preparing RVC v2 voice-model projects, with Step 5 (Export) available in the generated Kaggle Streamlit WebUI.

### Training Pipeline

1. **Model Settings** — Configure model name, sample rate (`32000`, `40000`, or `48000` Hz), vocoder (`HiFi-GAN`, `MRF HiFi-GAN`, or `RefineGAN`), CPU cores (`1–4`), GPU allocation (`0–1`, Dual T4 default), GPU hardware information, and existing Kaggle checkpoint folders for resuming training.
2. **Prepare Dataset** — Select or upload datasets, automatically detect `/kaggle/input` paths, and preprocess audio.
3. **Extract Features** — Perform F0 pitch extraction, speaker embeddings, and training filelist preparation.
4. **Train Model** — Configure batch size (`1–64`, default `2`) with live Effective Batch Allocation, Save Every Epoch (`1–100`, default `5`), Total Epoch (`1–10000`, default `200`), and Index Algorithm (`Auto`, `Faiss`, or `KMeans`), then launch training or index generation.
5. **Export Model (Kaggle WebUI)** — Download the trained `.pth` model and `.index` feature file.

### Advanced Training Controls

- Built-in OG Applio pretrained models are enabled by default (`Pretrained` ON) until Custom Pretrains is enabled or a custom/Legacy Core pretrain is selected.
- Configurable **Lossless Turbo Training** (default ON), **Save Only Latest** (default ON), **Save Every Weights** (default ON), **Pretrained** (default ON), **Fresh Training**, **Cache Dataset in GPU**, **Checkpointing**, **Shutdown after finishing**, and **Custom Pretrains**.
- Custom Pretrains supports G/D paths, URLs, ZIP archives, and Hugging Face repositories; automatic `G_` → `D_` pairing discovery and a Refresh pretrained list control are included.

### Single-Click Execution, Preflight Validation, and Browser-Clock ETA

- Single-click **Start Training** and **Stop Training** actions without double-click state lag.
- Kaggle preflight validation checks `sliced_audios/*.wav`, `extracted/*.npy`, and `filelist.txt` before training. Missing preprocessing data cannot silently produce a fake five-second completion.
- Automatic `extract_model` final-checkpoint rescue on Kaggle when weights have not yet been exported at completion.
- A self-contained browser-clock **Estimated Finish** card showing remaining time, **Your Local Time**, and **Finish Day** (Today, Tomorrow, or a date using the user's browser clock rather than UTC server time), EMA-smoothed speed, and calibrated **ETA Accuracy %** confidence.

### Zero-Lag Controls and Clean Inputs

- Smooth, transition-suppressed range sliders paired with `+/-` numeric stepper controls (`NumberStepperInput`) throughout the steps.
- Suppresses Streamlit “Press Enter to apply” prompts.
- Custom crimson scrollbars.

## 2. Dataset Preprocessing

Prepare, slice, filter, and normalize recordings before feature extraction.

- **Automatic Kaggle Dataset Path Detection** — Recursively scans `/kaggle/input` with `followlinks=True`, up to depth 5, on load and refresh. Supports a multi-dataset selector when multiple datasets are attached.
- **Dataset Creator & Refresh** — Toggleable Dataset Creator uploads raw clips directly to `assets/datasets/<model_name>` on Kaggle, alongside **Refresh models and datasets**.
- **Audio Cutting Modes** — `Automatic`, `Simple`, and `Skip`.
- **Slicing Controls** — Chunk length (`0.5–5.0 s`, default `3.0 s)) and overlap length (`0.0–0.4 s`, default `0.3 s`).
- **Silence Truncation** — Optional trimming with Silence duration (`50–3000 ms`, default `300 ms`).
- **Filtering and Normalization** — Optional Noise filter (off by default) and Loudness normalization (`-14 LUFS`, off by default).

## 3. Pitch and Feature Extraction

Extract F0 contours and speaker embeddings from preprocessed segments.

- **Pitch Algorithms** — `rmvpe` (default), `fcpe`, `swift-f0`, and `crepe`.
- **Embedder Models** — `contentvec` (default), `spin`, and `chinese-hubert-base`.
- **Silent Training Files** — Slider (`0–10`, default `0`) controlling silent negative-reference slices generated for training stability.

## 4. Samples and Voice Conversion

A four-step workspace for auditioning models and converting individual or batched audio.

### Step 1 — Voice Model

- Select a `.pth` model and auto-matched `.index` file.
- Unload Voice.
- Refresh models and indexes.

### Step 2 — Audio Source

- Audio Files (multi-clip queue) and Batch folder conversion.
- Drag-and-drop or upload `WAV`, `MP3`, `FLAC`, `OGG`, and `M4A`.
- Select existing recordings.
- Kaggle Quick Test Voice buttons using Free Samples.

### Step 3 — Pitch & Similarity

- Master Pitch: `-32` to `+32` semitones.
- Quick transpose buttons: `-12`, `-5`, `0`, `+5`, and `+12`.
- Voice Similarity (Index Rate): `0.00–1.00`, default `0.75`.

### Step 4 — Convert & Output

- **Convert** and **Convert All**, or individual **Convert / Re-Convert** actions.
- Each clip shows inheritance status (`Following Step 3` or `Custom Override`) and supports individual overrides.
- **Apply Step 3 Settings to All**.
- Displays the generated `core.py` CLI command.
- Output playback in **Recent Generated Samples** through the custom `CrimsonAudioPlayer`, with download and discard controls.

## 5. Advanced Voice Matching and Audio Settings

Fine-grained acoustic controls in Step 3's collapsible **Advanced Settings** drawer.

- **Voice Matching** — Volume Envelope (`0.00–1.00`, default `1.00`) and Protect Voiceless Consonants (`0.00–0.50`, default `0.50`).
- **Models & Algorithms** — Speaker ID (`0–100`), pitch extraction algorithm, and embedder model overrides.
- **Advanced Pitch & Tuning** — Autotune, Proposed Pitch, and Split Audio toggles.
- **Audio Cleanup** — Denoising toggle and reduction strength (`1–30 dB`, default `12 dB`, with Subtle, Balanced, and Strong labels).
- **Formant Shifting** — Toggleable formant preservation and shaping via Formant Quefrency (`0.0–16.0`) and Formant Timbre (`0.0–16.0`).
- **Output Settings** — Output Path, export format (`WAV`, `MP3`, `FLAC`), and Clear Outputs.

## 6. Models Manager

Manage downloaded voices, local uploads, and pretrained base checkpoints.

- **Download Model** — Fetch from Hugging Face, Pixeldrain, or Google Drive URLs, and inspect the preconfigured Boot Manifest.
- **Drop Files** — Upload or drag-and-drop `.pth`, `.index`, and `.zip` voice archives directly into `logs/` on Kaggle.
- **Download Pretrained Models** — Architectures include `Legacy Core 1.5 V2` at 32k (with a dedicated **Download Legacy Core 1.5 V2 (G + D)** action linked to Step 4 Custom Pretrains), `Titan`, `Snowie`, `KLM`, `Ov2Super`, and `Rin_E3` at `32k`, `40k`, and `48k`.
- OG Applio pretrains remain enabled by default unless Custom Pretrains or Legacy Core 1.5 V2 is enabled/downloaded.
- Custom Pretrains discovery supports Hugging Face repositories, ZIP bundles, direct `G_`/`D_` URLs, local uploads, and timestamped download logs.

## 7. Voice Blender

A three-step checkpoint-fusion workflow for merging two compatible models.

1. **Source Models** — Choose Model A and Model B via `.pth` picker or path.
2. **Blend Settings** — Set output name, target sample rate (`32k`, `40k`, `48k`), and Blend Ratio (`0.00–1.00`) for Model A/B weighting.
3. **Blend & Export** — Run Fusion, download the merged ZIP/checkpoint with **Download Model**, or inspect/copy the generated blend CLI command.

## 8. TTS Studio

Generate speech using Edge-TTS and convert it to a target RVC voice.

- **Base Voices** — Web Studio presets: `en-US-AndrewNeural`, `en-US-GuyNeural`, `en-US-AriaNeural`, `en-GB-RyanNeural`, and `en-US-JennyNeural`. Kaggle supports full Edge-TTS catalog synchronization.
- **Voice Conversion Controls** — Synced target RVC `.pth` and optional `.index`, Text to Speak, Pitch Shift (`-12` to `+12`), Speech Speed (`-50%` to `+50%`), Index Feature Ratio (`0.00–1.00`), and Consonant Protect (`0.00–0.50`).

## 9. Free Samples Studio

Bundled reference recordings for testing conversion without sourcing external audio.

- **Reference Clips** — Six dry, mono, 32 kHz WAV recordings (3 male and 3 female), with transcripts and gender filters (All, Male, Female).
- **Audition & Export** — Preview using `CrimsonAudioPlayer`, send clips using **Use in Samples**, queue all with **Load All into Samples**, download individual WAVs, or export all clips as `NoRegretsRVC_Free_Samples.zip`.

## 10. Diagnostics and Training Analysis

A multi-metric log and checkpoint analyzer for identifying undertraining, overtraining, dataset mismatch, and tail drift.

- **Multi-Metric Evaluation** — Parses training logs and live Kaggle TensorBoard/console metrics for Mel Reconstruction, KL Divergence, Feature Matching (FM), G/D Discriminator Balance, Gradient Norm stability, Dataset Alignment / Match %, Dataset Saturation (steps per clip and effective steps), and Tail Drift Watch.
- **Verdict & Risk Meters** — Verdict Hero Banner plus Undertrained Risk % and Overtrained Risk % meters.
- **Checkpoint Recommendations** — Recommended Best Epoch and Top Recommended Checkpoints table (Mel Loss, Tail Drift, Score /100), with Kaggle **Load in Samples** actions.
- **Interactive Chart & Importer** — SVG Observed Mel Loss chart marking Best Epoch and overtraining zones; Model Sound Feedback selector (Not auditioned yet, Undertrained, Clean & natural, Overtrained); and local `.ipynb`, `.log`, and `.txt` import.

## 11. Application Settings

Runtime, training, localization, and update preferences organized into General and Training tabs.

### General Settings

- **List Filtering** — Toggle `filter` (off by default) for file selectors.
- **Pitch Correction** — RMVPE high-register correction (`true_pitch` / `fold` mode), with F0 ceiling (`1000–2000 Hz`).
- **Discord Integration** — `discord_presence` (off by default).
- **Language** — UI localization across 10 languages, including RTL Arabic (`ar_SA`).
- **Updates & System** — GitHub release/tag/commit check (`/api/check-updates`), **Apply WebUI Update & Restart**, and **Restart Applio** on Kaggle.

### Training Settings

- **Model Author** — Default metadata embedded into exported checkpoints.
- **Training Precision** — `fp32` (default) or `fp16`.
- **Kaggle Tunnel & Security** — `Gradio + Cloudflare` or `Horizon` tunnel mode, custom fork URL, random 24-byte WebUI credentials, isolated `uv` Python 3.12 + CUDA 12.8 setup, and ONNX GPU verification toggles.
- **Lossless Turbo Training & 1-GPU Applio Batch Parity** — Controls for Lossless Turbo Training and **1-GPU Applio Batch Parity (<300 Epoch Fix)** (ON by default). Automatically splits batch sizes `>= 4` across dual T4 GPUs so sub-300-epoch runs retain standard 1-GPU Applio optimizer steps per epoch, with live effective GPU allocation.

---

# Kaggle Integration and Exclusive Features

## 12. One-Click Kaggle Notebook Launch

Deploy to Kaggle Dual Tesla T4 (`nvidiaTeslaT4`) using `NoRegretsRVC_Kaggle.ipynb` or `generateKaggleNotebookJson`.

The workflow provisions CUDA 12.8 (`torch` and `onnxruntime-gpu`), injects `noregrets_turbo.py` hooks into `rvc/train/train.py`, deploys `streamlit_app.py` and `.streamlit/config.toml`, and pre-fetches Free Samples and Boot Manifest items.

## 13. Pre-Packaged Notebook, WebUI, and Bundle Exporter

Deployment artifacts are generated directly through the project and `forkGenerator.ts`.

- `generateForkBundle()` creates a five-file deployment set: `assets/NoRegretsRVC_Kaggle.ipynb`, `streamlit_app.py`, `.streamlit/config.toml`, `noregrets_turbo.py`, and `run_commands.sh`.
- Repository root and `public/` provide ready-to-download assets served with `Content-Disposition: attachment` by `server.ts`: `NoRegretsRVC_Kaggle.ipynb`, `NoRegretsRVC-Beta-0.0.0.zip` (clean flat-root archive with fewer than 100 files for GitHub web uploads), and `NoRegretsRVC-Full-Repo.txt`.
- Kaggle Step 5 (Export Model) downloads completed `.pth` weights and `.index` files; Voice Blender exports fused models through **Download Model**.

## 14. Kaggle Live Monitor and Training Progress

A real-time notebook monitor in the final Kaggle execution cell, using an in-place HTML `DisplayHandle` (`nr_live_monitor`) to prevent scroll-jumping.

- **Crimson Styling & Progress Bar** — `#E51923` header and seamless HTML epoch progress bar (Current/Total and %), batch speed, and ETA.
- **Clean Epoch Table & Status Markers** — Suppresses noisy TensorFlow/PyTorch console output and presents a structured Epoch Table with inline `(saved)` indicators. Includes **Training Stopped** (two blank lines below the table), **Training Resumed** (two blank lines above the continued session table), and **Training Complete** (two blank lines below the final table) markers.
- **Zero-Epoch Failure Guard & Tunnel Links** — Prevents reporting completion when zero epochs ran; displays active WebUI Login, NoRegretsRVC/Cloudflare URL, and Local UI URL links after startup.

## 15. Turbo Training Optimizations

Runtime acceleration hooks (`noregrets_turbo.py`) are injected during Kaggle setup, with the stated aim of optimizing runtime without changing RVC math or output quality.

- **Vectorized AdamW & Zero-Grad** — Enables `foreach=True` on AdamW with fallback guards, `zero_grad(set_to_none=True)`, and `torch.backends.cudnn.benchmark = True`.
- **DataLoader & DDP Tuning** — Safe worker caps, `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=2`, `gradient_as_bucket_view=True`, and `flush_saves()` synchronization on `destroy_process_group`.
- **TensorBoard Overhead Reduction** — Skips costly `SummaryWriter` image and figure rendering (`add_image`/`add_figure`) during epoch loops.
- **Adaptive RAM Cache & Checkpoint Saves** — Caches audio/features in system RAM (up to `2 GB` per process while retaining at least `8 GB` of system RAM free), using `numpy.load`, `wavfile.read`, and `soundfile.read`. Resume checkpoints (`G_`/`D_`) are written asynchronously; exported inference `.pth` files are written synchronously and atomically.

## 16. Browser-Side DSP Audio Cleanup

Browser-local Web Audio API noise reduction (`audioCleanup.ts`) before inference or preview.

- **High-Pass & Noise-Floor Estimation** — First-order high-pass filter (`65–95 Hz`) and 20 ms frame RMS percentiles for ambient-noise estimation.
- **Soft-Knee Downward Expander** — Noise reduction from `1–30 dB` (8 ms attack / 120 ms release), retaining `12–26%` natural dry ambience and applying a `tanh` safety soft-clip.
- **PCM-16 Export & Cleanup** — Standard PCM-16 WAV encoding and revocation of unused `blob:` object URLs.

## 17. Live Dual-T4 VRAM Estimator and Kaggle Survival Predictor

Resource estimator (`rvcTuning.ts`) inside the Training workspace.

- **VRAM Budget & Batch Advisor** — Estimates usage against `16.0 GB`, reports Optimal, Tight, or Risk of OOM, recommends a safe maximum batch size, and displays Effective Batch (including Dual-T4 parity splits).
- **Disk & Session Predictor** — Estimates disk usage against `20.0 GB` (Safe, Watch Disk, or Risk of OOM/Full) and runtime against Kaggle's 12-hour cap (Fits 12h, Tight 12h, or May Exceed 12h).

## 18. Cloud Hardening, Preflight Guards, and Seamless Auto-Update

Reliability and security features for headless cloud notebooks.

- **Verified Tunnels & Random Authentication** — SHA-256 verification of downloaded `cloudflared` binaries and 24-byte `secrets.token_urlsafe` credentials for Streamlit (`APPLIO_AUTH_USERNAME`/`PASSWORD`).
- **Isolated Runtime & Port Management** — Dependency installation through `uv` in an isolated Python 3.12 + CUDA 12.8 environment, ONNX GPU verification, headless `_pick_py()`, and `_free_port_6969()` cleanup.
- **Revision-Aware Auto-Updates** — `NR_BUILD_REV` anti-downgrade tracking across GitHub raw/notebook sources, **Apply WebUI Update & Restart**, training preflight checks, and `extract_model` rescue exports.

## 19. Integrated GitHub Documentation

Direct `#readme` links in workspace headers and an Express `/api/check-updates` endpoint querying GitHub releases, tags, and `main` commit SHAs to show live version status in Settings.

---

# Tech Stack

- **Frontend Studio** — React 19, TypeScript (~7.0), Vite 8, Tailwind CSS v4 (`@tailwindcss/vite`), Lucide React, and Motion.
- **Backend & Server** — Express 4 and `tsx`; `server.ts` serves Vite middleware in development, production `dist/`, and attachment-download routes.
- **Audio, Charts & Testing** — Web Audio API (`audioCleanup.ts`), SVG and HTML5 audio (`LossChart`, `CrimsonAudioPlayer`), and Node built-in test runner (`node:test`).
- **Generated Kaggle Runtime** — Streamlit (`>=1.40.0`) with crimson `.streamlit/config.toml`, `uv`, Python 3.12, PyTorch (CUDA 12.8), Applio RVC core, `edge-tts`, `onnxruntime-gpu`, and `cloudflared`/`hrzn`.

---

# Getting Started

```bash
# Install dependencies
npm install

# Run the development server (Express + Vite on port 3000)
npm run dev

# Typecheck
npm run lint

# Build for production
npm run build

# Start the production server
npm start
```

---

# Project Structure

```text
NoRegretsRVC/
├── .env.example                     # Example environment variables
├── .gitignore
├── index.html                       # Vite HTML entry point
├── metadata.json                    # Application metadata
├── package.json                     # Scripts and dependencies
├── tsconfig.json
├── vite.config.ts
├── server.ts                        # Express server, update check and downloads
├── NoRegretsRVC_Kaggle.ipynb        # Pre-built Kaggle Dual-T4 notebook
├── assets/
│   └── NoRegretsRVC_Kaggle.ipynb    # Packaged notebook copy
├── public/
│   ├── NoRegretsRVC_Kaggle.ipynb
│   ├── NoRegretsRVC-Beta-0.0.0.zip # GitHub-ready flat-root archive
│   ├── NoRegretsRVC-Full-Repo.txt
│   └── assets/aistudio/*.wav        # Free Samples WAV recordings
└── src/
    ├── App.tsx                      # Studio layout, navigation and state
    ├── index.css                    # Tailwind v4 theme and crimson styling
    ├── main.tsx
    ├── components/                   # Workspace views, players, charts and settings
    ├── data/                         # Defaults, locales and Free Samples catalog
    ├── types/                        # TypeScript interfaces and training state
    └── utils/                         # Generators, DSP, diagnostics, tuning, i18n and tests
```

---

# Environment Variables

Defined in `.env.example`:

- `GEMINI_API_KEY` — Gemini API key (server-side secret, where applicable).
- `APP_URL` — Public base URL where the studio is hosted.

Keep secret values private. Do not commit API keys, credentials, or populated environment files.

---

# Upstream Project and Acknowledgements

NoRegretsRVC is built around the **Applio** RVC v2 voice conversion and training core (`rvc/train/train.py`, `core.py`), alongside Edge-TTS, RMVPE / FCPE / Swift-F0 / CREPE, ContentVec / SPIN / Chinese HuBERT, Faiss, and Cloudflare Tunnel.

Please review upstream projects and their respective licensing and notices when using or redistributing components.

---

# Version Information

- **Version:** `Beta 0.0` (`0.0.0`)
- **Generated Kaggle Build Revision (`NR_BUILD_REV`):** `2026112603`

---

# License

Refer to the repository license and upstream **Applio** license terms for runtime, training core, and model usage terms.
