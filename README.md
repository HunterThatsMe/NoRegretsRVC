# NoRegretsRVC

**NoRegretsRVC** is an RVC v2 voice-model training studio and configuration platform built around Applio-based Retrieval-based Voice Conversion workflows. It brings together a configurable training workspace, audio conversion tools, model management, diagnostics, and an integrated Kaggle deployment system in one interface.

**Current Version:** Beta 0.0

> **Project scope**
>
> This repository contains the NoRegretsRVC web interface, its configuration utilities, and its Kaggle notebook generator. It prepares training and conversion settings and generates configurable workflows for an Applio-based runtime.
>
> The repository does **not** bundle the complete Applio training runtime or pretrained model weights. Some workspaces configure settings or display project data, while actual training and conversion depend on the corresponding runtime and generated workflow.

---

## Contents

- [Features](#features)
  - [1. Training Workspace](#1-training-workspace)
  - [2. Dataset Preprocessing](#2-dataset-preprocessing)
  - [3. Pitch and Feature Extraction](#3-pitch--feature-extraction)
  - [4. Samples and Voice Conversion](#4-samples--voice-conversion)
  - [5. Advanced Voice Matching and Audio Settings](#5-advanced-voice-matching--audio-settings)
  - [6. Models Manager](#6-models-manager)
  - [7. Voice Blender](#7-voice-blender)
  - [8. TTS Studio](#8-tts-studio)
  - [9. Free Samples Studio](#9-free-samples-studio)
  - [10. Diagnostics and Training Analysis](#10-diagnostics--training-analysis)
  - [11. Application Settings](#11-application-settings)
- [Kaggle Integration and Exclusive Features](#kaggle-integration--exclusive-features)
  - [12. One-Click Kaggle Notebook Launch](#12-one-click-kaggle-notebook-launch)
  - [13. In-App Kaggle Notebook and Bundle Exporter](#13-in-app-kaggle-notebook--bundle-exporter)
  - [14. Kaggle Live Monitor and Training Progress](#14-kaggle-live-monitor--training-progress)
  - [15. Turbo Training Optimizations](#15-turbo-training-optimizations)
  - [16. Browser-Side DSP Audio Cleanup](#16-browser-side-dsp-audio-cleanup)
  - [17. Live Dual-T4 VRAM Estimator and Kaggle Survival Predictor](#17-live-dual-t4-vram-estimator--kaggle-survival-predictor)
  - [18. Cloud Hardening and Automated Checkpoint Backups](#18-cloud-hardening--automated-checkpoint-backups)
  - [19. Integrated GitHub Documentation](#19-integrated-github-documentation)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Environment Variables](#environment-variables)
- [Upstream Project and Acknowledgements](#upstream-project--acknowledgements)
- [Version Information](#version-information)
- [License](#license)

---

## Features

### 1. Training Workspace

A guided five-step pipeline for configuring and preparing RVC v2 voice-model projects.

#### Five-Step Training Pipeline

1. **Model Settings** — Configure the model architecture, sample rate, and training parameters.
2. **Prepare Dataset** — Upload, preprocess, and organize audio datasets.
3. **Extract Features** — Extract pitch and content features with supported algorithms.
4. **Train Model** — Configure training parameters and run the generated training workflow.
5. **Export Model** — Prepare completed voice models for export.

#### Available Training Configurations

| Category | Available options |
|---|---|
| Sample rates | 32,000 / 40,000 / 48,000 Hz |
| Vocoders | HiFi-GAN / MRF HiFi-GAN / RefineGAN |
| Optimizers | AdamW / RAdam |
| Loss functions | Single-Scale Mel / Multi-Scale Mel |
| Training precision | FP16 / FP32 |
| Index builders | Auto / Faiss / KMeans |

#### Additional Training Controls

- Configurable batch size and training epochs.
- Save interval and checkpoint preferences.
- Speaker ID selection.
- Pretrained architecture selection.
- Training precision configuration.
- T4 GPU memory estimation and configuration warnings.

#### Live Training Completion Estimate

A real-time training-time estimator that automatically appears beneath **Start Training** once training begins.

- Continuously recalculates the projected training time without adding separate timers or unnecessary interface elements.
- Displays a smoothed **Remaining Time** estimate, such as *Finishes in ~1 hr 14 min*.
- Converts the projected completion time into **Your Local Time**.
- Displays the expected **Finish Day**, identifying whether training is expected to finish Today, Tomorrow, or on a specific calendar date.
- Includes a calibrated **Accuracy %** confidence indicator. It starts relatively low during early epoch warmup and increases as training speeds become more consistent, reaching approximately 95% under stable conditions.
- Smooths the estimate to avoid unnecessary fluctuations as training progresses.

The displayed estimate is approximate, not a guarantee. Actual completion times can change because of GPU throttling, checkpoint disk writes, batch length, and other runtime conditions.

### 2. Dataset Preprocessing

Audio preparation tools for organizing datasets before feature extraction and training.

- Automatic, Simple, and Skip audio-cutting modes.
- Adjustable chunk and overlap durations.
- Silence truncation.
- Noise filtering.
- Automatic loudness normalization.
- Silent mute-file generation.
- Dataset balancing.

### 3. Pitch & Feature Extraction

Multiple algorithms are available for extracting pitch and content features.

#### Pitch Extractors

- RMVPE
- FCPE
- Swift-F0
- CREPE

#### Feature Embedders

- ContentVec
- SPIN
- Chinese-HuBERT-Base

Extractors and embedders can be selected according to the intended training configuration.

### 4. Samples & Voice Conversion

A configurable conversion workspace for managing voice models, input audio, and conversion settings.

#### Model Management

- Load and select .pth voice models.
- Select corresponding .index files.
- Refresh and unload loaded models.

#### Audio Input

- Single-file audio uploads.
- Multi-file audio uploads.
- Drag-and-drop support.
- Kaggle batch-path mode.
- Multi-clip conversion queue.

#### Conversion Controls

- Pitch shifting from -32 to +32 semitones.
- Adjustable index rate.
- Volume envelope.
- Voiceless consonant protection.
- Autotune and proposed pitch.
- Split Audio.
- Individual export-format selection.

#### Multi-Clip Queue

Every queued audio clip can use its own independent conversion configuration, including:

- Pitch.
- Index rate.
- Voice model.
- Index file.
- Protection.
- Volume envelope.
- Autotune.
- Proposed pitch.
- Split Audio.
- Export format.

Playback tools also include side-by-side A/B comparison between original and converted recordings, along with a **Recent Generated Samples** history drawer.

### 5. Advanced Voice Matching & Audio Settings

Advanced conversion controls are grouped into collapsible sections, keeping frequently used settings accessible without overcrowding the interface.

#### Voice Matching *(Expanded by Default)*

- Search Feature Ratio — default: 0.75.
- Volume Envelope — default: 1.0.
- Voiceless Consonant Protection — default: 0.5.

#### Models & Algorithms

- Speaker ID selection.
- Pitch extraction algorithm.
- Feature embedder selection.

#### Advanced Pitch & Tuning

- Autotune.
- Proposed Pitch.
- Split Audio.

#### Audio Cleanup

- Local input noise cleanup.
- Adjustable noise reduction from 1–30 dB, with a default of 12 dB.

#### Formant Shifting

- Male and female formant shifting.
- Adjustable Quefrency and Timbre sliders.

#### Output Settings

- Custom output directory.
- WAV, MP3, and FLAC export formats.
- Clear Outputs functionality.

### 6. Models Manager

A dedicated workspace for managing voice models, pretrained architectures, and supporting files.

- Remote model archive downloads through Hugging Face, Pixeldrain, and Google Drive links.
- Local .pth and .index file staging.
- Pretrained architecture selection:
  - Titan
  - Snowie
  - KLM
  - Ov2Super
  - Rin_E3
- Pre-boot download manifest for managing required model files.

### 7. Voice Blender

A model-blending utility for combining two RVC voice-model checkpoints.

- Merge two .pth model checkpoints.
- Adjustable blending ratio.
- Sample rate selection.
- Automatic CLI command generation.
- Exportable .zip blending recipes.

### 8. TTS Studio

An integrated text-to-speech workspace powered by Edge-TTS.

- Built-in Edge-TTS voice selector.
- Adjustable speaking rate and pitch.
- Configurable index rate and protection ratio.
- Direct routing of generated speech through a loaded RVC model.

### 9. Free Samples Studio

A built-in sample library with ready-to-use audio recordings for testing voice conversion.

#### Included Sample Library

- Six dry, mono, 32 kHz WAV recordings.
- Three male and three female recordings.
- Matching transcripts for every recording.

#### Playback & Downloads

- Integrated waveform playback.
- Individual audio downloads.
- One-click ZIP download.
- Instant queuing into the Samples workspace.
- Automatic GitHub sample retrieval inside Kaggle.

The sample library is available directly through the application and its generated Kaggle interface.

### 10. Diagnostics & Training Analysis

An integrated diagnostics workspace for reviewing training progress and identifying suitable checkpoints.

#### Supported Input

- Raw console stdout.
- Live tqdm progress bars.
- Jupyter Notebook (.ipynb) files.

#### Generated Analysis

- Interactive training loss curves.
- Undertraining checkpoint recommendations.
- Suggested Sweet Spot recommendations.
- Overtraining checkpoint recommendations.

Recommended epochs are aligned with the configured **Save Every Epoch** interval to make checkpoint selection easier.

### 11. Application Settings

A centralized settings workspace for configuring application behavior and training preferences.

- General and training preferences.
- RMVPE high-register correction toggle.
- Discord Rich Presence integration.
- Support for 10 UI languages.
- Model Author metadata.
- FP16 / FP32 precision switcher.
- Live release update checker.
- Configurable Kaggle notebook tabs.
- Tunnel and checkpoint backup preferences.

---

## Kaggle Integration & Exclusive Features

NoRegretsRVC includes a dedicated Kaggle deployment workflow, extending the application beyond its local configuration interface.

### 12. One-Click Kaggle Notebook Launch

**Branded NoRegretsRVC Kaggle deployment — assets/NoRegretsRVC_Kaggle.ipynb**

The generated notebook provides a branded NoRegretsRVC interface inside a Kaggle-hosted, Applio-based runtime.

Importing the notebook into Kaggle and launching its Public URL opens the NoRegretsRVC interface.

#### Interface Improvements

- Deep-crimson application theme.
- Gradio 6 compatibility.
- Replaces Applio's default interface with the NoRegretsRVC UI.
- Removes Preset Settings.
- Expanded pitch range from -32 to +32 semitones.
- Integrated Free Samples tab directly inside Gradio.

#### Preconfigured Dual-T4 Training Defaults

| Setting | Default |
|---|---:|
| Sample Rate | 32,000 Hz |
| Batch Size | 2 |
| Save Every | 5 epochs |

### 13. In-App Kaggle Notebook & Bundle Exporter

Available under **Settings → Training**, this exporter generates customized Kaggle deployment files directly through the application.

#### Available Exports

- Customized NoRegretsRVC_Kaggle.ipynb notebook.
- Complete .zip runtime bundle.

#### Preserved Configuration

- Tunnel configuration.
- Checkpoint backup preferences.
- Selected application tabs.

Generated exports retain the selected deployment preferences, reducing the need to manually reconfigure each notebook.

### 14. Kaggle Live Monitor & Training Progress

An improved monitoring interface designed to keep important deployment information accessible while reducing unnecessary console output.

#### Persistent Deployment Information

Displayed at the top of the Kaggle output (**Cell 2**):

- NoRegretsRVC Public URL.
- FileBrowser URL.
- Automatically generated 24-byte login credentials.

#### Training Monitor

- Filters unnecessary framework warnings.
- Displays clear [Stage] progress milestones.
- Live mid-epoch progress indicator.
- Simplified training statistics table:

| Speed | Step | Epoch | Lowest Loss |
|---|---|---|---|

### 15. Turbo Training Optimizations

**Training optimization module — noregrets_turbo.py**

An automatic Kaggle training optimization module designed to improve epoch speeds while preserving the intended acoustic training behavior.

**Expected performance improvement: approximately 18–25% faster epochs.**

#### Included Optimizations

- Persistent cudnn.benchmark = True on Linux and Kaggle, overriding Applio's Windows-only check.
- Fused AdamW optimizer.
- zero_grad(set_to_none=True).
- DDP gradient_as_bucket_view.
- Safe in-memory dataset audio caching.
- Asynchronous background .pth checkpoint writing.
- Disabled slow TensorBoard image rendering.

Actual speed improvements depend on hardware, dataset characteristics, and training configuration. The expected performance increase is not a guaranteed benchmark for every workload.

### 16. Browser-Side DSP Audio Cleanup

A built-in Web Audio PCM processing system designed to reduce background noise while preserving natural speech characteristics.

#### Processing Includes

- 65–95 Hz rumble-removal high-pass filtering.
- 15th-percentile noise-floor tracking.
- Soft-knee downward expansion.
- Dry ambience blending.

The system is designed to reduce background hiss and unwanted noise without unnecessarily chopping speech.

### 17. Live Dual-T4 VRAM Estimator & Kaggle Survival Predictor

A resource-monitoring system built specifically around Kaggle dual-T4 GPU environments.

#### GPU Monitoring

- Live T4 VRAM headroom monitoring.
- Training status indicators: Optimal, Tight, and Risk of OOM.
- Recommended batch-size calculations.

#### Kaggle Runtime Monitoring

- Estimated 12-hour session survival.
- 20 GB disk-capacity monitoring.
- Runtime and storage indicators tailored to dual-T4 environments.

### 18. Cloud Hardening & Automated Checkpoint Backups

A deployment protection and backup system designed to help preserve training sessions and model checkpoints across Kaggle runtime resets.

#### Environment Isolation

- Isolated Python 3.12 environment using uv.
- CUDA 12.8 configuration.
- ONNX Runtime GPU CUDA verification.

#### Security & Tunnel Protection

- Automatically randomized 24-byte WebUI passwords.
- Automatically randomized 24-byte FileBrowser passwords.
- SHA-256 verification for Cloudflare (cloudflared) and Horizon tunnels.

#### Automated Checkpoint Backups

- Hugging Face checkpoint synchronization.
- Google Drive backups through rclone.
- Background backup handling through core_backup.py.

These features help preserve trained models and checkpoints beyond individual Kaggle sessions.

### 19. Integrated GitHub Documentation

External Applio guide buttons have been replaced with direct links to the NoRegretsRVC GitHub README.

Available across:

- Training.
- Models.
- Diagnostics.
- Voice Blender.
- Kaggle.

---

## Tech Stack

NoRegretsRVC uses the following technologies for its web interface and supporting utilities:

- **React 19 + TypeScript** — application interface and logic.
- **Vite 8** — frontend development and build tooling.
- **Express** — backend server and application endpoints.
- **Tailwind CSS 4** — interface styling.
- **Lucide React** — icon library.
- **Motion** — interface animations.

The project also generates workflows that use Applio and its associated dependencies.

---

## Getting Started

### Requirements

- Windows, macOS, or Linux.
- Node.js 22 or newer.
- npm.

### Installation

Clone the repository:

    git clone https://github.com/HunterThatsMe/NoRegretsRVC.git
    cd NoRegretsRVC

Install dependencies:

    npm install

### Start the Development Server

    npm run dev

The development server prints its local address in the terminal when it starts.

### Build the Frontend

    npm run build

### Run the TypeScript Check

    npm run lint

---

## Project Structure

The current repository includes the web application, Kaggle notebook assets, sample recordings, and supporting utilities. The main directories and files are organized as follows:

    NoRegretsRVC/
    ├── assets/
    │   ├── audios/                 # Source audio recordings
    │   ├── free_samples/           # Free Samples Studio recordings
    │   ├── themes/                 # NoRegretsRVC theme files
    │   └── NoRegretsRVC_Kaggle.ipynb
    ├── public/
    │   ├── assets/
    │   │   └── aistudio/           # Public sample audio assets
    │   └── NoRegretsRVC_Kaggle.ipynb
    ├── src/
    │   ├── components/             # Main interface views and reusable UI
    │   ├── data/                   # Default settings and sample data
    │   ├── types/                  # TypeScript data models
    │   ├── utils/                  # Notebook generation, tuning, parsing, ZIP helpers
    │   ├── App.tsx
    │   ├── index.css
    │   └── main.tsx
    ├── .gitattributes
    ├── NoRegretsRVC_Kaggle.ipynb
    ├── app.py
    ├── index.html
    ├── metadata.json
    ├── noregrets_turbo.py
    ├── package.json
    ├── run_commands.sh
    ├── server.ts
    ├── tsconfig.json
    └── vite.config.ts

The original project layout notes also referenced a **docs/** directory, **.env.example** environment-variable template, and **.gitignore** file. Those paths were not present in the inspected main-branch file tree when this README was prepared, so check the current checkout before relying on them.

---

## Environment Variables

The original setup notes describe a starter .env.example file.

Copy it to .env only when you need to configure the corresponding environment values for your setup.

**Never commit .env files, API keys, access tokens, passwords, or private model credentials.** The original project notes state that the actual .env file is excluded through .gitignore; however, that ignore file was not present in the inspected main-branch tree when this README was prepared. Verify that your local ignore rules exclude sensitive files before committing.

---

## Upstream Project & Acknowledgements

NoRegretsRVC generates workflows that use [Applio](https://github.com/IAHispano/Applio).

Applio and its dependencies retain their respective licenses and notices. Review the upstream license and all third-party notices before redistributing generated environments or bundled dependencies.

NoRegretsRVC is an independent project. Ilaria RVC was used only as a reference for repository organization and documentation style; its source code and README text have not been copied into this repository.

---

## Version Information

| Component | Version |
|---|---|
| Application | Beta 0.0 |
| Repository Package | 0.0.0 |
| Kaggle Interface | NoRegretsRVC Beta |

See [CHANGELOG.md](CHANGELOG.md) for project changes and development notes.

---

## License

A project license has not been added yet.

Do not assume the repository is licensed for reuse or redistribution until the original project code and upstream-derived portions have been reviewed and an appropriate license has been selected.
