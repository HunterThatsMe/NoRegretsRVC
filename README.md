# NoRegretsRVC

**NoRegretsRVC** is a voice-model training studio and configuration tool for preparing Applio-based Retrieval-based Voice Conversion (RVC) workflows. The project is currently at **Beta 0.0**.

> **Current scope:** This repository contains the NoRegretsRVC web interface, its configuration utilities, and a Kaggle notebook generator. The interface prepares settings and generates a notebook/workflow; it does not bundle the full Applio training runtime or model weights.

## Features

- **Training workspace** — create and configure voice-model projects, including sample rate, vocoder, pitch extraction, embedder, batch size, epochs, and precision.
- **VRAM estimate** — estimate memory use for a selected training configuration and flag settings that may be too demanding for a T4 GPU.
- **Kaggle notebook generator** — generate a configurable `.ipynb` workflow for an Applio-based training environment.
- **Samples / conversion workspace** — configure conversion settings and manage sample-related project state.
- **Models and workbench tools** — configure model downloads, voice blending, and text-to-speech defaults.
- **TensorBoard view** — inspect parsed training information and loss data.
- **Settings** — configure notebook tabs, tunnel options, and checkpoint backup preferences.
- **Local project persistence** — the interface saves supported settings in the browser's local storage.

Some workspaces prepare settings or display project data rather than executing the underlying RVC training or conversion process locally. See the current code and generated notebook for the exact behavior of Beta 0.0.

## Tech stack

- React 19 + TypeScript
- Vite 8
- Express
- Tailwind CSS 4
- Lucide React and Motion

## Getting started

### Requirements

- Windows, macOS, or Linux
- Node.js 22 or newer
- npm

### Install dependencies

```bash
npm install
```

### Start the development server

```bash
npm run dev
```

The development server prints its local address in the terminal when it starts.

### Build the frontend

```bash
npm run build
```

### Run the TypeScript check

```bash
npm run lint
```

## Project layout

```text
NoRegretsRVC/
├── public/
│   └── assets/                 # Static assets and bundled notebook reference
├── src/
│   ├── components/             # Main interface views and reusable UI
│   ├── data/                   # Default settings and sample data
│   ├── types/                  # TypeScript data models
│   └── utils/                  # Notebook generation, tuning, parsing, ZIP helpers
├── docs/                       # Project and development notes
├── .env.example                # Environment-variable template (no real secrets)
├── .gitignore
├── index.html
├── package.json
├── server.ts                   # Express server and notebook/update endpoints
└── vite.config.ts
```

## Environment variables

A starter `.env.example` is included. Copy it to `.env` only if you need to configure the corresponding environment values for your own setup. **Never commit `.env`, API keys, access tokens, passwords, or private model credentials.** The real `.env` file is excluded by `.gitignore`.

## Upstream project and acknowledgements

NoRegretsRVC generates workflows that use [Applio](https://github.com/IAHispano/Applio). Applio and its dependencies retain their own licenses and notices. Review the upstream license and all third-party notices before redistributing generated environments or bundled dependencies.

NoRegretsRVC is an independent project. Ilaria RVC was used only as a reference for repository organization and documentation style; its source code and README text have not been copied into this repository.

## Version

- **Application version:** Beta 0.0
- **Repository package version:** 0.0.0
- See [CHANGELOG.md](CHANGELOG.md) for project notes.

## License

A project license has not been added yet. Do not assume that the repository is licensed for reuse or redistribution until the project's original code and upstream-derived portions have been reviewed and an appropriate license is chosen.
