# Project structure

This document describes the layout of the Beta 0.0 source tree.

- `src/App.tsx` — application state, navigation, and top-level view selection.
- `src/components/` — training, conversion, workbench, TensorBoard, settings, sidebar, and shared interface components.
- `src/data/defaults.ts` — application version, default project configuration, and initial studio state.
- `src/types/studio.ts` — TypeScript types for projects and settings.
- `src/utils/forkGenerator.ts` — generates the configurable Kaggle notebook and related workflow artifacts.
- `src/utils/rvcTuning.ts` — configuration validation, command generation, and T4 VRAM estimation helpers.
- `src/utils/logParser.ts` — parses training logs and notebook data.
- `server.ts` — Express endpoints for notebook output and upstream release/commit checks; Vite middleware is used in development.
- `public/assets/` — static assets, including the bundled notebook reference file.

The generated training workflow relies on upstream Applio and its own dependencies. This repository does not include the complete upstream runtime or trained voice models.
