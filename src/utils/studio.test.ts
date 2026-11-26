import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {
  APP_VERSION,
  DEFAULT_PROJECTS,
  DEFAULT_STUDIO_STATE,
  LEGACY_CORE_PRETRAIN_D,
  LEGACY_CORE_PRETRAIN_G,
  LEGACY_CORE_PRETRAIN_NAME,
  SAMPLE_OVERTRAIN_LOG,
  SAMPLE_TRAINING_LOG,
} from '../data/defaults.ts';
import { FREE_SAMPLES } from '../data/freeSamples.ts';
import {
  buildBlendCliCommand,
  buildInferCliCommand,
  buildTrainPipelineCommands,
  estimateKaggleSurvival,
  estimateT4VramUsage,
  estimateTrainingCompletion,
  extractPthFilenameFromUrl,
  inferMatchingDUrl,
  isHuggingFaceRepoUrl,
  isPretrainedZipUrl,
  resolveCustomPretrainedPath,
  resolvePerGpuBatchSize,
  validateForkConfig,
} from './rvcTuning.ts';
import { createTranslator, resolveLocale } from './i18n.ts';
import { analyzeTrainingZones, parseTrainingNotebookOrLog } from './logParser.ts';
import { generateForkBundle, generateTurboTrainerPython } from './forkGenerator.ts';
import {
  applyBalancedAudioCleanup,
  clampCleanupStrength,
  collectUnusedBlobUrls,
  computeCleanupParameters,
  encodePcm16Wav,
} from './audioCleanup.ts';
import { buildZipArchive } from './zipBuilder.ts';

test('default project uses empty model name, 32000 SR, batch 2, save every 5, total 200, fp32, and required checkbox defaults', () => {
  const def = DEFAULT_PROJECTS[0];
  assert.equal(APP_VERSION, '0.0.0');
  assert.equal(DEFAULT_STUDIO_STATE.convertAudio.volumeEnvelope, 1);
  assert.equal(DEFAULT_STUDIO_STATE.convertAudio.protect, 0.5);
  assert.equal(DEFAULT_STUDIO_STATE.ttsDefaults.protect, 0.5);
  assert.equal(def.name, '');
  assert.equal(def.samplingRate, '32000');
  assert.equal(def.batchSize, 2);
  assert.equal(def.saveEveryEpoch, 5);
  assert.equal(def.totalEpoch, 200);
  assert.equal(def.silentTrainingFiles, 0);
  assert.equal(def.fp16, false);
  assert.equal(def.truncateSilenceEnabled, false);
  assert.equal(def.noiseFilter, false);
  assert.equal(def.loudnessNorm, false);
  assert.equal(def.saveOnlyLatest, true);
  assert.equal(def.saveEveryWeights, true);
  assert.equal(def.pretrained, true);
  assert.equal(def.freshTraining, false);
  assert.equal(def.cacheDatasetGpu, false);
  assert.equal(def.checkpointing, false);
  assert.equal(def.shutdownAfterFinish, false);
  assert.equal(def.customPretrained, false);
  assert.equal(def.agreedTerms, true);
  assert.equal(DEFAULT_STUDIO_STATE.convertAudio.agreedTerms, true);
  assert.equal(DEFAULT_STUDIO_STATE.generalSettings.listFilterEnabled, false);
  assert.equal(DEFAULT_STUDIO_STATE.trainingSettings.precision, 'fp32');
  assert.equal(DEFAULT_STUDIO_STATE.forkConfig.turboTraining, true);
});

test('estimateT4VramUsage flags Risk of OOM for oversized RefineGAN FP32 batch and Optimal for default', () => {
  const heavy = {
    ...DEFAULT_PROJECTS[0],
    samplingRate: '48000' as const,
    vocoder: 'RefineGAN' as const,
    batchSize: 16,
    fp16: false,
  };
  const est = estimateT4VramUsage(heavy);
  assert.equal(est.fitStatus, 'Risk of OOM');
  assert.ok(est.estimatedVramGb > 14.5);

  const defEst = estimateT4VramUsage(DEFAULT_PROJECTS[0]);
  assert.equal(defEst.fitStatus, 'Optimal');
});

test('buildTrainPipelineCommands emits truncate silence & loudness flags when toggled on', () => {
  const withToggles = {
    ...DEFAULT_PROJECTS[0],
    name: 'TestVoice',
    datasetPath: '/kaggle/input/datasets/user/ds',
    truncateSilenceEnabled: true,
    silenceDurationMs: 450,
    loudnessNorm: true,
  };
  const trainCmd = buildTrainPipelineCommands(withToggles);
  assert.match(trainCmd, /--truncate_silence True --silence_duration_ms 450/);
  assert.match(trainCmd, /--process_effects True --loudness_target -14/);
  assert.match(trainCmd, /--fp16 False/);

  const inferCmd = buildInferCliCommand(DEFAULT_STUDIO_STATE.convertAudio);
  assert.match(inferCmd, /python core\.py infer/);
  assert.match(inferCmd, /--sid 0 --f0_method rmvpe --embedder_model contentvec/);
  assert.match(inferCmd, /--protect 0\.5 --volume_envelope 1/);
  assert.match(inferCmd, /--proposed_pitch False/);

  const enhancedInferCmd = buildInferCliCommand({
    ...DEFAULT_STUDIO_STATE.convertAudio,
    proposedPitch: true,
    noiseReduction: true,
    noiseReductionStrength: 18,
    formantShiftEnabled: true,
    formantQuefrency: 1.5,
    formantTimbre: 1.2,
  });
  assert.match(enhancedInferCmd, /--proposed_pitch True/);
  assert.match(enhancedInferCmd, /--clean_audio True --clean_strength 18/);
  assert.match(enhancedInferCmd, /--formant_shifting True --formant_qfrency 1\.5 --formant_timbre 1\.2/);

  const blendCmd = buildBlendCliCommand(DEFAULT_STUDIO_STATE.blender);
  assert.match(blendCmd, /python core\.py model_blend/);
});

test('validateForkConfig catches missing Horizon ID and invalid customRepoUrl', () => {
  const issues = validateForkConfig({
    ...DEFAULT_STUDIO_STATE.forkConfig,
    tunnelMode: 'Horizon',
    horizonId: '',
    customRepoUrl: 'not-a-github-url',
  });
  assert.ok(issues.some((i) => i.field === 'horizonId' && i.severity === 'warning'));
  assert.ok(issues.some((i) => i.field === 'customRepoUrl' && i.severity === 'warning'));
});

test('parseTrainingNotebookOrLog parses raw training stdout, live tqdm bar, and ipynb JSON accurately', () => {
  const rawResult = parseTrainingNotebookOrLog(SAMPLE_TRAINING_LOG);
  assert.ok(rawResult);
  assert.equal(rawResult.speakerName, 'StudioDemo');
  assert.equal(rawResult.segmentCount, 411);
  assert.equal(rawResult.epochsCompleted, 51);
  assert.equal(rawResult.lowestLoss, 14.616);

  const liveLog = [
    'DemoVoice | epoch=1 | step=107 | time=17:02:47 | training_speed=0:00:54',
    'DemoVoice | epoch=2 | step=214 | time=17:03:39 | training_speed=0:00:51 | lowest_value=29.003 (epoch 1 and step 104)',
    'epoch=3 | step=296 |  77%|█████████▏  | batch=82/107 [00:39<00:11,  2.17it/s]',
  ].join('\n');
  const parsedLive = parseTrainingNotebookOrLog(liveLog);
  assert.ok(parsedLive);
  assert.equal(parsedLive.points.length, 2);
  assert.equal(parsedLive.lowestLossEpoch, 1);
  assert.ok(parsedLive.liveProgress);
  assert.equal(parsedLive.liveProgress?.epoch, 3);
  assert.equal(parsedLive.liveProgress?.step, 296);
  assert.equal(parsedLive.liveProgress?.percent, 77);
  assert.equal(parsedLive.liveProgress?.batchDone, 82);
  assert.equal(parsedLive.liveProgress?.batchTotal, 107);
  assert.equal(parsedLive.liveProgress?.itPerSec, 2.17);
});

test('generateForkBundle produces locked NoRegretsRVC Streamlit WebUI, config.toml theme, turbo trainer, Samples & Models tabs, and ZIP archive', () => {
  const files = generateForkBundle(
    DEFAULT_STUDIO_STATE.forkConfig,
    DEFAULT_PROJECTS[0],
    DEFAULT_STUDIO_STATE.downloadManifest
  );
  assert.equal(files.length, 5);
  const nbFile = files.find((f) => f.id === 'notebook');
  assert.ok(nbFile);
  const parsedNb = JSON.parse(nbFile.content);
  assert.equal(parsedNb.nbformat, 4);
  assert.match(nbFile.content, /noregrets_turbo\.py/);
  assert.match(nbFile.content, /streamlit>=1\.40\.0/);
  assert.match(nbFile.content, /streamlit_app\.py/);
  assert.match(nbFile.content, /config\.toml/);
  assert.match(nbFile.content, /NoRegretsRVC Live Monitor/);
  assert.match(nbFile.content, /Lowest Loss/);
  assert.match(nbFile.content, /deque\(maxlen=25\)/);
  const cell1Py = parsedNb.cells[2].source.join('');
  const cell2Py = parsedNb.cells[4].source.join('');
  assert.match(cell2Py, /"streamlit", "run", "streamlit_app\.py"/);
  assert.match(cell1Py, /_cfg_d\["precision"\] = "fp32"/);
  assert.match(cell1Py, /_cfg_d\["discord_presence"\] = False/);
  assert.match(cell1Py, /\.streamlit/);
  assert.doesNotMatch(nbFile.content, /core_backup\.py/);
  assert.doesNotMatch(nbFile.content, /filebrowser/i);
  assert.doesNotMatch(nbFile.content, /hf_token/);

  const turboFile = files.find((f) => f.id === 'turbo-py');
  assert.ok(turboFile);
  assert.match(turboFile.content, /cudnn\.benchmark = True/);
  assert.match(turboFile.content, /kwargs\["foreach"\] = True/);
  assert.doesNotMatch(turboFile.content, /fused=True/);
  assert.match(turboFile.content, /NR_DDP_LR_SCALE/);
  assert.match(turboFile.content, /destroy_process_group/);
  assert.match(turboFile.content, /gradient_as_bucket_view/);
  assert.match(turboFile.content, /_MIN_SYSTEM_FREE_RAM_BYTES/);
  assert.doesNotMatch(turboFile.content, /allow_tf32/);

  const turboDisabledFiles = generateForkBundle(
    { ...DEFAULT_STUDIO_STATE.forkConfig, turboTraining: false },
    DEFAULT_PROJECTS[0],
    DEFAULT_STUDIO_STATE.downloadManifest
  );
  const disabledNb = turboDisabledFiles.find((f) => f.id === 'notebook');
  assert.ok(disabledNb);
  assert.doesNotMatch(disabledNb.content, /noregrets_turbo\.apply\(\)/);

  const turboPy = generateTurboTrainerPython();
  assert.match(turboPy, /def apply\(\) -> None:/);

  const themeFile = files.find((f) => f.id === 'theme-py');
  assert.ok(themeFile);
  assert.equal(themeFile.zipPath, '.streamlit/config.toml');
  assert.match(themeFile.content, /base = "dark"/);
  assert.match(themeFile.content, /primaryColor = "#E51923"/);
  assert.match(themeFile.content, /backgroundColor = "#0D0405"/);
  assert.match(themeFile.content, /secondaryBackgroundColor = "#160709"/);

  const appFile = files.find((f) => f.id === 'app-py');
  assert.ok(appFile);
  assert.equal(appFile.zipPath, 'streamlit_app.py');
  assert.match(appFile.content, /import streamlit as st/);
  assert.match(appFile.content, /st\.set_page_config/);
  assert.match(appFile.content, /nr-header-card/);
  assert.match(appFile.content, /\.applio-step/);
  assert.match(appFile.content, /"Samples"/);
  assert.match(appFile.content, /"Free Samples"/);
  assert.match(appFile.content, /"Models"/);
  assert.match(appFile.content, /"Diagnostics"/);
  assert.match(appFile.content, /"Voice Blender"/);
  assert.match(appFile.content, /"TTS"/);
  assert.match(appFile.content, /"Settings"/);
  assert.doesNotMatch(appFile.content, /active_tab == "Extra"/);
  assert.doesNotMatch(appFile.content, /active_tab == "TensorBoard"|"TensorBoard"/);
  assert.doesNotMatch(appFile.content, /Refresh Log/);
  assert.doesNotMatch(appFile.content, /Save Language/);
  assert.doesNotMatch(appFile.content, /none_strict/);
  assert.match(appFile.content, /nr-nav-marker/);
  assert.match(appFile.content, /"System Auto"/);
  assert.match(appFile.content, /def _resolve_lang\(\)/);
  assert.match(appFile.content, /direction:\s*rtl/);
  assert.match(appFile.content, /Download pretrained pair/);
  assert.match(appFile.content, /Source URL/);
  assert.match(appFile.content, /Advanced Settings \(\{f0_cur\} · index \{idx_cur:\.2f\}/);
  assert.match(appFile.content, /Switches the studio interface instantaneously/);
  assert.match(appFile.content, /Check for updates/);
  assert.match(appFile.content, /nr_precision_initialized/);
  assert.match(appFile.content, /"nr_silent_files": 0/);
  assert.match(appFile.content, /_synced_slider_num\(\s*"Batch Size"/);
  assert.match(appFile.content, /_synced_slider_num\(\s*"Save Every Epoch"/);
  assert.match(appFile.content, /_synced_slider_num\(\s*"Total Epoch"/);
  assert.match(appFile.content, /"nr_trunc_sil"/);
  assert.match(appFile.content, /"nr_loud_norm"/);
  assert.match(appFile.content, /\("en_US", "English"\)/);
  assert.match(appFile.content, /key="nr_train_terms"/);
  assert.match(appFile.content, /key="nr_inf_terms"/);
  assert.match(appFile.content, /start_new_session=True/);
  assert.match(appFile.content, /"nr_model_name": ""/);
  assert.doesNotMatch(appFile.content, /backdrop-filter:\s*blur/i);
  assert.match(appFile.content, /nr-toast-stack/);
  assert.match(appFile.content, /@st\.cache_data/);
  assert.match(appFile.content, /def _start_core_task_bg/);
  assert.match(appFile.content, /def _poll_core_tasks/);
  assert.match(appFile.content, /✔ Preprocess complete!/);
  assert.match(appFile.content, /✔ Feature extraction complete!/);
  assert.match(appFile.content, /✔ Index generation complete!/);
  assert.match(appFile.content, /✔ Training complete!/);
  assert.match(appFile.content, /Best Epoch Analyzer/);
  assert.match(appFile.content, /Waiting for Training Data/);
  assert.match(appFile.content, /python core\.py infer/);
  assert.match(appFile.content, /Load All 6 in Samples/);
  assert.match(appFile.content, /Download All 6 \(\.zip\)/);
  assert.match(appFile.content, /Automatically off by default to keep model and index selectors clean/);
  assert.match(appFile.content, /FP32 prioritizes maximum stability on T4 GPUs/);
  assert.match(appFile.content, /Verifies your installed NoRegretsRVC build/);
  assert.doesNotMatch(appFile.content, /kw = \{json\.dumps\(kwargs\)\}/);
  assert.doesNotMatch(appFile.content, /if _has_active_bg_work\(\):\s*time\.sleep\(2\.0\)\s*st\.rerun\(\)/);
  assert.match(appFile.content, /\.args\.json/);
  assert.doesNotMatch(appFile.content, /contextlib\.ExitStack/);
  assert.doesNotMatch(appFile.content, /_tab_slot\s*=\s*st\.empty/);
  assert.match(appFile.content, /def _render_active_tab/);
  assert.match(appFile.content, /def _select_tab/);
  assert.match(appFile.content, /on_click=_select_tab/);
  assert.match(appFile.content, /def _filter_status_lines/);
  assert.match(appFile.content, /def _start_pretrain_download_bg/);
  assert.match(appFile.content, /Pretrained Download Complete/);
  assert.match(appFile.content, /def _read_file_bytes_cached/);
  assert.match(appFile.content, /def _render_samples_workspace_frag/);
  assert.match(appFile.content, /def _sync_samples_state/);
  assert.match(appFile.content, /_nr_prev_\{def_key\}/);
  assert.match(appFile.content, /_nr_bg_watcher/);
  assert.match(appFile.content, /_nr_card\("nr_train_step_5"\)/);
  assert.match(appFile.content, /_nr_card\("nr_samples_step_1"\)/);
  assert.match(appFile.content, /_nr_card\("nr_samples_step_2"\)/);
  assert.match(appFile.content, /_nr_card\("nr_samples_step_3"\)/);
  assert.match(appFile.content, /_nr_card\("nr_samples_step_4"\)/);
  assert.match(appFile.content, /"Pitch & Settings"/);
  assert.match(appFile.content, /"Convert & Output"/);
  assert.match(appFile.content, /"Clip Pitch Override"/);
  assert.match(appFile.content, /"Override pitch, index, model & advanced \(optional\)"/);
  assert.match(appFile.content, /"View Generated core\.py CLI Command"/);
  assert.doesNotMatch(appFile.content, /"Default Pitch"/);
  assert.match(appFile.content, /key="nr_cp_g_up"/);
  assert.match(appFile.content, /key="nr_cp_d_up"/);
  assert.match(appFile.content, /overflow-anchor:\s*none/);
  assert.match(nbFile.content, /NameError:/);
  assert.match(cell2Py, /"--server\.fileWatcherType",\s*"none"/);
  assert.match(nbFile.content, /STREAMLIT_SERVER_FILE_WATCHER_TYPE/);
  assert.match(nbFile.content, /local_sources_watcher/);
  assert.match(nbFile.content, /⏳ Downloading pretrained/);

  // Performance & bugfix assertions
  assert.doesNotMatch(appFile.content, /background-attachment:\s*fixed/i);
  assert.match(appFile.content, /def _next_clip_id\(/);
  assert.match(appFile.content, /def _read_tail_text\(/);
  assert.match(appFile.content, /os\.walk\(str\(base\),\s*followlinks=True\)/);
  assert.match(appFile.content, /nr_inf_seen_ups/);
  assert.match(appFile.content, /nr_dl_sav_\{s_id\}/);
  assert.match(appFile.content, /st\.session_state\["nr_vb_ratio_sl"\] = 0\.5/);
  assert.match(appFile.content, /st\.session_state\["nr_cp_d_inp"\] = inferred_d/);
  assert.match(appFile.content, /\("nr_adv_idx",\s*"nr_def_idx_rate",\s*float,\s*0\.75\)/);

  const zipBytes = buildZipArchive(files.map((f) => ({ path: f.zipPath, content: f.content })));
  assert.equal(zipBytes[0], 0x50);
  assert.equal(zipBytes[1], 0x4b);
});

test('estimateKaggleSurvival stays Safe by default and flags Risk of Full if saveOnlyLatest is disabled', () => {
  const safeEst = estimateKaggleSurvival(DEFAULT_PROJECTS[0]);
  assert.equal(safeEst.diskStatus, 'Safe');
  assert.equal(safeEst.sessionStatus, 'Fits 12h');
  assert.ok(safeEst.estimatedDiskGb < 10);

  const diskHeavy = {
    ...DEFAULT_PROJECTS[0],
    saveOnlyLatest: false,
    saveEveryEpoch: 5,
    totalEpoch: 200,
  };
  const heavyEst = estimateKaggleSurvival(diskHeavy);
  assert.equal(heavyEst.diskStatus, 'Risk of Full');
  assert.ok(heavyEst.estimatedDiskGb > 20);
});

test('analyzeTrainingZones refuses fake pre-training epochs and pinpoints Best Epoch from real observed training data', () => {
  const projEvery5 = { ...DEFAULT_PROJECTS[0], name: 'TestVoice', saveEveryEpoch: 5, totalEpoch: 200 };

  // No training epochs yet -> untrained state with null epochs (no fabricated chart/zones)
  const noLogZones = analyzeTrainingZones(null, projEvery5, 'untested');
  assert.equal(noLogZones.confidence, 'untrained');
  assert.equal(noLogZones.saveEveryEpoch, 5);
  assert.equal(noLogZones.undertrainingEndEpoch, null);
  assert.equal(noLogZones.sweetSpotMinEpoch, null);
  assert.equal(noLogZones.sweetSpotMaxEpoch, null);
  assert.equal(noLogZones.sweetSpotEpoch, null);
  assert.equal(noLogZones.overtrainingStartEpoch, null);
  assert.deepEqual(noLogZones.candidateEpochs, []);

  // Still dropping sample log (saveEveryEpoch = 5) -> observed Best Epoch so far
  const parsedSample = parseTrainingNotebookOrLog(SAMPLE_TRAINING_LOG);
  assert.ok(parsedSample);
  const sampleZones = analyzeTrainingZones(parsedSample, projEvery5, 'untested');
  assert.equal(sampleZones.confidence, 'observed');
  assert.ok(sampleZones.sweetSpotEpoch !== null);
  assert.equal((sampleZones.sweetSpotEpoch ?? 0) % 5, 0);
  assert.equal((sampleZones.sweetSpotMinEpoch ?? 0) % 5, 0);
  assert.equal((sampleZones.sweetSpotMaxEpoch ?? 0) % 5, 0);

  // Plateau log with custom saveEveryEpoch = 10 -> every suggested epoch snaps to multiple of 10
  const projEvery10 = { ...DEFAULT_PROJECTS[0], saveEveryEpoch: 10, totalEpoch: 200 };
  const plateauLog = [
    'VoiceA | epoch=11 | step=1000 | time=10:01:00 | training_speed=0:00:45 | lowest_value=21.000 (epoch 11 and step 1000)',
    'VoiceA | epoch=42 | step=4000 | time=10:04:00 | training_speed=0:00:45 | lowest_value=15.200 (epoch 42 and step 4000)',
    'VoiceA | epoch=78 | step=8000 | time=10:08:00 | training_speed=0:00:45 | lowest_value=14.300 (epoch 78 and step 8000)',
    'VoiceA | epoch=122 | step=12000 | time=10:12:00 | training_speed=0:00:45 | lowest_value=14.250 (epoch 122 and step 12000)',
    'VoiceA | epoch=163 | step=16000 | time=10:16:00 | training_speed=0:00:45 | lowest_value=14.280 (epoch 163 and step 16000)',
  ].join('\n');
  const parsedPlateau = parseTrainingNotebookOrLog(plateauLog);
  assert.ok(parsedPlateau);
  const observedZones = analyzeTrainingZones(parsedPlateau, projEvery10, 'clean');
  assert.equal(observedZones.confidence, 'observed');
  assert.ok(observedZones.sweetSpotEpoch !== null);
  assert.equal((observedZones.sweetSpotEpoch ?? 0) % 10, 0);
  assert.equal((observedZones.undertrainingEndEpoch ?? 0) % 10, 0);
  assert.equal((observedZones.overtrainingStartEpoch ?? 0) % 10, 0);
  assert.ok((observedZones.candidateEpochs ?? []).includes(observedZones.sweetSpotEpoch!));

  // Overtrained sound feedback shifts sweet spot earlier and still aligns to saveEveryEpoch
  const overtrainedZones = analyzeTrainingZones(parsedPlateau, projEvery10, 'overtrained');
  assert.ok((overtrainedZones.sweetSpotEpoch ?? 999) < (observedZones.sweetSpotEpoch ?? 0));
  assert.equal((overtrainedZones.sweetSpotEpoch ?? 0) % 10, 0);
});

test('FREE_SAMPLES catalog provides 3 male and 3 female speaking clips backed by real WAV files', () => {
  assert.equal(FREE_SAMPLES.length, 6);
  assert.equal(FREE_SAMPLES.filter((s) => s.gender === 'male').length, 3);
  assert.equal(FREE_SAMPLES.filter((s) => s.gender === 'female').length, 3);

  const ids = new Set(FREE_SAMPLES.map((s) => s.id));
  const urls = new Set(FREE_SAMPLES.map((s) => s.url));
  assert.equal(ids.size, 6);
  assert.equal(urls.size, 6);

  for (const sample of FREE_SAMPLES) {
    assert.ok(sample.url.startsWith('/assets/aistudio/'));
    assert.ok(sample.fileName.endsWith('.wav'));
    const diskPath = path.join(process.cwd(), 'public', sample.url.replace(/^\//, ''));
    assert.ok(fs.existsSync(diskPath), `Missing sample file on disk: ${diskPath}`);
    assert.ok(fs.statSync(diskPath).size > 100_000);
  }
});

test('audioCleanup reduces quiet background hiss/rumble cleanly without crushing speech peaks', () => {
  assert.equal(clampCleanupStrength(0), 1);
  assert.equal(clampCleanupStrength(99), 30);
  assert.equal(computeCleanupParameters(12).tierLabel, 'Balanced');

  const sr = 32000;
  const totalSamples = sr; // 1.0s clip: first 0.5s quiet noise+rumble, second 0.5s speech-like tone
  const raw = new Float32Array(totalSamples);
  for (let i = 0; i < totalSamples; i++) {
    const t = i / sr;
    const rumble = 0.015 * Math.sin(2 * Math.PI * 35 * t);
    const hiss = 0.008 * Math.sin(2 * Math.PI * 5200 * t);
    const speech = i >= sr / 2 ? 0.45 * Math.sin(2 * Math.PI * 220 * t) : 0;
    raw[i] = rumble + hiss + speech;
  }

  const cleaned = applyBalancedAudioCleanup(raw, sr, 12);
  assert.equal(cleaned.length, raw.length);

  const rmsRange = (buf: Float32Array, from: number, to: number) => {
    let sum = 0;
    for (let i = from; i < to; i++) sum += buf[i] * buf[i];
    return Math.sqrt(sum / (to - from));
  };

  const quietBefore = rmsRange(raw, Math.round(sr * 0.1), Math.round(sr * 0.45));
  const quietAfter = rmsRange(cleaned, Math.round(sr * 0.1), Math.round(sr * 0.45));
  const speechBefore = rmsRange(raw, Math.round(sr * 0.6), Math.round(sr * 0.95));
  const speechAfter = rmsRange(cleaned, Math.round(sr * 0.6), Math.round(sr * 0.95));

  // Quiet floor should be meaningfully reduced (not underdoing it) yet remain non-zero (not hard-gated)
  assert.ok(quietAfter < quietBefore * 0.65, `Expected quiet RMS reduction, got ${quietAfter} vs ${quietBefore}`);
  assert.ok(quietAfter > 0.0005, `Expected natural floor retention without hard mute, got ${quietAfter}`);
  // Active vocal energy should remain intact (not overdoing it)
  assert.ok(speechAfter > speechBefore * 0.9, `Expected vocal preservation, got ${speechAfter} vs ${speechBefore}`);

  const wavBlob = encodePcm16Wav([cleaned], sr);
  assert.equal(wavBlob.type, 'audio/wav');
  assert.equal(wavBlob.size, 44 + totalSamples * 2);
});

test('estimateTrainingCompletion computes remaining time, Today/Tomorrow rollover, warmup vs stabilized accuracy %, and disclaimer', () => {
  const baseMorning = new Date(2026, 10, 20, 10, 0, 0).getTime();
  const warmup = estimateTrainingCompletion({
    currentEpoch: 1,
    totalEpoch: 50,
    batchDone: 10,
    batchTotal: 100,
    elapsedSec: 5,
    itPerSec: 2.0,
    nowMs: baseMorning,
  });
  assert.ok(warmup.remainingSeconds > 2400 && warmup.remainingSeconds < 2600);
  assert.match(warmup.remainingLabel, /~4[0-9] minutes/);
  assert.match(warmup.finishDayLabel, /^Today · /);
  assert.ok(warmup.finishTimeLabel.length >= 4);
  assert.ok(warmup.accuracyPercent >= 28 && warmup.accuracyPercent <= 45);
  assert.match(warmup.disclaimer, /rough estimate/i);

  const stable = estimateTrainingCompletion({
    currentEpoch: 25,
    totalEpoch: 50,
    batchDone: 50,
    batchTotal: 100,
    elapsedSec: 1200,
    itPerSec: 2.0,
    completedEpochDurationsSec: [50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
    nowMs: baseMorning,
  });
  assert.ok(stable.accuracyPercent >= 78, `Expected stabilized accuracy >= 78, got ${stable.accuracyPercent}`);
  assert.ok(stable.accuracyPercent > warmup.accuracyPercent);

  const lateNight = new Date(2026, 10, 20, 23, 50, 0).getTime();
  const rollover = estimateTrainingCompletion({
    currentEpoch: 1,
    totalEpoch: 30,
    batchDone: 0,
    batchTotal: 100,
    elapsedSec: 0,
    itPerSec: 1.0,
    nowMs: lateNight,
  });
  assert.match(rollover.finishDayLabel, /^Tomorrow · /);
});

test('i18n resolveLocale and createTranslator switch languages cleanly with Arabic RTL', () => {
  assert.equal(resolveLocale('System Auto', 'es-MX'), 'es_ES');
  assert.equal(resolveLocale('System Auto', 'pt-BR'), 'pt_BR');
  assert.equal(resolveLocale('fr_FR', 'en-US'), 'fr_FR');

  const es = createTranslator('es_ES');
  assert.equal(es.isRtl, false);
  assert.equal(es.t('nav.training'), 'Entrenamiento');
  assert.equal(es.t('nav.settings'), 'Ajustes');

  const ar = createTranslator('ar_SA');
  assert.equal(ar.isRtl, true);
  assert.equal(ar.t('nav.training'), 'التدريب');
  assert.equal(ar.t('customPretrain.uploadBtn'), 'رفع ملف .pth');
});

test('extractPthFilenameFromUrl, resolveCustomPretrainedPath, and pipeline commands support Hugging Face G & D URLs', () => {
  const gUrl = 'https://huggingface.co/lyery/test/resolve/main/G_2333333%20(6).pth?download=true';
  const dUrl = 'https://huggingface.co/lyery/test/resolve/main/D_2333333%20(6).pth?download=true';

  assert.equal(extractPthFilenameFromUrl(gUrl, 'custom_G.pth'), 'G_2333333 (6).pth');
  assert.equal(extractPthFilenameFromUrl(dUrl, 'custom_D.pth'), 'D_2333333 (6).pth');

  assert.equal(
    resolveCustomPretrainedPath(gUrl, 'G'),
    'rvc/models/pretraineds/custom/G_2333333 (6).pth'
  );
  assert.equal(
    resolveCustomPretrainedPath(dUrl, 'D'),
    'rvc/models/pretraineds/custom/D_2333333 (6).pth'
  );

  const cmd = buildTrainPipelineCommands({
    ...DEFAULT_PROJECTS[0],
    name: 'CustomPretrainTest',
    customPretrained: true,
    customPretrainedG: gUrl,
    customPretrainedD: dUrl,
  });
  assert.match(cmd, /curl -fL "https:\/\/huggingface\.co\/lyery\/test\/resolve\/main\/G_2333333%20\(6\)\.pth\?download=true" -o "rvc\/models\/pretraineds\/custom\/G_2333333 \(6\)\.pth"/);
  assert.match(cmd, /curl -fL "https:\/\/huggingface\.co\/lyery\/test\/resolve\/main\/D_2333333%20\(6\)\.pth\?download=true" -o "rvc\/models\/pretraineds\/custom\/D_2333333 \(6\)\.pth"/);
  assert.match(cmd, /--custom_pretrained True --g_pretrained_path "rvc\/models\/pretraineds\/custom\/G_2333333 \(6\)\.pth" --d_pretrained_path "rvc\/models\/pretraineds\/custom\/D_2333333 \(6\)\.pth"/);

  const bundle = generateForkBundle(
    DEFAULT_STUDIO_STATE.forkConfig,
    {
      ...DEFAULT_PROJECTS[0],
      customPretrained: true,
      customPretrainedG: gUrl,
      customPretrainedD: dUrl,
    },
    DEFAULT_STUDIO_STATE.downloadManifest
  );
  const nb = bundle.find((f) => f.id === 'notebook');
  assert.ok(nb);
  assert.match(nb.content, /custom_pretrain_dir/);
  assert.match(nb.content, /G_2333333%20\(6\)\.pth\?download=true/);
  assert.match(nb.content, /D_2333333%20\(6\)\.pth\?download=true/);
});

test('inferMatchingDUrl, HF repo detection, ZIP detection, auto-D CLI & Streamlit crimson scrollbar CSS work accurately', () => {
  const gUrl = 'https://huggingface.co/lyery/test/resolve/main/G_2333333%20(6).pth?download=true';
  const expectedDUrl = 'https://huggingface.co/lyery/test/resolve/main/D_2333333%20(6).pth?download=true';

  assert.equal(inferMatchingDUrl(gUrl), expectedDUrl);
  assert.equal(inferMatchingDUrl('rvc/models/pretraineds/custom/f0G40k.pth'), 'rvc/models/pretraineds/custom/f0D40k.pth');
  assert.equal(inferMatchingDUrl('https://huggingface.co/author/pretrained'), '');
  assert.equal(inferMatchingDUrl('https://huggingface.co/author/repo/resolve/main/weights.zip'), '');

  assert.equal(isHuggingFaceRepoUrl('https://huggingface.co/author/pretrained'), true);
  assert.equal(isHuggingFaceRepoUrl(gUrl), false);
  assert.equal(isPretrainedZipUrl('https://huggingface.co/author/repo/resolve/main/weights.zip?download=true'), true);
  assert.equal(isPretrainedZipUrl(gUrl), false);

  // Auto-discovers matching D in pipeline commands when D is left blank
  const autoCmd = buildTrainPipelineCommands({
    ...DEFAULT_PROJECTS[0],
    name: 'AutoDVoice',
    customPretrained: true,
    customPretrainedG: gUrl,
    customPretrainedD: '',
  });
  assert.match(autoCmd, /curl -fL "https:\/\/huggingface\.co\/lyery\/test\/resolve\/main\/D_2333333%20\(6\)\.pth\?download=true"/);
  assert.match(autoCmd, /--d_pretrained_path "rvc\/models\/pretraineds\/custom\/D_2333333 \(6\)\.pth"/);

  // HF repo skips direct curl in shell preview & uses repo helper in notebook
  const repoCmd = buildTrainPipelineCommands({
    ...DEFAULT_PROJECTS[0],
    name: 'RepoVoice',
    customPretrained: true,
    customPretrainedG: 'https://huggingface.co/author/pretrained',
    customPretrainedD: '',
  });
  assert.doesNotMatch(repoCmd, /curl -fL "https:\/\/huggingface\.co\/author\/pretrained"/);

  const bundle = generateForkBundle(
    DEFAULT_STUDIO_STATE.forkConfig,
    {
      ...DEFAULT_PROJECTS[0],
      customPretrained: true,
      customPretrainedG: gUrl,
      customPretrainedD: '',
    },
    DEFAULT_STUDIO_STATE.downloadManifest
  );
  const nb = bundle.find((f) => f.id === 'notebook');
  assert.ok(nb);
  assert.match(nb.content, /_download_hf_repo_pretrains/);
  assert.match(nb.content, /zipfile\.ZipFile/);
  assert.match(nb.content, /D_2333333%20\(6\)\.pth\?download=true/);

  const appPy = bundle.find((f) => f.id === 'app-py');
  assert.ok(appPy);
  assert.match(appPy.content, /scrollbar-color:\s*#e51923\s+#100506/i);
  assert.match(appPy.content, /::-webkit-scrollbar-thumb/);
});

test('collectUnusedBlobUrls protects duplicate queue items and Recent Generated Samples from premature revocation', () => {
  const sharedBlob = 'blob:http://localhost:3000/shared-source-1';
  const deadConverted = 'blob:http://localhost:3000/old-converted-2';
  const savedConverted = 'blob:http://localhost:3000/still-in-saved-3';
  const staticAsset = '/assets/aistudio/male_sample_1.wav';

  // Removing one duplicate clip must NOT revoke sharedBlob if another clip OR saved sample references it
  const remainingQueue = [{ sourceUrl: sharedBlob, convertedUrl: '' }];
  const remainingSaved = [{ url: savedConverted }];
  const dead = collectUnusedBlobUrls(
    [sharedBlob, deadConverted, savedConverted, staticAsset],
    remainingQueue,
    remainingSaved
  );
  assert.deepEqual(dead, [deadConverted]);

  // Discarding a converted output from Recent Generated Samples leaves nothing referencing it -> revokes once
  const deadAfterClear = collectUnusedBlobUrls(
    [sharedBlob, savedConverted, savedConverted],
    [],
    []
  );
  assert.deepEqual(deadAfterClear, [sharedBlob, savedConverted]);
});

test('multi-metric logParser scores datasetMatchPercent, stabilityScore, and metricBreakdown accurately', () => {
  const multiMetricLog = [
    'epoch=5 | step=100 | training_speed=0:00:45 | lowest_value=24.50 (epoch 5) | mel=24.5 kl=1.45 fm=6.8 d_total=2.15 g_total=32.1',
    'epoch=10 | step=200 | training_speed=0:00:44 | lowest_value=17.80 (epoch 10) | mel=17.8 kl=1.10 fm=4.9 d_total=1.92 g_total=25.4',
    'epoch=15 | step=300 | training_speed=0:00:44 | lowest_value=13.90 (epoch 15) | mel=13.9 kl=0.88 fm=3.8 d_total=1.88 g_total=20.2',
  ].join('\n');

  const analysis = parseTrainingNotebookOrLog(multiMetricLog);
  assert.ok(analysis);
  assert.equal(analysis.epochsCompleted, 15);
  assert.ok(analysis.avgKlLoss != null && analysis.avgKlLoss < 1.5);
  assert.ok(analysis.avgFmLoss != null && analysis.avgFmLoss < 7);
  assert.ok(analysis.avgDiscLoss != null);
  assert.ok(analysis.avgGenLoss != null);

  const zone = analyzeTrainingZones(analysis, DEFAULT_PROJECTS[0], 'clean');
  assert.equal(zone.confidence, 'observed');
  assert.ok(zone.datasetMatchPercent != null && zone.datasetMatchPercent >= 70 && zone.datasetMatchPercent <= 99);
  assert.ok(zone.stabilityScore != null && zone.stabilityScore >= 50 && zone.stabilityScore <= 99);
  assert.ok(Array.isArray(zone.metricBreakdown) && zone.metricBreakdown.length === 8);
  assert.ok(Array.isArray(zone.rankedCheckpoints) && zone.rankedCheckpoints.length > 0);
  assert.equal(typeof zone.undertrainingRiskPercent, 'number');
  assert.equal(typeof zone.overtrainingRiskPercent, 'number');
});

test('1-GPU Applio batch parity splits per-GPU batch on Dual T4 and overtraining detector catches tail rebound', () => {
  const split = resolvePerGpuBatchSize(8, 2, true);
  assert.equal(split.perGpuBatchSize, 4);
  assert.equal(split.totalEffectiveBatchSize, 8);
  assert.equal(split.isParitySplit, true);

  const rawDual = resolvePerGpuBatchSize(8, 2, false);
  assert.equal(rawDual.perGpuBatchSize, 8);
  assert.equal(rawDual.totalEffectiveBatchSize, 16);
  assert.equal(rawDual.isParitySplit, false);

  const cmd = buildTrainPipelineCommands({
    ...DEFAULT_PROJECTS[0],
    name: 'ParityModel',
    batchSize: 8,
    gpuCount: 2,
    batchParity1Gpu: true,
    totalEpoch: 200,
    saveEveryEpoch: 100,
  });
  assert.match(cmd, /--batch_size 4\b/);
  assert.match(cmd, /--save_every_epoch 100\b/);
  assert.match(cmd, /--total_epoch 200\b/);

  const overParsed = parseTrainingNotebookOrLog(SAMPLE_OVERTRAIN_LOG);
  assert.ok(overParsed);
  assert.equal(overParsed.epochsCompleted, 240);
  const overZone = analyzeTrainingZones(overParsed, { ...DEFAULT_PROJECTS[0], saveEveryEpoch: 20, totalEpoch: 240 }, 'untested');
  assert.equal(overZone.trainingVerdict, 'overtrained');
  assert.ok((overZone.overtrainingRiskPercent ?? 0) >= 50, `Expected high overtraining risk, got ${overZone.overtrainingRiskPercent}`);
  assert.ok((overZone.sweetSpotEpoch ?? 999) < 240, `Expected recommended epoch < 240, got ${overZone.sweetSpotEpoch}`);
  assert.ok(overZone.stepsPerClip !== null && overZone.stepsPerClip !== undefined && overZone.stepsPerClip > 5);
  assert.equal(overZone.rankedCheckpoints?.[0]?.epoch, overZone.sweetSpotEpoch);
});

test('generated Kaggle WebUI, noregrets_turbo.py, and notebook include all v0.0.0 fixes and Legacy Core 1.5 V2 defaults', () => {
  assert.equal(LEGACY_CORE_PRETRAIN_NAME, 'Legacy Core 1.5 V2');
  assert.match(LEGACY_CORE_PRETRAIN_G, /G_LegacyCore1\.5v2\.pth/);
  assert.match(LEGACY_CORE_PRETRAIN_D, /D_LegacyCore1\.5v2\.pth/);

  const bundle = generateForkBundle(
    DEFAULT_STUDIO_STATE.forkConfig,
    DEFAULT_PROJECTS[0],
    DEFAULT_STUDIO_STATE.downloadManifest
  );
  const appFile = bundle.find((f) => f.id === 'app-py');
  const nbFile = bundle.find((f) => f.id === 'notebook');
  const turboFile = bundle.find((f) => f.id === 'turbo-py');
  assert.ok(appFile && nbFile && turboFile);

  // Streamlit WebUI checks
  assert.match(appFile.content, /\[data-testid="InputInstructions"\]/);
  assert.match(appFile.content, /Lossless Turbo Training/);
  assert.match(appFile.content, /NR_TURBO_ENABLED/);
  assert.match(appFile.content, /Dual T4 Effective:/);
  assert.match(appFile.content, /1-GPU Applio Batch Parity/);
  assert.match(appFile.content, /nr_batch_parity/);
  assert.match(appFile.content, /Legacy Core 1\.5 V2/);
  assert.match(appFile.content, /G_LegacyCore1\.5v2\.pth/);
  assert.match(appFile.content, /D_LegacyCore1\.5v2\.pth/);
  assert.match(appFile.content, /ETA Accuracy:/);
  assert.match(appFile.content, /data-nr-fin-ms=/);
  assert.doesNotMatch(appFile.content, /window\.parent\.document/);
  assert.doesNotMatch(appFile.content, /Default 1-Click Pretrain/);
  assert.doesNotMatch(appFile.content, /key="nr_t_batch_parity"/);
  assert.match(appFile.content, /key="nr_set_batch_parity_chk"/);
  assert.match(appFile.content, /0 epochs completed/);
  assert.match(appFile.content, /Auto-detected Kaggle dataset folder/);
  assert.match(appFile.content, /Dataset Similarity Fit/);
  assert.match(appFile.content, /Apply WebUI Update & Restart/);
  assert.match(appFile.content, /st\.session_state\["nr_trn_log"\] = msg\n\s+st\.rerun\(\)/);
  assert.match(appFile.content, /def _resolve_pretrain_path/);
  assert.match(appFile.content, /SimpleNamespace/);
  assert.match(appFile.content, /Voice Similarity \(Index Rate\)/);
  assert.match(appFile.content, /Following Step 3/);
  assert.match(appFile.content, /Custom Override/);
  assert.match(appFile.content, /Load in Samples/);
  assert.match(appFile.content, /Undertrained Risk/);
  assert.match(appFile.content, /Overtrained Risk/);

  // noregrets_turbo.py sync exported weight save fix + NR_TURBO_ENABLED check + world_size DDP tuning
  assert.match(turboFile.content, /NR_TURBO_ENABLED/);
  assert.match(turboFile.content, /WORLD_SIZE/);
  assert.match(turboFile.content, /def flush_saves/);
  assert.match(turboFile.content, /fname\.startswith\(\("G_", "D_"\)\)/);

  // Kaggle Notebook Cell 1 & Cell 2 in-place HTML crimson monitor checks
  assert.match(nbFile.content, /credentials\.toml/);
  assert.match(nbFile.content, /raw\.githubusercontent\.com\/HunterThatsMe\/NoRegretsRVC\/main/);
  assert.match(nbFile.content, /display_id=\\"nr_live_monitor\\"/);
  assert.match(nbFile.content, /color:#E51923/);
  assert.match(nbFile.content, /background:#E51923/);
  assert.doesNotMatch(nbFile.content, /🟥|⬛/);
  assert.match(nbFile.content, /\(saved\)/);
  assert.match(nbFile.content, /Training Stopped/);
  assert.match(nbFile.content, /Training Resumed/);
  assert.match(nbFile.content, /Training Complete/);
  assert.match(nbFile.content, /Idle — waiting for next epoch batches\.\.\./);
  assert.match(nbFile.content, /_free_port_6969/);
  assert.match(nbFile.content, /_pick_py/);

  // Final-turn hardening checks
  assert.match(appFile.content, /nr_set_turbo_chk/);
  assert.match(appFile.content, /def _read_tb_metrics_for_model/);
  assert.match(appFile.content, /@st\.cache_data\(ttl=300, show_spinner=False\)\ndef _gpu_info_lines/);
  assert.match(appFile.content, /noregrets_turbo\.flush_saves\(\)/);
  assert.match(appFile.content, /extract_model/);
  assert.match(appFile.content, /NoRegretsRVC_Kaggle\.ipynb/);
  assert.match(nbFile.content, /_gh_updated_any/);

  // Pretrain download NameError regression checks
  assert.doesNotMatch(appFile.content, /\bLEGACY_CORE_PRETRAIN_G\b/);
  assert.doesNotMatch(appFile.content, /\bLEGACY_CORE_PRETRAIN_D\b/);
  assert.match(appFile.content, /json\.dumps\(LEGACY_CORE_G_URL\)/);
  assert.match(appFile.content, /json\.dumps\(LEGACY_CORE_D_URL\)/);
  assert.match(appFile.content, /def _dl_url/);
  assert.match(appFile.content, /def _is_d_file/);
});





