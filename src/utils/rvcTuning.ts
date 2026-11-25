import type {
  BlenderRecipe,
  ConvertAudioState,
  ForkConfig,
  KaggleSurvivalEstimate,
  T4VramEstimate,
  TrainingCompletionEstimate,
  ValidationIssue,
  VoiceModelProject,
} from '../types/studio.ts';

const T4_VRAM_CAP_GB = 16.0;
const KAGGLE_DISK_CAP_GB = 20.0;

const TIME_FORMATTER = new Intl.DateTimeFormat(undefined, {
  hour: 'numeric',
  minute: '2-digit',
});

const DAY_FORMATTER = new Intl.DateTimeFormat(undefined, {
  weekday: 'short',
  month: 'short',
  day: 'numeric',
});

export interface TrainingCompletionInput {
  currentEpoch: number;
  totalEpoch: number;
  batchDone: number;
  batchTotal: number;
  elapsedSec: number;
  itPerSec: number;
  completedEpochDurationsSec?: readonly number[];
  nowMs?: number;
}

export function formatRemainingDuration(totalSeconds: number): string {
  const safeSec = Math.max(0, Math.round(totalSeconds));
  if (safeSec < 60) {
    return safeSec <= 5 ? 'finishing now' : `~${safeSec}s (< 1 minute)`;
  }
  const totalMinutes = Math.max(1, Math.round(safeSec / 60));
  if (totalMinutes < 60) {
    return `~${totalMinutes} minute${totalMinutes === 1 ? '' : 's'}`;
  }
  const hours = Math.floor(totalMinutes / 60);
  const mins = totalMinutes % 60;
  if (hours < 24) {
    return mins > 0
      ? `~${hours} hr${hours === 1 ? '' : 's'} ${mins} min`
      : `~${hours} hr${hours === 1 ? '' : 's'}`;
  }
  const days = Math.floor(hours / 24);
  const remHours = hours % 24;
  return remHours > 0
    ? `~${days}d ${remHours}h ${mins}m`
    : `~${days}d ${mins}m`;
}

export function formatFinishDayLabel(nowMs: number, finishMs: number): string {
  const nowDate = new Date(nowMs);
  const finishDate = new Date(finishMs);
  const startToday = new Date(
    nowDate.getFullYear(),
    nowDate.getMonth(),
    nowDate.getDate()
  ).getTime();
  const startFinish = new Date(
    finishDate.getFullYear(),
    finishDate.getMonth(),
    finishDate.getDate()
  ).getTime();
  const dayDiff = Math.round((startFinish - startToday) / 86_400_000);
  const formatted = DAY_FORMATTER.format(finishDate);
  if (dayDiff <= 0) return `Today · ${formatted}`;
  if (dayDiff === 1) return `Tomorrow · ${formatted}`;
  return formatted;
}

export function estimateTrainingCompletion(
  input: TrainingCompletionInput
): TrainingCompletionEstimate {
  const totalEpoch = Math.max(1, Math.round(input.totalEpoch));
  const currentEpoch = Math.max(1, Math.min(totalEpoch, Math.round(input.currentEpoch)));
  const batchTotal = Math.max(1, Math.round(input.batchTotal));
  const batchDone = Math.max(0, Math.min(batchTotal, input.batchDone));
  const safeItPerSec = Math.max(0.15, input.itPerSec || 0.15);
  const completedEpochs = Math.max(0, currentEpoch - 1);
  const history = input.completedEpochDurationsSec ?? [];

  // Current epoch pace blended gently with actual elapsed batch pace once warmed up
  const instantaneousEpochSec = batchTotal / safeItPerSec;
  let currentEpochPaceSec = instantaneousEpochSec;
  if (completedEpochs === 0 && batchDone >= 4 && input.elapsedSec > 0.5) {
    const observedEpochSec = (input.elapsedSec / batchDone) * batchTotal;
    if (
      observedEpochSec >= instantaneousEpochSec * 0.45 &&
      observedEpochSec <= instantaneousEpochSec * 2.5
    ) {
      currentEpochPaceSec = observedEpochSec * 0.45 + instantaneousEpochSec * 0.55;
    }
  }

  // Exponentially smoothed completed epoch duration when history exists
  let smoothedEpochSec = currentEpochPaceSec;
  let cvPenalty = 0;
  if (history.length > 0) {
    let ema = history[0];
    for (let i = 1; i < history.length; i++) {
      ema = ema * 0.65 + history[i] * 0.35;
    }
    const historyWeight = Math.min(0.88, 0.45 + history.length * 0.12);
    smoothedEpochSec = ema * historyWeight + currentEpochPaceSec * (1 - historyWeight);

    if (history.length >= 2) {
      const mean = history.reduce((sum, v) => sum + v, 0) / history.length;
      const variance =
        history.reduce((sum, v) => sum + (v - mean) ** 2, 0) / history.length;
      const cv = mean > 0 ? Math.sqrt(variance) / mean : 0;
      cvPenalty = Math.min(22, cv * 65);
    }
  } else if (completedEpochs > 0 && input.elapsedSec > 0.5) {
    const progressInEpochs = completedEpochs + batchDone / batchTotal;
    const avgObservedEpochSec = input.elapsedSec / Math.max(0.25, progressInEpochs);
    if (
      avgObservedEpochSec >= instantaneousEpochSec * 0.45 &&
      avgObservedEpochSec <= instantaneousEpochSec * 2.5
    ) {
      const obsWeight = Math.min(0.75, 0.35 + completedEpochs * 0.1);
      smoothedEpochSec =
        avgObservedEpochSec * obsWeight + currentEpochPaceSec * (1 - obsWeight);
    }
  }

  const remainingCurrentEpochSec = Math.max(
    0,
    (batchTotal - batchDone) / safeItPerSec
  );
  const remainingFutureEpochsSec =
    Math.max(0, totalEpoch - currentEpoch) * smoothedEpochSec;
  const remainingSeconds = Math.max(
    0,
    Math.round(remainingCurrentEpochSec + remainingFutureEpochsSec)
  );

  const nowMs = input.nowMs ?? Date.now();
  const finishMs = nowMs + remainingSeconds * 1000;
  const finishDate = new Date(finishMs);

  // Calibrated confidence scale (28% warmup -> climbs cleanly up to ~95% as epochs stabilize)
  const batchWarmup = Math.min(1, batchDone / Math.max(1, batchTotal)) * 14;
  const sampleCount = Math.max(completedEpochs, history.length);
  const sampleConfidence = 30 + batchWarmup * (sampleCount === 0 ? 1 : 0.25) + 49 * (1 - Math.exp(-sampleCount / 3.5));
  const horizonHours = remainingSeconds / 3600;
  const horizonPenalty = Math.min(8, horizonHours * 0.65);
  const accuracyPercent = Math.max(
    28,
    Math.min(95, Math.round(sampleConfidence - cvPenalty - horizonPenalty))
  );

  return {
    remainingSeconds,
    remainingLabel: formatRemainingDuration(remainingSeconds),
    finishTimeLabel: TIME_FORMATTER.format(finishDate),
    finishDayLabel: formatFinishDayLabel(nowMs, finishMs),
    accuracyPercent,
    disclaimer:
      'Rough estimate only — real completion time may vary with GPU throttling, checkpoint disk saves, and batch length.',
  };
}

export interface PerGpuBatchResolution {
  requestedBatchSize: number;
  perGpuBatchSize: number;
  totalEffectiveBatchSize: number;
  gpuCount: number;
  isParitySplit: boolean;
}

export function resolvePerGpuBatchSize(
  requestedBatchSize: number,
  gpuCount: number,
  batchParity1Gpu = true
): PerGpuBatchResolution {
  const safeRequested = Math.max(1, Math.round(requestedBatchSize || 1));
  const safeGpus = Math.max(1, Math.round(gpuCount || 1));
  const isParitySplit = Boolean(batchParity1Gpu) && safeGpus >= 2 && safeRequested >= 4;
  const perGpuBatchSize = isParitySplit
    ? Math.max(2, Math.floor(safeRequested / safeGpus))
    : safeRequested;
  return {
    requestedBatchSize: safeRequested,
    perGpuBatchSize,
    totalEffectiveBatchSize: perGpuBatchSize * safeGpus,
    gpuCount: safeGpus,
    isParitySplit,
  };
}

export function estimateT4VramUsage(project: VoiceModelProject): T4VramEstimate {
  const srFactor =
    project.samplingRate === '48000'
      ? 1.28
      : project.samplingRate === '40000'
        ? 1.12
        : 1.0;
  const vocoderFactor =
    project.vocoder === 'RefineGAN'
      ? 1.45
      : project.vocoder === 'MRF HiFi-GAN'
        ? 1.22
        : 1.0;
  const melFactor = project.melLoss === 'Multi-Scale' ? 1.14 : 1.0;
  const fp16Factor = project.fp16 ? 1.0 : 1.55;
  const checkpointFactor = project.checkpointing ? 0.78 : 1.0;
  const batchAlloc = resolvePerGpuBatchSize(
    project.batchSize,
    project.gpuCount,
    project.batchParity1Gpu ?? true
  );

  const baseModelGb = 2.6 * srFactor * vocoderFactor;
  const perBatchGb = 0.72 * srFactor * vocoderFactor * melFactor * fp16Factor * checkpointFactor;
  const estimatedVramGb = Number(
    (baseModelGb + batchAlloc.perGpuBatchSize * perBatchGb).toFixed(2)
  );

  const safeBudget = 13.2 - baseModelGb;
  const recommendedBatchSize = Math.max(
    1,
    Math.min(64, Math.floor(safeBudget / perBatchGb))
  );

  let fitStatus: T4VramEstimate['fitStatus'] = 'Optimal';
  if (estimatedVramGb > 14.5) {
    fitStatus = 'Risk of OOM';
  } else if (estimatedVramGb > 12.6) {
    fitStatus = 'Tight';
  }

  const warnings: string[] = [];
  if (fitStatus === 'Risk of OOM') {
    warnings.push(
      `Estimated ${estimatedVramGb} GB approaches T4's 16 GB limit. Drop Batch Size to ${recommendedBatchSize} or enable Checkpointing / FP16.`
    );
  } else if (fitStatus === 'Tight') {
    warnings.push(
      `VRAM headroom is tight (${estimatedVramGb} GB / 16 GB). Recommended max batch: ${recommendedBatchSize}.`
    );
  }

  return {
    estimatedVramGb,
    vramCapGb: T4_VRAM_CAP_GB,
    fitStatus,
    recommendedBatchSize,
    warnings,
  };
}

export function estimateKaggleSurvival(
  project: VoiceModelProject
): KaggleSurvivalEstimate {
  const interval = Math.max(1, project.saveEveryEpoch);
  const saveEvents = Math.max(1, Math.floor(project.totalEpoch / interval));
  const checkpointCount = project.saveEveryWeights ? saveEvents : 1;

  const weightSizeGb = project.vocoder === 'RefineGAN' ? 0.08 : 0.055;
  const resumeStateGb = project.saveOnlyLatest ? 0.58 : saveEvents * 0.58;
  const baseEnvGb = 3.2; // venv, pretrained bases, extracted features
  const estimatedDiskGb = Number(
    (baseEnvGb + resumeStateGb + checkpointCount * weightSizeGb).toFixed(2)
  );

  let diskStatus: KaggleSurvivalEstimate['diskStatus'] = 'Safe';
  if (estimatedDiskGb >= 18.5) {
    diskStatus = 'Risk of Full';
  } else if (estimatedDiskGb >= 14.5) {
    diskStatus = 'Watch Disk';
  }

  const gpuMultiplier = project.gpuCount === 2 ? 0.62 : 1.0;
  const vocoderTimeFactor =
    project.vocoder === 'RefineGAN'
      ? 1.35
      : project.vocoder === 'MRF HiFi-GAN'
        ? 1.15
        : 1.0;
  const fpFactor = project.fp16 ? 0.78 : 1.0;
  const secPerEpoch = 46 * gpuMultiplier * vocoderTimeFactor * fpFactor;
  const estimatedHours = Number(
    ((project.totalEpoch * secPerEpoch) / 3600).toFixed(1)
  );

  let sessionStatus: KaggleSurvivalEstimate['sessionStatus'] = 'Fits 12h';
  if (estimatedHours > 11.5) {
    sessionStatus = 'May Exceed 12h';
  } else if (estimatedHours > 9.0) {
    sessionStatus = 'Tight 12h';
  }

  return {
    estimatedDiskGb,
    diskCapGb: KAGGLE_DISK_CAP_GB,
    diskStatus,
    checkpointCount,
    estimatedHours,
    sessionStatus,
  };
}

export function sanitizeModelName(name: string): string {
  const cleaned = name
    .trim()
    .replace(/[^a-zA-Z0-9_-]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return cleaned || 'MyVoice';
}

export function extractPthFilenameFromUrl(
  rawInput: string,
  fallbackName: string
): string {
  const trimmed = rawInput.trim();
  if (!trimmed) return fallbackName;
  try {
    const parsed = new URL(trimmed);
    const segment = parsed.pathname.split('/').filter(Boolean).pop() || fallbackName;
    const decoded = decodeURIComponent(segment);
    return decoded.endsWith('.pth') ? decoded : `${decoded}.pth`;
  } catch {
    const clean = trimmed.split('?')[0].split('/').filter(Boolean).pop() || fallbackName;
    try {
      return decodeURIComponent(clean);
    } catch {
      return clean;
    }
  }
}

export function isHuggingFaceRepoUrl(rawInput: string): boolean {
  const trimmed = rawInput.trim();
  if (!/^https?:\/\/huggingface\.co\//i.test(trimmed)) return false;
  try {
    const parsed = new URL(trimmed);
    const parts = parsed.pathname.split('/').filter(Boolean);
    return (
      parts.length === 2 &&
      !/\.(pth|zip|pt|safetensors|bin)$/i.test(parts[1]) &&
      !parsed.pathname.includes('/resolve/') &&
      !parsed.pathname.includes('/blob/')
    );
  } catch {
    return false;
  }
}

export function isPretrainedZipUrl(rawInput: string): boolean {
  const trimmed = rawInput.trim();
  return /^https?:\/\/.+\.zip(\?.*)?$/i.test(trimmed) || trimmed.toLowerCase().endsWith('.zip');
}

export function inferMatchingDUrl(gInput: string): string {
  const trimmed = gInput.trim();
  if (!trimmed || isHuggingFaceRepoUrl(trimmed) || isPretrainedZipUrl(trimmed)) {
    return '';
  }
  const replaceFilenamePrefix = (fname: string): string => {
    if (/^f0G/i.test(fname)) return fname.replace(/^f0G/i, 'f0D');
    if (/^G([_\d-]|$)/i.test(fname)) return fname.replace(/^G/i, 'D');
    return '';
  };

  if (/^https?:\/\//i.test(trimmed)) {
    try {
      const parsed = new URL(trimmed);
      const segs = parsed.pathname.split('/');
      const rawLast = segs[segs.length - 1] || '';
      const decoded = decodeURIComponent(rawLast);
      const nextDecoded = replaceFilenamePrefix(decoded);
      if (!nextDecoded) return '';
      segs[segs.length - 1] = nextDecoded.includes(' ')
        ? encodeURIComponent(nextDecoded)
        : nextDecoded;
      parsed.pathname = segs.join('/');
      return parsed.toString();
    } catch {
      return '';
    }
  }

  const slashIdx = trimmed.lastIndexOf('/');
  const dir = slashIdx >= 0 ? trimmed.slice(0, slashIdx + 1) : '';
  const base = slashIdx >= 0 ? trimmed.slice(slashIdx + 1) : trimmed;
  const nextBase = replaceFilenamePrefix(base);
  return nextBase ? `${dir}${nextBase}` : '';
}

export function resolveCustomPretrainedPath(
  rawInput: string,
  role: 'G' | 'D'
): string {
  const trimmed = rawInput.trim();
  if (!trimmed) return '';
  if (/^https?:\/\//i.test(trimmed)) {
    const fname = extractPthFilenameFromUrl(trimmed, `custom_${role}.pth`);
    return `rvc/models/pretraineds/custom/${fname}`;
  }
  return trimmed;
}

export function buildTrainPipelineCommands(project: VoiceModelProject): string {
  const safeName = sanitizeModelName(project.name);
  const dsPath = project.datasetPath.trim() || `/kaggle/input/datasets/your-username/${safeName}`;
  const gpus = project.gpuNumber.trim() || (project.gpuCount === 2 ? '0-1' : '0');
  const batchAlloc = resolvePerGpuBatchSize(
    project.batchSize,
    project.gpuCount,
    project.batchParity1Gpu ?? true
  );
  const silenceFlags = project.truncateSilenceEnabled
    ? ` --truncate_silence True --silence_duration_ms ${project.silenceDurationMs}`
    : '';
  const loudnessFlags = project.loudnessNorm
    ? ` --process_effects True --loudness_target ${project.loudnessTargetLufs}`
    : ` --process_effects False`;
  const rawG = project.customPretrainedG.trim();
  const autoD = !project.customPretrainedD.trim() ? inferMatchingDUrl(rawG) : '';
  const rawD = (project.customPretrainedD.trim() || autoD).trim();
  const resolvedG = resolveCustomPretrainedPath(rawG, 'G');
  const resolvedD = resolveCustomPretrainedPath(rawD, 'D');
  const customDownloadLines: string[] = [];
  if (project.customPretrained) {
    if (/^https?:\/\//i.test(rawG) && !isHuggingFaceRepoUrl(rawG)) {
      customDownloadLines.push(
        `mkdir -p rvc/models/pretraineds/custom && curl -fL "${rawG}" -o "${resolvedG}"`
      );
    }
    if (/^https?:\/\//i.test(rawD) && !isHuggingFaceRepoUrl(rawD)) {
      customDownloadLines.push(
        `mkdir -p rvc/models/pretraineds/custom && curl -fL "${rawD}" -o "${resolvedD}"`
      );
    }
  }
  const customPretrainedFlags = project.customPretrained
    ? ` --custom_pretrained True --g_pretrained_path "${resolvedG}" --d_pretrained_path "${resolvedD}"`
    : '';

  return [
    `# 1. Preprocess dataset`,
    `python core.py preprocess \\`,
    `  --model_name "${safeName}" --dataset_path "${dsPath}" \\`,
    `  --sample_rate ${project.samplingRate} --cpu_cores ${project.cpuCores} \\`,
    `  --cut_preprocess "${project.audioCutting}" --chunk_len ${project.chunkLengthSec} --overlap_len ${project.overlapLengthSec} \\`,
    `  --noise_reduction ${project.noiseFilter ? 'True' : 'False'}${loudnessFlags}${silenceFlags}`,
    ``,
    `# 2. Extract features (${project.pitchExtractor} + ${project.embedder})`,
    `python core.py extract \\`,
    `  --model_name "${safeName}" --f0_method ${project.pitchExtractor} \\`,
    `  --sample_rate ${project.samplingRate} --embedder_model ${project.embedder} \\`,
    `  --gpu ${gpus} --cpu_cores ${project.cpuCores} --include_mutes ${project.silentTrainingFiles}`,
    ``,
    ...(customDownloadLines.length > 0
      ? [`# 2b. Download custom G & D pretrained weights`, ...customDownloadLines, ``]
      : []),
    `# 3. Train ${project.vocoder} RVC model (${project.fp16 ? 'fp16' : 'fp32'})`,
    `python core.py train \\`,
    `  --model_name "${safeName}" --optimizer ${project.optimizer} --vocoder "${project.vocoder}" \\`,
    `  --save_every_epoch ${project.saveEveryEpoch} --total_epoch ${project.totalEpoch} --batch_size ${batchAlloc.perGpuBatchSize} \\`,
    `  --sample_rate ${project.samplingRate} --gpu ${gpus} --fp16 ${project.fp16 ? 'True' : 'False'} \\`,
    `  --save_only_latest ${project.saveOnlyLatest ? 'True' : 'False'} --save_every_weights ${project.saveEveryWeights ? 'True' : 'False'} \\`,
    `  --pretrained ${project.pretrained ? 'True' : 'False'} --checkpointing ${project.checkpointing ? 'True' : 'False'} \\`,
    `  --cache_data_in_gpu ${project.cacheDatasetGpu ? 'True' : 'False'}${customPretrainedFlags}`,
    ``,
    `# 4. Generate ${project.indexAlgorithm} feature index`,
    `python core.py index --model_name "${safeName}" --index_algorithm ${project.indexAlgorithm}`,
  ].join('\n');
}

export function buildInferCliCommand(convert: ConvertAudioState): string {
  const pth = convert.voiceModel.trim() || 'logs/Model/Model.pth';
  const idx = convert.indexFile.trim() || 'logs/Model/Model.index';
  const inputTarget =
    convert.sourceMode === 'batch'
      ? convert.batchFolderPath.trim() || '/kaggle/working/audio_inputs'
      : convert.audioFileName.trim() || 'sample_input.wav';
  const cmdName = convert.sourceMode === 'batch' ? 'batch_infer' : 'infer';
  const cleanFlags = convert.noiseReduction
    ? ` --clean_audio True --clean_strength ${convert.noiseReductionStrength}`
    : '';
  const formantFlags = convert.formantShiftEnabled
    ? ` --formant_shifting True --formant_qfrency ${convert.formantQuefrency} --formant_timbre ${convert.formantTimbre}`
    : '';

  return [
    `python core.py ${cmdName} \\`,
    `  --pth_path "${pth}" \\`,
    `  --index_path "${idx}" \\`,
    `  --input_path "${inputTarget}" \\`,
    `  --output_path "${convert.outputPath.trim() || '/kaggle/working/program_ml/assets/audios/output.wav'}" \\`,
    `  --sid ${convert.speakerId} --f0_method ${convert.pitchExtractor} --embedder_model ${convert.embedder} \\`,
    `  --pitch ${convert.pitch} --index_rate ${convert.indexRate} --protect ${convert.protect} --volume_envelope ${convert.volumeEnvelope} \\`,
    `  --f0_autotune ${convert.autotune ? 'True' : 'False'} --proposed_pitch ${convert.proposedPitch ? 'True' : 'False'} --split_audio ${convert.splitAudio ? 'True' : 'False'}${cleanFlags}${formantFlags} \\`,
    `  --export_format ${convert.outputFormat.toUpperCase()}`,
  ].join('\n');
}

export function buildBlendCliCommand(recipe: BlenderRecipe): string {
  const outName = sanitizeModelName(recipe.outputName);
  const ratio = Number(recipe.ratio.toFixed(2));
  return [
    `python core.py model_blend \\`,
    `  --model_name "${outName}" \\`,
    `  --pth_path_1 "${recipe.modelAPath.trim() || 'logs/ModelA.pth'}" \\`,
    `  --pth_path_2 "${recipe.modelBPath.trim() || 'logs/ModelB.pth'}" \\`,
    `  --ratio ${ratio}`,
  ].join('\n');
}

export function validateForkConfig(config: ForkConfig): ValidationIssue[] {
  const issues: ValidationIssue[] = [];

  const enabledCount = Object.values(config.activeTabs).filter(Boolean).length;
  if (enabledCount === 0) {
    issues.push({
      field: 'activeTabs',
      severity: 'error',
      message: 'Enable at least one WebUI tab so app.py can render.',
    });
  }

  if (config.tunnelMode === 'Horizon' && !config.horizonId.trim()) {
    issues.push({
      field: 'horizonId',
      severity: 'warning',
      message: 'Horizon tunnel requires your Horizon ID from hrzn.run/dashboard.',
    });
  }

  if (config.customRepoUrl.trim() && !/^https:\/\/github\.com\/[^/]+\/[^/]+/.test(config.customRepoUrl.trim())) {
    issues.push({
      field: 'customRepoUrl',
      severity: 'warning',
      message: 'Custom fork URL should be a valid https://github.com/user/repo link.',
    });
  }

  return issues;
}
