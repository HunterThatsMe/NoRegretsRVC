import {
  LiveEpochProgress,
  MetricBreakdownItem,
  ModelSoundFeedback,
  RankedCheckpointItem,
  TrainingAnalysis,
  TrainingEpochPoint,
  TrainingZoneInsight,
  VoiceModelProject,
} from '../types/studio';

function extractTextFromIpynb(raw: string): string | null {
  try {
    const parsed = JSON.parse(raw);
    if (!parsed || !Array.isArray(parsed.cells)) {
      return null;
    }
    const chunks: string[] = [];
    for (const cell of parsed.cells) {
      if (Array.isArray(cell.outputs)) {
        for (const out of cell.outputs) {
          if (typeof out.text === 'string') {
            chunks.push(out.text);
          } else if (Array.isArray(out.text)) {
            chunks.push(out.text.join(''));
          }
        }
      }
      if (Array.isArray(cell.source)) {
        chunks.push(cell.source.join(''));
      } else if (typeof cell.source === 'string') {
        chunks.push(cell.source);
      }
    }
    return chunks.join('\n');
  } catch {
    return null;
  }
}

function parseDurationToSeconds(hms: string): number {
  const parts = hms.split(':').map((n) => Number(n));
  if (parts.length === 3 && parts.every((n) => !Number.isNaN(n))) {
    return parts[0] * 3600 + parts[1] * 60 + parts[2];
  }
  return 0;
}

function matchFloat(line: string, patterns: RegExp[]): number | null {
  for (const re of patterns) {
    const m = line.match(re);
    if (m && m[1] !== undefined) {
      const n = Number(m[1]);
      if (Number.isFinite(n)) return n;
    }
  }
  return null;
}

export function parseTrainingNotebookOrLog(rawInput: string): TrainingAnalysis | null {
  const trimmed = rawInput.trim();
  if (!trimmed) return null;

  const ipynbText = trimmed.startsWith('{') ? extractTextFromIpynb(trimmed) : null;
  const text = ipynbText ?? trimmed;

  const durationMatch = text.match(/on\s+(\d{2}:\d{2}:\d{2})\s+seconds of audio/i);
  const segmentsMatch = text.match(/\((\d+)\/(\d+)\s+files successful\)/i);
  const vocoderMatch = text.match(/Using\s+([A-Za-z0-9-]+)\s+vocoder/i);
  const optimizerMatch = text.match(/Using\s+([A-Za-z0-9-]+)\s+optimizer/i);
  const gpuMatch = text.match(/on\s+(cuda:\d+(?:,\s*cuda:\d+)*)/i);
  const modelPreprocessMatch = text.match(/Model\s+([A-Za-z0-9_.-]+)\s+preprocessed successfully/i);

  const epochLineRegex =
    /(?:([A-Za-z0-9_.-]+)\s*\|\s*)?epoch=(\d+)\s*\|\s*step=(\d+)(?:\s*\|\s*time=\d{1,2}:\d{2}:\d{2})?\s*\|\s*training_speed=(\d{1,2}:\d{2}:\d{2})(?:\s*\|\s*lowest_value=([0-9.]+)(?:\s*\(epoch\s+(\d+)(?:\s+and\s+step\s+(\d+))?\))?)?/g;

  const lines = text.split(/\r?\n/);
  const multiScaleMelDetected = /multi[-_\s]?scale\s*mel|512\s*\/\s*1024\s*\/\s*2048|\bmel_ms\b/i.test(text);
  const globalKlVals: number[] = [];
  const globalFmVals: number[] = [];
  const globalDiscVals: number[] = [];
  const globalGenVals: number[] = [];
  const globalGradVals: number[] = [];
  const globalEmaVals: number[] = [];

  for (const line of lines) {
    const kl = matchFloat(line, [
      /(?:loss\/g\/kl|kl_loss|loss_kl|\bkl)\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)/i,
    ]);
    const fm = matchFloat(line, [
      /(?:loss\/g\/fm|fm_loss|loss_fm|\bfm)\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)/i,
    ]);
    const disc = matchFloat(line, [
      /(?:loss\/d\/total|loss_disc|disc_loss|d_loss|d_total)\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)/i,
    ]);
    const gen = matchFloat(line, [
      /(?:loss\/g\/total|loss_gen|gen_loss|g_loss|g_total)\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)/i,
    ]);
    const grad = matchFloat(line, [
      /(?:grad_norm(?:_[dg])?|\bgrad)\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)/i,
    ]);
    const emaVal = matchFloat(line, [
      /(?:ema_decay|gen_ema|\bema)\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)/i,
    ]);
    if (kl !== null) globalKlVals.push(kl);
    if (fm !== null) globalFmVals.push(fm);
    if (disc !== null) globalDiscVals.push(disc);
    if (gen !== null) globalGenVals.push(gen);
    if (grad !== null) globalGradVals.push(grad);
    if (emaVal !== null) globalEmaVals.push(emaVal);
  }

  const points: TrainingEpochPoint[] = [];
  let speakerFromEpochs = '';
  let lowestLoss: number | null = null;
  let lowestLossEpoch: number | null = null;
  let lowestLossStep: number | null = null;

  let match: RegExpExecArray | null;
  while ((match = epochLineRegex.exec(text)) !== null) {
    if (match[1]) speakerFromEpochs = match[1];
    const epoch = Number(match[2]);
    const step = Number(match[3]);
    const speedSec = parseDurationToSeconds(match[4]);
    const lowVal = match[5] ? Number(match[5]) : null;
    const lineStart = text.lastIndexOf('\n', match.index) + 1;
    const nextNl = text.indexOf('\n', match.index);
    const fullLine = text.slice(lineStart, nextNl === -1 ? text.length : nextNl);

    const melLoss = matchFloat(fullLine, [
      /(?:loss\/g\/mel_ms|loss\/g\/mel|mel_ms_loss|mel_loss|loss_mel|\bmel_ms|\bmel)\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)/i,
    ]);
    const klLoss = matchFloat(fullLine, [
      /(?:loss\/g\/kl|kl_loss|loss_kl|\bkl)\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)/i,
    ]);
    const fmLoss = matchFloat(fullLine, [
      /(?:loss\/g\/fm|fm_loss|loss_fm|\bfm)\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)/i,
    ]);
    const discLoss = matchFloat(fullLine, [
      /(?:loss\/d\/total|loss_disc|disc_loss|d_loss|d_total)\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)/i,
    ]);
    const genLoss = matchFloat(fullLine, [
      /(?:loss\/g\/total|loss_gen|gen_loss|g_loss|g_total)\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)/i,
    ]);
    const gradNorm = matchFloat(fullLine, [
      /(?:grad_norm(?:_[dg])?|\bgrad)\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)/i,
    ]);

    // Prefer explicit current-epoch mel= over cumulative lowest_value= when present so tail rebounds are visible!
    const currentLoss = melLoss ?? lowVal;
    const refLow = lowVal ?? melLoss;
    if (refLow !== null && (lowestLoss === null || refLow < lowestLoss)) {
      lowestLoss = refLow;
      lowestLossEpoch = match[6] ? Number(match[6]) : epoch;
      lowestLossStep = match[7] ? Number(match[7]) : step;
    }
    points.push({
      epoch,
      step,
      speedSec,
      lowestLoss: currentLoss,
      melLoss: currentLoss,
      klLoss,
      fmLoss,
      discLoss,
      genLoss,
      gradNorm,
    });
  }

  const liveRegex =
    /epoch=(\d+)\s*\|\s*step=(\d+)\s*\|\s*(\d+)%\|[^|]*\|\s*batch=(\d+)\/(\d+)\s*\[([^<\]]+)<([^,\]]+),\s*([0-9.]+)it\/s\]/g;
  let liveProgress: LiveEpochProgress | null = null;
  let liveMatch: RegExpExecArray | null;
  while ((liveMatch = liveRegex.exec(text)) !== null) {
    liveProgress = {
      epoch: Number(liveMatch[1]),
      step: Number(liveMatch[2]),
      percent: Number(liveMatch[3]),
      batchDone: Number(liveMatch[4]),
      batchTotal: Number(liveMatch[5]),
      elapsedLabel: liveMatch[6].trim(),
      etaLabel: liveMatch[7].trim(),
      itPerSec: Number(liveMatch[8]),
    };
  }

  if (points.length === 0 && !durationMatch && !segmentsMatch && !liveProgress) {
    return null;
  }

  const avgSpeedSec =
    points.length > 0
      ? Math.round(points.reduce((acc, p) => acc + p.speedSec, 0) / points.length)
      : 0;

  const avg = (arr: number[]) =>
    arr.length > 0 ? Number((arr.reduce((s, v) => s + v, 0) / arr.length).toFixed(3)) : null;

  const gpuDevices = gpuMatch
    ? gpuMatch[1].split(',').map((g) => g.trim())
    : ['cuda:0'];

  return {
    speakerName:
      speakerFromEpochs || (modelPreprocessMatch ? modelPreprocessMatch[1] : 'ImportedRun'),
    audioDurationLabel: durationMatch ? durationMatch[1] : 'Unknown',
    segmentCount: segmentsMatch ? Number(segmentsMatch[1]) : 0,
    epochsCompleted: points.length > 0 ? points[points.length - 1].epoch : (liveProgress?.epoch ?? 0),
    latestStep: points.length > 0 ? points[points.length - 1].step : (liveProgress?.step ?? 0),
    avgSpeedSec,
    lowestLoss,
    lowestLossEpoch,
    lowestLossStep,
    avgKlLoss: avg(globalKlVals),
    avgFmLoss: avg(globalFmVals),
    avgDiscLoss: avg(globalDiscVals),
    avgGenLoss: avg(globalGenVals),
    latestGradNorm:
      globalGradVals.length > 0 ? Number(globalGradVals[globalGradVals.length - 1].toFixed(3)) : null,
    emaDecay:
      globalEmaVals.length > 0 ? Number(globalEmaVals[globalEmaVals.length - 1].toFixed(4)) : null,
    multiScaleMel: multiScaleMelDetected,
    gpuDevices,
    vocoderDetected: vocoderMatch ? vocoderMatch[1] : 'HiFi-GAN',
    optimizerDetected: optimizerMatch ? optimizerMatch[1] : 'AdamW',
    points,
    liveProgress,
  };
}

function snapToSaveInterval(
  epoch: number,
  saveEvery: number,
  minSaved: number,
  maxSaved: number
): number {
  const safeStep = Math.max(1, saveEvery);
  const rounded = Math.round(epoch / safeStep) * safeStep;
  return Math.max(minSaved, Math.min(maxSaved, rounded));
}

function uniqueSortedEpochs(epochs: number[]): number[] {
  return Array.from(new Set(epochs)).sort((a, b) => a - b);
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n));
}

function nearestPointForEpoch(
  points: (TrainingEpochPoint & { lowestLoss: number })[],
  targetEpoch: number
): TrainingEpochPoint & { lowestLoss: number } {
  let best = points[0];
  let bestDist = Math.abs(points[0].epoch - targetEpoch);
  for (let i = 1; i < points.length; i++) {
    const dist = Math.abs(points[i].epoch - targetEpoch);
    if (dist < bestDist) {
      best = points[i];
      bestDist = dist;
    }
  }
  return best;
}

export function analyzeTrainingZones(
  analysis: TrainingAnalysis | null,
  project?: VoiceModelProject,
  soundCheck: ModelSoundFeedback = 'untested'
): TrainingZoneInsight {
  const saveEvery = Math.max(1, Math.round(project?.saveEveryEpoch ?? 5));
  const totalEpoch = Math.max(saveEvery, Math.round(project?.totalEpoch ?? 200));
  const maxSaved = Math.max(saveEvery, Math.floor(totalEpoch / saveEvery) * saveEvery);
  const modelName = project?.name.trim() || 'Active Model';
  const modelSlug = (modelName.replace(/[^a-zA-Z0-9_-]+/g, '-') || 'Model');
  const vocoder = analysis?.vocoderDetected || project?.vocoder || 'HiFi-GAN';
  const sr = project?.samplingRate || '32000';
  const modelContext = `${modelName} · ${vocoder} · ${sr} Hz · Save every ${saveEvery} ep`;

  const lossPoints = analysis
    ? analysis.points.filter(
        (p): p is TrainingEpochPoint & { lowestLoss: number } => p.lowestLoss !== null
      )
    : [];
  const allPoints = analysis?.points ?? [];

  if (!analysis || (lossPoints.length === 0 && allPoints.length === 0)) {
    return {
      confidence: 'untrained',
      saveEveryEpoch: saveEvery,
      modelContext,
      datasetSummary: analysis?.segmentCount
        ? `${analysis.segmentCount} dataset segments (${analysis.audioDurationLabel})`
        : 'Waiting for training epochs',
      bestLoss: null,
      candidateEpochs: [],
      datasetMatchPercent: null,
      stabilityScore: null,
      convergenceLabel: 'Awaiting Data',
      trainingVerdict: 'untrained',
      verdictTitle: 'Waiting for Training Data',
      verdictGuidance:
        'Start training on Kaggle or import a console log / .ipynb file below to check for undertraining, overtraining, and exact checkpoint quality.',
      undertrainingRiskPercent: 0,
      overtrainingRiskPercent: 0,
      effectiveStepsCompleted: 0,
      stepsPerClip: null,
      rankedCheckpoints: [],
      metricBreakdown: [],
      summary: `Waiting for real training data for ${modelName}. Once epochs are logged (or a .log/.ipynb is imported), Diagnostics will analyze mel loss, KL/FM acoustic convergence, G/D balance, gradient stability, and dataset fit to pinpoint the Best Epoch.`,
      undertrainingLabel: 'Awaiting training data',
      undertrainingEndEpoch: null,
      sweetSpotLabel: 'Awaiting training data',
      sweetSpotEpoch: null,
      sweetSpotMinEpoch: null,
      sweetSpotMaxEpoch: null,
      overtrainingLabel: 'Awaiting training data',
      overtrainingStartEpoch: null,
    };
  }

  const segCount = analysis.segmentCount || 0;
  const effBatch = Math.max(1, (project?.batchSize ?? 2) * (project?.gpuCount ?? 1));
  const datasetTier =
    segCount > 0
      ? segCount < 140
        ? `Small dataset (${segCount} clips · converges quickly)`
        : segCount >= 450
          ? `Large dataset (${segCount} clips · tolerates deeper epochs)`
          : `Balanced dataset (${segCount} clips)`
      : `Batch ${effBatch} eff. · ${analysis.audioDurationLabel !== 'Unknown' ? analysis.audioDurationLabel : `${vocoder} @ ${sr} Hz`}`;

  const soundNote =
    soundCheck === 'undertrained'
      ? 'You marked this checkpoint undertrained (robotic/flat) — weighting toward higher saved epochs or continuing training if loss is still dropping.'
      : soundCheck === 'overtrained'
        ? 'You marked this checkpoint overtrained (metallic/harsh) — pulling the Best Epoch back earlier before memorization sets in.'
        : soundCheck === 'clean'
          ? 'You confirmed it sounds clean & natural — locking near your lowest stable multi-metric checkpoint.'
          : '';

  const maxObservedEpoch = Math.max(
    saveEvery,
    allPoints.length > 0 ? allPoints[allPoints.length - 1].epoch : analysis.epochsCompleted
  );
  const maxObservedSaved = Math.max(saveEvery, Math.floor(maxObservedEpoch / saveEvery) * saveEvery);
  const latestStepObserved = allPoints.length > 0 ? allPoints[allPoints.length - 1].step || 0 : analysis.latestStep || 0;
  const stepsPerClip = segCount > 0 ? Number((latestStepObserved / Math.max(1, segCount)).toFixed(1)) : null;

  // Sparse log (1 loss point): report real observed progress only
  if (lossPoints.length < 2) {
    const curBestEp = lossPoints[0]?.epoch ?? maxObservedEpoch;
    const bestSnap = snapToSaveInterval(curBestEp, saveEvery, saveEvery, Math.max(saveEvery, maxObservedSaved));
    const bestVal = lossPoints[0]?.lowestLoss ?? analysis.lowestLoss ?? null;
    const warmupMatch = clamp(
      Math.round(32 + (bestVal !== null ? Math.max(0, 26 - bestVal) * 1.4 : 0) + (segCount > 0 ? 6 : 0)),
      18,
      62
    );
    const saturationPct = stepsPerClip !== null ? clamp(Math.round((stepsPerClip / 18) * 100), 5, 100) : 15;
    const sparseRank: RankedCheckpointItem[] = [
      {
        epoch: bestSnap,
        step: lossPoints[0]?.step ?? latestStepObserved,
        checkpointName: `${modelSlug}_${bestSnap}e.pth`,
        melLoss: bestVal,
        driftPercent: 0,
        score: warmupMatch,
        tag: 'Warmup Baseline',
        status: 'watch',
      },
    ];
    return {
      confidence: 'observed',
      saveEveryEpoch: saveEvery,
      modelContext,
      datasetSummary: datasetTier,
      bestLoss: bestVal,
      candidateEpochs: [bestSnap],
      datasetMatchPercent: warmupMatch,
      stabilityScore: 55,
      convergenceLabel: 'Warming Up',
      trainingVerdict: 'undertrained',
      verdictTitle: 'Warming Up — Keep Training',
      verdictGuidance:
        'Only 1 epoch point is logged so far. Keep training until mel & KL losses settle and dataset saturation climbs.',
      undertrainingRiskPercent: 82,
      overtrainingRiskPercent: 5,
      effectiveStepsCompleted: latestStepObserved,
      stepsPerClip,
      rankedCheckpoints: sparseRank,
      metricBreakdown: [
        {
          label: 'Mel Reconstruction',
          value: bestVal !== null ? bestVal.toFixed(3) : 'Warming',
          status: 'watch',
          detail: 'Initial spectral baseline recorded',
        },
        {
          label: 'Dataset Alignment',
          value: `${warmupMatch}% fit`,
          status: 'watch',
          detail: datasetTier,
        },
        {
          label: 'Dataset Saturation',
          value: stepsPerClip !== null ? `${stepsPerClip} steps/clip` : `${saturationPct}% warmed`,
          status: 'watch',
          detail: `${latestStepObserved} optimizer steps logged`,
        },
        {
          label: 'Tail Drift Watch',
          value: '0.0% baseline',
          status: 'optimal',
          detail: 'No rebound at warmup',
        },
      ],
      summary: `Early training data recorded (${allPoints.length} epoch${allPoints.length === 1 ? '' : 's'}, best observed loss ${bestVal !== null ? bestVal.toFixed(3) : 'N/A'} at epoch ${curBestEp}). Let more saved epochs log so mel, KL, and stability convergence can be measured.${soundNote ? ` ${soundNote}` : ''}`,
      undertrainingLabel: `Epoch 1–${maxObservedEpoch} (warming up)`,
      undertrainingEndEpoch: bestSnap,
      sweetSpotLabel: `Best so far: Epoch ${bestSnap}`,
      sweetSpotEpoch: bestSnap,
      sweetSpotMinEpoch: bestSnap,
      sweetSpotMaxEpoch: bestSnap,
      overtrainingLabel: 'Not yet reached',
      overtrainingStartEpoch: null,
    };
  }

  const first = lossPoints[0];
  let minIdx = 0;
  for (let i = 1; i < lossPoints.length; i++) {
    if (lossPoints[i].lowestLoss < lossPoints[minIdx].lowestLoss) {
      minIdx = i;
    }
  }

  const minPoint = lossPoints[minIdx];
  const lastPoint = lossPoints[lossPoints.length - 1];
  const totalDrop = Math.max(0.01, first.lowestLoss - minPoint.lowestLoss);
  const relDropRatio = totalDrop / Math.max(1, first.lowestLoss);

  // Tail stability & jitter across last 4 points
  const tailWindow = lossPoints.slice(-Math.min(4, lossPoints.length));
  const tailSlope =
    tailWindow.length >= 2
      ? (tailWindow[0].lowestLoss - tailWindow[tailWindow.length - 1].lowestLoss) /
        Math.max(1, tailWindow[tailWindow.length - 1].epoch - tailWindow[0].epoch)
      : 0;
  const reboundFromMin = Math.max(0, lastPoint.lowestLoss - minPoint.lowestLoss);
  const reboundPct = Number(
    ((reboundFromMin / Math.max(0.1, minPoint.lowestLoss)) * 100).toFixed(1)
  );
  let upwardJitterSum = 0;
  for (let i = 1; i < tailWindow.length; i++) {
    const upwardStep = tailWindow[i].lowestLoss - tailWindow[i - 1].lowestLoss;
    if (upwardStep > 0) {
      upwardJitterSum += upwardStep;
    }
  }
  const avgUpwardJitter = tailWindow.length > 1 ? upwardJitterSum / (tailWindow.length - 1) : 0;

  // Secondary TensorBoard / stdout losses if present
  const klPoints = allPoints.map((p) => p.klLoss).filter((v): v is number => v !== null && v !== undefined);
  const fmPoints = allPoints.map((p) => p.fmLoss).filter((v): v is number => v !== null && v !== undefined);
  const discPoints = allPoints.map((p) => p.discLoss).filter((v): v is number => v !== null && v !== undefined);
  const genPoints = allPoints.map((p) => p.genLoss).filter((v): v is number => v !== null && v !== undefined);
  const gradPoints = allPoints.map((p) => p.gradNorm).filter((v): v is number => v !== null && v !== undefined);

  const avgKl = klPoints.length > 0 ? klPoints[klPoints.length - 1] : analysis.avgKlLoss ?? null;
  const avgFm = fmPoints.length > 0 ? fmPoints[fmPoints.length - 1] : analysis.avgFmLoss ?? null;
  const avgDisc = discPoints.length > 0 ? discPoints[discPoints.length - 1] : analysis.avgDiscLoss ?? null;
  const avgGen = genPoints.length > 0 ? genPoints[genPoints.length - 1] : analysis.avgGenLoss ?? null;
  const gradNorm = gradPoints.length > 0 ? gradPoints[gradPoints.length - 1] : analysis.latestGradNorm ?? null;

  // Stability score (0–99) — calibrated for clipped grad norms (<= 65 optimal) & Generator EMA smoothing
  const emaActive = Boolean(analysis.emaDecay) || project?.melLoss === 'Multi-Scale' || Boolean(analysis.multiScaleMel);
  const reboundPenalty = reboundFromMin * (emaActive ? 34 : 38);
  const jitterPenalty = avgUpwardJitter * (emaActive ? 15 : 18);
  const gradPenalty = gradNorm !== null && gradNorm > 85 ? Math.min(22, (gradNorm - 85) * 0.28) : 0;
  const discPenalty = avgDisc !== null ? Math.min(14, Math.abs(avgDisc - 1.9) * 5.5) : 0;
  const emaStabilityBonus = analysis.emaDecay ? 2 : 0;
  const stabilityScore = clamp(
    Math.round(97 + emaStabilityBonus - reboundPenalty - jitterPenalty - gradPenalty - discPenalty),
    35,
    99
  );

  // Dataset alignment & perceptual match score (0–99%)
  const melScore = clamp(
    Math.round(62 + relDropRatio * 55 + Math.max(0, 17 - minPoint.lowestLoss) * 2.2),
    35,
    98
  );
  const klScore =
    avgKl !== null
      ? clamp(Math.round(100 - avgKl * 18), 40, 99)
      : clamp(Math.round(58 + relDropRatio * 68), 42, 92);
  const fmScore =
    avgFm !== null
      ? clamp(Math.round(100 - Math.max(0, avgFm - 1.5) * 8), 42, 99)
      : clamp(Math.round(60 + relDropRatio * 62), 45, 93);
  const gdRatio = avgDisc !== null && avgGen !== null && avgDisc > 0 ? avgGen / avgDisc : null;
  const gdScore =
    gdRatio !== null
      ? clamp(Math.round(98 - Math.abs(gdRatio - 1.1) * 18), 40, 98)
      : clamp(stabilityScore - 4, 45, 94);

  const stepsObserved = lastPoint.step || latestStepObserved;
  const targetStepsPerClip = segCount > 0 ? stepsObserved / Math.max(1, segCount) : maxObservedEpoch * 0.8;
  const coverageBonus = clamp(targetStepsPerClip * 1.1, 0, 12);
  const soundAdjust =
    soundCheck === 'clean' ? 6 : soundCheck === 'overtrained' ? -10 : soundCheck === 'undertrained' ? -8 : 0;
  const rawMatch =
    melScore * 0.36 +
    klScore * 0.2 +
    fmScore * 0.18 +
    gdScore * 0.14 +
    stabilityScore * 0.12 +
    coverageBonus -
    (soundAdjust < 0 ? 0 : -soundAdjust);
  const datasetMatchPercent = clamp(Math.round(rawMatch + (soundAdjust < 0 ? soundAdjust : 0)), 25, 99);

  // Dataset saturation & dual risk meters (0-100%)
  const saturationPct =
    stepsPerClip !== null
      ? clamp(Math.round((stepsPerClip / 18) * 100), 10, 100)
      : clamp(Math.round((maxObservedEpoch / 160) * 100), 12, 100);

  const lossGapRisk = clamp((minPoint.lowestLoss - 13.5) * 7.5, 0, 48);
  const stepGapRisk =
    stepsPerClip !== null
      ? clamp((10 - stepsPerClip) * 5.5, 0, 35)
      : clamp((70 - maxObservedEpoch) * 0.55, 0, 35);
  const activeDropRisk = tailSlope > 0.03 ? clamp(tailSlope * 180, 0, 22) : 0;
  const soundUnderBoost =
    soundCheck === 'undertrained' ? 28 : soundCheck === 'clean' ? -18 : soundCheck === 'overtrained' ? -25 : 0;
  const undertrainingRiskPercent = clamp(
    Math.round(lossGapRisk + stepGapRisk + activeDropRisk + soundUnderBoost),
    4,
    96
  );

  const reboundRisk = clamp(reboundFromMin * 44, 0, 48);
  const pastBestEpochsRisk = clamp((maxObservedEpoch - minPoint.epoch) * 0.45, 0, 26);
  const discCollapseRisk =
    avgDisc !== null && avgDisc < 1.65 ? clamp((1.65 - avgDisc) * 55, 0, 24) : 0;
  const gradSpikeRisk =
    gradNorm !== null && gradNorm > 85 ? clamp((gradNorm - 85) * 0.52, 0, 22) : 0;
  const soundOverBoost =
    soundCheck === 'overtrained' ? 32 : soundCheck === 'clean' ? -15 : soundCheck === 'undertrained' ? -20 : 0;
  const overtrainingRiskPercent = clamp(
    Math.round(reboundRisk + pastBestEpochsRisk + discCollapseRisk + gradSpikeRisk + soundOverBoost),
    3,
    97
  );

  // When ~72% of observed mel-loss reduction finishes, voice timbre enters usable clarity
  const underThreshold = first.lowestLoss - totalDrop * 0.72;
  let underIdx = 0;
  for (let i = 0; i < lossPoints.length; i++) {
    if (lossPoints[i].lowestLoss <= underThreshold) {
      underIdx = i;
      break;
    }
  }
  const underEndEpoch = snapToSaveInterval(lossPoints[underIdx].epoch, saveEvery, saveEvery, maxSaved);

  const hasPlateauOrRebound = minIdx < lossPoints.length - 1 || tailSlope < 0.025;

  // Earliest epoch reaching within 4.5% of total mel-loss gain
  const sweetCap = minPoint.lowestLoss + totalDrop * 0.045;
  let balancedPoint = minPoint;
  for (let i = underIdx; i <= minIdx; i++) {
    if (lossPoints[i].lowestLoss <= sweetCap) {
      balancedPoint = lossPoints[i];
      break;
    }
  }

  let datasetOffsetEpochs = 0;
  if (segCount > 0 && segCount < 140) datasetOffsetEpochs -= saveEvery;
  else if (segCount >= 450 && !hasPlateauOrRebound) datasetOffsetEpochs += saveEvery;
  if (sr === '48000' && soundCheck !== 'overtrained') datasetOffsetEpochs += saveEvery;

  const upperObservedBound = Math.max(saveEvery, maxObservedSaved);

  let bestEpoch: number;
  let sweetMin: number;
  let sweetMax: number;
  let overStartEpoch: number | null;
  let overLabel: string;
  let sweetLabel: string;
  let convergenceLabel: string;
  let candidates: number[];

  if (hasPlateauOrRebound) {
    bestEpoch = snapToSaveInterval(
      balancedPoint.epoch + datasetOffsetEpochs,
      saveEvery,
      saveEvery,
      upperObservedBound
    );
    if (soundCheck === 'overtrained') {
      bestEpoch = snapToSaveInterval(bestEpoch - saveEvery * 2, saveEvery, saveEvery, upperObservedBound);
    } else if (soundCheck === 'undertrained') {
      bestEpoch = snapToSaveInterval(minPoint.epoch + saveEvery, saveEvery, saveEvery, upperObservedBound);
    } else if (soundCheck === 'clean') {
      bestEpoch = snapToSaveInterval(
        Math.round((balancedPoint.epoch + minPoint.epoch) / 2),
        saveEvery,
        saveEvery,
        upperObservedBound
      );
    }
    sweetMin = snapToSaveInterval(bestEpoch - saveEvery, saveEvery, saveEvery, bestEpoch);
    sweetMax = snapToSaveInterval(bestEpoch + saveEvery, saveEvery, bestEpoch, upperObservedBound);
    overStartEpoch = snapToSaveInterval(
      Math.max(sweetMax + saveEvery, minPoint.epoch + saveEvery),
      saveEvery,
      Math.min(maxSaved, sweetMax + saveEvery),
      Math.max(maxSaved, sweetMax + saveEvery)
    );
    overLabel =
      minIdx < lossPoints.length - 1 || soundCheck === 'overtrained'
        ? `Past Epoch ${overStartEpoch} (plateau/rebound)`
        : `Risk past Epoch ${overStartEpoch}`;
    sweetLabel =
      sweetMin < sweetMax ? `Best: Epoch ${bestEpoch} (test e${sweetMin}–e${sweetMax})` : `Best: Epoch ${bestEpoch}`;
    convergenceLabel =
      reboundFromMin > 0.25 || soundCheck === 'overtrained'
        ? 'Plateau / Mild Overfit'
        : 'Sweet-Spot Converged';
    candidates = uniqueSortedEpochs(
      [sweetMin, bestEpoch, sweetMax].filter((e) => e >= saveEvery && e <= upperObservedBound)
    );
  } else {
    bestEpoch = snapToSaveInterval(minPoint.epoch, saveEvery, saveEvery, upperObservedBound);
    if (soundCheck === 'overtrained') {
      bestEpoch = snapToSaveInterval(bestEpoch - saveEvery * 2, saveEvery, saveEvery, upperObservedBound);
    }
    sweetMin = snapToSaveInterval(bestEpoch - saveEvery, saveEvery, saveEvery, bestEpoch);
    sweetMax = snapToSaveInterval(bestEpoch, saveEvery, sweetMin, upperObservedBound);
    overStartEpoch = null;
    overLabel = 'None observed yet (still improving)';
    sweetLabel = `Best so far: Epoch ${bestEpoch}`;
    convergenceLabel = 'Converging';
    candidates = uniqueSortedEpochs(
      [sweetMin, bestEpoch].filter((e) => e >= saveEvery && e <= upperObservedBound)
    );
  }

  let trainingVerdict: TrainingZoneInsight['trainingVerdict'] = 'sweet_spot';
  let verdictTitle = 'Optimal Sweet Spot Reached';
  let verdictGuidance = `Audition ${modelSlug}_${bestEpoch}e.pth first — it captures peak natural timbre (${datasetMatchPercent}% dataset match) before overtraining creep begins.`;

  if (overtrainingRiskPercent >= 50 || soundCheck === 'overtrained' || reboundFromMin >= 0.45) {
    trainingVerdict = 'overtrained';
    verdictTitle = 'Overtraining Detected — Roll Back to Earlier Checkpoint';
    verdictGuidance = `Later epochs show +${reboundPct}% mel drift${avgDisc !== null && avgDisc < 1.65 ? ` and discriminator tightening (D ${avgDisc.toFixed(2)})` : ''}. Export Epoch ${bestEpoch} instead of the final epoch to keep vocals natural and artifact-free.`;
  } else if (!hasPlateauOrRebound && (undertrainingRiskPercent >= 48 || maxObservedEpoch < 50 || soundCheck === 'undertrained')) {
    trainingVerdict = 'undertrained';
    verdictTitle = 'Still Undertrained — Keep Training';
    verdictGuidance = `Acoustic losses are still falling cleanly (${maxObservedEpoch} ep${stepsPerClip !== null ? `, ${stepsPerClip} steps/clip` : ''}). Keep training past Epoch ${bestEpoch} until mel loss flattens.`;
  }

  // Build Ranked Checkpoints leaderboard around saved candidates + min/latest
  const rankEpochPool = uniqueSortedEpochs(
    [
      ...candidates,
      snapToSaveInterval(minPoint.epoch, saveEvery, saveEvery, upperObservedBound),
      snapToSaveInterval(underEndEpoch, saveEvery, saveEvery, upperObservedBound),
      upperObservedBound,
    ].filter((e) => e >= saveEvery && e <= upperObservedBound)
  );

  const rankedCheckpoints: RankedCheckpointItem[] = rankEpochPool
    .map((ep) => {
      const nearest = nearestPointForEpoch(lossPoints, ep);
      const drift = Number(
        (((nearest.lowestLoss - minPoint.lowestLoss) / Math.max(0.1, minPoint.lowestLoss)) * 100).toFixed(1)
      );
      const isBest = ep === bestEpoch;
      const distPenalty = Math.abs(ep - bestEpoch) * 0.25;
      const underPenalty = ep < underEndEpoch ? 18 : 0;
      const overPenalty =
        (overStartEpoch !== null && ep >= overStartEpoch) || ep > minPoint.epoch + saveEvery * 2
          ? Math.max(12, drift * 2.2)
          : 0;
      const score = isBest
        ? clamp(Math.max(datasetMatchPercent, 76), 50, 99)
        : clamp(Math.round(datasetMatchPercent - 3 - distPenalty - drift * 1.4 - underPenalty - overPenalty), 22, 97);

      let tag = 'Clean Alternative';
      let status: RankedCheckpointItem['status'] = 'optimal';
      if (isBest) {
        tag = 'Best Recommended';
        status = 'optimal';
      } else if (ep < underEndEpoch) {
        tag = 'Undertrained';
        status = 'watch';
      } else if ((overStartEpoch !== null && ep >= overStartEpoch) || drift > 2.5) {
        tag = 'Overfit Risk';
        status = 'high';
      } else if (ep < bestEpoch) {
        tag = 'Softer / Safer';
        status = 'optimal';
      }

      return {
        epoch: ep,
        step: nearest.step ?? 0,
        checkpointName: `${modelSlug}_${ep}e.pth`,
        melLoss: Number(nearest.lowestLoss.toFixed(3)),
        driftPercent: Math.max(0, drift),
        score,
        tag,
        status,
      };
    })
    .sort((a, b) => {
      if (a.epoch === bestEpoch) return -1;
      if (b.epoch === bestEpoch) return 1;
      return b.score - a.score;
    })
    .slice(0, 4);

  const isMultiScale = Boolean(analysis.multiScaleMel) || project?.melLoss === 'Multi-Scale';
  const metricBreakdown: MetricBreakdownItem[] = [
    {
      label: isMultiScale ? 'Multi-Scale Mel (512/1024/2048)' : 'Mel Reconstruction',
      value: `${minPoint.lowestLoss.toFixed(3)} (-${totalDrop.toFixed(2)})`,
      status: minPoint.lowestLoss <= 16 ? 'optimal' : minPoint.lowestLoss <= 22 ? 'watch' : 'high',
      detail: isMultiScale
        ? `Multi-band FFT convergence · best at ep ${minPoint.epoch}`
        : `Best spectral error at ep ${minPoint.epoch}`,
    },
    {
      label: 'KL Latent Alignment',
      value: avgKl !== null ? `${avgKl.toFixed(3)} kl` : `${klScore}% inferred`,
      status: klScore >= 78 ? 'optimal' : klScore >= 60 ? 'watch' : 'high',
      detail: avgKl !== null ? 'Measured phoneme/timbre KL divergence' : 'Estimated from mel convergence curve',
    },
    {
      label: 'Feature Matching (FM)',
      value: avgFm !== null ? `${avgFm.toFixed(3)} fm` : `${fmScore}% harmonic fit`,
      status: fmScore >= 78 ? 'optimal' : fmScore >= 60 ? 'watch' : 'high',
      detail: avgFm !== null ? 'Multi-layer discriminator feature match' : 'Vocoder harmonic alignment',
    },
    {
      label: 'G / D Adversary Balance',
      value:
        avgGen !== null && avgDisc !== null
          ? `${avgGen.toFixed(2)} / ${avgDisc.toFixed(2)}`
          : `${gdScore}% balanced`,
      status: gdScore >= 75 ? 'optimal' : gdScore >= 58 ? 'watch' : 'high',
      detail: 'Generator vs. Multi-Period Discriminator parity',
    },
    {
      label: 'EMA & Clipped Grad Stability',
      value:
        gradNorm !== null
          ? `${stabilityScore}% (grad ${gradNorm}${analysis.emaDecay ? ` · ema ${analysis.emaDecay}` : ''})`
          : analysis.emaDecay
            ? `${stabilityScore}% (EMA ${analysis.emaDecay})`
            : `${stabilityScore}% steady`,
      status: stabilityScore >= 80 ? 'optimal' : stabilityScore >= 62 ? 'watch' : 'high',
      detail:
        reboundFromMin > 0.12
          ? 'Tail rebound detected'
          : gradNorm !== null && gradNorm <= 65
            ? 'Clipped grad norm healthy · EMA smoothed'
            : 'Low epoch-to-epoch variance',
    },
    {
      label: 'Dataset Fit Accuracy',
      value: `${datasetMatchPercent}% match`,
      status: datasetMatchPercent >= 80 ? 'optimal' : datasetMatchPercent >= 62 ? 'watch' : 'high',
      detail: datasetTier,
    },
    {
      label: 'Dataset Saturation',
      value: stepsPerClip !== null ? `${stepsPerClip} steps/clip` : `${saturationPct}% saturated`,
      status: saturationPct >= 65 ? 'optimal' : saturationPct >= 35 ? 'watch' : 'high',
      detail: `${stepsObserved} effective optimizer steps`,
    },
    {
      label: 'Tail Drift & Overfit Watch',
      value: reboundPct > 0.8 ? `+${reboundPct}% rebound` : '0.0% (at minimum)',
      status: reboundPct <= 1.2 ? 'optimal' : reboundPct <= 3.5 ? 'watch' : 'high',
      detail:
        reboundPct > 1.2
          ? `Loss rose past e${minPoint.epoch} min — roll back`
          : 'No late-epoch loss rebound detected',
    },
  ];

  const firstSaved = snapToSaveInterval(first.epoch, saveEvery, saveEvery, underEndEpoch);

  return {
    confidence: 'observed',
    saveEveryEpoch: saveEvery,
    modelContext,
    datasetSummary: datasetTier,
    bestLoss: minPoint.lowestLoss,
    candidateEpochs: candidates,
    datasetMatchPercent,
    stabilityScore,
    convergenceLabel,
    trainingVerdict,
    verdictTitle,
    verdictGuidance,
    undertrainingRiskPercent,
    overtrainingRiskPercent,
    effectiveStepsCompleted: stepsObserved,
    stepsPerClip,
    rankedCheckpoints,
    metricBreakdown,
    summary:
      hasPlateauOrRebound
        ? `Best checkpoint: Epoch ${bestEpoch} (${datasetMatchPercent}% dataset match, ${stabilityScore}% stability · mel ${minPoint.lowestLoss.toFixed(3)} near ep ${minPoint.epoch}, dropped ${totalDrop.toFixed(2)} pts across ${datasetTier.toLowerCase()}).${soundNote ? ` ${soundNote}` : ''}`
        : `Mel loss is still steadily dropping through Epoch ${lastPoint.epoch} (${lastPoint.lowestLoss.toFixed(3)} · ${datasetMatchPercent}% dataset fit, ${stabilityScore}% stability). Best checkpoint so far: Epoch ${bestEpoch}.${soundNote ? ` ${soundNote}` : ''}`,
    undertrainingLabel:
      firstSaved < underEndEpoch ? `Epochs ${firstSaved}–${underEndEpoch}` : `Up to Epoch ${underEndEpoch}`,
    undertrainingEndEpoch: underEndEpoch,
    sweetSpotLabel: sweetLabel,
    sweetSpotEpoch: bestEpoch,
    sweetSpotMinEpoch: sweetMin,
    sweetSpotMaxEpoch: sweetMax,
    overtrainingLabel: overLabel,
    overtrainingStartEpoch: overStartEpoch,
  };
}
