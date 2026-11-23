import {
  LiveEpochProgress,
  ModelSoundFeedback,
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
    /([A-Za-z0-9_.-]+)\s*\|\s*epoch=(\d+)\s*\|\s*step=(\d+)\s*\|\s*time=\d{1,2}:\d{2}:\d{2}\s*\|\s*training_speed=(\d{1,2}:\d{2}:\d{2})(?:\s*\|\s*lowest_value=([0-9.]+)\s*\(epoch\s+(\d+)\s+and\s+step\s+(\d+)\))?/g;

  const points: TrainingEpochPoint[] = [];
  let speakerFromEpochs = '';
  let lowestLoss: number | null = null;
  let lowestLossEpoch: number | null = null;
  let lowestLossStep: number | null = null;

  let match: RegExpExecArray | null;
  while ((match = epochLineRegex.exec(text)) !== null) {
    speakerFromEpochs = match[1];
    const epoch = Number(match[2]);
    const step = Number(match[3]);
    const speedSec = parseDurationToSeconds(match[4]);
    const lowVal = match[5] ? Number(match[5]) : null;
    if (lowVal !== null && (lowestLoss === null || lowVal < lowestLoss)) {
      lowestLoss = lowVal;
      lowestLossEpoch = match[6] ? Number(match[6]) : epoch;
      lowestLossStep = match[7] ? Number(match[7]) : step;
    }
    points.push({
      epoch,
      step,
      speedSec,
      lowestLoss: lowVal,
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

function computeModelBaselineSweetRange(
  project: VoiceModelProject | undefined,
  segmentCount: number,
  soundCheck: ModelSoundFeedback
): { low: number; high: number; underEnd: number; overStart: number } {
  const saveEvery = Math.max(1, Math.round(project?.saveEveryEpoch ?? 5));
  const totalEpoch = Math.max(saveEvery, Math.round(project?.totalEpoch ?? 200));
  const maxSaved = Math.max(saveEvery, Math.floor(totalEpoch / saveEvery) * saveEvery);

  // Base sweet spot tuned by model architecture, sample rate, pretrained usage, and batch size
  let center = 65;
  if (project) {
    if (project.vocoder === 'RefineGAN') center += 20;
    else if (project.vocoder === 'MRF HiFi-GAN') center += 10;

    if (project.samplingRate === '48000') center += 15;
    else if (project.samplingRate === '40000') center += 8;

    if (!project.pretrained) center = Math.round(center * 1.85);

    const effBatch = Math.max(1, project.batchSize * project.gpuCount);
    if (effBatch >= 8) center += 15;
    else if (effBatch <= 2) center -= 10;
  }

  if (segmentCount > 0) {
    if (segmentCount < 150) center -= 15;
    else if (segmentCount >= 550) center += 35;
    else if (segmentCount >= 300) center += 15;
  }

  if (soundCheck === 'undertrained') center += Math.max(20, saveEvery * 4);
  if (soundCheck === 'overtrained') center -= Math.max(15, saveEvery * 3);

  const halfWidth = Math.max(saveEvery * 2, Math.round((center * 0.2) / saveEvery) * saveEvery);
  let low = snapToSaveInterval(center - halfWidth, saveEvery, saveEvery, maxSaved);
  let high = snapToSaveInterval(center + halfWidth, saveEvery, Math.min(maxSaved, low + saveEvery), maxSaved);
  if (high <= low && low + saveEvery <= maxSaved) {
    high = low + saveEvery;
  }

  const underEnd = snapToSaveInterval(
    Math.max(saveEvery, low - saveEvery),
    saveEvery,
    saveEvery,
    Math.max(saveEvery, low)
  );
  const overStart = snapToSaveInterval(
    high + Math.max(saveEvery * 4, Math.round((high * 0.45) / saveEvery) * saveEvery),
    saveEvery,
    Math.min(maxSaved, high + saveEvery),
    Math.max(maxSaved, high + saveEvery)
  );

  return { low, high, underEnd, overStart };
}

function uniqueSortedEpochs(epochs: number[]): number[] {
  return Array.from(new Set(epochs)).sort((a, b) => a - b);
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
  const vocoder = analysis?.vocoderDetected || project?.vocoder || 'HiFi-GAN';
  const sr = project?.samplingRate || '32000';
  const modelContext = `${modelName} · ${vocoder} · ${sr} Hz · Save every ${saveEvery} ep`;

  const lossPoints = analysis
    ? analysis.points.filter(
        (p): p is TrainingEpochPoint & { lowestLoss: number } => p.lowestLoss !== null
      )
    : [];
  const allPoints = analysis?.points ?? [];

  // Never fabricate epoch numbers or charts before a model has real training epochs logged
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
      summary: `Waiting for real training data for ${modelName}. Once epochs are logged (or a .log/.ipynb is imported), Diagnostics will analyze mel loss, dataset size, architecture, and how your checkpoints sound to pinpoint the Best Epoch.`,
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
          ? 'You confirmed it sounds clean & natural — locking near your lowest stable mel-loss checkpoint.'
          : '';

  const maxObservedEpoch = Math.max(
    saveEvery,
    allPoints.length > 0 ? allPoints[allPoints.length - 1].epoch : analysis.epochsCompleted
  );
  const maxObservedSaved = Math.max(saveEvery, Math.floor(maxObservedEpoch / saveEvery) * saveEvery);

  // Sparse log (1–3 loss points): report real observed progress only, never fabricate future curves
  if (lossPoints.length < 2) {
    const curBestEp = lossPoints[0]?.epoch ?? maxObservedEpoch;
    const bestSnap = snapToSaveInterval(curBestEp, saveEvery, saveEvery, Math.max(saveEvery, maxObservedSaved));
    const bestVal = lossPoints[0]?.lowestLoss ?? analysis.lowestLoss ?? null;
    return {
      confidence: 'observed',
      saveEveryEpoch: saveEvery,
      modelContext,
      datasetSummary: datasetTier,
      bestLoss: bestVal,
      candidateEpochs: [bestSnap],
      summary: `Early training data recorded (${allPoints.length} epoch${allPoints.length === 1 ? '' : 's'}, best observed loss ${bestVal !== null ? bestVal.toFixed(3) : 'N/A'} at epoch ${curBestEp}). Let more saved epochs log so mel-loss stabilization can be measured.${soundNote ? ` ${soundNote}` : ''}`,
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

  const tailWindow = lossPoints.slice(-Math.min(3, lossPoints.length));
  const tailSlope =
    tailWindow.length >= 2
      ? (tailWindow[0].lowestLoss - tailWindow[tailWindow.length - 1].lowestLoss) /
        Math.max(1, tailWindow[tailWindow.length - 1].epoch - tailWindow[0].epoch)
      : 0;
  const hasPlateauOrRebound = minIdx < lossPoints.length - 1 || tailSlope < 0.025;

  // Earliest epoch reaching within 4.5% of total mel-loss gain (avoids overfitting tail noise)
  const sweetCap = minPoint.lowestLoss + totalDrop * 0.045;
  let balancedPoint = minPoint;
  for (let i = underIdx; i <= minIdx; i++) {
    if (lossPoints[i].lowestLoss <= sweetCap) {
      balancedPoint = lossPoints[i];
      break;
    }
  }

  // Factor dataset size & sample rate slightly around observed balanced checkpoint
  let datasetOffsetEpochs = 0;
  if (segCount > 0 && segCount < 140) datasetOffsetEpochs -= saveEvery;
  else if (segCount >= 450 && !hasPlateauOrRebound) datasetOffsetEpochs += saveEvery;
  if (sr === '48000' && soundCheck !== 'overtrained') datasetOffsetEpochs += saveEvery;

  const upperObservedBound = Math.max(saveEvery, maxObservedSaved);

  if (hasPlateauOrRebound) {
    let bestEpoch = snapToSaveInterval(
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

    const sweetMin = snapToSaveInterval(bestEpoch - saveEvery, saveEvery, saveEvery, bestEpoch);
    const sweetMax = snapToSaveInterval(bestEpoch + saveEvery, saveEvery, bestEpoch, upperObservedBound);
    const overStartEpoch = snapToSaveInterval(
      Math.max(sweetMax + saveEvery, minPoint.epoch + saveEvery),
      saveEvery,
      Math.min(maxSaved, sweetMax + saveEvery),
      Math.max(maxSaved, sweetMax + saveEvery)
    );
    const firstSaved = snapToSaveInterval(first.epoch, saveEvery, saveEvery, underEndEpoch);
    const candidates = uniqueSortedEpochs([sweetMin, bestEpoch, sweetMax].filter((e) => e >= saveEvery && e <= upperObservedBound));

    return {
      confidence: 'observed',
      saveEveryEpoch: saveEvery,
      modelContext,
      datasetSummary: datasetTier,
      bestLoss: minPoint.lowestLoss,
      candidateEpochs: candidates,
      summary: `Best checkpoint: Epoch ${bestEpoch} (lowest mel loss ${minPoint.lowestLoss.toFixed(3)} near ep ${minPoint.epoch}, dropped ${totalDrop.toFixed(2)} pts total across ${datasetTier.toLowerCase()}).${soundNote ? ` ${soundNote}` : ''}`,
      undertrainingLabel:
        firstSaved < underEndEpoch ? `Epochs ${firstSaved}–${underEndEpoch}` : `Up to Epoch ${underEndEpoch}`,
      undertrainingEndEpoch: underEndEpoch,
      sweetSpotLabel:
        sweetMin < sweetMax ? `Best: Epoch ${bestEpoch} (test e${sweetMin}–e${sweetMax})` : `Best: Epoch ${bestEpoch}`,
      sweetSpotEpoch: bestEpoch,
      sweetSpotMinEpoch: sweetMin,
      sweetSpotMaxEpoch: sweetMax,
      overtrainingLabel:
        minIdx < lossPoints.length - 1 || soundCheck === 'overtrained'
          ? `Past Epoch ${overStartEpoch} (plateau/rebound)`
          : `Risk past Epoch ${overStartEpoch}`,
      overtrainingStartEpoch: overStartEpoch,
    };
  }

  // Mel loss is still actively improving through the latest logged epoch
  let currentBest = snapToSaveInterval(minPoint.epoch, saveEvery, saveEvery, upperObservedBound);
  if (soundCheck === 'overtrained') {
    currentBest = snapToSaveInterval(currentBest - saveEvery * 2, saveEvery, saveEvery, upperObservedBound);
  }
  const sweetMin = snapToSaveInterval(currentBest - saveEvery, saveEvery, saveEvery, currentBest);
  const sweetMax = snapToSaveInterval(currentBest, saveEvery, sweetMin, upperObservedBound);
  const firstSaved = snapToSaveInterval(first.epoch, saveEvery, saveEvery, currentBest);
  const candidates = uniqueSortedEpochs([sweetMin, currentBest].filter((e) => e >= saveEvery && e <= upperObservedBound));

  return {
    confidence: 'observed',
    saveEveryEpoch: saveEvery,
    modelContext,
    datasetSummary: datasetTier,
    bestLoss: minPoint.lowestLoss,
    candidateEpochs: candidates,
    summary:
      soundCheck === 'undertrained'
        ? `Best saved checkpoint so far is Epoch ${currentBest} (${lastPoint.lowestLoss.toFixed(3)}), and mel loss is still actively dropping across ${datasetTier.toLowerCase()} — keep training past e${currentBest} until loss flattens.`
        : `Mel loss is still steadily dropping through Epoch ${lastPoint.epoch} (${lastPoint.lowestLoss.toFixed(3)}) without overtraining signs yet. Best checkpoint so far: Epoch ${currentBest} (also audition e${candidates[0] ?? currentBest}).${soundNote ? ` ${soundNote}` : ''}`,
    undertrainingLabel:
      firstSaved < underEndEpoch ? `Epochs ${firstSaved}–${underEndEpoch}` : `Up to Epoch ${underEndEpoch}`,
    undertrainingEndEpoch: underEndEpoch,
    sweetSpotLabel: `Best so far: Epoch ${currentBest}`,
    sweetSpotEpoch: currentBest,
    sweetSpotMinEpoch: sweetMin,
    sweetSpotMaxEpoch: sweetMax,
    overtrainingLabel: 'None observed yet (still improving)',
    overtrainingStartEpoch: null,
  };
}
