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

export function analyzeTrainingZones(
  analysis: TrainingAnalysis | null,
  project?: VoiceModelProject,
  soundCheck: ModelSoundFeedback = 'untested'
): TrainingZoneInsight {
  const saveEvery = Math.max(1, Math.round(project?.saveEveryEpoch ?? 5));
  const totalEpoch = Math.max(saveEvery, Math.round(project?.totalEpoch ?? 200));
  const maxSaved = Math.max(saveEvery, Math.floor(totalEpoch / saveEvery) * saveEvery);
  const modelName = project?.name.trim() || 'Active Model';
  const modelContext = project
    ? `${modelName} · ${project.vocoder} · ${project.samplingRate} Hz · Save every ${saveEvery} ep`
    : `Save every ${saveEvery} ep`;

  const lossPoints = analysis
    ? analysis.points.filter(
        (p): p is TrainingEpochPoint & { lowestLoss: number } => p.lowestLoss !== null
      )
    : [];

  const soundNote =
    soundCheck === 'undertrained'
      ? 'Based on your note that the voice still sounds robotic/undertrained, aim toward the upper end of the sweet window.'
      : soundCheck === 'overtrained'
        ? 'Based on your note that the voice sounds metallic/harsh, step back to an earlier saved checkpoint.'
        : soundCheck === 'clean'
          ? 'Your checkpoint already sounds clean—compare adjacent saves around this range before training further.'
          : '';

  // No log or sparse milestones: give a clear model-driven estimate strictly aligned to saveEveryEpoch
  if (!analysis || lossPoints.length < 4) {
    const baseline = computeModelBaselineSweetRange(
      project,
      analysis?.segmentCount ?? 0,
      soundCheck
    );
    const mid = snapToSaveInterval(
      (baseline.low + baseline.high) / 2,
      saveEvery,
      baseline.low,
      baseline.high
    );

    const baseReason = !analysis
      ? `No training log loaded yet—showing an estimate tuned to ${modelName} (${project?.vocoder ?? 'HiFi-GAN'}, ${project?.samplingRate ?? '32000'} Hz, save every ${saveEvery} epochs).`
      : `Only ${lossPoints.length} loss milestone${lossPoints.length === 1 ? '' : 's'} logged so far, so a plateau cannot be measured yet—showing an estimate aligned to every ${saveEvery} epochs.`;

    return {
      confidence: 'estimate',
      saveEveryEpoch: saveEvery,
      modelContext,
      summary: soundNote ? `${baseReason} ${soundNote}` : baseReason,
      undertrainingLabel:
        baseline.underEnd > saveEvery
          ? `Est. Epochs ${saveEvery}–${baseline.underEnd}`
          : `Est. Up to Epoch ${baseline.underEnd}`,
      undertrainingEndEpoch: baseline.underEnd,
      sweetSpotLabel:
        baseline.low === baseline.high
          ? `Est. Epoch ${baseline.low}`
          : `Est. Epochs ${baseline.low}–${baseline.high}`,
      sweetSpotEpoch: mid,
      sweetSpotMinEpoch: baseline.low,
      sweetSpotMaxEpoch: baseline.high,
      overtrainingLabel:
        soundCheck === 'overtrained'
          ? `Past Epoch ${baseline.high} (harshness reported)`
          : `Est. Past Epoch ${baseline.overStart}`,
      overtrainingStartEpoch: baseline.overStart,
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

  // Undertraining ends once ~75% of the observed mel-loss drop has completed
  const underThreshold = first.lowestLoss - totalDrop * 0.75;
  let underIdx = 0;
  for (let i = 0; i < lossPoints.length; i++) {
    if (lossPoints[i].lowestLoss <= underThreshold) {
      underIdx = i;
      break;
    }
  }
  const rawUnderEnd = lossPoints[underIdx].epoch;
  const underEndEpoch = snapToSaveInterval(rawUnderEnd, saveEvery, saveEvery, maxSaved);

  const tailSlope =
    lossPoints.length >= 2
      ? (lossPoints[lossPoints.length - 2].lowestLoss - lastPoint.lowestLoss) /
        Math.max(1, lastPoint.epoch - lossPoints[lossPoints.length - 2].epoch)
      : 0;
  const stalledOrReversed = minIdx < lossPoints.length - 1 || tailSlope < 0.03;

  const sweetCap = minPoint.lowestLoss + totalDrop * 0.06;
  let sweetPoint = minPoint;
  for (let i = underIdx; i <= minIdx; i++) {
    if (lossPoints[i].lowestLoss <= sweetCap) {
      sweetPoint = lossPoints[i];
      break;
    }
  }

  if (stalledOrReversed) {
    let snappedSweet = snapToSaveInterval(sweetPoint.epoch, saveEvery, saveEvery, maxSaved);
    if (soundCheck === 'overtrained') {
      snappedSweet = snapToSaveInterval(
        snappedSweet - saveEvery * 2,
        saveEvery,
        saveEvery,
        maxSaved
      );
    } else if (soundCheck === 'undertrained') {
      snappedSweet = snapToSaveInterval(
        snappedSweet + saveEvery * 2,
        saveEvery,
        saveEvery,
        maxSaved
      );
    }

    const rangePad = saveEvery * 2;
    const sweetMin = snapToSaveInterval(
      snappedSweet - rangePad,
      saveEvery,
      Math.min(maxSaved, underEndEpoch + saveEvery),
      snappedSweet
    );
    const sweetMax = snapToSaveInterval(
      snappedSweet + rangePad,
      saveEvery,
      snappedSweet,
      maxSaved
    );
    const overStartEpoch = snapToSaveInterval(
      Math.max(sweetMax + saveEvery * 2, minPoint.epoch + saveEvery),
      saveEvery,
      Math.min(maxSaved, sweetMax + saveEvery),
      Math.max(maxSaved, sweetMax + saveEvery)
    );
    const firstSaved = snapToSaveInterval(first.epoch, saveEvery, saveEvery, underEndEpoch);

    return {
      confidence: 'observed',
      saveEveryEpoch: saveEvery,
      modelContext,
      summary: `Mel loss dropped ${totalDrop.toFixed(2)} pts overall and flattened near saved checkpoint Epoch ${snappedSweet} (${sweetPoint.lowestLoss.toFixed(3)}).${soundNote ? ` ${soundNote}` : ''}`,
      undertrainingLabel:
        firstSaved < underEndEpoch
          ? `Epochs ${firstSaved}–${underEndEpoch}`
          : `Up to Epoch ${underEndEpoch}`,
      undertrainingEndEpoch: underEndEpoch,
      sweetSpotLabel:
        sweetMin < sweetMax
          ? `Epochs ${sweetMin}–${sweetMax} (best ~${snappedSweet})`
          : `Epoch ${snappedSweet} (observed)`,
      sweetSpotEpoch: snappedSweet,
      sweetSpotMinEpoch: sweetMin,
      sweetSpotMaxEpoch: sweetMax,
      overtrainingLabel:
        minIdx < lossPoints.length - 1 || soundCheck === 'overtrained'
          ? `Past Epoch ${overStartEpoch} (stalled/harsh)`
          : `Past Epoch ${overStartEpoch}`,
      overtrainingStartEpoch: overStartEpoch,
    };
  }

  // Still dropping cleanly at the end of the log: snap projected range strictly to saveEvery
  const lastSnapped = snapToSaveInterval(lastPoint.epoch, saveEvery, saveEvery, maxSaved);
  const projectedLow = snapToSaveInterval(
    Math.max(lastSnapped + saveEvery * 2, lastPoint.epoch * 1.25),
    saveEvery,
    saveEvery,
    maxSaved
  );
  const projectedHigh = snapToSaveInterval(
    projectedLow + Math.max(saveEvery * 4, 30),
    saveEvery,
    projectedLow,
    maxSaved
  );
  const projectedSweet = snapToSaveInterval(
    (projectedLow + projectedHigh) / 2,
    saveEvery,
    projectedLow,
    projectedHigh
  );
  const projectedOverStart = snapToSaveInterval(
    projectedHigh + Math.max(saveEvery * 6, 40),
    saveEvery,
    Math.min(maxSaved, projectedHigh + saveEvery),
    Math.max(maxSaved, projectedHigh + saveEvery)
  );
  const firstSaved = snapToSaveInterval(first.epoch, saveEvery, saveEvery, lastSnapped);

  return {
    confidence: 'estimate',
    saveEveryEpoch: saveEvery,
    modelContext,
    summary: `Mel loss is still dropping through Epoch ${lastPoint.epoch} (${lastPoint.lowestLoss.toFixed(3)}) without flattening yet—showing the next checkpoints aligned to every ${saveEvery} epochs.${soundNote ? ` ${soundNote}` : ''}`,
    undertrainingLabel:
      firstSaved < lastSnapped
        ? `Epochs ${firstSaved}–${lastSnapped} (still improving)`
        : `Up to Epoch ${lastSnapped} (still improving)`,
    undertrainingEndEpoch: lastSnapped,
    sweetSpotLabel:
      projectedLow < projectedHigh
        ? `Est. Epochs ${projectedLow}–${projectedHigh}`
        : `Est. Epoch ${projectedLow}`,
    sweetSpotEpoch: projectedSweet,
    sweetSpotMinEpoch: projectedLow,
    sweetSpotMaxEpoch: projectedHigh,
    overtrainingLabel: `Est. Past Epoch ${projectedOverStart}`,
    overtrainingStartEpoch: projectedOverStart,
  };
}
