import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ExternalLink,
  Info,
} from 'lucide-react';
import { ACCENT_HEX, REPO_README_URL } from '../data/defaults';
import { VoiceModelProject } from '../types/studio';
import {
  buildTrainPipelineCommands,
  estimateKaggleSurvival,
  estimateT4VramUsage,
  estimateTrainingCompletion,
  sanitizeModelName,
} from '../utils/rvcTuning';
import { useI18n } from '../utils/i18n';
import {
  Step1ModelSettingsCard,
  Step2PrepareDatasetCard,
  Step3ExtractFeaturesCard,
  Step4TrainModelCard,
} from './TrainStepCards';

interface TrainWorkspaceViewProps {
  activeProject: VoiceModelProject;
  onUpdateProject: (updated: VoiceModelProject) => void;
}

export const TrainWorkspaceView: React.FC<TrainWorkspaceViewProps> = React.memo(({
  activeProject,
  onUpdateProject,
}) => {
  const { t } = useI18n();
  const [showInfoBar, setShowInfoBar] = useState(false);

  const [cliPreviewOpen, setCliPreviewOpen] = useState(false);

  const [isTraining, setIsTraining] = useState(false);
  const [trainingRun, setTrainingRun] = useState({
    epoch: 1,
    step: 0,
    batchDone: 0,
    batchTotal: 60,
    elapsedSec: 0,
    itPerSec: 2.6,
    epochDurationsSec: [] as number[],
  });
  const [refreshStatus, setRefreshStatus] = useState('');

  const safeName = useMemo(
    () => sanitizeModelName(activeProject.name),
    [activeProject.name]
  );

  // Lazy Model & VRAM Info calculation: skip work when panel is closed
  const infoMetrics = useMemo(() => {
    if (!showInfoBar) return null;
    const vram = estimateT4VramUsage(activeProject);
    const survival = estimateKaggleSurvival(activeProject);
    const vramPct = Math.min(
      100,
      Math.round((vram.estimatedVramGb / vram.vramCapGb) * 100)
    );
    return { vram, survival, vramPct };
  }, [showInfoBar, activeProject]);

  const pipelineCommands = useMemo(
    () => (cliPreviewOpen ? buildTrainPipelineCommands(activeProject) : ''),
    [cliPreviewOpen, activeProject]
  );

  // Quantized to whole seconds / epoch transitions so live completion stays smooth & zero-overhead
  const elapsedWholeSec = Math.floor(trainingRun.elapsedSec);
  const completionEstimate = useMemo(() => {
    if (!isTraining) return null;
    return estimateTrainingCompletion({
      currentEpoch: trainingRun.epoch,
      totalEpoch: activeProject.totalEpoch,
      batchDone: trainingRun.batchDone,
      batchTotal: trainingRun.batchTotal,
      elapsedSec: elapsedWholeSec,
      itPerSec: trainingRun.itPerSec,
      completedEpochDurationsSec: trainingRun.epochDurationsSec,
    });
  }, [
    isTraining,
    trainingRun.epoch,
    trainingRun.batchDone,
    trainingRun.batchTotal,
    trainingRun.itPerSec,
    trainingRun.epochDurationsSec,
    elapsedWholeSec,
    activeProject.totalEpoch,
  ]);

  const activeProjectRef = useRef(activeProject);
  const onUpdateProjectRef = useRef(onUpdateProject);
  const safeNameRef = useRef(safeName);
  const isTrainingRef = useRef(isTraining);
  const currentEpochRef = useRef(trainingRun.epoch);
  const trainingRunRef = useRef(trainingRun);
  const refreshTimerRef = useRef<number | null>(null);

  activeProjectRef.current = activeProject;
  onUpdateProjectRef.current = onUpdateProject;
  safeNameRef.current = safeName;
  isTrainingRef.current = isTraining;
  currentEpochRef.current = trainingRun.epoch;
  trainingRunRef.current = trainingRun;

  useEffect(() => {
    return () => {
      if (refreshTimerRef.current !== null) {
        window.clearTimeout(refreshTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!isTraining) return;
    const timer = window.setInterval(() => {
      const proj = activeProjectRef.current;
      const name = safeNameRef.current;
      const prev = trainingRunRef.current;
      const effBatch = Math.max(1, proj.batchSize * proj.gpuCount);
      const batchTotal = Math.max(24, Math.min(120, Math.round(240 / effBatch)));
      const batchStep = Math.max(4, Math.round(batchTotal / 6));
      const baseItPerSec =
        proj.vocoder === 'RefineGAN' ? 1.85 : proj.vocoder === 'MRF HiFi-GAN' ? 2.25 : 2.65;
      const nextElapsed = Number((prev.elapsedSec + 0.25).toFixed(1));
      const rawBatch = prev.batchDone + batchStep;

      if (rawBatch < batchTotal) {
        setTrainingRun({
          epoch: prev.epoch,
          step: prev.step + batchStep,
          batchDone: rawBatch,
          batchTotal,
          elapsedSec: nextElapsed,
          itPerSec: baseItPerSec,
          epochDurationsSec: prev.epochDurationsSec,
        });
        return;
      }

      const finishedEpoch = prev.epoch;
      const completedEpochDuration = Number(
        (batchTotal / Math.max(0.15, baseItPerSec)).toFixed(2)
      );
      const nextDurations = [...prev.epochDurationsSec.slice(-11), completedEpochDuration];
      if (finishedEpoch >= proj.totalEpoch) {
        setIsTraining(false);
        setTrainingRun({
          epoch: proj.totalEpoch,
          step: prev.step + (batchTotal - prev.batchDone),
          batchDone: batchTotal,
          batchTotal,
          elapsedSec: nextElapsed,
          itPerSec: baseItPerSec,
          epochDurationsSec: nextDurations,
        });
        onUpdateProjectRef.current({
          ...proj,
          trainingOutput: `Model ${name} completed ${proj.totalEpoch} epochs on GPU ${proj.gpuNumber}. Saved logs/${name}/${name}_${proj.totalEpoch}e.pth`,
          exportPthFile: `logs/${name}/${name}_${proj.totalEpoch}e.pth`,
          exportIndexFile: `logs/${name}/${name}.index`,
        });
        return;
      }

      setTrainingRun({
        epoch: finishedEpoch + 1,
        step: prev.step + (batchTotal - prev.batchDone),
        batchDone: 0,
        batchTotal,
        elapsedSec: nextElapsed,
        itPerSec: baseItPerSec,
        epochDurationsSec: nextDurations,
      });
    }, 250);
    return () => window.clearInterval(timer);
  }, [isTraining]);

  const patchProject = useCallback((patch: Partial<VoiceModelProject>) => {
    onUpdateProjectRef.current({ ...activeProjectRef.current, ...patch });
  }, []);

  const handleRefreshDatasets = useCallback(() => {
    setRefreshStatus('Refreshed Kaggle dataset directory & model registry.');
    if (refreshTimerRef.current !== null) {
      window.clearTimeout(refreshTimerRef.current);
    }
    refreshTimerRef.current = window.setTimeout(() => {
      setRefreshStatus('');
      refreshTimerRef.current = null;
    }, 2400);
  }, []);

  const handlePreprocessDataset = useCallback(() => {
    const proj = activeProjectRef.current;
    const name = safeNameRef.current;
    const dsPath = proj.datasetPath.trim();
    if (!dsPath) {
      onUpdateProjectRef.current({
        ...proj,
        preprocessOutput: 'Enter a valid Kaggle dataset path before preprocessing.',
      });
      return;
    }
    const notes: string[] = [
      `cut=${proj.audioCutting}`,
      proj.truncateSilenceEnabled
        ? `truncate_silence=${proj.silenceDurationMs}ms`
        : 'truncate_silence=off',
      proj.noiseFilter ? 'noise_filter=on' : 'noise_filter=off',
      proj.loudnessNorm
        ? `loudness=${proj.loudnessTargetLufs} LUFS`
        : 'loudness=off',
    ];
    onUpdateProjectRef.current({
      ...proj,
      preprocessOutput: `Model ${name} preprocessed successfully (${notes.join(', ')}).`,
    });
  }, []);

  const handleExtractFeatures = useCallback(() => {
    const proj = activeProjectRef.current;
    const name = safeNameRef.current;
    onUpdateProjectRef.current({
      ...proj,
      extractOutput: `Model ${name} extracted successfully (${proj.pitchExtractor} + ${proj.embedder}, silent files: ${proj.silentTrainingFiles}).`,
    });
  }, []);

  const handleToggleTraining = useCallback(() => {
    const proj = activeProjectRef.current;
    const name = safeNameRef.current;
    if (!proj.agreedTerms) {
      onUpdateProjectRef.current({
        ...proj,
        trainingOutput: 'Please agree to the terms of use before starting training.',
      });
      return;
    }
    if (isTrainingRef.current) {
      setIsTraining(false);
      onUpdateProjectRef.current({
        ...proj,
        trainingOutput: `Training stopped at epoch ${currentEpochRef.current}/${proj.totalEpoch}. Latest weights kept in logs/${name}/.`,
      });
    } else {
      const effBatch = Math.max(1, proj.batchSize * proj.gpuCount);
      const batchTotal = Math.max(24, Math.min(120, Math.round(240 / effBatch)));
      const baseItPerSec = proj.vocoder === 'RefineGAN' ? 1.85 : proj.vocoder === 'MRF HiFi-GAN' ? 2.25 : 2.65;
      setTrainingRun({
        epoch: 1,
        step: 0,
        batchDone: 0,
        batchTotal,
        elapsedSec: 0,
        itPerSec: baseItPerSec,
        epochDurationsSec: [],
      });
      setIsTraining(true);
      onUpdateProjectRef.current({
        ...proj,
        trainingOutput: `Training ${name} (${proj.vocoder}, ${proj.samplingRate} Hz, batch ${proj.batchSize}, ${
          proj.fp16 ? 'fp16' : 'fp32'
        }) started...`,
      });
    }
  }, []);

  const handleGenerateIndex = useCallback(() => {
    const proj = activeProjectRef.current;
    const name = safeNameRef.current;
    onUpdateProjectRef.current({
      ...proj,
      trainingOutput: `Index file for model ${name} generated successfully (${proj.indexAlgorithm}).`,
      exportIndexFile: `logs/${name}/${name}.index`,
    });
  }, []);

  const handleToggleCliPreview = useCallback(() => {
    setCliPreviewOpen((v) => !v);
  }, []);

  return (
    <div className="space-y-6">
      {/* Training Header */}
      <div className="flex flex-wrap items-start justify-between gap-4 px-1">
        <div>
          <h2 className="font-display text-2xl font-bold text-[#FDF7F7] tracking-tight">
            {t('train.title')}
          </h2>
          <p className="text-sm text-[#DFBFC3] mt-1">
            {t('train.subtitle')}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2 shrink-0">
          <button
            type="button"
            onClick={() => setShowInfoBar((v) => !v)}
            aria-expanded={showInfoBar}
            className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
              showInfoBar
                ? 'border-[#E51923] bg-[#200B0E] text-white'
                : 'border-[#47161D] bg-[#160709] text-[#DFBFC3] hover:text-white hover:border-[#E51923]'
            }`}
          >
            <Info className="h-3.5 w-3.5 text-[#E51923]" />
            <span>{t('train.vramInfo')}</span>
          </button>

          <a
            href={REPO_README_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 rounded-lg border border-[#47161D] bg-[#160709] px-3 py-1.5 text-xs font-medium text-[#DFBFC3] hover:text-white hover:border-[#E51923] whitespace-nowrap shrink-0"
          >
            <span>{t('train.readme')}</span>
            <ExternalLink className="h-3.5 w-3.5" />
          </a>
        </div>
      </div>

      {/* Optional Model & T4 VRAM Informational Bar */}
      {showInfoBar && infoMetrics && (
        <section
          aria-label="Model and T4 VRAM information bar"
          className="rounded-xl border border-[#47161D] bg-[#160709] p-4 sm:p-5"
        >
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="text-xs font-medium text-[#FDF7F7]">
              {activeProject.name.trim() || 'Untitled Model'} · {activeProject.samplingRate} Hz · {activeProject.vocoder}
            </div>

            <div className="flex flex-wrap sm:flex-nowrap items-center gap-4 min-w-[260px] sm:min-w-[300px]">
              <div className="flex-1 min-w-[200px]">
                <div className="flex items-center justify-between gap-3 text-[11px] mb-1.5">
                  <span className="text-[#AD8288] whitespace-nowrap">
                    T4 VRAM ({activeProject.gpuCount === 2 ? 'Dual 16GB' : 'Single 16GB'}) · {activeProject.batchSize}×{activeProject.gpuCount} = {activeProject.batchSize * activeProject.gpuCount} eff. batch
                  </span>
                  <span
                    className={`font-mono font-semibold tabular-nums whitespace-nowrap ${
                      infoMetrics.vram.fitStatus === 'Risk of OOM'
                        ? 'text-[#FF2B35]'
                        : infoMetrics.vram.fitStatus === 'Tight'
                          ? 'text-[#F59E0B]'
                          : 'text-[#FDF7F7]'
                    }`}
                  >
                    {infoMetrics.vram.estimatedVramGb} / {infoMetrics.vram.vramCapGb} GB ({infoMetrics.vram.fitStatus})
                  </span>
                </div>
                <div className="h-1.5 w-full rounded-full bg-[#100506] overflow-hidden border border-[#47161D]">
                  <div
                    className="h-full transition-all"
                    style={{
                      width: `${infoMetrics.vramPct}%`,
                      backgroundColor:
                        infoMetrics.vram.fitStatus === 'Risk of OOM'
                          ? '#FF2B35'
                          : infoMetrics.vram.fitStatus === 'Tight'
                            ? '#F59E0B'
                            : ACCENT_HEX,
                    }}
                  />
                </div>
                <p className="mt-1.5 text-[11px] font-mono text-[#AD8288] tabular-nums">
                  Disk ~{infoMetrics.survival.estimatedDiskGb} / {infoMetrics.survival.diskCapGb} GB ({infoMetrics.survival.checkpointCount} ckpt{infoMetrics.survival.checkpointCount === 1 ? '' : 's'} · {infoMetrics.survival.diskStatus}) · ~{infoMetrics.survival.estimatedHours}h ({infoMetrics.survival.sessionStatus})
                </p>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* STEP 1: Model Settings */}
      <Step1ModelSettingsCard
        name={activeProject.name}
        samplingRate={activeProject.samplingRate}
        vocoder={activeProject.vocoder}
        cpuCores={activeProject.cpuCores}
        gpuNumber={activeProject.gpuNumber}
        onPatchProject={patchProject}
      />

      {/* STEP 2: Prepare Dataset */}
      <Step2PrepareDatasetCard
        datasetPath={activeProject.datasetPath}
        datasetCreator={activeProject.datasetCreator}
        audioCutting={activeProject.audioCutting}
        chunkLengthSec={activeProject.chunkLengthSec}
        overlapLengthSec={activeProject.overlapLengthSec}
        truncateSilenceEnabled={activeProject.truncateSilenceEnabled}
        silenceDurationMs={activeProject.silenceDurationMs}
        noiseFilter={activeProject.noiseFilter}
        loudnessNorm={activeProject.loudnessNorm}
        loudnessTargetLufs={activeProject.loudnessTargetLufs}
        preprocessOutput={activeProject.preprocessOutput}
        refreshStatus={refreshStatus}
        onPatchProject={patchProject}
        onRefreshDatasets={handleRefreshDatasets}
        onPreprocessDataset={handlePreprocessDataset}
      />

      {/* STEP 3: Extract Features */}
      <Step3ExtractFeaturesCard
        pitchExtractor={activeProject.pitchExtractor}
        embedder={activeProject.embedder}
        silentTrainingFiles={activeProject.silentTrainingFiles}
        extractOutput={activeProject.extractOutput}
        onPatchProject={patchProject}
        onExtractFeatures={handleExtractFeatures}
      />

      {/* STEP 4: Train Model */}
      <Step4TrainModelCard
        batchSize={activeProject.batchSize}
        saveEveryEpoch={activeProject.saveEveryEpoch}
        totalEpoch={activeProject.totalEpoch}
        saveOnlyLatest={activeProject.saveOnlyLatest}
        saveEveryWeights={activeProject.saveEveryWeights}
        pretrained={activeProject.pretrained}
        freshTraining={activeProject.freshTraining}
        cacheDatasetGpu={activeProject.cacheDatasetGpu}
        checkpointing={activeProject.checkpointing}
        shutdownAfterFinish={activeProject.shutdownAfterFinish}
        customPretrained={activeProject.customPretrained}
        customPretrainedG={activeProject.customPretrainedG}
        customPretrainedD={activeProject.customPretrainedD}
        indexAlgorithm={activeProject.indexAlgorithm}
        agreedTerms={activeProject.agreedTerms}
        trainingOutput={activeProject.trainingOutput}
        isTraining={isTraining}
        trainingElapsedSec={trainingRun.elapsedSec}
        currentEpoch={trainingRun.epoch}
        currentStep={trainingRun.step}
        batchDone={trainingRun.batchDone}
        batchTotal={trainingRun.batchTotal}
        itPerSec={trainingRun.itPerSec}
        completionEstimate={completionEstimate}
        cliPreviewOpen={cliPreviewOpen}
        pipelineCommands={pipelineCommands}
        onToggleCliPreview={handleToggleCliPreview}
        onPatchProject={patchProject}
        onToggleTraining={handleToggleTraining}
        onGenerateIndex={handleGenerateIndex}
      />
    </div>
  );
});
