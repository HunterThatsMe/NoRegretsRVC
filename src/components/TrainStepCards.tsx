import React, { useMemo, useState } from 'react';
import {
  ChevronDown,
  ChevronRight,
  Clock,
  RefreshCw,
  RotateCcw,
  Terminal,
} from 'lucide-react';
import { ACCENT_HEX } from '../data/defaults';
import {
  AudioCuttingMethod,
  EmbedderModel,
  IndexAlgorithm,
  PitchExtractor,
  SamplingRate,
  TrainingCompletionEstimate,
  Vocoder,
  VoiceModelProject,
} from '../types/studio';
import { useI18n } from '../utils/i18n';
import {
  inferMatchingDUrl,
  isHuggingFaceRepoUrl,
  isPretrainedZipUrl,
  resolvePerGpuBatchSize,
} from '../utils/rvcTuning';
import { CustomPretrainedPair } from './CustomPretrainedPair';
import { NumberStepperInput } from './NumberStepperInput';

export const StepBadge: React.FC<{ num: number }> = ({ num }) => (
  <span
    className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-xs font-bold text-white tabular-nums"
    style={{ backgroundColor: ACCENT_HEX }}
  >
    {num}
  </span>
);

function getGpuInfoLines(gpuNumber: string): string {
  const parts = gpuNumber
    .split('-')
    .map((s) => s.trim())
    .filter(Boolean);
  if (parts.length === 0) return 'No GPU specified (CPU fallback).';
  return parts.map((id) => `${id}: Tesla T4 (14 GB)`).join('\n');
}

type PatchProjectFn = (patch: Partial<VoiceModelProject>) => void;

interface Step1Props {
  name: string;
  samplingRate: SamplingRate;
  vocoder: Vocoder;
  cpuCores: number;
  gpuNumber: string;
  onPatchProject: PatchProjectFn;
}

export const Step1ModelSettingsCard: React.FC<Step1Props> = React.memo(
  ({ name, samplingRate, vocoder, cpuCores, gpuNumber, onPatchProject }) => {
    const { t } = useI18n();
    const [step1AdvOpen, setStep1AdvOpen] = useState(false);
    const gpuInfoLines = useMemo(() => getGpuInfoLines(gpuNumber), [gpuNumber]);

    return (
      <section className="rounded-xl border border-[#47161D] bg-[#160709] p-6 space-y-5">
        <div className="flex items-start gap-3">
          <StepBadge num={1} />
          <div>
            <h3 className="font-display text-[17px] font-semibold text-[#FDF7F7]">
              {t('train.step1.title')}
            </h3>
            <p className="text-[13.5px] text-[#C89EA4] mt-0.5">
              {t('train.step1.desc')}
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label
              htmlFor="train-model-name"
              className="block text-[13.5px] font-semibold text-[#FDF7F7]"
            >
              Model Name
            </label>
            <p className="text-[13px] text-[#C89EA4] mb-1.5">
              Identifier used for saved checkpoints, index files, and log directories.
            </p>
            <input
              id="train-model-name"
              name="nr-model-name-field"
              type="text"
              value={name}
              placeholder="Enter model name (e.g. VocalModel-32k)"
              autoComplete="off"
              autoCorrect="off"
              autoCapitalize="none"
              spellCheck={false}
              data-lpignore="true"
              data-1p-ignore="true"
              onChange={(e) => onPatchProject({ name: e.target.value })}
              className="w-full rounded-lg bg-[#100506] border border-[#47161D] px-3 py-2 text-sm text-[#FDF7F7] placeholder-[#AD8288]/45 focus:outline-none focus:border-[#E51923]"
            />
          </div>

          <div>
            <label
              htmlFor="train-sampling-rate"
              className="block text-[13.5px] font-semibold text-[#FDF7F7]"
            >
              Sampling Rate
            </label>
            <p className="text-[13px] text-[#C89EA4] mb-1.5">
              Target audio resolution in Hz for preprocessed slices and model synthesis.
            </p>
            <select
              id="train-sampling-rate"
              value={samplingRate}
              onChange={(e) =>
                onPatchProject({ samplingRate: e.target.value as SamplingRate })
              }
              className="w-full rounded-lg bg-[#100506] border border-[#47161D] px-3 py-2 text-sm text-[#FDF7F7] focus:outline-none focus:border-[#E51923]"
            >
              <option value="32000">32000</option>
              <option value="40000">40000</option>
              <option value="48000">48000</option>
            </select>
          </div>
        </div>

        <div>
          <label
            htmlFor="train-vocoder"
            className="block text-[13.5px] font-semibold text-[#FDF7F7]"
          >
            Vocoder
          </label>
          <p className="text-[13px] text-[#C89EA4] mt-0.5 mb-2 leading-relaxed">
            Waveform synthesizer architecture: HiFi-GAN works across all RVC players; MRF HiFi-GAN boosts harmonic detail for Applio clients; RefineGAN maximizes clarity with extra VRAM demand.
          </p>
          <select
            id="train-vocoder"
            value={vocoder}
            onChange={(e) =>
              onPatchProject({ vocoder: e.target.value as Vocoder })
            }
            className="w-full rounded-lg bg-[#100506] border border-[#47161D] px-3 py-2 text-sm text-[#FDF7F7] focus:outline-none focus:border-[#E51923]"
          >
            <option value="HiFi-GAN">HiFi-GAN</option>
            <option value="MRF HiFi-GAN">MRF HiFi-GAN</option>
            <option value="RefineGAN">RefineGAN</option>
          </select>
        </div>

        {/* Step 1 Advanced Settings */}
        <div className="rounded-lg border border-[#47161D] bg-[#100506]">
          <button
            type="button"
            onClick={() => setStep1AdvOpen((v) => !v)}
            aria-expanded={step1AdvOpen}
            className="w-full flex items-center justify-between px-4 py-2.5 text-xs font-semibold text-[#DFBFC3] hover:text-white"
          >
            <span>Advanced Settings</span>
            {step1AdvOpen ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
          </button>
          {step1AdvOpen && (
            <div className="border-t border-[#47161D] p-4 space-y-4">
              {/* CPU Cores */}
              <div className="rounded-lg border border-[#47161D] bg-[#160709] p-3.5">
                <div className="flex flex-wrap items-center justify-between gap-3 mb-2">
                  <div>
                    <label
                      htmlFor="train-cpu-cores"
                      className="block text-[13.5px] font-semibold text-[#FDF7F7]"
                    >
                      CPU Cores
                    </label>
                    <p className="text-[13px] text-[#C89EA4] mt-0.5">
                      Parallel CPU worker threads allocated for audio slicing and pitch extraction. Using all 4 Kaggle vCPUs keeps preprocessing fast.
                    </p>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <NumberStepperInput
                      value={cpuCores}
                      min={1}
                      max={4}
                      ariaLabel="CPU Cores value"
                      unitName="CPU cores"
                      inputWidthClass="w-10"
                      onChange={(next) => onPatchProject({ cpuCores: next })}
                    />
                    <button
                      type="button"
                      onClick={() => onPatchProject({ cpuCores: 4 })}
                      aria-label="Reset CPU cores to 4"
                      className="rounded-lg border border-[#47161D] bg-[#200B0E] p-1.5 text-[#DFBFC3] hover:text-white hover:border-[#E51923]"
                    >
                      <RotateCcw className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>
                <input
                  id="train-cpu-cores"
                  type="range"
                  min={1}
                  max={4}
                  step={1}
                  value={cpuCores}
                  onChange={(e) =>
                    onPatchProject({ cpuCores: Number(e.target.value) })
                  }
                  className="w-full scarlet-slider"
                />
                <div className="flex justify-between text-xs font-mono text-[#C89EA4] mt-1 tabular-nums">
                  <span>1</span>
                  <span>4</span>
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label
                    htmlFor="train-gpu-number"
                    className="block text-[13.5px] font-semibold text-[#FDF7F7]"
                  >
                    GPU Number
                  </label>
                  <p className="text-[13px] text-[#C89EA4] mb-1.5">
                    Target CUDA device IDs separated by hyphens (for dual T4 sessions keep 0-1).
                  </p>
                  <input
                    id="train-gpu-number"
                    type="text"
                    value={gpuNumber}
                    onChange={(e) => {
                      const val = e.target.value;
                      const count =
                        val.split('-').filter((x) => x.trim()).length >= 2 ? 2 : 1;
                      onPatchProject({ gpuNumber: val, gpuCount: count });
                    }}
                    className="w-full rounded-lg bg-[#160709] border border-[#47161D] px-3 py-2 text-xs font-mono text-[#FDF7F7] focus:outline-none focus:border-[#E51923]"
                  />
                </div>
                <div>
                  <p className="text-xs font-semibold text-[#FDF7F7]">
                    GPU Information
                  </p>
                  <p className="text-[13px] text-[#C89EA4] mb-1.5">
                    Detected CUDA accelerator allocation for this session.
                  </p>
                  <pre className="rounded-lg border border-[#47161D] bg-[#160709] px-3 py-2 text-xs font-mono text-[#DFBFC3] leading-relaxed">
                    {gpuInfoLines}
                  </pre>
                </div>
              </div>
            </div>
          )}
        </div>
      </section>
    );
  }
);

interface Step2Props {
  datasetPath: string;
  datasetCreator: boolean;
  audioCutting: AudioCuttingMethod;
  chunkLengthSec: number;
  overlapLengthSec: number;
  truncateSilenceEnabled: boolean;
  silenceDurationMs: number;
  noiseFilter: boolean;
  loudnessNorm: boolean;
  loudnessTargetLufs: number;
  preprocessOutput: string;
  refreshStatus: string;
  onPatchProject: PatchProjectFn;
  onRefreshDatasets: () => void;
  onPreprocessDataset: () => void;
}

export const Step2PrepareDatasetCard: React.FC<Step2Props> = React.memo(
  ({
    datasetPath,
    datasetCreator,
    audioCutting,
    chunkLengthSec,
    overlapLengthSec,
    truncateSilenceEnabled,
    silenceDurationMs,
    noiseFilter,
    loudnessNorm,
    loudnessTargetLufs,
    preprocessOutput,
    refreshStatus,
    onPatchProject,
    onRefreshDatasets,
    onPreprocessDataset,
  }) => {
    const { t } = useI18n();
    const [step2AdvOpen, setStep2AdvOpen] = useState(false);

    return (
      <section className="rounded-xl border border-[#47161D] bg-[#160709] p-6 space-y-5">
        <div className="flex items-start gap-3">
          <StepBadge num={2} />
          <div>
            <h3 className="font-display text-[17px] font-semibold text-[#FDF7F7]">
              {t('train.step2.title')}
            </h3>
            <p className="text-[13.5px] text-[#C89EA4] mt-0.5">
              {t('train.step2.desc')}
            </p>
          </div>
        </div>

        <div>
          <label
            htmlFor="train-dataset-path"
            className="block text-[13.5px] font-semibold text-[#FDF7F7]"
          >
            Dataset Path
          </label>
          <p className="text-[13px] text-[#C89EA4] mb-1.5">
            Mounted Kaggle input path containing your dry vocal recordings.
          </p>
          <input
            id="train-dataset-path"
            type="text"
            value={datasetPath}
            onChange={(e) => onPatchProject({ datasetPath: e.target.value })}
            placeholder=""
            className="w-full rounded-lg bg-[#100506] border border-[#47161D] px-3 py-2 text-sm font-mono text-[#FDF7F7] placeholder-[#AD8288]/45 focus:outline-none focus:border-[#E51923]"
          />
          {!datasetPath.trim() && (
            <p
              data-testid="dataset-not-detected-note"
              className="text-xs text-[#F59E0B] mt-1.5"
            >
              No dataset detected yet — enter your mounted Kaggle dataset folder path above or click Refresh.
            </p>
          )}
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3">
          <label className="inline-flex items-center gap-2.5 text-xs font-medium text-[#DFBFC3] cursor-pointer">
            <input
              type="checkbox"
              checked={datasetCreator}
              onChange={(e) =>
                onPatchProject({ datasetCreator: e.target.checked })
              }
              className="h-4 w-4 rounded accent-[#E51923]"
            />
            <span>Dataset Creator</span>
          </label>

          <button
            type="button"
            onClick={onRefreshDatasets}
            className="inline-flex items-center gap-2 rounded-lg border border-[#47161D] bg-[#200B0E] px-3.5 py-2 text-xs font-medium text-[#FDF7F7] hover:border-[#E51923] whitespace-nowrap shrink-0"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            <span>{t('train.step2.refresh')}</span>
          </button>
        </div>

        {refreshStatus && (
          <p className="text-xs text-[#DFBFC3]" role="status">
            {refreshStatus}
          </p>
        )}

        {/* Step 2 Advanced Settings */}
        <div className="rounded-lg border border-[#47161D] bg-[#100506]">
          <button
            type="button"
            onClick={() => setStep2AdvOpen((v) => !v)}
            aria-expanded={step2AdvOpen}
            className="w-full flex items-center justify-between px-4 py-2.5 text-xs font-semibold text-[#DFBFC3] hover:text-white"
          >
            <span>Advanced Settings</span>
            {step2AdvOpen ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
          </button>
          {step2AdvOpen && (
            <div className="border-t border-[#47161D] p-4 space-y-4">
              <div>
                <label
                  htmlFor="train-audio-cutting"
                  className="block text-[13.5px] font-semibold text-[#FDF7F7]"
                >
                  Audio cutting
                </label>
                <p className="text-[13px] text-[#C89EA4] mt-0.5 mb-1.5">
                  Choose how long recordings are divided: Automatic detects quiet gaps dynamically, Simple cuts fixed chunks, and Skip leaves pre-split clips untouched.
                </p>
                <select
                  id="train-audio-cutting"
                  value={audioCutting}
                  onChange={(e) =>
                    onPatchProject({
                      audioCutting: e.target.value as AudioCuttingMethod,
                    })
                  }
                  className="w-full rounded-lg bg-[#160709] border border-[#47161D] px-3 py-2 text-xs text-[#FDF7F7] focus:outline-none focus:border-[#E51923]"
                >
                  <option value="Automatic">Automatic</option>
                  <option value="Simple">Simple</option>
                  <option value="Skip">Skip</option>
                </select>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="rounded-lg border border-[#47161D] bg-[#160709] p-3.5">
                  <div className="flex items-center justify-between gap-2 mb-1.5">
                    <div>
                      <label
                        htmlFor="train-chunk-len"
                        className="block text-[13.5px] font-semibold text-[#FDF7F7]"
                      >
                        Chunk length (sec)
                      </label>
                      <p className="text-[13px] text-[#C89EA4]">
                        Duration of each split audio window when Simple cutting is selected.
                      </p>
                    </div>
                    <div className="flex items-center gap-1.5 shrink-0">
                      <NumberStepperInput
                        value={chunkLengthSec}
                        min={0.5}
                        max={5}
                        step={0.1}
                        ariaLabel="Chunk length value"
                        unitName="chunk length"
                        inputWidthClass="w-12"
                        onChange={(next) => onPatchProject({ chunkLengthSec: next })}
                      />
                      <button
                        type="button"
                        onClick={() => onPatchProject({ chunkLengthSec: 3 })}
                        aria-label="Reset chunk length to 3"
                        className="rounded-md border border-[#47161D] bg-[#200B0E] p-1 text-[#DFBFC3] hover:text-white"
                      >
                        <RotateCcw className="h-3 w-3" />
                      </button>
                    </div>
                  </div>
                  <input
                    id="train-chunk-len"
                    type="range"
                    min={0.5}
                    max={5}
                    step={0.1}
                    value={chunkLengthSec}
                    onChange={(e) =>
                      onPatchProject({ chunkLengthSec: Number(e.target.value) })
                    }
                    className="w-full scarlet-slider"
                  />
                  <div className="flex justify-between text-xs font-mono text-[#C89EA4] mt-1 tabular-nums">
                    <span>0.5</span>
                    <span>5</span>
                  </div>
                </div>

                <div className="rounded-lg border border-[#47161D] bg-[#160709] p-3.5">
                  <div className="flex items-center justify-between gap-2 mb-1.5">
                    <div>
                      <label
                        htmlFor="train-overlap-len"
                        className="block text-[13.5px] font-semibold text-[#FDF7F7]"
                      >
                        Overlap length (sec)
                      </label>
                      <p className="text-[13px] text-[#C89EA4]">
                        Crossfade overlap between consecutive Simple slices to avoid abrupt cuts.
                      </p>
                    </div>
                    <div className="flex items-center gap-1.5 shrink-0">
                      <NumberStepperInput
                        value={overlapLengthSec}
                        min={0}
                        max={0.4}
                        step={0.05}
                        ariaLabel="Overlap length value"
                        unitName="overlap length"
                        inputWidthClass="w-12"
                        onChange={(next) => onPatchProject({ overlapLengthSec: next })}
                      />
                      <button
                        type="button"
                        onClick={() =>
                          onPatchProject({ overlapLengthSec: 0.3 })
                        }
                        aria-label="Reset overlap length to 0.3"
                        className="rounded-md border border-[#47161D] bg-[#200B0E] p-1 text-[#DFBFC3] hover:text-white"
                      >
                        <RotateCcw className="h-3 w-3" />
                      </button>
                    </div>
                  </div>
                  <input
                    id="train-overlap-len"
                    type="range"
                    min={0}
                    max={0.4}
                    step={0.05}
                    value={overlapLengthSec}
                    onChange={(e) =>
                      onPatchProject({
                        overlapLengthSec: Number(e.target.value),
                      })
                    }
                    className="w-full scarlet-slider"
                  />
                  <div className="flex justify-between text-xs font-mono text-[#C89EA4] mt-1 tabular-nums">
                    <span>0</span>
                    <span>0.4</span>
                  </div>
                </div>
              </div>

              {/* Compact Silence Truncation strip (off by default) */}
              <div className="rounded-lg border border-[#47161D] bg-[#160709] px-3.5 py-2.5 flex flex-wrap items-center justify-between gap-3">
                <label className="inline-flex items-center gap-2.5 text-xs text-[#FDF7F7] cursor-pointer">
                  <input
                    type="checkbox"
                    checked={truncateSilenceEnabled}
                    onChange={(e) =>
                      onPatchProject({
                        truncateSilenceEnabled: e.target.checked,
                      })
                    }
                    className="h-4 w-4 rounded accent-[#E51923]"
                  />
                  <span>Truncate silence</span>
                </label>

                <div className="flex items-center gap-2">
                  <label
                    htmlFor="train-silence-duration"
                    className="text-[13px] text-[#C89EA4]"
                  >
                    Silence duration (ms)
                  </label>
                  <NumberStepperInput
                    id="train-silence-duration"
                    value={silenceDurationMs}
                    min={50}
                    max={3000}
                    step={50}
                    disabled={!truncateSilenceEnabled}
                    ariaLabel="Silence duration in ms"
                    unitName="silence duration"
                    inputWidthClass="w-14"
                    onChange={(next) =>
                      onPatchProject({ silenceDurationMs: next })
                    }
                  />
                </div>
              </div>

              {/* Noise Filter (off) & Loudness Normalization (-20 LUFS ON by default) */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <label className="flex items-start gap-2.5 rounded-lg border border-[#47161D] bg-[#160709] p-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={noiseFilter}
                    onChange={(e) =>
                      onPatchProject({ noiseFilter: e.target.checked })
                    }
                    className="mt-0.5 h-4 w-4 rounded accent-[#E51923]"
                  />
                  <div>
                    <span className="block text-[13.5px] font-semibold text-[#FDF7F7]">
                      Noise filter
                    </span>
                    <span className="block text-[13px] text-[#C89EA4] mt-0.5">
                      Applies spectral denoising to raw clips. Keep unchecked if your vocals are already clean or denoised.
                    </span>
                  </div>
                </label>

                <label className="flex items-start gap-2.5 rounded-lg border border-[#47161D] bg-[#160709] p-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={loudnessNorm}
                    onChange={(e) =>
                      onPatchProject({ loudnessNorm: e.target.checked })
                    }
                    className="mt-0.5 h-4 w-4 rounded accent-[#E51923]"
                  />
                  <div>
                    <span className="block text-[13.5px] font-semibold text-[#FDF7F7]">
                      Loudness normalization ({loudnessTargetLufs} LUFS)
                    </span>
                    <span className="block text-[13px] text-[#C89EA4] mt-0.5">
                      On by default: balances clips to a clean, transparent {loudnessTargetLufs} LUFS without loudness distortion.
                    </span>
                  </div>
                </label>
              </div>
            </div>
          )}
        </div>

        <div>
          <p className="text-xs font-medium text-[#DFBFC3] mb-1.5">
            Preprocess output
          </p>
          <div className="rounded-lg border border-[#47161D] bg-[#100506] px-3.5 py-2.5 text-xs font-mono text-[#DFBFC3]">
            {preprocessOutput || 'The output information will be displayed here.'}
          </div>
        </div>

        <button
          type="button"
          onClick={onPreprocessDataset}
          className="w-full rounded-lg py-2.5 text-sm font-semibold text-white transition-colors hover:bg-[#FF2B35]"
          style={{ backgroundColor: ACCENT_HEX }}
        >
          {t('train.step2.btn')}
        </button>
      </section>
    );
  }
);

interface Step3Props {
  pitchExtractor: PitchExtractor;
  embedder: EmbedderModel;
  silentTrainingFiles: number;
  extractOutput: string;
  onPatchProject: PatchProjectFn;
  onExtractFeatures: () => void;
}

export const Step3ExtractFeaturesCard: React.FC<Step3Props> = React.memo(
  ({
    pitchExtractor,
    embedder,
    silentTrainingFiles,
    extractOutput,
    onPatchProject,
    onExtractFeatures,
  }) => {
    const { t } = useI18n();
    return (
    <section className="cv-card rounded-xl border border-[#47161D] bg-[#160709] p-6 space-y-5">
      <div className="flex items-start gap-3">
        <StepBadge num={3} />
        <div>
          <h3 className="font-display text-[17px] font-semibold text-[#FDF7F7]">
            {t('train.step3.title')}
          </h3>
          <p className="text-[13.5px] text-[#C89EA4] mt-0.5">
            {t('train.step3.desc')}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label
            htmlFor="train-pitch-extractor"
            className="block text-[13.5px] font-semibold text-[#FDF7F7]"
          >
            Pitch extraction algorithm
          </label>
          <p className="text-[13px] text-[#C89EA4] mb-1.5">
            F0 tracker used to capture vocal melody. RMVPE offers the sharpest accuracy across speech and singing.
          </p>
          <select
            id="train-pitch-extractor"
            value={pitchExtractor}
            onChange={(e) =>
              onPatchProject({
                pitchExtractor: e.target.value as PitchExtractor,
              })
            }
            className="w-full rounded-lg bg-[#100506] border border-[#47161D] px-3 py-2 text-sm text-[#FDF7F7] focus:outline-none focus:border-[#E51923]"
          >
            <option value="rmvpe">rmvpe</option>
            <option value="fcpe">fcpe</option>
            <option value="swift-f0">swift-f0</option>
            <option value="crepe">crepe</option>
          </select>
        </div>

        <div>
          <label
            htmlFor="train-embedder"
            className="block text-[13.5px] font-semibold text-[#FDF7F7]"
          >
            Embedder Model
          </label>
          <p className="text-[13px] text-[#C89EA4] mb-1.5">
            Speech encoder that extracts phonetic timbre representations from audio.
          </p>
          <select
            id="train-embedder"
            value={embedder}
            onChange={(e) =>
              onPatchProject({ embedder: e.target.value as EmbedderModel })
            }
            className="w-full rounded-lg bg-[#100506] border border-[#47161D] px-3 py-2 text-sm text-[#FDF7F7] focus:outline-none focus:border-[#E51923]"
          >
            <option value="contentvec">contentvec</option>
            <option value="spin">spin</option>
            <option value="chinese-hubert-base">chinese-hubert-base</option>
          </select>
        </div>
      </div>

      <div className="rounded-lg border border-[#47161D] bg-[#100506] p-4">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-2">
          <div>
            <label
              htmlFor="train-silent-files"
              className="block text-[13.5px] font-semibold text-[#FDF7F7]"
            >
              Silent training files
            </label>
            <p className="text-[13px] text-[#C89EA4] mt-0.5">
              Injects blank silent samples to help prevent breath artifacts during pauses. Leave at 0 if your dataset already includes natural quiet gaps.
            </p>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <NumberStepperInput
              value={silentTrainingFiles}
              min={0}
              max={10}
              step={1}
              ariaLabel="Silent training files value"
              unitName="silent training files"
              inputWidthClass="w-10"
              onChange={(next) => onPatchProject({ silentTrainingFiles: next })}
            />
            <button
              type="button"
              onClick={() => onPatchProject({ silentTrainingFiles: 0 })}
              aria-label="Reset silent training files to 0"
              className="rounded-md border border-[#47161D] bg-[#200B0E] p-1.5 text-[#DFBFC3] hover:text-white"
            >
              <RotateCcw className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
        <input
          id="train-silent-files"
          type="range"
          min={0}
          max={10}
          step={1}
          value={silentTrainingFiles}
          onChange={(e) =>
            onPatchProject({ silentTrainingFiles: Number(e.target.value) })
          }
          className="w-full scarlet-slider"
        />
        <div className="flex justify-between text-xs font-mono text-[#C89EA4] mt-1 tabular-nums">
          <span>0</span>
          <span>10</span>
        </div>
      </div>

      <div>
        <p className="text-xs font-medium text-[#DFBFC3] mb-1.5">
          Feature extraction output
        </p>
        <div className="rounded-lg border border-[#47161D] bg-[#100506] px-3.5 py-2.5 text-xs font-mono text-[#DFBFC3]">
          {extractOutput || 'The output information will be displayed here.'}
        </div>
      </div>

      <button
        type="button"
        onClick={onExtractFeatures}
        className="w-full rounded-lg py-2.5 text-sm font-semibold text-white transition-colors hover:bg-[#FF2B35]"
        style={{ backgroundColor: ACCENT_HEX }}
      >
        {t('train.step3.btn')}
      </button>
    </section>
    );
  }
);

interface Step4Props {
  batchSize: number;
  gpuCount?: 1 | 2;
  saveEveryEpoch: number;
  totalEpoch: number;
  turboTraining?: boolean;
  onToggleTurboTraining?: (enabled: boolean) => void;
  saveOnlyLatest: boolean;
  saveEveryWeights: boolean;
  pretrained: boolean;
  freshTraining: boolean;
  cacheDatasetGpu: boolean;
  checkpointing: boolean;
  shutdownAfterFinish: boolean;
  customPretrained: boolean;
  customPretrainedG: string;
  customPretrainedD: string;
  indexAlgorithm: IndexAlgorithm;
  agreedTerms: boolean;
  trainingOutput: string;
  isTraining: boolean;
  trainingElapsedSec: number;
  currentEpoch: number;
  currentStep: number;
  batchDone: number;
  batchTotal: number;
  itPerSec: number;
  completionEstimate: TrainingCompletionEstimate | null;
  cliPreviewOpen: boolean;
  pipelineCommands: string;
  onToggleCliPreview: () => void;
  onPatchProject: PatchProjectFn;
  onToggleTraining: () => void;
  onGenerateIndex: () => void;
}

const STEP4_TOGGLES: {
  key:
    | 'saveOnlyLatest'
    | 'saveEveryWeights'
    | 'pretrained'
    | 'freshTraining'
    | 'cacheDatasetGpu'
    | 'checkpointing'
    | 'shutdownAfterFinish'
    | 'customPretrained';
  title: string;
  desc: string;
}[] = [
  {
    key: 'saveOnlyLatest',
    title: 'Save Only Latest',
    desc: 'Overwrites previous G and D resume states on each save to prevent Kaggle disk overflow.',
  },
  {
    key: 'saveEveryWeights',
    title: 'Save Every Weights',
    desc: 'Exports a standalone inference .pth weight file every time the save interval triggers.',
  },
  {
    key: 'pretrained',
    title: 'Pretrained',
    desc: 'Initializes training from base RVC weights for faster convergence and cleaner timbre.',
  },
  {
    key: 'freshTraining',
    title: 'Fresh Training',
    desc: 'Clears prior checkpoints and TensorBoard history inside the model folder before starting.',
  },
  {
    key: 'cacheDatasetGpu',
    title: 'Cache Dataset in GPU',
    desc: 'Pins sliced audio tensors directly in VRAM to reduce disk reads on smaller datasets.',
  },
  {
    key: 'checkpointing',
    title: 'Checkpointing',
    desc: 'Trades compute speed for lower peak VRAM consumption via gradient checkpointing.',
  },
  {
    key: 'shutdownAfterFinish',
    title: 'Shutdown after finishing',
    desc: 'Powers down the runtime environment automatically once the final epoch wraps up.',
  },
  {
    key: 'customPretrained',
    title: 'Custom Pretrains',
    desc: 'Points training to your own custom G and D base weights instead of the built-in defaults.',
  },
];

export const Step4TrainModelCard: React.FC<Step4Props> = React.memo((props) => {
  const {
    batchSize,
    gpuCount = 2,
    saveEveryEpoch,
    totalEpoch,
    turboTraining = true,
    onToggleTurboTraining,
    customPretrained,
    customPretrainedG,
    customPretrainedD,
    indexAlgorithm,
    agreedTerms,
    trainingOutput,
    isTraining,
    trainingElapsedSec,
    currentEpoch,
    currentStep,
    batchDone,
    batchTotal,
    itPerSec,
    completionEstimate,
    cliPreviewOpen,
    pipelineCommands,
    onToggleCliPreview,
    onPatchProject,
    onToggleTraining,
    onGenerateIndex,
  } = props;
  const { t } = useI18n();
  const [step4AdvOpen, setStep4AdvOpen] = useState(false);
  const [pretrainRefreshMsg, setPretrainRefreshMsg] = useState('');
  const batchAlloc = useMemo(
    () => resolvePerGpuBatchSize(batchSize, gpuCount),
    [batchSize, gpuCount]
  );
  const safeBatchTotal = Math.max(1, batchTotal);
  const batchPct = Math.min(100, Math.round((batchDone / safeBatchTotal) * 100));
  const etaSec = Math.max(0, (safeBatchTotal - batchDone) / Math.max(0.1, itPerSec));

  const handleRefreshPretrains = () => {
    const gTrim = customPretrainedG.trim();
    const dTrim = customPretrainedD.trim();
    if (!gTrim && !dTrim) {
      setPretrainRefreshMsg(
        'Scanned rvc/models/pretraineds/custom/. Paste a link or upload a G/D pair to register custom pretrains.'
      );
      return;
    }
    if (gTrim && !dTrim) {
      if (isHuggingFaceRepoUrl(gTrim) || isPretrainedZipUrl(gTrim)) {
        setPretrainRefreshMsg(
          'Synced custom source to rvc/models/pretraineds/custom/. Matching G & D weights will be auto-discovered on boot.'
        );
        return;
      }
      const inferred = inferMatchingDUrl(gTrim);
      if (inferred) {
        onPatchProject({ customPretrainedD: inferred });
        setPretrainRefreshMsg(
          `Refreshed list and auto-discovered matching D weight (${inferred.split('?')[0].split('/').pop()}).`
        );
        return;
      }
    }
    setPretrainRefreshMsg(
      'Refreshed custom pretrain list in rvc/models/pretraineds/custom/.'
    );
  };

  return (
    <section className="cv-card rounded-xl border border-[#47161D] bg-[#160709] p-6 space-y-5">
      <div className="flex items-start gap-3">
        <StepBadge num={4} />
        <div>
          <h3 className="font-display text-[17px] font-semibold text-[#FDF7F7]">
            {t('train.step4.title')}
          </h3>
          <p className="text-[13.5px] text-[#C89EA4] mt-0.5">
            {t('train.step4.desc')}
          </p>
        </div>
      </div>

      <div className="space-y-4">
        {/* Batch Size (default 2) */}
        <div className="rounded-lg border border-[#47161D] bg-[#100506] p-4">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-2">
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <label
                  htmlFor="train-batch-size"
                  className="block text-[13.5px] font-semibold text-[#FDF7F7]"
                >
                  Batch Size
                </label>
                <span
                  data-testid="effective-batch-badge"
                  className="rounded-md border border-[#E51923]/55 bg-[#200B0E] px-2 py-0.5 text-[11px] font-mono font-semibold text-[#FDF7F7] tabular-nums"
                >
                  {gpuCount > 1
                    ? `Batch ${batchSize} · Dual T4 Effective: ${batchAlloc.totalEffectiveBatchSize}`
                    : `Effective Batch: ${batchSize}`}
                </span>
              </div>
              <p className="text-[13px] text-[#C89EA4] mt-0.5">
                {gpuCount > 1
                  ? `Per-GPU batch size. On Dual T4 GPUs, effective total batch is doubled (${batchSize} -> ${batchAlloc.totalEffectiveBatchSize}).`
                  : `Per-GPU batch size (${batchSize} effective).`}
              </p>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <NumberStepperInput
                value={batchSize}
                min={1}
                max={64}
                ariaLabel="Batch Size value"
                unitName="batch size"
                inputWidthClass="w-12"
                onChange={(next) => onPatchProject({ batchSize: next })}
              />
              <button
                type="button"
                onClick={() => onPatchProject({ batchSize: 2 })}
                aria-label="Reset batch size to 2"
                className="rounded-lg border border-[#47161D] bg-[#200B0E] p-1.5 text-[#DFBFC3] hover:text-white hover:border-[#E51923]"
              >
                <RotateCcw className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
          <input
            id="train-batch-size"
            type="range"
            min={1}
            max={64}
            step={1}
            value={batchSize}
            onChange={(e) =>
              onPatchProject({ batchSize: Number(e.target.value) })
            }
            className="w-full scarlet-slider"
          />
          <div className="flex justify-between text-xs font-mono text-[#C89EA4] mt-1 tabular-nums">
            <span>1</span>
            <span>64</span>
          </div>
        </div>

        {/* Save Every Epoch (default 5) */}
        <div className="rounded-lg border border-[#47161D] bg-[#100506] p-4">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-2">
            <div>
              <label
                htmlFor="train-save-every"
                className="block text-[13.5px] font-semibold text-[#FDF7F7]"
              >
                Save Every Epoch
              </label>
              <p className="text-[13px] text-[#C89EA4] mt-0.5">
                Interval of completed epochs between saved checkpoint snapshots.
              </p>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <NumberStepperInput
                value={saveEveryEpoch}
                min={1}
                max={100}
                ariaLabel="Save Every Epoch value"
                unitName="save every epoch"
                inputWidthClass="w-12"
                onChange={(next) => onPatchProject({ saveEveryEpoch: next })}
              />
              <button
                type="button"
                onClick={() => onPatchProject({ saveEveryEpoch: 5 })}
                aria-label="Reset save every epoch to 5"
                className="rounded-lg border border-[#47161D] bg-[#200B0E] p-1.5 text-[#DFBFC3] hover:text-white hover:border-[#E51923]"
              >
                <RotateCcw className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
          <input
            id="train-save-every"
            type="range"
            min={1}
            max={100}
            step={1}
            value={saveEveryEpoch}
            onChange={(e) =>
              onPatchProject({ saveEveryEpoch: Number(e.target.value) })
            }
            className="w-full scarlet-slider"
          />
          <div className="flex justify-between text-xs font-mono text-[#C89EA4] mt-1 tabular-nums">
            <span>1</span>
            <span>100</span>
          </div>
        </div>

        {/* Total Epoch (default 200) */}
        <div className="rounded-lg border border-[#47161D] bg-[#100506] p-4">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-2">
            <div>
              <label
                htmlFor="train-total-epoch"
                className="block text-[13.5px] font-semibold text-[#FDF7F7]"
              >
                Total Epoch
              </label>
              <p className="text-[13px] text-[#C89EA4] mt-0.5">
                Target number of full dataset passes before training stops automatically.
              </p>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <NumberStepperInput
                value={totalEpoch}
                min={1}
                max={10000}
                ariaLabel="Total Epoch value"
                unitName="total epoch"
                inputWidthClass="w-16"
                onChange={(next) => onPatchProject({ totalEpoch: next })}
              />
              <button
                type="button"
                onClick={() => onPatchProject({ totalEpoch: 200 })}
                aria-label="Reset total epoch to 200"
                className="rounded-lg border border-[#47161D] bg-[#200B0E] p-1.5 text-[#DFBFC3] hover:text-white hover:border-[#E51923]"
              >
                <RotateCcw className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
          <input
            id="train-total-epoch"
            type="range"
            min={1}
            max={10000}
            step={1}
            value={totalEpoch}
            onChange={(e) =>
              onPatchProject({ totalEpoch: Number(e.target.value) })
            }
            className="w-full scarlet-slider"
          />
          <div className="flex justify-between text-xs font-mono text-[#C89EA4] mt-1 tabular-nums">
            <span>1</span>
            <span>10000</span>
          </div>
        </div>
      </div>

      {/* Step 4 Advanced Settings */}
      <div className="rounded-lg border border-[#47161D] bg-[#100506]">
        <button
          type="button"
          onClick={() => setStep4AdvOpen((v) => !v)}
          aria-expanded={step4AdvOpen}
          className="w-full flex items-center justify-between px-4 py-2.5 text-xs font-semibold text-[#DFBFC3] hover:text-white"
        >
          <span>{t('train.step4.adv')}</span>
          {step4AdvOpen ? (
            <ChevronDown className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
        </button>
        {step4AdvOpen && (
          <div className="border-t border-[#47161D] p-4 space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {STEP4_TOGGLES.map((opt) => (
                <label
                  key={opt.key}
                  className="flex items-start gap-2.5 rounded-lg border border-[#47161D] bg-[#160709] p-3 cursor-pointer"
                >
                  <input
                    type="checkbox"
                    checked={props[opt.key]}
                    onChange={(e) =>
                      onPatchProject({ [opt.key]: e.target.checked })
                    }
                    className="mt-0.5 h-4 w-4 rounded accent-[#E51923]"
                  />
                  <div>
                    <span className="block text-[13.5px] font-semibold text-[#FDF7F7]">
                      {opt.title}
                    </span>
                    <span className="block text-[13px] text-[#C89EA4] mt-0.5 leading-snug">
                      {opt.desc}
                    </span>
                  </div>
                </label>
              ))}
              <label className="flex items-start gap-2.5 rounded-lg border border-[#E51923]/55 bg-[#160709] p-3 cursor-pointer md:col-span-2">
                <input
                  type="checkbox"
                  checked={turboTraining}
                  onChange={(e) => onToggleTurboTraining?.(e.target.checked)}
                  className="mt-0.5 h-4 w-4 rounded accent-[#E51923]"
                />
                <div>
                  <span className="block text-[13.5px] font-semibold text-[#FDF7F7]">
                    Lossless Turbo Training
                  </span>
                  <span className="block text-[13px] text-[#C89EA4] mt-0.5 leading-snug">
                    Always on by default. Enables cuDNN autotune, IEEE-exact foreach AdamW steps, pinned memory caching, and atomic background saving without degrading model precision.
                  </span>
                </div>
              </label>
            </div>

            {customPretrained && (
              <div className="rounded-lg border border-[#47161D] bg-[#160709] p-4 space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="text-xs font-semibold text-[#FDF7F7]">
                    {t('customPretrain.title')}
                  </span>
                  <button
                    type="button"
                    onClick={handleRefreshPretrains}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-[#47161D] bg-[#200B0E] px-3 py-1.5 text-xs font-medium text-[#FDF7F7] hover:border-[#E51923] whitespace-nowrap shrink-0"
                  >
                    <RefreshCw className="h-3.5 w-3.5 text-[#E51923]" />
                    <span>{t('train.step4.refreshPretrains')}</span>
                  </button>
                </div>
                {pretrainRefreshMsg && (
                  <p className="text-[11px] font-mono text-[#DFBFC3]" role="status">
                    {pretrainRefreshMsg}
                  </p>
                )}
                <CustomPretrainedPair
                  idPrefix="step4-custom-pretrained"
                  gValue={customPretrainedG}
                  dValue={customPretrainedD}
                  onChangeG={(next) => onPatchProject({ customPretrainedG: next })}
                  onChangeD={(next) => onPatchProject({ customPretrainedD: next })}
                />
              </div>
            )}

            <div>
              <label
                htmlFor="train-index-algorithm"
                className="block text-[13.5px] font-semibold text-[#FDF7F7]"
              >
                Index Algorithm
              </label>
              <p className="text-[13px] text-[#C89EA4] mt-0.5 mb-1.5">
                Vector clustering method used when building your .index retrieval file. Auto selects the fastest option for your dataset size.
              </p>
              <select
                id="train-index-algorithm"
                value={indexAlgorithm}
                onChange={(e) =>
                  onPatchProject({
                    indexAlgorithm: e.target.value as IndexAlgorithm,
                  })
                }
                className="w-full rounded-lg bg-[#160709] border border-[#47161D] px-3 py-2 text-xs text-[#FDF7F7] focus:outline-none focus:border-[#E51923]"
              >
                <option value="Auto">Auto</option>
                <option value="Faiss">Faiss</option>
                <option value="KMeans">KMeans</option>
              </select>
            </div>
          </div>
        )}
      </div>

      {/* Terms of use (automatically checked) */}
      <div className="rounded-lg border border-[#47161D] bg-[#100506] p-4">
        <label className="flex items-start gap-3 cursor-pointer">
          <input
            type="checkbox"
            checked={agreedTerms}
            onChange={(e) =>
              onPatchProject({ agreedTerms: e.target.checked })
            }
            className="mt-0.5 h-4 w-4 rounded accent-[#E51923]"
          />
          <div>
            <span className="block text-[13.5px] font-semibold text-[#FDF7F7]">
              I agree to the terms of use
            </span>
            <span className="block text-[13px] text-[#C89EA4] mt-0.5">
              Confirm that your training audio follows the usage guidelines for NoRegretsRVC.
            </span>
          </div>
        </label>
      </div>

      {isTraining && (
        <div
          aria-label="Live training progress"
          className="rounded-lg border border-[#E51923]/50 bg-[#200B0E] px-4 py-3 space-y-2"
        >
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
            <span className="text-[#FDF7F7] font-semibold tabular-nums">
              Epoch {currentEpoch}/{totalEpoch} · Step {currentStep}
            </span>
            <span className="text-[#E51923] font-semibold tabular-nums">
              Running · {trainingElapsedSec.toFixed(1)}s
            </span>
          </div>

          <div className="h-1.5 w-full rounded-full bg-[#100506] overflow-hidden border border-[#47161D]">
            <div
              className="h-full bg-[#E51923] transition-all duration-150"
              style={{ width: `${batchPct}%` }}
            />
          </div>

          <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono text-[#DFBFC3] tabular-nums">
            <span>
              Batch {batchDone}/{safeBatchTotal} ({batchPct}%) · {itPerSec.toFixed(2)} it/s
            </span>
            <span className="text-[#AD8288]">Epoch ETA ~{etaSec.toFixed(1)}s</span>
          </div>
        </div>
      )}

      <div>
        <p className="text-xs font-medium text-[#DFBFC3] mb-1.5">
          Training output
        </p>
        <div className="rounded-lg border border-[#47161D] bg-[#100506] px-3.5 py-2.5 text-xs font-mono text-[#DFBFC3]">
          {trainingOutput || 'The output information will be displayed here.'}
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <button
          type="button"
          onClick={onToggleTraining}
          className="rounded-lg py-2.5 px-4 text-sm font-semibold text-white transition-colors hover:bg-[#FF2B35] whitespace-nowrap"
          style={{ backgroundColor: isTraining ? '#991B1B' : ACCENT_HEX }}
        >
          {isTraining ? t('train.step4.stop') : t('train.step4.start')}
        </button>
        <button
          type="button"
          onClick={onGenerateIndex}
          className="rounded-lg border border-[#47161D] bg-[#200B0E] py-2.5 px-4 text-sm font-semibold text-[#FDF7F7] hover:border-[#E51923] whitespace-nowrap"
        >
          {t('train.step4.index')}
        </button>
      </div>

      {isTraining && completionEstimate && (
        <section
          aria-label={t('train.eta.title')}
          className="rounded-xl border border-[#47161D] bg-[#100506] p-4 space-y-3"
        >
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-[#E51923] shrink-0" />
              <span className="text-xs font-semibold text-[#FDF7F7]">
                {t('train.eta.title')}
              </span>
              <span className="text-[11px] font-mono text-[#AD8288]">
                {t('train.eta.live')}
              </span>
            </div>

            <div className="flex items-center gap-2.5">
              <span className="text-[11px] font-medium text-[#AD8288]">
                {t('train.eta.accuracy')}
              </span>
              <div className="h-1.5 w-20 rounded-full bg-[#160709] border border-[#47161D] overflow-hidden">
                <div
                  className="h-full transition-all duration-300"
                  style={{
                    width: `${completionEstimate.accuracyPercent}%`,
                    backgroundColor:
                      completionEstimate.accuracyPercent >= 75
                        ? '#10B981'
                        : completionEstimate.accuracyPercent >= 50
                          ? '#F59E0B'
                          : ACCENT_HEX,
                  }}
                />
              </div>
              <span className="font-mono text-xs font-semibold text-[#FDF7F7] tabular-nums">
                {completionEstimate.accuracyPercent}%
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
            <div className="rounded-lg border border-[#47161D] bg-[#160709] px-3 py-2">
              <div className="text-[11px] uppercase tracking-wider text-[#C89EA4]">
                {t('train.eta.remaining')}
              </div>
              <div className="mt-0.5 text-sm font-semibold text-[#FDF7F7] tabular-nums">
                Finishes in {completionEstimate.remainingLabel}
              </div>
            </div>

            <div className="rounded-lg border border-[#47161D] bg-[#160709] px-3 py-2">
              <div className="text-[11px] uppercase tracking-wider text-[#C89EA4]">
                {t('train.eta.localTime')}
              </div>
              <div className="mt-0.5 text-sm font-semibold text-[#FDF7F7] tabular-nums">
                {completionEstimate.finishTimeLabel}
              </div>
            </div>

            <div className="rounded-lg border border-[#47161D] bg-[#160709] px-3 py-2">
              <div className="text-[11px] uppercase tracking-wider text-[#C89EA4]">
                {t('train.eta.finishDay')}
              </div>
              <div className="mt-0.5 text-sm font-semibold text-[#FDF7F7] tabular-nums">
                {completionEstimate.finishDayLabel}
              </div>
            </div>
          </div>

          <p className="text-[13px] text-[#C89EA4] leading-relaxed">
            {completionEstimate.disclaimer}
          </p>
        </section>
      )}
      <div className="rounded-lg border border-[#47161D] bg-[#100506]">
        <button
          type="button"
          onClick={onToggleCliPreview}
          className="w-full flex items-center justify-between px-4 py-2.5 text-xs font-medium text-[#AD8288] hover:text-[#FDF7F7]"
        >
          <span className="inline-flex items-center gap-2">
            <Terminal className="h-3.5 w-3.5" />
            <span>{t('train.step4.cli')}</span>
          </span>
          {cliPreviewOpen ? (
            <ChevronDown className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
        </button>
        {cliPreviewOpen && (
          <pre className="border-t border-[#47161D] p-4 text-[11px] font-mono text-[#DFBFC3] overflow-x-auto leading-relaxed">
            {pipelineCommands}
          </pre>
        )}
      </div>
    </section>
  );
});
