import React, { useMemo, useRef, useState } from 'react';
import {
  Activity,
  Award,
  ChevronDown,
  ChevronRight,
  ExternalLink,
  FileUp,
  RotateCcw,
  Sparkles,
} from 'lucide-react';
import { AlertTriangle, CheckCircle2, Clock } from 'lucide-react';
import {
  REPO_README_URL,
  SAMPLE_OVERTRAIN_LOG,
  SAMPLE_TRAINING_LOG,
} from '../data/defaults';
import {
  ModelSoundFeedback,
  TrainingAnalysis,
  VoiceModelProject,
} from '../types/studio';
import { analyzeTrainingZones, parseTrainingNotebookOrLog } from '../utils/logParser';
import { resolvePerGpuBatchSize, sanitizeModelName } from '../utils/rvcTuning';
import { useI18n } from '../utils/i18n';
import { LossChart } from './LossChart';

interface TensorBoardViewProps {
  activeProject: VoiceModelProject;
  lastAnalysis: TrainingAnalysis | null;
  onAnalysisChange: (analysis: TrainingAnalysis | null) => void;
}

const SOUND_FEEDBACK_OPTIONS: { id: ModelSoundFeedback; label: string }[] = [
  { id: 'untested', label: 'Not auditioned yet' },
  { id: 'undertrained', label: 'Undertrained (robotic / flat)' },
  { id: 'clean', label: 'Clean & natural' },
  { id: 'overtrained', label: 'Overtrained (metallic / harsh)' },
];

export const TensorBoardView: React.FC<TensorBoardViewProps> = React.memo(({
  activeProject,
  lastAnalysis,
  onAnalysisChange,
}) => {
  const { t } = useI18n();
  const [soundCheck, setSoundCheck] = useState<ModelSoundFeedback>('untested');
  const [logPanelOpen, setLogPanelOpen] = useState(false);
  const [logInput, setLogInput] = useState('');
  const [parseError, setParseError] = useState('');
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const zoneInsight = useMemo(
    () => analyzeTrainingZones(lastAnalysis, activeProject, soundCheck),
    [lastAnalysis, activeProject, soundCheck]
  );

  const runParse = (raw: string) => {
    const parsed = parseTrainingNotebookOrLog(raw);
    if (!parsed) {
      setParseError(
        'Could not detect RVC training metrics. Paste Applio/Kaggle console output or upload a .ipynb / .log file.'
      );
      return;
    }
    setParseError('');
    onAnalysisChange(parsed);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const text = typeof reader.result === 'string' ? reader.result : '';
      setLogInput(text);
      runParse(text);
    };
    reader.readAsText(file);
  };

  const batchAlloc = resolvePerGpuBatchSize(
    activeProject.batchSize,
    activeProject.gpuCount,
    activeProject.batchParity1Gpu ?? true
  );
  const effectiveBatch = batchAlloc.totalEffectiveBatchSize;
  const activeModelTitle = activeProject.name.trim() || 'Untitled Model';
  const safeModelSlug = sanitizeModelName(activeProject.name);
  const isTrained = Boolean(lastAnalysis && zoneInsight.confidence !== 'untrained');
  const observedLossCount =
    lastAnalysis?.points.filter((p) => p.lowestLoss !== null).length ?? 0;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4 px-1">
        <div>
          <h2 className="font-display text-2xl font-bold text-[#FDF7F7] tracking-tight">
            {t('diagnostics.title')}
          </h2>
          <p className="text-[15px] text-[#DFBFC3] mt-1">
            Evaluates mel reconstruction, KL divergence, feature matching, G/D balance, and dataset fit against your logged runs.
          </p>
        </div>

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

      <section
        aria-label={t('diagnostics.guide')}
        className="rounded-xl border border-[#47161D] bg-[#160709] p-6 space-y-5 shadow-[0_12px_30px_-16px_rgba(0,0,0,0.8)]"
      >
        <div className="flex flex-wrap items-start justify-between gap-3 border-b border-[#47161D] pb-4">
          <div className="flex items-start gap-3">
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-[#E51923]/50 bg-[#200B0E] text-[#E51923]">
              <Activity className="h-4 w-4" />
            </span>
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h3 className="font-display text-[17px] font-semibold text-[#FDF7F7]">
                  Best Epoch Diagnostics
                </h3>
                <span className="rounded-md border border-[#E51923]/40 bg-[#200B0E] px-2.5 py-0.5 text-xs font-mono text-[#FDF7F7]">
                  Save Every {activeProject.saveEveryEpoch} Epochs
                </span>
              </div>
              <p className="text-[13.5px] text-[#C89EA4] mt-1">
                Evaluates only real logged epochs against your dataset clips, sample rate, vocoder, and listening feedback.
              </p>
            </div>
          </div>

          {isTrained && (
            <button
              type="button"
              onClick={() => {
                onAnalysisChange(null);
                setSoundCheck('untested');
              }}
              className="inline-flex items-center gap-1.5 rounded-lg border border-[#47161D] bg-[#100506] px-3 py-1.5 text-xs font-medium text-[#DFBFC3] hover:text-white hover:border-[#E51923] whitespace-nowrap shrink-0"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              <span>Clear Loaded Run</span>
            </button>
          )}
        </div>

        {/* Active Model Reference Bar */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5">
          <div className="rounded-lg border border-[#47161D] bg-[#100506] px-3.5 py-2.5">
            <p className="text-xs text-[#C89EA4]">Active Model</p>
            <p className="text-[13.5px] font-semibold text-[#FDF7F7] truncate mt-0.5">
              {activeModelTitle}
            </p>
          </div>
          <div className="rounded-lg border border-[#47161D] bg-[#100506] px-3.5 py-2.5">
            <p className="text-xs text-[#C89EA4]">Vocoder</p>
            <p className="text-[13.5px] font-semibold text-[#FDF7F7] truncate mt-0.5">
              {lastAnalysis?.vocoderDetected || activeProject.vocoder}
            </p>
          </div>
          <div className="rounded-lg border border-[#47161D] bg-[#100506] px-3.5 py-2.5">
            <p className="text-xs text-[#C89EA4]">Sample Rate</p>
            <p className="text-[13.5px] font-mono font-semibold text-[#FDF7F7] tabular-nums mt-0.5">
              {activeProject.samplingRate} Hz
            </p>
          </div>
          <div className="rounded-lg border border-[#47161D] bg-[#100506] px-3.5 py-2.5">
            <p className="text-xs text-[#C89EA4]">Effective Batch</p>
            <p className="text-[13.5px] font-mono font-semibold text-[#FDF7F7] tabular-nums mt-0.5">
              {effectiveBatch}
            </p>
          </div>
          <div className="rounded-lg border border-[#47161D] bg-[#100506] px-3.5 py-2.5">
            <p className="text-xs text-[#C89EA4]">Save Interval</p>
            <p className="text-[13.5px] font-mono font-semibold text-[#E51923] tabular-nums mt-0.5">
              Every {activeProject.saveEveryEpoch} ep
            </p>
          </div>
          <div className="rounded-lg border border-[#47161D] bg-[#100506] px-3.5 py-2.5">
            <p className="text-xs text-[#C89EA4]">Total Target</p>
            <p className="text-[13.5px] font-mono font-semibold text-[#FDF7F7] tabular-nums mt-0.5">
              {activeProject.totalEpoch} ep
            </p>
          </div>
        </div>

        {!isTrained ? (
          <div
            data-testid="diagnostics-untrained-state"
            className="rounded-xl border border-dashed border-[#47161D] bg-[#100506] p-6 text-center space-y-2.5"
          >
            <p className="text-[15px] font-semibold text-[#FDF7F7]">
              Waiting for Training Data
            </p>
            <p className="text-[13.5px] text-[#C89EA4] max-w-2xl mx-auto leading-relaxed">
              {zoneInsight.summary}
            </p>
            <p className="text-xs font-mono text-[#AD8288] pt-1">
              On Kaggle, Diagnostics activates automatically as soon as your model finishes its first epoch and logs mel loss.
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {/* Training Verdict Hero Banner + Dual Under/Over Risk Meters */}
            <div
              data-testid="diagnostics-verdict-banner"
              className={`rounded-xl border p-4 space-y-3.5 ${
                zoneInsight.trainingVerdict === 'overtrained'
                  ? 'border-[#E51923] bg-[#24090D]'
                  : zoneInsight.trainingVerdict === 'undertrained'
                    ? 'border-[#F59E0B]/70 bg-[#1B1207]'
                    : 'border-[#10B981]/60 bg-[#081A14]'
              }`}
            >
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="flex items-start gap-3">
                  <span
                    className={`mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border ${
                      zoneInsight.trainingVerdict === 'overtrained'
                        ? 'border-[#E51923]/60 bg-[#160709] text-[#FF5A62]'
                        : zoneInsight.trainingVerdict === 'undertrained'
                          ? 'border-[#F59E0B]/60 bg-[#160709] text-[#F59E0B]'
                          : 'border-[#10B981]/60 bg-[#160709] text-[#10B981]'
                    }`}
                  >
                    {zoneInsight.trainingVerdict === 'overtrained' ? (
                      <AlertTriangle className="h-4 w-4" />
                    ) : zoneInsight.trainingVerdict === 'undertrained' ? (
                      <Clock className="h-4 w-4" />
                    ) : (
                      <CheckCircle2 className="h-4 w-4" />
                    )}
                  </span>
                  <div>
                    <p className="font-display text-[16px] font-bold text-[#FDF7F7]">
                      {zoneInsight.verdictTitle}
                    </p>
                    <p className="text-[13px] text-[#DFBFC3] mt-0.5 leading-relaxed">
                      {zoneInsight.verdictGuidance}
                    </p>
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-2 font-mono text-xs shrink-0">
                  {zoneInsight.stepsPerClip !== null && (
                    <span className="rounded-md border border-[#47161D] bg-[#100506] px-2.5 py-1 text-[#FDF7F7] tabular-nums">
                      {zoneInsight.stepsPerClip} steps/clip
                    </span>
                  )}
                  <span className="rounded-md border border-[#47161D] bg-[#100506] px-2.5 py-1 text-[#DFBFC3] tabular-nums">
                    {zoneInsight.effectiveStepsCompleted} eff. steps
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="rounded-lg border border-[#47161D] bg-[#100506] p-3">
                  <div className="flex items-center justify-between text-xs mb-1.5">
                    <span className="font-semibold text-[#F59E0B]">Undertrained Risk</span>
                    <span className="font-mono font-bold text-[#FDF7F7] tabular-nums">
                      {zoneInsight.undertrainingRiskPercent}%
                    </span>
                  </div>
                  <div className="h-2 w-full rounded-sm bg-[#160709] overflow-hidden border border-[#47161D]">
                    <div
                      className="h-full bg-[#F59E0B] transition-all duration-200"
                      style={{
                        width: `${Math.min(100, Math.max(0, zoneInsight.undertrainingRiskPercent ?? 0))}%`,
                      }}
                    />
                  </div>
                </div>

                <div className="rounded-lg border border-[#47161D] bg-[#100506] p-3">
                  <div className="flex items-center justify-between text-xs mb-1.5">
                    <span className="font-semibold text-[#FF5A62]">Overtrained Risk</span>
                    <span className="font-mono font-bold text-[#FDF7F7] tabular-nums">
                      {zoneInsight.overtrainingRiskPercent ?? 0}%
                    </span>
                  </div>
                  <div className="h-2 w-full rounded-sm bg-[#160709] overflow-hidden border border-[#47161D]">
                    <div
                      className="h-full bg-[#E51923] transition-all duration-200"
                      style={{
                        width: `${Math.min(100, Math.max(0, zoneInsight.overtrainingRiskPercent ?? 0))}%`,
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Sound Feedback Selector (available once real model checkpoints exist) */}
            <div className="space-y-2">
              <p className="text-[13.5px] font-semibold text-[#FDF7F7]">
                How does this model sound in Samples?
              </p>
              <div className="flex flex-wrap gap-2">
                {SOUND_FEEDBACK_OPTIONS.map((opt) => (
                  <button
                    key={opt.id}
                    type="button"
                    onClick={() => setSoundCheck(opt.id)}
                    aria-pressed={soundCheck === opt.id}
                    className={`rounded-lg border px-3.5 py-2 text-[13px] font-medium transition-colors whitespace-nowrap ${
                      soundCheck === opt.id
                        ? 'border-[#E51923] bg-[#200B0E] text-white'
                        : 'border-[#47161D] bg-[#100506] text-[#DFBFC3] hover:text-white'
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Recommended Best Epoch Hero Card */}
            <div
              data-testid="diagnostics-best-epoch-card"
              className="rounded-xl border border-[#E51923]/70 bg-[#200B0E] p-5 space-y-4"
            >
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div className="space-y-1">
                  <div className="inline-flex items-center gap-2 text-xs font-mono uppercase tracking-wider text-[#FF5A62]">
                    <Award className="h-4 w-4 text-[#E51923]" />
                    <span>Recommended Best Epoch</span>
                  </div>
                  <p className="font-display text-2xl font-bold text-[#FDF7F7] tabular-nums">
                    Epoch {zoneInsight.sweetSpotEpoch}
                    <span className="ml-2.5 font-mono text-sm font-normal text-[#DFBFC3]">
                      ({safeModelSlug}_{zoneInsight.sweetSpotEpoch}e.pth)
                    </span>
                  </p>
                </div>

                <div className="flex flex-wrap gap-2">
                  {(zoneInsight.candidateEpochs ?? []).map((ep) => (
                    <span
                      key={ep}
                      className={`rounded-lg border px-2.5 py-1 font-mono text-xs tabular-nums ${
                        ep === zoneInsight.sweetSpotEpoch
                          ? 'border-[#E51923] bg-[#E51923] text-white font-semibold'
                          : 'border-[#47161D] bg-[#100506] text-[#DFBFC3]'
                      }`}
                    >
                      e{ep}
                    </span>
                  ))}
                </div>
              </div>

              <p className="text-[13.5px] text-[#FDF7F7]/90 leading-relaxed">
                {zoneInsight.summary}
              </p>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-1">
                <div className="rounded-lg border border-[#47161D] bg-[#100506] p-3.5">
                  <p className="text-xs font-semibold text-[#F59E0B]">
                    Early / Undertrained Range
                  </p>
                  <p className="text-[14px] font-mono font-bold text-[#FDF7F7] tabular-nums mt-1">
                    {zoneInsight.undertrainingLabel}
                  </p>
                  <p className="text-[13px] text-[#C89EA4] mt-1">
                    Timbre is still forming relative to your dataset.
                  </p>
                </div>

                <div className="rounded-lg border border-[#47161D] bg-[#100506] p-3.5">
                  <p className="text-xs font-semibold text-[#10B981]">
                    Dataset Match &amp; Lowest Mel
                  </p>
                  <p className="text-[14px] font-mono font-bold text-[#FDF7F7] tabular-nums mt-1">
                    {zoneInsight.datasetMatchPercent != null
                      ? `${zoneInsight.datasetMatchPercent}% dataset match`
                      : zoneInsight.bestLoss != null
                        ? `${zoneInsight.bestLoss.toFixed(3)} loss`
                        : 'Observed'}
                  </p>
                  <p className="text-[13px] text-[#C89EA4] mt-1">
                    {zoneInsight.bestLoss != null
                      ? `Min mel ${zoneInsight.bestLoss.toFixed(3)} · ${zoneInsight.datasetSummary || 'Measured from epochs'}`
                      : zoneInsight.datasetSummary || 'Measured from observed training epochs.'}
                  </p>
                </div>

                <div className="rounded-lg border border-[#47161D] bg-[#100506] p-3.5">
                  <p className="text-xs font-semibold text-[#FF2B35]">
                    Overtraining Threshold
                  </p>
                  <p className="text-[14px] font-mono font-bold text-[#FDF7F7] tabular-nums mt-1">
                    {zoneInsight.overtrainingLabel}
                  </p>
                  <p className="text-[13px] text-[#C89EA4] mt-1">
                    Past this point, weights risk metallic artifacts or overfitting.
                  </p>
                </div>
              </div>

              {zoneInsight.metricBreakdown && zoneInsight.metricBreakdown.length > 0 && (
                <div className="pt-2 border-t border-[#47161D]/70 space-y-2.5">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <p className="text-xs font-semibold uppercase tracking-wider text-[#DFBFC3]">
                      Multi-Metric &amp; Dataset Fit Breakdown
                    </p>
                    <div className="flex items-center gap-2 text-xs font-mono">
                      {zoneInsight.stabilityScore != null && (
                        <span className="rounded border border-[#47161D] bg-[#100506] px-2 py-0.5 text-[#DFBFC3] tabular-nums">
                          Stability {zoneInsight.stabilityScore}%
                        </span>
                      )}
                      {zoneInsight.convergenceLabel && (
                        <span className="rounded border border-[#E51923]/45 bg-[#100506] px-2 py-0.5 text-[#FF5A62]">
                          {zoneInsight.convergenceLabel}
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
                    {zoneInsight.metricBreakdown.map((item) => (
                      <div
                        key={item.label}
                        className="rounded-lg border border-[#47161D] bg-[#100506] px-3.5 py-2.5"
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-xs font-medium text-[#C89EA4]">
                            {item.label}
                          </span>
                          <span
                            className={`text-xs font-mono font-semibold tabular-nums ${
                              item.status === 'optimal'
                                ? 'text-[#10B981]'
                                : item.status === 'watch'
                                  ? 'text-[#F59E0B]'
                                  : 'text-[#FF5A62]'
                            }`}
                          >
                            {item.value}
                          </span>
                        </div>
                        <p className="text-xs text-[#AD8288] mt-1">{item.detail}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {(zoneInsight.rankedCheckpoints?.length ?? 0) > 0 && (
                <div className="pt-2 border-t border-[#47161D]/70 space-y-2.5">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <p className="text-xs font-semibold uppercase tracking-wider text-[#DFBFC3]">
                      Top Recommended Checkpoints to Export &amp; Audition
                    </p>
                    <span className="text-xs font-mono text-[#AD8288]">
                      Ranked by Mel convergence + tail stability
                    </span>
                  </div>
                  <div className="overflow-x-auto rounded-lg border border-[#47161D] bg-[#100506]">
                    <table className="w-full text-left text-xs">
                      <thead className="border-b border-[#47161D] text-[#AD8288] font-mono">
                        <tr>
                          <th className="py-2 px-3">Rank</th>
                          <th className="py-2 px-3">Epoch</th>
                          <th className="py-2 px-3">Checkpoint File</th>
                          <th className="py-2 px-3">Mel Loss</th>
                          <th className="py-2 px-3">Tail Drift</th>
                          <th className="py-2 px-3">Score</th>
                          <th className="py-2 px-3">Verdict</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-[#47161D]/60">
                        {(zoneInsight.rankedCheckpoints ?? []).map((rk, rIdx) => {
                          const isTopPick = rk.tag === 'Best Recommended';
                          return (
                            <tr key={rk.checkpointName} className={isTopPick ? 'bg-[#200B0E]/70' : ''}>
                              <td className="py-2 px-3 font-mono font-semibold text-[#FDF7F7] tabular-nums">
                                #{rIdx + 1}
                              </td>
                              <td className="py-2 px-3 font-mono text-[#FDF7F7] tabular-nums">
                                e{rk.epoch}
                              </td>
                              <td className="py-2 px-3 font-mono text-[#DFBFC3]">
                                {rk.checkpointName}
                              </td>
                              <td className="py-2 px-3 font-mono text-[#FDF7F7] tabular-nums">
                                {rk.melLoss !== null ? rk.melLoss.toFixed(3) : '—'}
                              </td>
                              <td className="py-2 px-3 font-mono text-[#C89EA4] tabular-nums">
                                +{rk.driftPercent.toFixed(1)}%
                              </td>
                              <td className="py-2 px-3 font-mono font-semibold text-[#FDF7F7] tabular-nums">
                                {rk.score}/100
                              </td>
                              <td className="py-2 px-3">
                                <span
                                  className={`inline-block rounded px-2 py-0.5 font-mono text-[11px] font-semibold ${
                                    isTopPick
                                      ? 'bg-[#E51923] text-white'
                                      : rk.status === 'optimal'
                                        ? 'border border-[#10B981]/50 bg-[#081A14] text-[#10B981]'
                                        : rk.status === 'watch'
                                          ? 'border border-[#F59E0B]/50 bg-[#1B1207] text-[#F59E0B]'
                                          : 'border border-[#E51923]/60 bg-[#24080D] text-[#FF5A62]'
                                  }`}
                                >
                                  {rk.tag}
                                </span>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>

            {lastAnalysis?.liveProgress && (
              <div className="rounded-lg border border-[#E51923]/45 bg-[#100506] p-4 space-y-2">
                <div className="flex flex-wrap items-center justify-between gap-2 text-[13px] font-mono">
                  <span className="font-semibold text-[#FDF7F7] tabular-nums">
                    Epoch {lastAnalysis.liveProgress.epoch} · Step {lastAnalysis.liveProgress.step} · Batch{' '}
                    {lastAnalysis.liveProgress.batchDone}/{lastAnalysis.liveProgress.batchTotal} (
                    {lastAnalysis.liveProgress.percent}%)
                  </span>
                  <span className="text-[#DFBFC3] tabular-nums">
                    {lastAnalysis.liveProgress.itPerSec.toFixed(2)} it/s ·{' '}
                    {lastAnalysis.liveProgress.elapsedLabel} &lt; {lastAnalysis.liveProgress.etaLabel}
                  </span>
                </div>
                <div className="h-1.5 w-full rounded-full bg-[#160709] overflow-hidden border border-[#47161D]">
                  <div
                    className="h-full bg-[#E51923] transition-all duration-200"
                    style={{
                      width: `${Math.min(100, Math.max(0, lastAnalysis.liveProgress.percent))}%`,
                    }}
                  />
                </div>
              </div>
            )}

            {observedLossCount >= 2 && (
              <LossChart
                points={lastAnalysis?.points ?? []}
                zoneInsight={zoneInsight}
              />
            )}

            {lastAnalysis && lastAnalysis.avgSpeedSec > 0 && (
              <p className="font-mono text-xs text-[#C89EA4] tabular-nums text-right">
                Avg {lastAnalysis.avgSpeedSec}s/ep · {lastAnalysis.epochsCompleted} epochs observed
                {lastAnalysis.lowestLossEpoch !== null
                  ? ` · Min mel loss at ep ${lastAnalysis.lowestLossEpoch}`
                  : ''}
              </p>
            )}
          </div>
        )}

        {/* Lightweight Collapsible Run Log Importer */}
        <div className="rounded-lg border border-[#47161D] bg-[#100506]">
          <button
            type="button"
            onClick={() => setLogPanelOpen((v) => !v)}
            aria-expanded={logPanelOpen}
            className="w-full flex items-center justify-between px-4 py-3 text-[13px] font-semibold text-[#DFBFC3] hover:text-white"
          >
            <span>
              {lastAnalysis
                ? `Imported Training Run (${lastAnalysis.speakerName} · ${lastAnalysis.epochsCompleted} ep)`
                : 'Import Training Log or .ipynb to Analyze a Trained Model'}
            </span>
            {logPanelOpen ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
          </button>

          {logPanelOpen && (
            <div className="border-t border-[#47161D] p-4 space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".ipynb,.txt,.log"
                  className="hidden"
                  onChange={handleFileUpload}
                />
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-[#47161D] bg-[#200B0E] px-3.5 py-2 text-xs font-medium text-[#FDF7F7] hover:border-[#E51923] whitespace-nowrap"
                >
                  <FileUp className="h-3.5 w-3.5" />
                  <span>Import .ipynb / Log</span>
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setLogInput(SAMPLE_TRAINING_LOG);
                    runParse(SAMPLE_TRAINING_LOG);
                  }}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-[#47161D] bg-[#160709] px-3.5 py-2 text-xs font-medium text-[#DFBFC3] hover:text-white whitespace-nowrap"
                >
                  <Sparkles className="h-3.5 w-3.5" />
                  <span>Load Sample Run</span>
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setLogInput(SAMPLE_OVERTRAIN_LOG);
                    runParse(SAMPLE_OVERTRAIN_LOG);
                  }}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-[#E51923]/50 bg-[#200B0E] px-3.5 py-2 text-xs font-medium text-[#FDF7F7] hover:border-[#E51923] whitespace-nowrap"
                >
                  <AlertTriangle className="h-3.5 w-3.5 text-[#E51923]" />
                  <span>Load Overtrained Run</span>
                </button>
              </div>

              <textarea
                value={logInput}
                onChange={(e) => setLogInput(e.target.value)}
                rows={3}
                placeholder="Paste Kaggle training stdout (epoch=... | step=... | lowest_value=...) or upload your .ipynb file..."
                aria-label="Training log or notebook JSON input"
                className="w-full rounded-lg bg-[#160709] border border-[#47161D] p-3 text-xs font-mono text-[#FDF7F7] placeholder-[#AD8288]/55 focus:outline-none focus:border-[#E51923]"
              />

              <div className="flex items-center justify-between gap-3">
                {parseError ? (
                  <p className="text-[13px] text-[#FF2B35]" role="alert">
                    {parseError}
                  </p>
                ) : (
                  <span className="text-[13px] text-[#C89EA4]">
                    Calculates your model&apos;s Best Epoch from real observed mel loss and dataset segment stats.
                  </span>
                )}
                <button
                  type="button"
                  onClick={() => runParse(logInput)}
                  className="rounded-lg border border-[#47161D] bg-[#200B0E] px-4 py-2 text-xs font-semibold text-[#FDF7F7] hover:border-[#E51923] whitespace-nowrap shrink-0"
                >
                  Analyze Training Run
                </button>
              </div>
            </div>
          )}
        </div>
      </section>
    </div>
  );
});
