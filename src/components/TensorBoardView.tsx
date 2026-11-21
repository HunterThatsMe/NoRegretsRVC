import React, { useMemo, useRef, useState } from 'react';
import {
  ChevronDown,
  ChevronRight,
  Compass,
  ExternalLink,
  FileUp,
  Sparkles,
} from 'lucide-react';
import { REPO_README_URL, SAMPLE_TRAINING_LOG } from '../data/defaults';
import {
  ModelSoundFeedback,
  TrainingAnalysis,
  VoiceModelProject,
} from '../types/studio';
import { analyzeTrainingZones, parseTrainingNotebookOrLog } from '../utils/logParser';
import { useI18n } from '../utils/i18n';
import { LossChart } from './LossChart';

interface TensorBoardViewProps {
  activeProject: VoiceModelProject;
  lastAnalysis: TrainingAnalysis | null;
  onAnalysisChange: (analysis: TrainingAnalysis | null) => void;
}

const SOUND_FEEDBACK_OPTIONS: { id: ModelSoundFeedback; label: string }[] = [
  { id: 'untested', label: 'Not tested yet' },
  { id: 'undertrained', label: 'Undertrained (robotic / flat)' },
  { id: 'clean', label: 'Sounds clean & natural' },
  { id: 'overtrained', label: 'Overtrained (metallic / harsh)' },
];

export const TensorBoardView: React.FC<TensorBoardViewProps> = React.memo(({
  activeProject,
  lastAnalysis,
  onAnalysisChange,
}) => {
  const { t } = useI18n();
  const [showSuggestion, setShowSuggestion] = useState(true);
  const [soundCheck, setSoundCheck] = useState<ModelSoundFeedback>('untested');
  const [logPanelOpen, setLogPanelOpen] = useState(false);
  const [logInput, setLogInput] = useState('');
  const [parseError, setParseError] = useState('');
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const zoneInsight = useMemo(
    () =>
      showSuggestion
        ? analyzeTrainingZones(lastAnalysis, activeProject, soundCheck)
        : null,
    [showSuggestion, lastAnalysis, activeProject, soundCheck]
  );

  const runParse = (raw: string) => {
    const parsed = parseTrainingNotebookOrLog(raw);
    if (!parsed) {
      setParseError(
        'Could not detect RVC training metrics. Paste Applio console stdout or upload a .ipynb file.'
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

  const effectiveBatch = activeProject.batchSize * activeProject.gpuCount;
  const activeModelTitle = activeProject.name.trim() || 'Untitled Model';

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4 px-1">
        <div>
          <h2 className="font-display text-2xl font-bold text-[#FDF7F7] tracking-tight">
            {t('diagnostics.title')}
          </h2>
          <p className="text-sm text-[#DFBFC3] mt-1">
            Model-specific Training Zone Guide aligned to your saved checkpoint schedule.
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

      {/* Training Zone Guide (Optional Suggestion ON by default + Red Graph) */}
      <section
        aria-label={t('diagnostics.guide')}
        className="rounded-xl border border-[#47161D] bg-[#160709] p-6 space-y-5"
      >
        <div className="flex flex-wrap items-start justify-between gap-3 border-b border-[#47161D] pb-4">
          <div className="flex items-start gap-3">
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-[#E51923]/50 bg-[#200B0E] text-[#E51923]">
              <Compass className="h-4 w-4" />
            </span>
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h3 className="font-display text-base font-semibold text-[#FDF7F7]">
                  {t('diagnostics.guide')}
                </h3>
                <span className="rounded-md border border-[#E51923]/40 bg-[#200B0E] px-2 py-0.5 text-[10px] font-mono text-[#FDF7F7]">
                  Save Every {activeProject.saveEveryEpoch} Epochs
                </span>
              </div>
              <p className="text-xs text-[#AD8288] mt-0.5">
                Shows Undertraining, Suggested Sweet Spot, and Overtraining checkpoints tailored to your active model.
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={() => setShowSuggestion((v) => !v)}
            aria-pressed={showSuggestion}
            className={`rounded-lg border px-3.5 py-1.5 text-xs font-semibold transition-colors whitespace-nowrap shrink-0 ${
              showSuggestion
                ? 'border-[#E51923] bg-[#E51923] text-white'
                : 'border-[#47161D] bg-[#100506] text-[#DFBFC3] hover:text-white hover:border-[#E51923]'
            }`}
          >
            {showSuggestion ? 'Optional Suggestion: On' : 'Optional Suggestion: Off'}
          </button>
        </div>

        {/* Active Model Reference Bar */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5">
          <div className="rounded-lg border border-[#47161D] bg-[#100506] px-3 py-2">
            <p className="text-[10px] text-[#AD8288]">Active Model</p>
            <p className="text-xs font-semibold text-[#FDF7F7] truncate mt-0.5">
              {activeModelTitle}
            </p>
          </div>
          <div className="rounded-lg border border-[#47161D] bg-[#100506] px-3 py-2">
            <p className="text-[10px] text-[#AD8288]">Vocoder</p>
            <p className="text-xs font-semibold text-[#FDF7F7] truncate mt-0.5">
              {activeProject.vocoder}
            </p>
          </div>
          <div className="rounded-lg border border-[#47161D] bg-[#100506] px-3 py-2">
            <p className="text-[10px] text-[#AD8288]">Sample Rate</p>
            <p className="text-xs font-mono font-semibold text-[#FDF7F7] tabular-nums mt-0.5">
              {activeProject.samplingRate} Hz
            </p>
          </div>
          <div className="rounded-lg border border-[#47161D] bg-[#100506] px-3 py-2">
            <p className="text-[10px] text-[#AD8288]">Effective Batch</p>
            <p className="text-xs font-mono font-semibold text-[#FDF7F7] tabular-nums mt-0.5">
              {effectiveBatch}
            </p>
          </div>
          <div className="rounded-lg border border-[#47161D] bg-[#100506] px-3 py-2">
            <p className="text-[10px] text-[#AD8288]">Save Interval</p>
            <p className="text-xs font-mono font-semibold text-[#E51923] tabular-nums mt-0.5">
              Every {activeProject.saveEveryEpoch} ep
            </p>
          </div>
          <div className="rounded-lg border border-[#47161D] bg-[#100506] px-3 py-2">
            <p className="text-[10px] text-[#AD8288]">Total Target</p>
            <p className="text-xs font-mono font-semibold text-[#FDF7F7] tabular-nums mt-0.5">
              {activeProject.totalEpoch} ep
            </p>
          </div>
        </div>

        {/* Model Sound Feedback */}
        <div className="space-y-2">
          <p className="text-xs font-semibold text-[#FDF7F7]">
            How does your current model checkpoint sound?
          </p>
          <div className="flex flex-wrap gap-2">
            {SOUND_FEEDBACK_OPTIONS.map((opt) => (
              <button
                key={opt.id}
                type="button"
                onClick={() => setSoundCheck(opt.id)}
                aria-pressed={soundCheck === opt.id}
                className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap ${
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

        {zoneInsight ? (
          <div className="rounded-lg border border-[#47161D] bg-[#100506] p-4 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="text-xs font-semibold text-[#FDF7F7]">
                Optional Training Zone Suggestion ·{' '}
                <span className="font-normal text-[#AD8288]">
                  {zoneInsight.confidence === 'observed'
                    ? 'Observed from log curve & active model'
                    : 'Estimated from active model settings'}
                </span>
              </p>
              <span className="font-mono text-[11px] text-[#DFBFC3] tabular-nums">
                Checkpoints snapped to ×{zoneInsight.saveEveryEpoch} epochs
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="rounded-md border border-[#47161D] bg-[#160709] p-3.5">
                <p className="text-[11px] text-[#F59E0B] font-semibold">
                  Undertraining Zone
                </p>
                <p className="text-sm font-mono font-bold text-[#FDF7F7] tabular-nums mt-1">
                  {zoneInsight.undertrainingLabel}
                </p>
                <p className="text-[11px] text-[#AD8288] mt-1">
                  Voice still lacks target timbre or sounds muffled.
                </p>
              </div>

              <div className="rounded-md border border-[#E51923] bg-[#200B0E] p-3.5">
                <p className="text-[11px] text-[#FDF7F7] font-semibold">
                  Suggested Sweet Spot
                </p>
                <p className="text-sm font-mono font-bold text-[#FDF7F7] tabular-nums mt-1">
                  {zoneInsight.sweetSpotLabel}
                </p>
                <p className="text-[11px] text-[#DFBFC3] mt-1">
                  Valid saved checkpoints to audition in Samples.
                </p>
              </div>

              <div className="rounded-md border border-[#47161D] bg-[#160709] p-3.5">
                <p className="text-[11px] text-[#FF2B35] font-semibold">
                  Overtraining Zone
                </p>
                <p className="text-sm font-mono font-bold text-[#FDF7F7] tabular-nums mt-1">
                  {zoneInsight.overtrainingLabel}
                </p>
                <p className="text-[11px] text-[#AD8288] mt-1">
                  High risk of metallic ringing or dataset memorization.
                </p>
              </div>
            </div>

            {/* Live Mid-Epoch Progress from Imported Active Run */}
            {lastAnalysis?.liveProgress && (
              <div className="rounded-lg border border-[#E51923]/45 bg-[#160709] p-3.5 space-y-2">
                <div className="flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
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
                <div className="h-1.5 w-full rounded-full bg-[#100506] overflow-hidden border border-[#47161D]">
                  <div
                    className="h-full bg-[#E51923] transition-all duration-200"
                    style={{
                      width: `${Math.min(100, Math.max(0, lastAnalysis.liveProgress.percent))}%`,
                    }}
                  />
                </div>
              </div>
            )}

            {/* Compact Red Graph under Optional Training Zone Suggestion */}
            <LossChart
              points={lastAnalysis?.points ?? []}
              zoneInsight={zoneInsight}
            />

            <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-[#DFBFC3] leading-relaxed">
              <p>{zoneInsight.summary}</p>
              {lastAnalysis && lastAnalysis.avgSpeedSec > 0 && (
                <p className="font-mono text-[11px] text-[#AD8288] tabular-nums shrink-0">
                  Avg {lastAnalysis.avgSpeedSec}s/ep
                  {lastAnalysis.lowestLossEpoch !== null &&
                  lastAnalysis.epochsCompleted > lastAnalysis.lowestLossEpoch
                    ? ` · ${lastAnalysis.epochsCompleted - lastAnalysis.lowestLossEpoch} ep since best (ep ${lastAnalysis.lowestLossEpoch})`
                    : ''}
                </p>
              )}
            </div>
          </div>
        ) : (
          <div className="rounded-lg border border-[#47161D] bg-[#100506] p-4 text-xs text-[#AD8288]">
            Optional Training Zone Suggestion is turned off. Click{' '}
            <strong className="text-[#FDF7F7]">Optional Suggestion: Off</strong> above anytime to show checkpoint recommendations and graph.
          </div>
        )}

        {/* Lightweight Collapsible Run Log Importer */}
        <div className="rounded-lg border border-[#47161D] bg-[#100506]">
          <button
            type="button"
            onClick={() => setLogPanelOpen((v) => !v)}
            aria-expanded={logPanelOpen}
            className="w-full flex items-center justify-between px-4 py-2.5 text-xs font-semibold text-[#DFBFC3] hover:text-white"
          >
            <span>
              {lastAnalysis
                ? `Optional: Imported Run Data (${lastAnalysis.speakerName} · ${lastAnalysis.epochsCompleted} ep)`
                : 'Optional: Import Training Log or .ipynb to Refine Zone Graph'}
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
                  className="inline-flex items-center gap-1.5 rounded-lg border border-[#47161D] bg-[#200B0E] px-3 py-1.5 text-xs font-medium text-[#FDF7F7] hover:border-[#E51923] whitespace-nowrap"
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
                  className="inline-flex items-center gap-1.5 rounded-lg border border-[#47161D] bg-[#160709] px-3 py-1.5 text-xs font-medium text-[#DFBFC3] hover:text-white whitespace-nowrap"
                >
                  <Sparkles className="h-3.5 w-3.5" />
                  <span>Load Sample Run</span>
                </button>
              </div>

              <textarea
                value={logInput}
                onChange={(e) => setLogInput(e.target.value)}
                rows={3}
                placeholder="Paste Kaggle notebook JSON or Applio training stdout (epoch=... | step=... | lowest_value=...)..."
                aria-label="Training log or notebook JSON input"
                className="w-full rounded-lg bg-[#160709] border border-[#47161D] p-3 text-xs font-mono text-[#FDF7F7] placeholder-[#AD8288]/45 focus:outline-none focus:border-[#E51923]"
              />

              <div className="flex items-center justify-between gap-3">
                {parseError ? (
                  <p className="text-xs text-[#FF2B35]" role="alert">
                    {parseError}
                  </p>
                ) : (
                  <span className="text-xs text-[#AD8288]">
                    Parsed runs automatically refine the red zone graph and checkpoint numbers above.
                  </span>
                )}
                <button
                  type="button"
                  onClick={() => runParse(logInput)}
                  className="rounded-lg border border-[#47161D] bg-[#200B0E] px-4 py-1.5 text-xs font-semibold text-[#FDF7F7] hover:border-[#E51923] whitespace-nowrap shrink-0"
                >
                  Parse Training Run
                </button>
              </div>
            </div>
          )}
        </div>
      </section>
    </div>
  );
});
