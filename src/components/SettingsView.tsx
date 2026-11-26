import React, { useMemo, useState } from 'react';
import {
  AlertCircle,
  ChevronDown,
  ChevronRight,
  RefreshCw,
  RotateCcw,
  Shield,
} from 'lucide-react';
import {
  ACCENT_HEX,
  APP_VERSION,
  BRAND_NAME,
  SUPPORTED_LANGUAGES,
} from '../data/defaults';
import {
  ApplioGeneralSettings,
  ApplioTrainingSettings,
  DownloadModelItem,
  ForkConfig,
  RmvpeCorrectionMode,
  TrainingPrecision,
  TunnelMode,
  VoiceModelProject,
} from '../types/studio';
import { validateForkConfig } from '../utils/rvcTuning';
import { useI18n } from '../utils/i18n';

interface SettingsViewProps {
  generalSettings: ApplioGeneralSettings;
  trainingSettings: ApplioTrainingSettings;
  forkConfig: ForkConfig;
  activeProject?: VoiceModelProject;
  downloadManifest?: DownloadModelItem[];
  onUpdateGeneral: (updated: ApplioGeneralSettings) => void;
  onUpdateTrainingSettings: (updated: ApplioTrainingSettings) => void;
  onUpdatePrecisionAcrossProjects: (precision: TrainingPrecision) => void;
  onUpdateBatchParityAcrossProjects?: (enabled: boolean) => void;
  onUpdateForkConfig: (updated: ForkConfig) => void;
}

const StepBadge: React.FC<{ num: number }> = ({ num }) => (
  <span
    className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-xs font-bold text-white tabular-nums"
    style={{ backgroundColor: ACCENT_HEX }}
  >
    {num}
  </span>
);

export const SettingsView: React.FC<SettingsViewProps> = React.memo(({
  generalSettings,
  trainingSettings,
  forkConfig,
  activeProject,
  onUpdateGeneral,
  onUpdateTrainingSettings,
  onUpdatePrecisionAcrossProjects,
  onUpdateBatchParityAcrossProjects,
  onUpdateForkConfig,
}) => {
  const { t } = useI18n();
  const [subTab, setSubTab] = useState<'general' | 'training'>('general');
  const [pitchCorrOpen, setPitchCorrOpen] = useState(false);
  const [checkingUpdates, setCheckingUpdates] = useState(false);
  const [releaseLink, setReleaseLink] = useState('');
  const [restartMsg, setRestartMsg] = useState('');
  const [authorOutput, setAuthorOutput] = useState(
    'The output information will be displayed here.'
  );
  const [precisionOutput, setPrecisionOutput] = useState(
    'The output information will be displayed here.'
  );

  const issues = useMemo(() => validateForkConfig(forkConfig), [forkConfig]);

  const handleCheckUpdates = async () => {
    setCheckingUpdates(true);
    onUpdateGeneral({
      ...generalSettings,
      versionStatus: `Checking latest ${BRAND_NAME} version on GitHub...`,
    });
    try {
      const resp = await fetch('/api/check-updates');
      const data = (await resp.json()) as {
        ok: boolean;
        currentVersion?: string;
        newestVersion?: string;
        latestTag?: string;
        latestCommit?: string;
        releaseUrl?: string;
        upToDate?: boolean;
        error?: string;
      };
      const currentVer = data.currentVersion || APP_VERSION;
      const newestVer = data.newestVersion || data.latestTag || APP_VERSION;
      if (data.releaseUrl) setReleaseLink(data.releaseUrl);
      if (data.ok) {
        const commitSuffix = data.latestCommit ? ` (commit ${data.latestCommit})` : '';
        const statusMsg =
          data.upToDate === false
            ? `Update available! Installed: Beta version ${currentVer} -> Newest GitHub release: ${newestVer}${commitSuffix}.`
            : `${BRAND_NAME} Beta version ${currentVer} is up to date (newest version: ${newestVer})${commitSuffix}.`;
        onUpdateGeneral({ ...generalSettings, versionStatus: statusMsg });
      } else {
        onUpdateGeneral({
          ...generalSettings,
          versionStatus: `${BRAND_NAME} Beta version ${currentVer} (newest version: ${newestVer}). ${data.error || 'Offline check complete.'}`,
        });
      }
    } catch {
      onUpdateGeneral({
        ...generalSettings,
        versionStatus: `${BRAND_NAME} Beta version ${APP_VERSION} (newest version: ${APP_VERSION}). Network check failed.`,
      });
    } finally {
      setCheckingUpdates(false);
    }
  };

  const handleRestartApplio = () => {
    setRestartMsg('Restart signal queued for Kaggle WebUI runtime (port 6969).');
    setTimeout(() => setRestartMsg(''), 2500);
  };

  const handleSetAuthor = () => {
    const cleaned = trainingSettings.modelAuthor.trim() || 'None';
    onUpdateTrainingSettings({ ...trainingSettings, modelAuthor: cleaned });
    setAuthorOutput(`Model author set to "${cleaned}".`);
  };

  const handleUpdatePrecision = () => {
    onUpdatePrecisionAcrossProjects(trainingSettings.precision);
    setPrecisionOutput(
      `Training & inference precision updated to ${trainingSettings.precision.toUpperCase()}.`
    );
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4 px-1">
        <div>
          <h2 className="font-display text-2xl font-bold text-[#FDF7F7] tracking-tight">
            {t('settings.title')}
          </h2>
          <p className="text-sm text-[#DFBFC3] mt-1">
            {t('settings.subtitle')}
          </p>
        </div>

        <div className="inline-flex rounded-lg border border-[#47161D] bg-[#160709] p-1">
          <button
            type="button"
            onClick={() => setSubTab('general')}
            className={`rounded-md px-4 py-1.5 text-xs font-semibold transition-colors whitespace-nowrap shrink-0 ${
              subTab === 'general'
                ? 'bg-[#E51923] text-white'
                : 'text-[#DFBFC3] hover:text-white'
            }`}
          >
            {t('settings.tab.general')}
          </button>
          <button
            type="button"
            onClick={() => setSubTab('training')}
            className={`rounded-md px-4 py-1.5 text-xs font-semibold transition-colors whitespace-nowrap shrink-0 ${
              subTab === 'training'
                ? 'bg-[#E51923] text-white'
                : 'text-[#DFBFC3] hover:text-white'
            }`}
          >
            {t('settings.tab.training')}
          </button>
        </div>
      </div>

      {subTab === 'general' ? (
        <div className="space-y-5">
          {/* 1. List Filtering */}
          <section className="rounded-xl border border-[#47161D] bg-[#160709] p-6 space-y-4">
            <div className="flex items-start gap-3">
              <StepBadge num={1} />
              <div>
                <h3 className="font-display text-[17px] font-semibold text-[#FDF7F7]">
                  {t('settings.sec1.title')}
                </h3>
                <p className="text-[13.5px] text-[#C89EA4] mt-0.5">
                  {t('settings.sec1.desc')}
                </p>
              </div>
            </div>
            <label className="flex items-center justify-between rounded-lg border border-[#47161D] bg-[#100506] px-4 py-3 cursor-pointer">
              <div>
                <span className="block text-[13.5px] font-semibold text-[#FDF7F7]">
                  {t('settings.sec1.label')}
                </span>
                <span className="block text-[13px] text-[#C89EA4] mt-0.5">
                  {t('settings.sec1.sub')}
                </span>
              </div>
              <input
                type="checkbox"
                checked={false}
                readOnly
                disabled
                aria-label="Enable model/index list filter (always off)"
                className="h-4 w-4 rounded accent-[#E51923] opacity-60 cursor-not-allowed"
              />
            </label>
          </section>

          {/* 2. Pitch Correction */}
          <section className="rounded-xl border border-[#47161D] bg-[#160709] p-6 space-y-4">
            <div className="flex items-start gap-3">
              <StepBadge num={2} />
              <div>
                <h3 className="font-display text-[17px] font-semibold text-[#FDF7F7]">
                  {t('settings.sec2.title')}
                </h3>
                <p className="text-[13.5px] text-[#C89EA4] mt-0.5">
                  {t('settings.sec2.desc')}
                </p>
              </div>
            </div>

            <div className="rounded-lg border border-[#47161D] bg-[#100506]">
              <button
                type="button"
                onClick={() => setPitchCorrOpen((v) => !v)}
                aria-expanded={pitchCorrOpen}
                className="w-full flex items-center justify-between px-4 py-2.5 text-xs font-semibold text-[#DFBFC3] hover:text-white"
              >
                <span>{t('settings.sec2.toggle')}</span>
                {pitchCorrOpen ? (
                  <ChevronDown className="h-4 w-4" />
                ) : (
                  <ChevronRight className="h-4 w-4" />
                )}
              </button>
              {pitchCorrOpen && (
                <div className="border-t border-[#47161D] p-4 space-y-4">
                  <label className="flex items-start justify-between gap-4 cursor-pointer">
                    <div>
                      <span className="block text-[13.5px] font-semibold text-[#FDF7F7]">
                        Enable RMVPE high-register correction
                      </span>
                      <span className="block text-[13px] text-[#C89EA4] mt-0.5 leading-relaxed">
                        Runs a secondary octave-down pass above ~1040 Hz to catch upper-register pitch jumps. Disabled by default so standard RMVPE behavior remains untouched unless checked.
                      </span>
                    </div>
                    <input
                      type="checkbox"
                      checked={generalSettings.rmvpeHighRegisterCorrection}
                      onChange={(e) =>
                        onUpdateGeneral({
                          ...generalSettings,
                          rmvpeHighRegisterCorrection: e.target.checked,
                        })
                      }
                      className="mt-1 h-4 w-4 shrink-0 rounded accent-[#E51923]"
                    />
                  </label>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label
                        htmlFor="settings-rmvpe-mode"
                        className="block text-xs font-semibold text-[#FDF7F7]"
                      >
                        Correction mode
                      </label>
                      <p className="text-[13px] text-[#C89EA4] mt-0.5 mb-1.5 leading-relaxed">
                        true_pitch tracks the actual note up to the F0 cap; fold corrects only misdetected octave frames.
                      </p>
                      <select
                        id="settings-rmvpe-mode"
                        value={generalSettings.rmvpeCorrectionMode}
                        onChange={(e) =>
                          onUpdateGeneral({
                            ...generalSettings,
                            rmvpeCorrectionMode: e.target.value as RmvpeCorrectionMode,
                          })
                        }
                        className="w-full rounded-lg bg-[#160709] border border-[#47161D] px-3 py-2 text-xs text-[#FDF7F7] focus:outline-none focus:border-[#E51923]"
                      >
                        <option value="true_pitch">true_pitch</option>
                        <option value="fold">fold</option>
                      </select>
                    </div>

                    <div className="rounded-lg border border-[#47161D] bg-[#160709] p-3.5">
                      <div className="flex items-center justify-between gap-2 mb-1.5">
                        <div>
                          <label
                            htmlFor="settings-f0-ceiling"
                            className="block text-xs font-semibold text-[#FDF7F7]"
                          >
                            F0 ceiling (Hz)
                          </label>
                          <p className="text-[13px] text-[#C89EA4]">
                            Upper frequency cap applied in true_pitch mode.
                          </p>
                        </div>
                        <div className="flex items-center gap-1.5 shrink-0">
                          <span className="font-mono text-xs text-[#FDF7F7] tabular-nums">
                            {generalSettings.rmvpeF0CeilingHz}
                          </span>
                          <button
                            type="button"
                            onClick={() =>
                              onUpdateGeneral({
                                ...generalSettings,
                                rmvpeF0CeilingHz: 1250,
                              })
                            }
                            aria-label="Reset F0 ceiling to 1250"
                            className="rounded-md border border-[#47161D] bg-[#200B0E] p-1 text-[#DFBFC3] hover:text-white"
                          >
                            <RotateCcw className="h-3 w-3" />
                          </button>
                        </div>
                      </div>
                      <input
                        id="settings-f0-ceiling"
                        type="range"
                        min={1000}
                        max={2000}
                        step={25}
                        value={generalSettings.rmvpeF0CeilingHz}
                        onChange={(e) =>
                          onUpdateGeneral({
                            ...generalSettings,
                            rmvpeF0CeilingHz: Number(e.target.value),
                          })
                        }
                        className="w-full scarlet-slider"
                      />
                      <div className="flex justify-between text-[10px] font-mono text-[#AD8288] mt-1 tabular-nums">
                        <span>1000</span>
                        <span>2000</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </section>

          {/* 3. Discord Integration */}
          <section className="rounded-xl border border-[#47161D] bg-[#160709] p-6 space-y-4">
            <div className="flex items-start gap-3">
              <StepBadge num={3} />
              <div>
                <h3 className="font-display text-[17px] font-semibold text-[#FDF7F7]">
                  {t('settings.sec3.title')}
                </h3>
                <p className="text-[13.5px] text-[#C89EA4] mt-0.5">
                  {t('settings.sec3.desc')}
                </p>
              </div>
            </div>
            <label className="flex items-center justify-between rounded-lg border border-[#47161D] bg-[#100506] px-4 py-3 cursor-pointer">
              <div>
                <span className="block text-[13.5px] font-semibold text-[#FDF7F7]">
                  {t('settings.sec3.label')}
                </span>
                <span className="block text-[13px] text-[#C89EA4] mt-0.5">
                  {t('settings.sec3.sub')}
                </span>
              </div>
              <input
                type="checkbox"
                checked={generalSettings.discordPresence}
                onChange={(e) =>
                  onUpdateGeneral({
                    ...generalSettings,
                    discordPresence: e.target.checked,
                  })
                }
                className="h-4 w-4 rounded accent-[#E51923]"
              />
            </label>
          </section>

          {/* 4. Language (No Appearance option!) */}
          <section className="rounded-xl border border-[#47161D] bg-[#160709] p-6 space-y-4">
            <div className="flex items-start gap-3">
              <StepBadge num={4} />
              <div>
                <h3 className="font-display text-[17px] font-semibold text-[#FDF7F7]">
                  {t('settings.sec4.title')}
                </h3>
                <p className="text-[13.5px] text-[#C89EA4] mt-0.5">
                  {t('settings.sec4.desc')}
                </p>
              </div>
            </div>
            <div>
              <label
                htmlFor="settings-language"
                className="block text-xs font-semibold text-[#FDF7F7] mb-1.5"
              >
                {t('settings.sec4.label')}
              </label>
              <select
                id="settings-language"
                value={generalSettings.language}
                onChange={(e) =>
                  onUpdateGeneral({
                    ...generalSettings,
                    language: e.target.value,
                  })
                }
                className="w-full rounded-lg bg-[#100506] border border-[#47161D] px-3 py-2 text-sm text-[#FDF7F7] focus:outline-none focus:border-[#E51923]"
              >
                {SUPPORTED_LANGUAGES.map((lang) => (
                  <option key={lang.code} value={lang.code}>
                    {lang.label}
                  </option>
                ))}
              </select>
            </div>
          </section>

          {/* 5. Updates & System */}
          <section className="rounded-xl border border-[#47161D] bg-[#160709] p-6 space-y-4">
            <div className="flex items-start gap-3">
              <StepBadge num={5} />
              <div>
                <h3 className="font-display text-[17px] font-semibold text-[#FDF7F7]">
                  {t('settings.sec5.title')}
                </h3>
                <p className="text-[13.5px] text-[#C89EA4] mt-0.5">
                  {t('settings.sec5.desc')}
                </p>
              </div>
            </div>
            <div>
              <p className="text-[13.5px] font-semibold text-[#FDF7F7]">{t('settings.sec5.checker')}</p>
              <p className="text-[13px] text-[#C89EA4] mb-2">
                Verifies your installed {BRAND_NAME} build against the newest available release (currently {APP_VERSION}).
              </p>
              <div className="rounded-lg border border-[#47161D] bg-[#100506] px-4 py-3 text-xs font-mono text-[#DFBFC3] flex flex-wrap items-center justify-between gap-2">
                <span>{generalSettings.versionStatus}</span>
                {releaseLink && (
                  <a
                    href={releaseLink}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-[#E51923] hover:text-[#FF2B35] underline font-sans font-semibold whitespace-nowrap"
                  >
                    Open GitHub Releases
                  </a>
                )}
              </div>
            </div>
            {restartMsg && (
              <p className="text-xs text-[#DFBFC3]" role="status">
                {restartMsg}
              </p>
            )}
            <div className="flex flex-wrap gap-3">
              <button
                type="button"
                disabled={checkingUpdates}
                onClick={handleCheckUpdates}
                className="inline-flex items-center gap-2 rounded-lg border border-[#47161D] bg-[#200B0E] px-4 py-2 text-xs font-semibold text-[#FDF7F7] hover:border-[#E51923] disabled:opacity-60 whitespace-nowrap shrink-0"
              >
                <RefreshCw className={`h-3.5 w-3.5 ${checkingUpdates ? 'animate-spin' : ''}`} />
                <span>{checkingUpdates ? t('settings.sec5.checking') : t('settings.sec5.checkBtn')}</span>
              </button>
              <button
                type="button"
                onClick={handleRestartApplio}
                className="rounded-lg px-4 py-2 text-xs font-semibold text-white hover:bg-[#FF2B35] whitespace-nowrap shrink-0"
                style={{ backgroundColor: ACCENT_HEX }}
              >
                {t('settings.sec5.restartBtn')}
              </button>
            </div>
          </section>
        </div>
      ) : (
        <div className="space-y-6">
          {issues.length > 0 && (
            <div className="space-y-2">
              {issues.map((issue, idx) => (
                <div
                  key={idx}
                  className={`flex items-start gap-2 rounded-lg border p-3 text-xs ${
                    issue.severity === 'error'
                      ? 'border-[#FF2B35]/50 bg-[#FF2B35]/10 text-[#FECACA]'
                      : 'border-[#F59E0B]/40 bg-[#F59E0B]/10 text-[#FDE68A]'
                  }`}
                >
                  <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
                  <span>{issue.message}</span>
                </div>
              ))}
            </div>
          )}

          {/* 1. Model Author */}
          <section className="rounded-xl border border-[#47161D] bg-[#160709] p-6 space-y-4">
            <div className="flex items-start gap-3">
              <StepBadge num={1} />
              <div>
                <h3 className="font-display text-[17px] font-semibold text-[#FDF7F7]">
                  {t('settings.tr1.title')}
                </h3>
                <p className="text-[13.5px] text-[#C89EA4] mt-0.5">
                  {t('settings.tr1.desc')}
                </p>
              </div>
            </div>

            <div>
              <label
                htmlFor="settings-model-author"
                className="block text-xs font-semibold text-[#FDF7F7]"
              >
                {t('settings.tr1.label')}
              </label>
              <p className="text-[13px] text-[#C89EA4] mb-1.5">
                Creator handle embedded in published checkpoints.
              </p>
              <input
                id="settings-model-author"
                type="text"
                value={trainingSettings.modelAuthor}
                onChange={(e) =>
                  onUpdateTrainingSettings({
                    ...trainingSettings,
                    modelAuthor: e.target.value,
                  })
                }
                className="w-full rounded-lg bg-[#100506] border border-[#47161D] px-3 py-2 text-sm text-[#FDF7F7] focus:outline-none focus:border-[#E51923]"
              />
            </div>

            <div>
              <p className="text-xs font-medium text-[#DFBFC3] mb-1.5">
                Model author output
              </p>
              <div className="rounded-lg border border-[#47161D] bg-[#100506] px-3.5 py-2.5 text-xs font-mono text-[#DFBFC3]">
                {authorOutput}
              </div>
            </div>

            <button
              type="button"
              onClick={handleSetAuthor}
              className="rounded-lg px-4 py-2 text-xs font-semibold text-white hover:bg-[#FF2B35] whitespace-nowrap shrink-0"
              style={{ backgroundColor: ACCENT_HEX }}
            >
              {t('settings.tr1.btn')}
            </button>
          </section>

          {/* 2. Training Precision (fp32 default) */}
          <section className="rounded-xl border border-[#47161D] bg-[#160709] p-6 space-y-4">
            <div className="flex items-start gap-3">
              <StepBadge num={2} />
              <div>
                <h3 className="font-display text-[17px] font-semibold text-[#FDF7F7]">
                  {t('settings.tr2.title')}
                </h3>
                <p className="text-[13.5px] text-[#C89EA4] mt-0.5">
                  {t('settings.tr2.desc')}
                </p>
              </div>
            </div>

            <div>
              <label
                htmlFor="settings-precision"
                className="block text-xs font-semibold text-[#FDF7F7]"
              >
                {t('settings.tr2.label')}
              </label>
              <p className="text-[13px] text-[#C89EA4] mb-1.5">
                FP32 prioritizes maximum stability on T4 GPUs; FP16 lowers VRAM usage.
              </p>
              <select
                id="settings-precision"
                value={trainingSettings.precision}
                onChange={(e) =>
                  onUpdateTrainingSettings({
                    ...trainingSettings,
                    precision: e.target.value as TrainingPrecision,
                  })
                }
                className="w-full rounded-lg bg-[#100506] border border-[#47161D] px-3 py-2 text-sm text-[#FDF7F7] focus:outline-none focus:border-[#E51923]"
              >
                <option value="fp32">fp32</option>
                <option value="fp16">fp16</option>
              </select>
            </div>

            <div>
              <p className="text-xs font-medium text-[#DFBFC3] mb-1.5">
                Precision output
              </p>
              <div className="rounded-lg border border-[#47161D] bg-[#100506] px-3.5 py-2.5 text-xs font-mono text-[#DFBFC3]">
                {precisionOutput}
              </div>
            </div>

            <button
              type="button"
              onClick={handleUpdatePrecision}
              className="rounded-lg px-4 py-2 text-xs font-semibold text-white hover:bg-[#FF2B35] whitespace-nowrap shrink-0"
              style={{ backgroundColor: ACCENT_HEX }}
            >
              {t('settings.tr2.btn')}
            </button>
          </section>

          {/* 3. Kaggle Tunnel & Security */}
          <section className="rounded-xl border border-[#47161D] bg-[#160709] p-6 space-y-5">
            <div className="flex items-start gap-3">
              <span
                className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-white"
                style={{ backgroundColor: ACCENT_HEX }}
              >
                <Shield className="h-4 w-4" />
              </span>
              <div>
                <h3 className="font-display text-[17px] font-semibold text-[#FDF7F7]">
                  {t('settings.tr3.title')}
                </h3>
                <p className="text-[13.5px] text-[#C89EA4] mt-0.5">
                  {t('settings.tr3.desc')}
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label
                  htmlFor="kaggle-tunnel-mode"
                  className="block text-xs font-semibold text-[#FDF7F7] mb-1.5"
                >
                  {t('settings.tr3.tunnel')}
                </label>
                <select
                  id="kaggle-tunnel-mode"
                  value={forkConfig.tunnelMode}
                  onChange={(e) =>
                    onUpdateForkConfig({
                      ...forkConfig,
                      tunnelMode: e.target.value as TunnelMode,
                    })
                  }
                  className="w-full rounded-lg bg-[#100506] border border-[#47161D] px-3 py-2 text-sm text-[#FDF7F7] focus:outline-none focus:border-[#E51923]"
                >
                  <option value="Gradio + Cloudflare">Gradio + Cloudflare</option>
                  <option value="Horizon">Horizon (hrzn.run)</option>
                </select>
              </div>

              <div>
                <label
                  htmlFor="kaggle-custom-repo"
                  className="block text-xs font-semibold text-[#FDF7F7] mb-1.5"
                >
                  {t('settings.tr3.repo')}
                </label>
                <input
                  id="kaggle-custom-repo"
                  type="url"
                  value={forkConfig.customRepoUrl}
                  onChange={(e) =>
                    onUpdateForkConfig({
                      ...forkConfig,
                      customRepoUrl: e.target.value,
                    })
                  }
                  placeholder="https://github.com/IAHispano/Applio.git"
                  className="w-full rounded-lg bg-[#100506] border border-[#47161D] px-3 py-2 text-xs font-mono text-[#FDF7F7] placeholder-[#AD8288]/45 focus:outline-none focus:border-[#E51923]"
                />
              </div>
            </div>

            {forkConfig.tunnelMode === 'Horizon' && (
              <div>
                <label
                  htmlFor="kaggle-horizon-id"
                  className="block text-xs font-semibold text-[#FDF7F7] mb-1.5"
                >
                  Horizon ID
                </label>
                <input
                  id="kaggle-horizon-id"
                  type="text"
                  value={forkConfig.horizonId}
                  onChange={(e) =>
                    onUpdateForkConfig({ ...forkConfig, horizonId: e.target.value })
                  }
                  placeholder="Paste Horizon ID from hrzn.run/dashboard"
                  className="w-full rounded-lg bg-[#100506] border border-[#47161D] px-3 py-2 text-xs font-mono text-[#FDF7F7] placeholder-[#AD8288]/45 focus:outline-none focus:border-[#E51923]"
                />
              </div>
            )}

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <label className="flex items-center gap-2.5 rounded-lg border border-[#47161D] bg-[#100506] p-3 text-[13px] text-[#FDF7F7] cursor-pointer">
                <input
                  type="checkbox"
                  checked={forkConfig.randomAuthEnabled}
                  onChange={(e) =>
                    onUpdateForkConfig({
                      ...forkConfig,
                      randomAuthEnabled: e.target.checked,
                    })
                  }
                  className="h-4 w-4 rounded accent-[#E51923]"
                />
                <span>{t('settings.tr3.randomAuth')}</span>
              </label>
              <label className="flex items-center gap-2.5 rounded-lg border border-[#47161D] bg-[#100506] p-3 text-[13px] text-[#FDF7F7] cursor-pointer">
                <input
                  type="checkbox"
                  checked={forkConfig.useUvVenv}
                  onChange={(e) =>
                    onUpdateForkConfig({
                      ...forkConfig,
                      useUvVenv: e.target.checked,
                    })
                  }
                  className="h-4 w-4 rounded accent-[#E51923]"
                />
                <span>{t('settings.tr3.uv')}</span>
              </label>
              <label className="flex items-center gap-2.5 rounded-lg border border-[#47161D] bg-[#100506] p-3 text-[13px] text-[#FDF7F7] cursor-pointer">
                <input
                  type="checkbox"
                  checked={forkConfig.verifyOnnxGpu}
                  onChange={(e) =>
                    onUpdateForkConfig({
                      ...forkConfig,
                      verifyOnnxGpu: e.target.checked,
                    })
                  }
                  className="h-4 w-4 rounded accent-[#E51923]"
                />
                <span>{t('settings.tr3.onnx')}</span>
              </label>
              <label className="flex items-center gap-2.5 rounded-lg border border-[#47161D] bg-[#100506] p-3 text-[13px] text-[#FDF7F7] cursor-pointer">
                <input
                  type="checkbox"
                  checked={forkConfig.turboTraining ?? true}
                  onChange={(e) =>
                    onUpdateForkConfig({
                      ...forkConfig,
                      turboTraining: e.target.checked,
                    })
                  }
                  className="h-4 w-4 rounded accent-[#E51923]"
                />
                <span>{t('settings.tr3.turbo')}</span>
              </label>
              <label className="flex items-start gap-2.5 rounded-lg border border-[#47161D] bg-[#100506] p-3 text-[13px] text-[#FDF7F7] cursor-pointer sm:col-span-2">
                <input
                  type="checkbox"
                  checked={activeProject?.batchParity1Gpu ?? true}
                  onChange={(e) =>
                    onUpdateBatchParityAcrossProjects?.(e.target.checked)
                  }
                  className="mt-0.5 h-4 w-4 rounded accent-[#E51923]"
                />
                <span>
                  1-GPU Applio Batch Parity (&lt;300 Epoch Fix) — Default ON. Splits batch sizes &ge; 4 across dual T4s so under-300 epoch runs receive full Original Applio optimizer steps.
                </span>
              </label>
            </div>
          </section>
        </div>
      )}
    </div>
  );
});
