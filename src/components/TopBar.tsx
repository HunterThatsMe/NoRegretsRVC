import React from 'react';
import {
  Activity,
  AudioWaveform,
  Combine,
  Download,
  Mic,
  Settings,
  Sliders,
  Volume2,
} from 'lucide-react';
import { ACCENT_HEX, APP_VERSION, BRAND_NAME } from '../data/defaults';
import { StudioSection, VoiceModelProject } from '../types/studio';
import { TranslationKey, useI18n } from '../utils/i18n';

interface TopBarProps {
  activeSection: StudioSection;
  onSelectSection: (section: StudioSection) => void;
  onPrefetchSection?: (section: StudioSection) => void;
  activeProject: VoiceModelProject;
}

const TOP_TABS: {
  id: StudioSection;
  labelKey: TranslationKey;
  icon: React.ComponentType<{ className?: string }>;
}[] = [
  { id: 'training', labelKey: 'nav.training', icon: AudioWaveform },
  { id: 'models', labelKey: 'nav.models', icon: Download },
  { id: 'samples', labelKey: 'nav.samples', icon: Mic },
  { id: 'free_samples', labelKey: 'nav.free_samples', icon: Volume2 },
  { id: 'tensorboard', labelKey: 'nav.diagnostics', icon: Activity },
  { id: 'blender', labelKey: 'nav.blender', icon: Combine },
  { id: 'tts', labelKey: 'nav.tts', icon: Sliders },
  { id: 'settings', labelKey: 'nav.settings', icon: Settings },
];

export const TopBar: React.FC<TopBarProps> = React.memo(
  ({
    activeSection,
    onSelectSection,
    onPrefetchSection,
    activeProject,
  }) => {
    const { t } = useI18n();

    return (
      <header className="w-full max-w-[1440px] mx-auto px-4 md:px-8 pt-5 pb-2 space-y-3">
        {/* Top Framed Brand & Active Model Banner */}
        <div className="rounded-xl border border-[#47161D] bg-[#160709] px-5 py-4 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3.5 min-w-0">
            <span
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg text-white font-display font-bold text-base"
              style={{ backgroundColor: ACCENT_HEX }}
            >
              NR
            </span>
            <div className="min-w-0">
              <div className="flex flex-wrap items-baseline gap-2.5">
                <h1 className="font-display text-xl sm:text-2xl font-bold tracking-tight text-[#FDF7F7] leading-none">
                  {BRAND_NAME}
                </h1>
                <span className="text-[11px] font-medium text-[#DFBFC3]/55">
                  Beta version {APP_VERSION}
                </span>
              </div>
              <p className="text-xs text-[#AD8288] mt-1 truncate">
                {t('topbar.subtitle')}
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <div className="rounded-lg border border-[#47161D] bg-[#100506] px-3.5 py-2 text-right">
              <p className="text-xs font-semibold text-[#FDF7F7]">
                {activeProject.name.trim() || t('topbar.untitled')} ({activeProject.samplingRate})
              </p>
              <p className="text-[11px] font-mono text-[#AD8288] tabular-nums mt-0.5">
                {activeProject.vocoder} · Batch {activeProject.batchSize} ({activeProject.batchSize * activeProject.gpuCount} eff.) · {activeProject.totalEpoch} ep
              </p>
            </div>
          </div>
        </div>

        {/* Horizontal Tab Navigation Bar (Training -> Models -> Samples -> Diagnostics -> Voice Blender -> TTS -> Settings) */}
        <nav
          aria-label="Workspace sections"
          className="rounded-xl border border-[#47161D] bg-[#160709] p-1.5 flex flex-wrap items-center gap-1.5"
        >
          {TOP_TABS.map((item) => {
            const Icon = item.icon;
            const isActive = activeSection === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onMouseEnter={() => onPrefetchSection?.(item.id)}
                onFocus={() => onPrefetchSection?.(item.id)}
                onClick={() => onSelectSection(item.id)}
                aria-current={isActive ? 'page' : undefined}
                className={`inline-flex items-center gap-2 rounded-lg px-3.5 py-2 text-xs sm:text-sm font-semibold transition-colors whitespace-nowrap shrink-0 ${
                  isActive
                    ? 'text-white'
                    : 'text-[#DFBFC3]/85 hover:text-[#FDF7F7] hover:bg-[#200B0E]'
                }`}
                style={isActive ? { backgroundColor: ACCENT_HEX } : undefined}
              >
                <Icon className="h-3.5 w-3.5 shrink-0" />
                <span>{t(item.labelKey)}</span>
              </button>
            );
          })}
        </nav>
      </header>
    );
  }
);
