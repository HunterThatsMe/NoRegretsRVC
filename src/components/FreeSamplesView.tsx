import React, { useMemo, useState } from 'react';
import { ArrowRight, Download, FolderArchive, Mic, Users } from 'lucide-react';
import { ACCENT_HEX } from '../data/defaults';
import { FREE_SAMPLES, FreeSampleItem } from '../data/freeSamples';
import { triggerBlobDownload } from '../utils/download';
import { useI18n } from '../utils/i18n';
import { CrimsonAudioPlayer } from './CrimsonAudioPlayer';

interface FreeSamplesViewProps {
  queuedFileNames: string[];
  onUseSample: (sample: FreeSampleItem) => void;
  onUseAllSamples: (samples: FreeSampleItem[]) => void;
}

type GenderFilter = 'all' | 'male' | 'female';

export const FreeSamplesView: React.FC<FreeSamplesViewProps> = React.memo(({
  queuedFileNames,
  onUseSample,
  onUseAllSamples,
}) => {
  const { t } = useI18n();
  const [genderFilter, setGenderFilter] = useState<GenderFilter>('all');
  const [isZipping, setIsZipping] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');

  const maleSamples = useMemo(
    () => FREE_SAMPLES.filter((s) => s.gender === 'male'),
    []
  );
  const femaleSamples = useMemo(
    () => FREE_SAMPLES.filter((s) => s.gender === 'female'),
    []
  );

  const visibleGroups = useMemo(() => {
    const groups: { title: string; subtitle: string; items: FreeSampleItem[] }[] = [];
    if (genderFilter === 'all' || genderFilter === 'male') {
      groups.push({
        title: 'Male Speaking Samples',
        subtitle: 'Dry mono 32 kHz WAV recordings across narration, voiceover, and casual dialogue.',
        items: maleSamples,
      });
    }
    if (genderFilter === 'all' || genderFilter === 'female') {
      groups.push({
        title: 'Female Speaking Samples',
        subtitle: 'Dry mono 32 kHz WAV recordings across calm support, clear reading, and assistant speech.',
        items: femaleSamples,
      });
    }
    return groups;
  }, [genderFilter, maleSamples, femaleSamples]);

  const handleDownloadAllZip = async () => {
    if (isZipping) return;
    setIsZipping(true);
    setStatusMessage('');
    try {
      const entries = await Promise.all(
        FREE_SAMPLES.map(async (sample) => {
          const res = await fetch(sample.url);
          if (!res.ok) {
            throw new Error(`Failed to fetch ${sample.fileName}`);
          }
          const buf = await res.arrayBuffer();
          return {
            path: `NoRegretsRVC_Free_Samples/${sample.fileName}`,
            content: new Uint8Array(buf),
          };
        })
      );
      const { buildZipArchive } = await import('../utils/zipBuilder');
      const zipBytes = buildZipArchive(entries);
      triggerBlobDownload(
        'NoRegretsRVC_Free_Samples.zip',
        new Uint8Array(zipBytes).buffer,
        'application/zip'
      );
      setStatusMessage('Downloaded NoRegretsRVC_Free_Samples.zip (all 6 mono WAV clips).');
    } catch {
      setStatusMessage('Could not bundle one or more WAV files. Try downloading individual clips below.');
    } finally {
      setIsZipping(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-4 px-1">
        <div>
          <h2 className="font-display text-2xl font-bold text-[#FDF7F7] tracking-tight">
            {t('freeSamples.title')}
          </h2>
          <p className="text-sm text-[#DFBFC3] mt-1">
            Ready-to-use dry mono talking recordings (3 male, 3 female) to preview, download for Kaggle, or load straight into Samples.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2 shrink-0">
          <button
            type="button"
            onClick={() => onUseAllSamples(FREE_SAMPLES)}
            className="inline-flex items-center gap-1.5 rounded-lg px-3.5 py-2 text-xs font-semibold text-white transition-colors hover:bg-[#FF2B35] whitespace-nowrap shrink-0"
            style={{ backgroundColor: ACCENT_HEX }}
          >
            <Mic className="h-3.5 w-3.5" />
            <span>{t('freeSamples.loadAll')}</span>
          </button>

          <button
            type="button"
            onClick={handleDownloadAllZip}
            disabled={isZipping}
            className="inline-flex items-center gap-1.5 rounded-lg border border-[#47161D] bg-[#200B0E] px-3.5 py-2 text-xs font-semibold text-[#FDF7F7] hover:border-[#E51923] disabled:opacity-60 whitespace-nowrap shrink-0"
          >
            <FolderArchive className="h-3.5 w-3.5 text-[#E51923]" />
            <span>{isZipping ? 'Packing ZIP...' : t('freeSamples.zipAll')}</span>
          </button>
        </div>
      </div>

      {/* Filter Bar */}
      <div className="rounded-xl border border-[#47161D] bg-[#160709] p-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-xs text-[#DFBFC3]">
          <Users className="h-4 w-4 text-[#E51923] shrink-0" />
          <span>
            Clean, dry mono WAVs (32 kHz) with zero background music or reverb — ideal for RVC conversion tests.
          </span>
        </div>

        <div className="inline-flex rounded-lg border border-[#47161D] bg-[#100506] p-1 shrink-0">
          {(
            [
              { id: 'all', label: `All (${FREE_SAMPLES.length})` },
              { id: 'male', label: `Male (${maleSamples.length})` },
              { id: 'female', label: `Female (${femaleSamples.length})` },
            ] as const
          ).map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setGenderFilter(tab.id)}
              className={`rounded-md px-3 py-1.5 text-xs font-semibold transition-colors whitespace-nowrap shrink-0 ${
                genderFilter === tab.id
                  ? 'bg-[#E51923] text-white'
                  : 'text-[#DFBFC3] hover:text-white'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {statusMessage && (
        <div
          role="status"
          className="rounded-lg border border-[#47161D] bg-[#160709] px-4 py-2.5 text-xs font-mono text-[#DFBFC3]"
        >
          {statusMessage}
        </div>
      )}

      {/* Sample Groups */}
      {visibleGroups.map((group) => (
        <section
          key={group.title}
          className="rounded-xl border border-[#47161D] bg-[#160709] p-6 space-y-4"
        >
          <div>
            <h3 className="font-display text-base font-semibold text-[#FDF7F7]">
              {group.title} ({group.items.length})
            </h3>
            <p className="text-xs text-[#AD8288] mt-0.5">{group.subtitle}</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {group.items.map((sample) => {
              const isInQueue = queuedFileNames.includes(sample.fileName);
              return (
                <article
                  key={sample.id}
                  className="rounded-xl border border-[#47161D] bg-[#100506] p-4 flex flex-col justify-between gap-3.5"
                >
                  <div className="space-y-2.5">
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <h4 className="font-display text-sm font-semibold text-[#FDF7F7]">
                          {sample.title}
                        </h4>
                        <p className="text-[11px] text-[#AD8288]">
                          {sample.style} · {sample.durationLabel}
                        </p>
                      </div>
                      <span className="rounded-md border border-[#47161D] bg-[#160709] px-2 py-0.5 font-mono text-[10px] text-[#DFBFC3] whitespace-nowrap shrink-0">
                        Mono 32k WAV
                      </span>
                    </div>

                    <p className="rounded-lg border border-[#47161D]/80 bg-[#160709] px-3 py-2 text-xs text-[#DFBFC3] leading-relaxed">
                      &ldquo;{sample.transcript}&rdquo;
                    </p>
                  </div>

                  <div className="space-y-2.5">
                    <CrimsonAudioPlayer
                      src={sample.url}
                      label={sample.fileName}
                      compact
                    />

                    <div className="grid grid-cols-2 gap-2">
                      <button
                        type="button"
                        onClick={() => onUseSample(sample)}
                        className="inline-flex items-center justify-center gap-1.5 rounded-lg px-3 py-2 text-xs font-semibold text-white transition-colors hover:bg-[#FF2B35] whitespace-nowrap"
                        style={{ backgroundColor: ACCENT_HEX }}
                      >
                        <span>{isInQueue ? 'Add Again' : 'Use in Samples'}</span>
                        <ArrowRight className="h-3.5 w-3.5 shrink-0" />
                      </button>

                      <a
                        href={sample.url}
                        download={sample.fileName}
                        className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-[#47161D] bg-[#200B0E] px-3 py-2 text-xs font-semibold text-[#FDF7F7] hover:border-[#E51923] whitespace-nowrap"
                      >
                        <Download className="h-3.5 w-3.5 shrink-0" />
                        <span>Download</span>
                      </a>
                    </div>
                  </div>
                </article>
              );
            })}
          </div>
        </section>
      ))}
    </div>
  );
});
