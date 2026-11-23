import React, { useMemo, useRef, useState } from 'react';
import {
  ChevronDown,
  ChevronRight,
  Download,
  Play,
  RefreshCw,
  RotateCcw,
  SlidersHorizontal,
  Trash2,
  Upload,
  X,
} from 'lucide-react';
import { ACCENT_HEX } from '../data/defaults';
import {
  ConvertAudioState,
  EmbedderModel,
  GeneratedSampleItem,
  ImportedLocalModel,
  PitchExtractor,
  SampleClipItem,
  VoiceModelProject,
} from '../types/studio';
import { computeCleanupParameters, processAudioUrlWithCleanup } from '../utils/audioCleanup';
import { buildInferCliCommand, sanitizeModelName } from '../utils/rvcTuning';
import { useI18n } from '../utils/i18n';
import { CrimsonAudioPlayer } from './CrimsonAudioPlayer';

interface ConvertAudioViewProps {
  convertAudio: ConvertAudioState;
  projects: VoiceModelProject[];
  importedModels: ImportedLocalModel[];
  sampleQueue: SampleClipItem[];
  onSetSampleQueue: React.Dispatch<React.SetStateAction<SampleClipItem[]>>;
  onRegisterBlobUrl: (url: string) => string;
  onRevokeUrl: (url: string) => void;
  onUpdateConvertAudio: (updated: ConvertAudioState) => void;
}

const StepBadge: React.FC<{ num: number }> = ({ num }) => (
  <span
    className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-xs font-bold text-white tabular-nums"
    style={{ backgroundColor: ACCENT_HEX }}
  >
    {num}
  </span>
);

export const ConvertAudioView: React.FC<ConvertAudioViewProps> = React.memo(({
  convertAudio,
  projects,
  importedModels,
  sampleQueue,
  onSetSampleQueue: setSampleQueue,
  onRegisterBlobUrl: registerUrl,
  onRevokeUrl: revokeUrl,
  onUpdateConvertAudio,
}) => {
  const { t } = useI18n();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [advOpen, setAdvOpen] = useState(false);
  const [subOpen, setSubOpen] = useState<Record<string, boolean>>({
    voiceMatching: true,
    modelsAlgorithms: false,
    pitchTuning: false,
    audioCleanup: false,
    formantShifting: false,
    outputSettings: false,
  });
  const [isConverting, setIsConverting] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [refreshMsg, setRefreshMsg] = useState('');
  const [expandedClipAdv, setExpandedClipAdv] = useState<Record<string, boolean>>({});

  const cleanupCalibration = useMemo(
    () => computeCleanupParameters(convertAudio.noiseReductionStrength),
    [convertAudio.noiseReductionStrength]
  );

  const toggleSub = (key: string) =>
    setSubOpen((prev) => ({ ...prev, [key]: !prev[key] }));

  const safeRevokeConvertedUrl = (url: string, queue = sampleQueue) => {
    if (!url) return;
    const stillUsedAsSource = queue.some((c) => c.sourceUrl === url);
    if (!stillUsedAsSource) {
      revokeUrl(url);
    }
  };

  const modelOptions = useMemo(() => {
    const projectModels = projects.map((p) => {
      const safe = sanitizeModelName(p.name);
      return p.exportPthFile || `logs/${safe}/${safe}_${p.totalEpoch}e.pth`;
    });
    return Array.from(
      new Set([...projectModels, ...importedModels.map((m) => m.pthPath)].filter(Boolean))
    );
  }, [projects, importedModels]);

  const indexOptions = useMemo(() => {
    const projectIndexes = projects.map((p) => {
      const safe = sanitizeModelName(p.name);
      return p.exportIndexFile || `logs/${safe}/${safe}.index`;
    });
    return Array.from(
      new Set([...projectIndexes, ...importedModels.map((m) => m.indexPath)].filter(Boolean))
    );
  }, [projects, importedModels]);

  const inferCliCommand = useMemo(
    () => buildInferCliCommand(convertAudio),
    [convertAudio]
  );

  const buildQueueItem = (
    fileName: string,
    sourceUrl: string,
    offsetIdx: number
  ): SampleClipItem => ({
    id: `clip-${Date.now()}-${offsetIdx}-${Math.random().toString(36).slice(2, 6)}`,
    fileName,
    sourceUrl,
    pitch: convertAudio.pitch,
    indexRate: convertAudio.indexRate,
    voiceModel: convertAudio.voiceModel || modelOptions[0] || '',
    indexFile: convertAudio.indexFile || '',
    protect: convertAudio.protect,
    volumeEnvelope: convertAudio.volumeEnvelope,
    speakerId: convertAudio.speakerId,
    pitchExtractor: convertAudio.pitchExtractor,
    embedder: convertAudio.embedder,
    splitAudio: convertAudio.splitAudio,
    autotune: convertAudio.autotune,
    proposedPitch: convertAudio.proposedPitch,
    outputFormat: convertAudio.outputFormat,
    convertedUrl: '',
    convertedName: '',
    status: 'queued',
  });

  const handleFilesChosen = (files: FileList | null | undefined) => {
    if (!files || files.length === 0) return;
    const fileArray = Array.from(files);
    const addedClips: SampleClipItem[] = fileArray.map((file, idx) => {
      const objUrl = registerUrl(URL.createObjectURL(file));
      return buildQueueItem(file.name, objUrl, idx);
    });

    setSampleQueue((prev) => [...prev, ...addedClips]);

    const first = addedClips[0];
    onUpdateConvertAudio({
      ...convertAudio,
      audioFileName:
        addedClips.length === 1
          ? first.fileName
          : `${sampleQueue.length + addedClips.length} audio clips loaded`,
      audioObjectUrl: first.sourceUrl,
      conversionOutput:
        addedClips.length === 1
          ? `Added "${first.fileName}" to sample queue. Adjust its Pitch & Index or click Convert.`
          : `Added ${addedClips.length} audio clips to sample queue. Customize each clip individually below.`,
    });
  };

  const handleDiscardClip = (clipId: string) => {
    setSampleQueue((prev) => {
      const target = prev.find((c) => c.id === clipId);
      if (target) {
        revokeUrl(target.sourceUrl);
        revokeUrl(target.convertedUrl);
      }
      const remaining = prev.filter((c) => c.id !== clipId);
      const nextActive = remaining[0];
      onUpdateConvertAudio({
        ...convertAudio,
        audioFileName:
          remaining.length === 0
            ? ''
            : remaining.length === 1
              ? nextActive.fileName
              : `${remaining.length} audio clips loaded`,
        audioObjectUrl: nextActive ? nextActive.sourceUrl : '',
        conversionOutput:
          remaining.length === 0
            ? 'Discarded sample recording. Upload or drop new audio files when ready.'
            : `Discarded "${target?.fileName ?? 'clip'}". ${remaining.length} clip${remaining.length === 1 ? '' : 's'} remaining.`,
      });
      return remaining;
    });
  };

  const handleClearAllClips = () => {
    sampleQueue.forEach((c) => {
      revokeUrl(c.sourceUrl);
      revokeUrl(c.convertedUrl);
    });
    setSampleQueue([]);
    onUpdateConvertAudio({
      ...convertAudio,
      audioFileName: '',
      audioObjectUrl: '',
      conversionOutput: 'Cleared all queued audio recordings.',
    });
  };

  const handleDiscardSavedSample = (sampleId: string) => {
    const target = convertAudio.savedSamples.find((s) => s.id === sampleId);
    if (target) {
      safeRevokeConvertedUrl(target.url);
    }
    const nextSaved = convertAudio.savedSamples.filter((s) => s.id !== sampleId);
    onUpdateConvertAudio({
      ...convertAudio,
      savedSamples: nextSaved,
      conversionOutput: target
        ? `Discarded generated sample "${target.name}".`
        : convertAudio.conversionOutput,
    });
  };

  const handleClearAllOutputs = () => {
    convertAudio.savedSamples.forEach((s) => safeRevokeConvertedUrl(s.url));
    setSampleQueue((prev) =>
      prev.map((c) => {
        safeRevokeConvertedUrl(c.convertedUrl, prev);
        return {
          ...c,
          convertedUrl: '',
          convertedName: '',
          status: 'queued',
        };
      })
    );
    onUpdateConvertAudio({
      ...convertAudio,
      savedSamples: [],
      hasConvertedSample: false,
      conversionOutput: 'Cleared all generated outputs in assets/audios.',
    });
  };

  const handleUpdateClip = (
    clipId: string,
    patch: Partial<SampleClipItem>
  ) => {
    setSampleQueue((prev) =>
      prev.map((c) => (c.id === clipId ? { ...c, ...patch } : c))
    );
  };

  const handleApplyDefaultsToQueue = () => {
    setSampleQueue((prev) =>
      prev.map((c) => ({
        ...c,
        pitch: convertAudio.pitch,
        indexRate: convertAudio.indexRate,
        voiceModel: convertAudio.voiceModel || c.voiceModel,
        indexFile: convertAudio.indexFile,
        protect: convertAudio.protect,
        volumeEnvelope: convertAudio.volumeEnvelope,
        speakerId: convertAudio.speakerId,
        pitchExtractor: convertAudio.pitchExtractor,
        embedder: convertAudio.embedder,
        splitAudio: convertAudio.splitAudio,
        autotune: convertAudio.autotune,
        proposedPitch: convertAudio.proposedPitch,
        outputFormat: convertAudio.outputFormat,
      }))
    );
    onUpdateConvertAudio({
      ...convertAudio,
      conversionOutput: `Applied default settings (Pitch ${convertAudio.pitch >= 0 ? `+${convertAudio.pitch}` : convertAudio.pitch}, Index ${convertAudio.indexRate.toFixed(2)}) to all ${sampleQueue.length} queued clips.`,
    });
  };

  const handleUnloadVoice = () => {
    onUpdateConvertAudio({
      ...convertAudio,
      voiceModel: '',
      indexFile: '',
      conversionOutput: 'Voice model unloaded from VRAM.',
    });
  };

  const handleRefreshModels = () => {
    setRefreshMsg('Refreshed trained and imported .pth models and .index files.');
    setTimeout(() => setRefreshMsg(''), 2200);
  };

  const renderClipSample = async (
    clip: SampleClipItem
  ): Promise<{
    updatedClip: SampleClipItem;
    sampleRecord: GeneratedSampleItem;
    cleanedApplied: boolean;
  }> => {
    let outUrl = clip.sourceUrl;
    let cleanedApplied = false;

    if (convertAudio.noiseReduction) {
      try {
        const cleanedBlob = await processAudioUrlWithCleanup(
          clip.sourceUrl,
          convertAudio.noiseReductionStrength
        );
        outUrl = registerUrl(URL.createObjectURL(cleanedBlob));
        cleanedApplied = true;
      } catch {
        outUrl = clip.sourceUrl;
      }
    }

    const baseName = (clip.fileName || 'voice_sample').replace(/\.[^.]+$/, '');
    const pitchTag = clip.pitch >= 0 ? `+${clip.pitch}` : `${clip.pitch}`;
    const cleanTag = cleanedApplied ? '_clean' : '';
    const sampleName = `${baseName}_p${pitchTag}${cleanTag}.${clip.outputFormat}`;
    const chosenModel = clip.voiceModel || convertAudio.voiceModel;
    const modelLabel = chosenModel.split('/').pop() || chosenModel;

    const updatedClip: SampleClipItem = {
      ...clip,
      convertedUrl: outUrl,
      convertedName: sampleName,
      status: 'ready',
    };

    const sampleRecord: GeneratedSampleItem = {
      id: `sample-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      name: sampleName,
      pitch: clip.pitch,
      format: clip.outputFormat,
      url: outUrl,
      createdAt: new Date().toLocaleTimeString(),
      voiceModel: modelLabel,
      indexRate: clip.indexRate,
    };

    return { updatedClip, sampleRecord, cleanedApplied };
  };

  const handleConvertSingleClip = async (clipId: string) => {
    if (isConverting) return;
    if (!convertAudio.agreedTerms) {
      onUpdateConvertAudio({
        ...convertAudio,
        conversionOutput: 'Please agree to the terms of use before generating samples.',
      });
      return;
    }

    const clip = sampleQueue.find((c) => c.id === clipId);
    if (!clip) return;

    const chosenModel = clip.voiceModel || convertAudio.voiceModel;
    if (!chosenModel.trim()) {
      onUpdateConvertAudio({
        ...convertAudio,
        conversionOutput: 'Select a Voice Model (.pth) in Step 1 or on the clip first.',
      });
      return;
    }

    setIsConverting(true);
    try {
      const { updatedClip, sampleRecord, cleanedApplied } = await renderClipSample(clip);
      setSampleQueue((prev) => prev.map((c) => (c.id === clipId ? updatedClip : c)));

      const nextSamples = [sampleRecord, ...convertAudio.savedSamples].slice(0, 12);
      const evicted = convertAudio.savedSamples.slice(11);
      evicted.forEach((s) => safeRevokeConvertedUrl(s.url));

      const cleanupNote = cleanedApplied
        ? ` · Audio cleanup ${cleanupCalibration.tierLabel} (${convertAudio.noiseReductionStrength} dB)`
        : '';

      onUpdateConvertAudio({
        ...convertAudio,
        hasConvertedSample: true,
        savedSamples: nextSamples,
        conversionOutput: `Converted "${sampleRecord.name}" (Pitch ${clip.pitch >= 0 ? `+${clip.pitch}` : clip.pitch}, Index ${clip.indexRate.toFixed(2)}${cleanupNote}).`,
      });
    } finally {
      setIsConverting(false);
    }
  };

  const handleConvert = async () => {
    if (isConverting) return;
    if (!convertAudio.agreedTerms) {
      onUpdateConvertAudio({
        ...convertAudio,
        conversionOutput: 'Please agree to the terms of use before generating samples.',
      });
      return;
    }
    if (convertAudio.sourceMode === 'batch') {
      if (!convertAudio.voiceModel.trim()) {
        onUpdateConvertAudio({
          ...convertAudio,
          conversionOutput: 'Select a Voice Model (.pth) in Step 1 first.',
        });
        return;
      }
      onUpdateConvertAudio({
        ...convertAudio,
        hasConvertedSample: true,
        conversionOutput: `Batch conversion queued for "${convertAudio.batchFolderPath}" using ${convertAudio.voiceModel}.`,
      });
      return;
    }

    if (sampleQueue.length === 0) {
      onUpdateConvertAudio({
        ...convertAudio,
        conversionOutput: 'Upload audio files in Step 2 first.',
      });
      return;
    }

    const missingModel = sampleQueue.some(
      (c) => !(c.voiceModel || convertAudio.voiceModel).trim()
    );
    if (missingModel) {
      onUpdateConvertAudio({
        ...convertAudio,
        conversionOutput: 'Select a Voice Model (.pth) in Step 1 before converting.',
      });
      return;
    }

    setIsConverting(true);
    try {
      const rendered = await Promise.all(sampleQueue.map((c) => renderClipSample(c)));
      setSampleQueue(rendered.map((r) => r.updatedClip));

      const newRecords = rendered.map((r) => r.sampleRecord).reverse();
      const nextSamples = [...newRecords, ...convertAudio.savedSamples].slice(0, 12);
      const evicted = [...newRecords, ...convertAudio.savedSamples].slice(12);
      evicted.forEach((s) => safeRevokeConvertedUrl(s.url));

      const anyCleaned = rendered.some((r) => r.cleanedApplied);
      const cleanSummary = anyCleaned
        ? ` with ${cleanupCalibration.tierLabel} audio cleanup (${convertAudio.noiseReductionStrength} dB)`
        : '';

      onUpdateConvertAudio({
        ...convertAudio,
        hasConvertedSample: true,
        savedSamples: nextSamples,
        conversionOutput:
          rendered.length === 1
            ? `Sample "${rendered[0].sampleRecord.name}" ready${cleanSummary}. Play or click Save to download.`
            : `Converted all ${rendered.length} samples${cleanSummary}.`,
      });
    } finally {
      setIsConverting(false);
    }
  };

  const handleSaveSample = (sampleUrl?: string, sampleFilename?: string) => {
    if (!sampleUrl) return;
    const baseName = (convertAudio.audioFileName || 'converted_sample').replace(/\.[^.]+$/, '');
    const filename = sampleFilename || `${baseName}_sample.${convertAudio.outputFormat}`;
    const a = document.createElement('a');
    a.href = sampleUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  return (
    <div className="space-y-6">
      <div className="px-1">
        <h2 className="font-display text-2xl font-bold text-[#FDF7F7] tracking-tight">
          {t('samples.title')}
        </h2>
        <p className="text-sm text-[#DFBFC3] mt-1">
          {t('samples.subtitle')}
        </p>
      </div>

      {/* STEP 1: Voice Model */}
      <section className="rounded-xl border border-[#47161D] bg-[#160709] p-6 space-y-4">
        <div className="flex items-start gap-3">
          <StepBadge num={1} />
          <div>
            <h3 className="font-display text-[17px] font-semibold text-[#FDF7F7]">
              Voice Model
            </h3>
            <p className="text-[13.5px] text-[#C89EA4] mt-0.5">
              Pick the default .pth voice checkpoint and optional FAISS .index file for new clips.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label
              htmlFor="convert-voice-model"
              className="block text-xs font-semibold text-[#FDF7F7] mb-1.5"
            >
              Voice Model
            </label>
            <select
              id="convert-voice-model"
              value={convertAudio.voiceModel}
              onChange={(e) =>
                onUpdateConvertAudio({ ...convertAudio, voiceModel: e.target.value })
              }
              className="w-full rounded-lg bg-[#100506] border border-[#47161D] px-3 py-2 text-xs font-mono text-[#FDF7F7] focus:outline-none focus:border-[#E51923]"
            >
              <option value="">-- Select Voice Model (.pth) --</option>
              {modelOptions.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label
              htmlFor="convert-index-file"
              className="block text-xs font-semibold text-[#FDF7F7] mb-1.5"
            >
              Index File
            </label>
            <select
              id="convert-index-file"
              value={convertAudio.indexFile}
              onChange={(e) =>
                onUpdateConvertAudio({ ...convertAudio, indexFile: e.target.value })
              }
              className="w-full rounded-lg bg-[#100506] border border-[#47161D] px-3 py-2 text-xs font-mono text-[#FDF7F7] focus:outline-none focus:border-[#E51923]"
            >
              <option value="">-- Select Index File (.index) --</option>
              {indexOptions.map((idx) => (
                <option key={idx} value={idx}>
                  {idx}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={handleUnloadVoice}
            className="rounded-lg border border-[#47161D] bg-[#200B0E] px-4 py-2 text-xs font-semibold text-[#FDF7F7] hover:border-[#E51923] whitespace-nowrap shrink-0"
          >
            Unload Voice
          </button>
          <button
            type="button"
            onClick={handleRefreshModels}
            className="inline-flex items-center gap-2 rounded-lg border border-[#47161D] bg-[#200B0E] px-4 py-2 text-xs font-semibold text-[#FDF7F7] hover:border-[#E51923] whitespace-nowrap shrink-0"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            <span>Refresh models and indexes</span>
          </button>
          {refreshMsg && (
            <span className="text-xs text-[#DFBFC3]" role="status">
              {refreshMsg}
            </span>
          )}
        </div>
      </section>

      {/* STEP 2: Audio Source & Multi-Sample Queue */}
      <section className="rounded-xl border border-[#47161D] bg-[#160709] p-6 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-start gap-3">
            <StepBadge num={2} />
            <div>
              <h3 className="font-display text-[17px] font-semibold text-[#FDF7F7]">
                Audio Source
              </h3>
              <p className="text-[13.5px] text-[#C89EA4] mt-0.5">
                Drop or select one or multiple recordings, customize Pitch &amp; Index per sample, or point to a Kaggle folder.
              </p>
            </div>
          </div>

          <div className="inline-flex rounded-lg border border-[#47161D] bg-[#100506] p-1">
            {(['single', 'batch'] as const).map((mode) => (
              <button
                key={mode}
                type="button"
                onClick={() =>
                  onUpdateConvertAudio({ ...convertAudio, sourceMode: mode })
                }
                className={`rounded-md px-3.5 py-1.5 text-xs font-semibold transition-colors whitespace-nowrap shrink-0 ${
                  convertAudio.sourceMode === mode
                    ? 'bg-[#E51923] text-white'
                    : 'text-[#DFBFC3] hover:text-white'
                }`}
              >
                {mode === 'single' ? 'Audio Files' : 'Batch'}
              </button>
            ))}
          </div>
        </div>

        {convertAudio.sourceMode === 'single' ? (
          <div className="space-y-4">
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={() => setDragActive(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragActive(false);
                handleFilesChosen(e.dataTransfer.files);
              }}
              className={`rounded-xl border-2 border-dashed p-6 text-center transition-colors ${
                dragActive
                  ? 'border-[#E51923] bg-[#200B0E]'
                  : 'border-[#47161D] bg-[#100506] hover:border-[#6D202B]'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                multiple
                className="hidden"
                onChange={(e) => {
                  handleFilesChosen(e.target.files);
                  e.target.value = '';
                }}
              />
              <Upload className="h-6 w-6 mx-auto text-[#E51923] mb-2" />
              <p className="text-sm font-medium text-[#FDF7F7]">
                {sampleQueue.length > 0
                  ? `${sampleQueue.length} recording${sampleQueue.length === 1 ? '' : 's'} ready in queue`
                  : 'Drop Audio Here - or - Click to Upload'}
              </p>
              <p className="text-xs text-[#AD8288] mt-1">
                Select one or multiple WAV, MP3, FLAC, or OGG recordings at once.
              </p>
              <div className="mt-4 flex flex-wrap items-center justify-center gap-2.5">
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="rounded-lg px-4 py-2 text-xs font-semibold text-white hover:bg-[#FF2B35] whitespace-nowrap shrink-0"
                  style={{ backgroundColor: ACCENT_HEX }}
                >
                  Upload Audio (Single / Multi)
                </button>
              </div>
            </div>

            {sampleQueue.length > 0 && (
              <div className="space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="text-xs font-semibold text-[#DFBFC3]">
                    Queued Audio Samples ({sampleQueue.length})
                  </p>
                  <div className="flex flex-wrap items-center gap-2">
                    {sampleQueue.length > 1 && (
                      <button
                        type="button"
                        onClick={handleApplyDefaultsToQueue}
                        className="rounded-md border border-[#47161D] bg-[#200B0E] px-2.5 py-1 text-[11px] font-medium text-[#DFBFC3] hover:text-white hover:border-[#E51923] whitespace-nowrap"
                      >
                        Apply Default Pitch/Index to All
                      </button>
                    )}
                    <button
                      type="button"
                      onClick={handleClearAllClips}
                      className="inline-flex items-center gap-1 rounded-md border border-[#47161D] bg-[#100506] px-2.5 py-1 text-[11px] font-medium text-[#AD8288] hover:text-white hover:border-[#E51923] whitespace-nowrap"
                    >
                      <X className="h-3 w-3" />
                      <span>Clear All</span>
                    </button>
                  </div>
                </div>

                {sampleQueue.map((clip, idx) => {
                  const isAdvExpanded = Boolean(expandedClipAdv[clip.id]);
                  const pitchText = clip.pitch > 0 ? `+${clip.pitch}` : `${clip.pitch}`;
                  return (
                    <div
                      key={clip.id}
                      className="rounded-xl border border-[#47161D] bg-[#100506] p-4 space-y-3"
                    >
                      <div className="flex items-center justify-between gap-3">
                        <div className="flex items-center gap-2.5 min-w-0">
                          <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-[#200B0E] border border-[#47161D] font-mono text-[11px] font-semibold text-[#FDF7F7] tabular-nums">
                            {idx + 1}
                          </span>
                          <div className="min-w-0">
                            <p className="font-mono text-xs font-semibold text-[#FDF7F7] truncate">
                              {clip.fileName}
                            </p>
                            <p className="text-[13px] text-[#C89EA4] truncate tabular-nums">
                              Pitch {pitchText} · Index {clip.indexRate.toFixed(2)} · {clip.outputFormat.toUpperCase()}
                              {clip.status === 'ready' ? ' · Converted' : ''}
                            </p>
                          </div>
                        </div>

                        <div className="flex items-center gap-2 shrink-0">
                          <button
                            type="button"
                            onClick={() => handleConvertSingleClip(clip.id)}
                            className="inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-semibold text-white hover:bg-[#FF2B35] whitespace-nowrap"
                            style={{ backgroundColor: ACCENT_HEX }}
                          >
                            <Play className="h-3 w-3" />
                            <span>{clip.status === 'ready' ? 'Re-Convert' : 'Convert'}</span>
                          </button>
                          <button
                            type="button"
                            onClick={() => handleDiscardClip(clip.id)}
                            aria-label={`Discard ${clip.fileName}`}
                            className="rounded-md border border-[#47161D] bg-[#160709] p-1.5 text-[#DFBFC3] hover:text-white hover:border-[#E51923]"
                          >
                            <X className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      </div>

                      <CrimsonAudioPlayer
                        src={clip.sourceUrl}
                        label={`Source · ${clip.fileName}`}
                        compact
                      />

                      {/* Easy Per-Sample Pitch & Index Controls */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1">
                        <div className="rounded-lg border border-[#47161D] bg-[#160709] px-3 py-2.5">
                          <div className="flex items-center justify-between gap-2 mb-1.5">
                            <label
                              htmlFor={`clip-pitch-${clip.id}`}
                              className="text-xs font-semibold text-[#FDF7F7]"
                            >
                              Pitch
                            </label>
                            <div className="flex items-center gap-1.5">
                              <span className="font-mono text-xs text-[#FDF7F7] tabular-nums">
                                {pitchText}
                              </span>
                              <button
                                type="button"
                                onClick={() => handleUpdateClip(clip.id, { pitch: 0 })}
                                aria-label={`Reset pitch for ${clip.fileName}`}
                                className="rounded p-1 text-[#AD8288] hover:text-white"
                              >
                                <RotateCcw className="h-3 w-3" />
                              </button>
                            </div>
                          </div>
                          <input
                            id={`clip-pitch-${clip.id}`}
                            type="range"
                            min={-32}
                            max={32}
                            step={1}
                            value={clip.pitch}
                            onChange={(e) =>
                              handleUpdateClip(clip.id, { pitch: Number(e.target.value) })
                            }
                            className="w-full scarlet-slider"
                          />
                        </div>

                        <div className="rounded-lg border border-[#47161D] bg-[#160709] px-3 py-2.5">
                          <div className="flex items-center justify-between gap-2 mb-1.5">
                            <label
                              htmlFor={`clip-index-${clip.id}`}
                              className="text-xs font-semibold text-[#FDF7F7]"
                            >
                              Index Rate
                            </label>
                            <span className="font-mono text-xs text-[#FDF7F7] tabular-nums">
                              {clip.indexRate.toFixed(2)}
                            </span>
                          </div>
                          <input
                            id={`clip-index-${clip.id}`}
                            type="range"
                            min={0}
                            max={1}
                            step={0.05}
                            value={clip.indexRate}
                            onChange={(e) =>
                              handleUpdateClip(clip.id, { indexRate: Number(e.target.value) })
                            }
                            className="w-full scarlet-slider"
                          />
                        </div>
                      </div>

                      {/* Optional per-sample full overrides */}
                      <div>
                        <button
                          type="button"
                          onClick={() =>
                            setExpandedClipAdv((prev) => ({
                              ...prev,
                              [clip.id]: !prev[clip.id],
                            }))
                          }
                          aria-expanded={isAdvExpanded}
                          className="inline-flex items-center gap-1.5 text-xs font-medium text-[#C89EA4] hover:text-[#FDF7F7]"
                        >
                          <SlidersHorizontal className="h-3 w-3 text-[#E51923]" />
                          <span>
                            {isAdvExpanded
                              ? 'Hide clip model & advanced overrides'
                              : 'Customize model, protect, format & advanced'}
                          </span>
                        </button>

                        {isAdvExpanded && (
                          <div className="mt-2.5 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 rounded-lg border border-[#47161D] bg-[#160709] p-3">
                            <div>
                              <label
                                htmlFor={`clip-model-${clip.id}`}
                                className="block text-xs font-medium text-[#C89EA4] mb-1"
                              >
                                Clip Voice Model
                              </label>
                              <select
                                id={`clip-model-${clip.id}`}
                                value={clip.voiceModel}
                                onChange={(e) =>
                                  handleUpdateClip(clip.id, { voiceModel: e.target.value })
                                }
                                className="w-full rounded-md bg-[#100506] border border-[#47161D] px-2.5 py-1.5 text-xs font-mono text-[#FDF7F7]"
                              >
                                <option value="">Use Default (Step 1)</option>
                                {modelOptions.map((m) => (
                                  <option key={m} value={m}>
                                    {m}
                                  </option>
                                ))}
                              </select>
                            </div>

                            <div>
                              <label
                                htmlFor={`clip-idxfile-${clip.id}`}
                                className="block text-xs font-medium text-[#C89EA4] mb-1"
                              >
                                Clip Index File
                              </label>
                              <select
                                id={`clip-idxfile-${clip.id}`}
                                value={clip.indexFile}
                                onChange={(e) =>
                                  handleUpdateClip(clip.id, { indexFile: e.target.value })
                                }
                                className="w-full rounded-md bg-[#100506] border border-[#47161D] px-2.5 py-1.5 text-xs font-mono text-[#FDF7F7]"
                              >
                                <option value="">Use Default / None</option>
                                {indexOptions.map((idxOpt) => (
                                  <option key={idxOpt} value={idxOpt}>
                                    {idxOpt}
                                  </option>
                                ))}
                              </select>
                            </div>

                            <div>
                              <label
                                htmlFor={`clip-format-${clip.id}`}
                                className="block text-xs font-medium text-[#C89EA4] mb-1"
                              >
                                Export Format
                              </label>
                              <select
                                id={`clip-format-${clip.id}`}
                                value={clip.outputFormat}
                                onChange={(e) =>
                                  handleUpdateClip(clip.id, {
                                    outputFormat: e.target.value as 'wav' | 'mp3' | 'flac',
                                  })
                                }
                                className="w-full rounded-md bg-[#100506] border border-[#47161D] px-2.5 py-1.5 text-xs text-[#FDF7F7]"
                              >
                                <option value="wav">WAV</option>
                                <option value="mp3">MP3</option>
                                <option value="flac">FLAC</option>
                              </select>
                            </div>

                            <div>
                              <div className="flex justify-between text-xs mb-1">
                                <span className="text-[#C89EA4]">Protect</span>
                                <span className="font-mono text-[#FDF7F7] tabular-nums">
                                  {clip.protect.toFixed(2)}
                                </span>
                              </div>
                              <input
                                type="range"
                                min={0}
                                max={0.5}
                                step={0.01}
                                value={clip.protect}
                                aria-label={`Protect consonants for ${clip.fileName}`}
                                onChange={(e) =>
                                  handleUpdateClip(clip.id, { protect: Number(e.target.value) })
                                }
                                className="w-full scarlet-slider"
                              />
                            </div>

                            <div>
                              <div className="flex justify-between text-xs mb-1">
                                <span className="text-[#C89EA4]">Volume Envelope</span>
                                <span className="font-mono text-[#FDF7F7] tabular-nums">
                                  {clip.volumeEnvelope.toFixed(2)}
                                </span>
                              </div>
                              <input
                                type="range"
                                min={0}
                                max={1}
                                step={0.05}
                                value={clip.volumeEnvelope}
                                aria-label={`Volume envelope for ${clip.fileName}`}
                                onChange={(e) =>
                                  handleUpdateClip(clip.id, { volumeEnvelope: Number(e.target.value) })
                                }
                                className="w-full scarlet-slider"
                              />
                            </div>

                            <div className="flex flex-wrap items-center gap-3.5 pt-2">
                              <label className="inline-flex items-center gap-1.5 text-xs text-[#DFBFC3] cursor-pointer">
                                <input
                                  type="checkbox"
                                  checked={clip.autotune}
                                  onChange={(e) =>
                                    handleUpdateClip(clip.id, { autotune: e.target.checked })
                                  }
                                  className="h-3.5 w-3.5 rounded accent-[#E51923]"
                                />
                                <span>Autotune</span>
                              </label>
                              <label className="inline-flex items-center gap-1.5 text-xs text-[#DFBFC3] cursor-pointer">
                                <input
                                  type="checkbox"
                                  checked={clip.proposedPitch}
                                  onChange={(e) =>
                                    handleUpdateClip(clip.id, { proposedPitch: e.target.checked })
                                  }
                                  className="h-3.5 w-3.5 rounded accent-[#E51923]"
                                />
                                <span>Proposed Pitch</span>
                              </label>
                              <label className="inline-flex items-center gap-1.5 text-xs text-[#DFBFC3] cursor-pointer">
                                <input
                                  type="checkbox"
                                  checked={clip.splitAudio}
                                  onChange={(e) =>
                                    handleUpdateClip(clip.id, { splitAudio: e.target.checked })
                                  }
                                  className="h-3.5 w-3.5 rounded accent-[#E51923]"
                                />
                                <span>Split</span>
                              </label>
                            </div>
                          </div>
                        )}
                      </div>

                      {clip.convertedUrl && (
                        <div className="rounded-lg border border-[#E51923]/50 bg-[#200B0E] p-3 space-y-2">
                          <div className="flex items-center justify-between gap-2">
                            <span className="text-xs font-semibold text-[#FDF7F7]">
                              Converted Sample ({clip.convertedName})
                            </span>
                            <button
                              type="button"
                              onClick={() =>
                                handleSaveSample(clip.convertedUrl, clip.convertedName)
                              }
                              className="inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[11px] font-semibold text-white hover:bg-[#FF2B35] whitespace-nowrap"
                              style={{ backgroundColor: ACCENT_HEX }}
                            >
                              <Download className="h-3 w-3" />
                              <span>Save (.{clip.outputFormat})</span>
                            </button>
                          </div>
                          <CrimsonAudioPlayer
                            src={clip.convertedUrl}
                            compact
                            onDiscard={() => {
                              safeRevokeConvertedUrl(clip.convertedUrl);
                              handleUpdateClip(clip.id, {
                                convertedUrl: '',
                                convertedName: '',
                                status: 'queued',
                              });
                            }}
                            discardLabel={`Discard converted output for ${clip.fileName}`}
                          />
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        ) : (
          <div>
            <label
              htmlFor="convert-batch-folder"
              className="block text-xs font-semibold text-[#FDF7F7] mb-1.5"
            >
              Input Folder Path (Batch)
            </label>
            <input
              id="convert-batch-folder"
              type="text"
              value={convertAudio.batchFolderPath}
              onChange={(e) =>
                onUpdateConvertAudio({
                  ...convertAudio,
                  batchFolderPath: e.target.value,
                })
              }
              className="w-full rounded-lg bg-[#100506] border border-[#47161D] px-3 py-2 text-xs font-mono text-[#FDF7F7] focus:outline-none focus:border-[#E51923]"
            />
          </div>
        )}
      </section>

      {/* STEP 3: Default Conversion Parameters & Run */}
      <section className="rounded-xl border border-[#47161D] bg-[#160709] p-6 space-y-5">
        <div className="flex items-start gap-3">
          <StepBadge num={3} />
          <div>
            <h3 className="font-display text-[17px] font-semibold text-[#FDF7F7]">
              Conversion Parameters
            </h3>
            <p className="text-[13.5px] text-[#C89EA4] mt-0.5">
              Set default Pitch, Index Rate, and export options for incoming clips or batch folders.
            </p>
          </div>
        </div>

        {/* Pitch Slider */}
        <div className="rounded-lg border border-[#47161D] bg-[#100506] p-4">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-2">
            <div>
              <label
                htmlFor="convert-pitch-slider"
                className="block text-[13.5px] font-semibold text-[#FDF7F7]"
              >
                Default Pitch
              </label>
              <p className="text-[13px] text-[#C89EA4] mt-0.5">
                Shift the vocal register up or down in semitones between -32 and +32.
              </p>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <span className="rounded-md border border-[#47161D] bg-[#160709] px-2.5 py-1 font-mono text-xs text-[#FDF7F7] tabular-nums">
                {convertAudio.pitch > 0 ? `+${convertAudio.pitch}` : convertAudio.pitch}
              </span>
              <button
                type="button"
                onClick={() => onUpdateConvertAudio({ ...convertAudio, pitch: 0 })}
                aria-label="Reset pitch to 0"
                className="rounded-md border border-[#47161D] bg-[#200B0E] p-1.5 text-[#DFBFC3] hover:text-white"
              >
                <RotateCcw className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
          <input
            id="convert-pitch-slider"
            type="range"
            min={-32}
            max={32}
            step={1}
            value={convertAudio.pitch}
            onChange={(e) =>
              onUpdateConvertAudio({
                ...convertAudio,
                pitch: Number(e.target.value),
              })
            }
            className="w-full scarlet-slider"
          />
          <div className="flex justify-between text-xs font-mono text-[#C89EA4] mt-1 tabular-nums">
            <span>-32</span>
            <span>0</span>
            <span>+32</span>
          </div>
        </div>

        {/* Collapsible Advanced Settings */}
        <div className="rounded-lg border border-[#47161D] bg-[#100506]">
          <button
            type="button"
            onClick={() => {
              const next = !advOpen;
              setAdvOpen(next);
              if (next) {
                setSubOpen({
                  voiceMatching: true,
                  modelsAlgorithms: false,
                  pitchTuning: false,
                  audioCleanup: false,
                  formantShifting: false,
                  outputSettings: false,
                });
              }
            }}
            aria-expanded={advOpen}
            className="w-full flex items-center justify-between px-4 py-3 text-xs font-semibold text-[#FDF7F7] hover:text-white"
          >
            <div className="flex items-center gap-2">
              <SlidersHorizontal className="h-3.5 w-3.5 text-[#E51923]" />
              <span>Advanced Settings</span>
              <span className="font-mono text-[11px] font-normal text-[#AD8288] hidden sm:inline">
                ({convertAudio.pitchExtractor} · index {convertAudio.indexRate.toFixed(2)} · env {convertAudio.volumeEnvelope} · protect {convertAudio.protect})
              </span>
            </div>
            {advOpen ? (
              <ChevronDown className="h-4 w-4 text-[#DFBFC3]" />
            ) : (
              <ChevronRight className="h-4 w-4 text-[#DFBFC3]" />
            )}
          </button>

          {advOpen && (
            <div className="border-t border-[#47161D] p-4 space-y-3">
              {/* 1. Voice Matching */}
              <div className="rounded-lg border border-[#47161D] bg-[#160709]">
                <button
                  type="button"
                  onClick={() => toggleSub('voiceMatching')}
                  aria-expanded={subOpen.voiceMatching}
                  className="w-full flex items-center justify-between px-3.5 py-2.5 text-xs font-semibold text-[#FDF7F7]"
                >
                  <span>Voice Matching</span>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[11px] font-normal text-[#AD8288] tabular-nums">
                      {convertAudio.indexRate.toFixed(2)} · {convertAudio.volumeEnvelope} · {convertAudio.protect}
                    </span>
                    {subOpen.voiceMatching ? (
                      <ChevronDown className="h-3.5 w-3.5 text-[#AD8288]" />
                    ) : (
                      <ChevronRight className="h-3.5 w-3.5 text-[#AD8288]" />
                    )}
                  </div>
                </button>

                {subOpen.voiceMatching && (
                  <div className="border-t border-[#47161D] p-3.5 space-y-4">
                    {/* Search Feature Ratio */}
                    <div>
                      <div className="flex flex-wrap items-center justify-between gap-2 mb-1">
                        <label htmlFor="conv-index-rate" className="text-[13.5px] font-semibold text-[#FDF7F7]">
                          Search Feature Ratio
                        </label>
                        <div className="flex items-center gap-1.5">
                          <span className="rounded border border-[#47161D] bg-[#100506] px-2 py-0.5 font-mono text-xs text-[#FDF7F7] tabular-nums">
                            {convertAudio.indexRate}
                          </span>
                          <button
                            type="button"
                            onClick={() => onUpdateConvertAudio({ ...convertAudio, indexRate: 0.75 })}
                            aria-label="Reset search feature ratio to 0.75"
                            className="rounded border border-[#47161D] bg-[#200B0E] p-1 text-[#DFBFC3] hover:text-white"
                          >
                            <RotateCcw className="h-3 w-3" />
                          </button>
                        </div>
                      </div>
                      <p className="text-[13px] text-[#C89EA4] mb-2">
                        Influence exerted by the index file; higher values give stronger accent/timbre transfer, while lower values help mitigate audio artifacts.
                      </p>
                      <input
                        id="conv-index-rate"
                        type="range"
                        min={0}
                        max={1}
                        step={0.05}
                        value={convertAudio.indexRate}
                        onChange={(e) =>
                          onUpdateConvertAudio({ ...convertAudio, indexRate: Number(e.target.value) })
                        }
                        className="w-full scarlet-slider"
                      />
                      <div className="flex justify-between text-xs font-mono text-[#C89EA4] mt-1 tabular-nums">
                        <span>0</span>
                        <span>1</span>
                      </div>
                    </div>

                    {/* Volume Envelope */}
                    <div>
                      <div className="flex flex-wrap items-center justify-between gap-2 mb-1">
                        <label htmlFor="conv-vol-env" className="text-[13.5px] font-semibold text-[#FDF7F7]">
                          Volume Envelope
                        </label>
                        <div className="flex items-center gap-1.5">
                          <span className="rounded border border-[#47161D] bg-[#100506] px-2 py-0.5 font-mono text-xs text-[#FDF7F7] tabular-nums">
                            {convertAudio.volumeEnvelope}
                          </span>
                          <button
                            type="button"
                            onClick={() => onUpdateConvertAudio({ ...convertAudio, volumeEnvelope: 1 })}
                            aria-label="Reset volume envelope to 1"
                            className="rounded border border-[#47161D] bg-[#200B0E] p-1 text-[#DFBFC3] hover:text-white"
                          >
                            <RotateCcw className="h-3 w-3" />
                          </button>
                        </div>
                      </div>
                      <p className="text-[13px] text-[#C89EA4] mb-2">
                        Substitute or blend with the volume envelope of the output. Closer to 1 employs more of the output envelope.
                      </p>
                      <input
                        id="conv-vol-env"
                        type="range"
                        min={0}
                        max={1}
                        step={0.05}
                        value={convertAudio.volumeEnvelope}
                        onChange={(e) =>
                          onUpdateConvertAudio({ ...convertAudio, volumeEnvelope: Number(e.target.value) })
                        }
                        className="w-full scarlet-slider"
                      />
                      <div className="flex justify-between text-xs font-mono text-[#C89EA4] mt-1 tabular-nums">
                        <span>0</span>
                        <span>1</span>
                      </div>
                    </div>

                    {/* Protect Voiceless Consonants */}
                    <div>
                      <div className="flex flex-wrap items-center justify-between gap-2 mb-1">
                        <label htmlFor="conv-protect" className="text-[13.5px] font-semibold text-[#FDF7F7]">
                          Protect Voiceless Consonants
                        </label>
                        <div className="flex items-center gap-1.5">
                          <span className="rounded border border-[#47161D] bg-[#100506] px-2 py-0.5 font-mono text-xs text-[#FDF7F7] tabular-nums">
                            {convertAudio.protect}
                          </span>
                          <button
                            type="button"
                            onClick={() => onUpdateConvertAudio({ ...convertAudio, protect: 0.5 })}
                            aria-label="Reset protect voiceless consonants to 0.5"
                            className="rounded border border-[#47161D] bg-[#200B0E] p-1 text-[#DFBFC3] hover:text-white"
                          >
                            <RotateCcw className="h-3 w-3" />
                          </button>
                        </div>
                      </div>
                      <p className="text-[13px] text-[#C89EA4] mb-2">
                        Safeguard distinct consonants and breathing sounds to prevent electro-acoustic tearing. Pulling to 0.5 offers maximum protection.
                      </p>
                      <input
                        id="conv-protect"
                        type="range"
                        min={0}
                        max={0.5}
                        step={0.01}
                        value={convertAudio.protect}
                        onChange={(e) =>
                          onUpdateConvertAudio({ ...convertAudio, protect: Number(e.target.value) })
                        }
                        className="w-full scarlet-slider"
                      />
                      <div className="flex justify-between text-xs font-mono text-[#C89EA4] mt-1 tabular-nums">
                        <span>0</span>
                        <span>0.5</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* 2. Models & Algorithms */}
              <div className="rounded-lg border border-[#47161D] bg-[#160709]">
                <button
                  type="button"
                  onClick={() => toggleSub('modelsAlgorithms')}
                  aria-expanded={subOpen.modelsAlgorithms}
                  className="w-full flex items-center justify-between px-3.5 py-2.5 text-xs font-semibold text-[#FDF7F7]"
                >
                  <span>Models &amp; Algorithms</span>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[11px] font-normal text-[#AD8288]">
                      sid {convertAudio.speakerId} · {convertAudio.pitchExtractor} · {convertAudio.embedder}
                    </span>
                    {subOpen.modelsAlgorithms ? (
                      <ChevronDown className="h-3.5 w-3.5 text-[#AD8288]" />
                    ) : (
                      <ChevronRight className="h-3.5 w-3.5 text-[#AD8288]" />
                    )}
                  </div>
                </button>

                {subOpen.modelsAlgorithms && (
                  <div className="border-t border-[#47161D] p-3.5 grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <label htmlFor="conv-speaker-id" className="block text-[13.5px] font-semibold text-[#FDF7F7]">
                        Speaker ID
                      </label>
                      <p className="text-[13px] text-[#C89EA4] mb-1.5">
                        Speaker ID to use for conversion.
                      </p>
                      <input
                        id="conv-speaker-id"
                        type="number"
                        min={0}
                        max={100}
                        value={convertAudio.speakerId}
                        onChange={(e) =>
                          onUpdateConvertAudio({
                            ...convertAudio,
                            speakerId: Math.max(0, Number(e.target.value) || 0),
                          })
                        }
                        className="w-full rounded-lg bg-[#100506] border border-[#47161D] px-3 py-2 text-xs font-mono text-[#FDF7F7] focus:outline-none focus:border-[#E51923]"
                      />
                    </div>

                    <div>
                      <label htmlFor="conv-f0-method" className="block text-[13.5px] font-semibold text-[#FDF7F7]">
                        Pitch extraction algorithm
                      </label>
                      <p className="text-[13px] text-[#C89EA4] mb-1.5">
                        rmvpe is recommended for most cases.
                      </p>
                      <select
                        id="conv-f0-method"
                        value={convertAudio.pitchExtractor}
                        onChange={(e) =>
                          onUpdateConvertAudio({
                            ...convertAudio,
                            pitchExtractor: e.target.value as PitchExtractor,
                          })
                        }
                        className="w-full rounded-lg bg-[#100506] border border-[#47161D] px-3 py-2 text-xs font-mono text-[#FDF7F7] focus:outline-none focus:border-[#E51923]"
                      >
                        <option value="rmvpe">rmvpe</option>
                        <option value="fcpe">fcpe</option>
                        <option value="swift-f0">swift-f0</option>
                        <option value="crepe">crepe</option>
                      </select>
                    </div>

                    <div>
                      <label htmlFor="conv-embedder" className="block text-[13.5px] font-semibold text-[#FDF7F7]">
                        Embedder Model
                      </label>
                      <p className="text-[13px] text-[#C89EA4] mb-1.5">
                        Model used for speaker embedding.
                      </p>
                      <select
                        id="conv-embedder"
                        value={convertAudio.embedder}
                        onChange={(e) =>
                          onUpdateConvertAudio({
                            ...convertAudio,
                            embedder: e.target.value as EmbedderModel,
                          })
                        }
                        className="w-full rounded-lg bg-[#100506] border border-[#47161D] px-3 py-2 text-xs font-mono text-[#FDF7F7] focus:outline-none focus:border-[#E51923]"
                      >
                        <option value="contentvec">contentvec</option>
                        <option value="spin">spin</option>
                        <option value="chinese-hubert-base">chinese-hubert-base</option>
                      </select>
                    </div>
                  </div>
                )}
              </div>

              {/* 3. Advanced Pitch & Tuning */}
              <div className="rounded-lg border border-[#47161D] bg-[#160709]">
                <button
                  type="button"
                  onClick={() => toggleSub('pitchTuning')}
                  aria-expanded={subOpen.pitchTuning}
                  className="w-full flex items-center justify-between px-3.5 py-2.5 text-xs font-semibold text-[#FDF7F7]"
                >
                  <span>Advanced Pitch &amp; Tuning</span>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[11px] font-normal text-[#AD8288]">
                      {[
                        convertAudio.autotune && 'Autotune',
                        convertAudio.proposedPitch && 'Proposed Pitch',
                        convertAudio.splitAudio && 'Split',
                      ]
                        .filter(Boolean)
                        .join(' · ') || 'Standard'}
                    </span>
                    {subOpen.pitchTuning ? (
                      <ChevronDown className="h-3.5 w-3.5 text-[#AD8288]" />
                    ) : (
                      <ChevronRight className="h-3.5 w-3.5 text-[#AD8288]" />
                    )}
                  </div>
                </button>

                {subOpen.pitchTuning && (
                  <div className="border-t border-[#47161D] p-3.5 grid grid-cols-1 md:grid-cols-3 gap-3">
                    <label className="flex items-start gap-2.5 rounded-lg border border-[#47161D] bg-[#100506] p-3 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={convertAudio.autotune}
                        onChange={(e) =>
                          onUpdateConvertAudio({ ...convertAudio, autotune: e.target.checked })
                        }
                        className="mt-0.5 h-4 w-4 rounded accent-[#E51923]"
                      />
                      <div>
                        <span className="block text-[13.5px] font-semibold text-[#FDF7F7]">Autotune</span>
                        <span className="block text-[13px] text-[#C89EA4] mt-0.5">
                          Apply soft autotune; recommended for singing conversions.
                        </span>
                      </div>
                    </label>

                    <label className="flex items-start gap-2.5 rounded-lg border border-[#47161D] bg-[#100506] p-3 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={convertAudio.proposedPitch}
                        onChange={(e) =>
                          onUpdateConvertAudio({
                            ...convertAudio,
                            proposedPitch: e.target.checked,
                          })
                        }
                        className="mt-0.5 h-4 w-4 rounded accent-[#E51923]"
                      />
                      <div>
                        <span className="block text-[13.5px] font-semibold text-[#FDF7F7]">
                          Proposed Pitch
                        </span>
                        <span className="block text-[13px] text-[#C89EA4] mt-0.5">
                          Auto-adjust input pitch to match the voice model range.
                        </span>
                      </div>
                    </label>

                    <label className="flex items-start gap-2.5 rounded-lg border border-[#47161D] bg-[#100506] p-3 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={convertAudio.splitAudio}
                        onChange={(e) =>
                          onUpdateConvertAudio({ ...convertAudio, splitAudio: e.target.checked })
                        }
                        className="mt-0.5 h-4 w-4 rounded accent-[#E51923]"
                      />
                      <div>
                        <span className="block text-[13.5px] font-semibold text-[#FDF7F7]">
                          Split Audio
                        </span>
                        <span className="block text-[13px] text-[#C89EA4] mt-0.5">
                          Split long clips on silence for cleaner VRAM usage.
                        </span>
                      </div>
                    </label>
                  </div>
                )}
              </div>

              {/* 4. Audio Cleanup */}
              <div className="rounded-lg border border-[#47161D] bg-[#160709]">
                <button
                  type="button"
                  onClick={() => toggleSub('audioCleanup')}
                  aria-expanded={subOpen.audioCleanup}
                  className="w-full flex items-center justify-between px-3.5 py-2.5 text-xs font-semibold text-[#FDF7F7]"
                >
                  <span>Audio Cleanup</span>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[11px] font-normal text-[#AD8288]">
                      {convertAudio.noiseReduction
                        ? `On · ${convertAudio.noiseReductionStrength} dB (${cleanupCalibration.tierLabel})`
                        : 'Off'}
                    </span>
                    {subOpen.audioCleanup ? (
                      <ChevronDown className="h-3.5 w-3.5 text-[#AD8288]" />
                    ) : (
                      <ChevronRight className="h-3.5 w-3.5 text-[#AD8288]" />
                    )}
                  </div>
                </button>

                {subOpen.audioCleanup && (
                  <div className="border-t border-[#47161D] p-3.5 space-y-3.5">
                    <label className="flex items-start gap-2.5 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={convertAudio.noiseReduction}
                        onChange={(e) =>
                          onUpdateConvertAudio({
                            ...convertAudio,
                            noiseReduction: e.target.checked,
                          })
                        }
                        className="mt-0.5 h-4 w-4 rounded accent-[#E51923]"
                      />
                      <div>
                        <span className="block text-[13.5px] font-semibold text-[#FDF7F7]">
                          Audio cleanup ({convertAudio.noiseReduction ? 'On' : 'Off'})
                        </span>
                        <span className="block text-[13px] text-[#C89EA4] mt-0.5">
                          Cleans input before conversion using balanced rumble high-pass and smooth downward expansion so voices stay natural without harsh gating.
                        </span>
                      </div>
                    </label>

                    <div className={convertAudio.noiseReduction ? '' : 'opacity-50'}>
                      <div className="flex flex-wrap items-center justify-between gap-2 mb-1">
                        <label
                          htmlFor="conv-clean-strength"
                          className="text-[13.5px] font-semibold text-[#FDF7F7]"
                        >
                          Noise reduction limit (dB)
                        </label>
                        <div className="flex items-center gap-1.5">
                          <span className="rounded border border-[#47161D] bg-[#100506] px-2 py-0.5 font-mono text-[11px] text-[#DFBFC3]">
                            {cleanupCalibration.tierLabel}
                          </span>
                          <span className="rounded border border-[#47161D] bg-[#100506] px-2 py-0.5 font-mono text-xs text-[#FDF7F7] tabular-nums">
                            {convertAudio.noiseReductionStrength} dB
                          </span>
                          <button
                            type="button"
                            onClick={() =>
                              onUpdateConvertAudio({ ...convertAudio, noiseReductionStrength: 12 })
                            }
                            aria-label="Reset noise reduction limit to 12 dB"
                            className="rounded border border-[#47161D] bg-[#200B0E] p-1 text-[#DFBFC3] hover:text-white"
                          >
                            <RotateCcw className="h-3 w-3" />
                          </button>
                        </div>
                      </div>
                      <p className="text-[13px] text-[#C89EA4] mb-2">
                        {cleanupCalibration.description} ({cleanupCalibration.highpassHz} Hz rumble filter, up to -{cleanupCalibration.maxAttenuationDb} dB quiet-floor reduction).
                      </p>
                      <input
                        id="conv-clean-strength"
                        type="range"
                        min={1}
                        max={30}
                        step={1}
                        disabled={!convertAudio.noiseReduction}
                        value={convertAudio.noiseReductionStrength}
                        onChange={(e) =>
                          onUpdateConvertAudio({
                            ...convertAudio,
                            noiseReductionStrength: Number(e.target.value),
                          })
                        }
                        className="w-full scarlet-slider"
                      />
                      <div className="flex justify-between text-xs font-mono text-[#C89EA4] mt-1 tabular-nums">
                        <span>1 (Subtle)</span>
                        <span>12 (Balanced)</span>
                        <span>30 (Strong)</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* 5. Formant Shifting */}
              <div className="rounded-lg border border-[#47161D] bg-[#160709]">
                <button
                  type="button"
                  onClick={() => toggleSub('formantShifting')}
                  aria-expanded={subOpen.formantShifting}
                  className="w-full flex items-center justify-between px-3.5 py-2.5 text-xs font-semibold text-[#FDF7F7]"
                >
                  <span>Formant Shifting</span>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[11px] font-normal text-[#AD8288]">
                      {convertAudio.formantShiftEnabled
                        ? `On · Q ${convertAudio.formantQuefrency} · T ${convertAudio.formantTimbre}`
                        : 'Off'}
                    </span>
                    {subOpen.formantShifting ? (
                      <ChevronDown className="h-3.5 w-3.5 text-[#AD8288]" />
                    ) : (
                      <ChevronRight className="h-3.5 w-3.5 text-[#AD8288]" />
                    )}
                  </div>
                </button>

                {subOpen.formantShifting && (
                  <div className="border-t border-[#47161D] p-3.5 space-y-3.5">
                    <label className="flex items-start gap-2.5 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={convertAudio.formantShiftEnabled}
                        onChange={(e) =>
                          onUpdateConvertAudio({
                            ...convertAudio,
                            formantShiftEnabled: e.target.checked,
                          })
                        }
                        className="mt-0.5 h-4 w-4 rounded accent-[#E51923]"
                      />
                      <div>
                        <span className="block text-[13.5px] font-semibold text-[#FDF7F7]">
                          Formant Shifting
                        </span>
                        <span className="block text-[13px] text-[#C89EA4] mt-0.5">
                          Enable formant shifting for male-to-female and female-to-male conversions.
                        </span>
                      </div>
                    </label>

                    {convertAudio.formantShiftEnabled && (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">
                        <div>
                          <div className="flex items-center justify-between text-xs mb-1">
                            <label htmlFor="conv-formant-q" className="text-[#DFBFC3] font-medium">
                              Formant Quefrency
                            </label>
                            <span className="font-mono text-[#FDF7F7] tabular-nums">
                              {convertAudio.formantQuefrency.toFixed(1)}
                            </span>
                          </div>
                          <input
                            id="conv-formant-q"
                            type="range"
                            min={0}
                            max={16}
                            step={0.1}
                            value={convertAudio.formantQuefrency}
                            onChange={(e) =>
                              onUpdateConvertAudio({
                                ...convertAudio,
                                formantQuefrency: Number(e.target.value),
                              })
                            }
                            className="w-full scarlet-slider"
                          />
                        </div>

                        <div>
                          <div className="flex items-center justify-between text-xs mb-1">
                            <label htmlFor="conv-formant-t" className="text-[#DFBFC3] font-medium">
                              Formant Timbre
                            </label>
                            <span className="font-mono text-[#FDF7F7] tabular-nums">
                              {convertAudio.formantTimbre.toFixed(1)}
                            </span>
                          </div>
                          <input
                            id="conv-formant-t"
                            type="range"
                            min={0}
                            max={16}
                            step={0.1}
                            value={convertAudio.formantTimbre}
                            onChange={(e) =>
                              onUpdateConvertAudio({
                                ...convertAudio,
                                formantTimbre: Number(e.target.value),
                              })
                            }
                            className="w-full scarlet-slider"
                          />
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* 6. Output Settings */}
              <div className="rounded-lg border border-[#47161D] bg-[#160709]">
                <button
                  type="button"
                  onClick={() => toggleSub('outputSettings')}
                  aria-expanded={subOpen.outputSettings}
                  className="w-full flex items-center justify-between px-3.5 py-2.5 text-xs font-semibold text-[#FDF7F7]"
                >
                  <span>Output Settings</span>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[11px] font-normal text-[#AD8288]">
                      {convertAudio.outputFormat.toUpperCase()}
                    </span>
                    {subOpen.outputSettings ? (
                      <ChevronDown className="h-3.5 w-3.5 text-[#AD8288]" />
                    ) : (
                      <ChevronRight className="h-3.5 w-3.5 text-[#AD8288]" />
                    )}
                  </div>
                </button>

                {subOpen.outputSettings && (
                  <div className="border-t border-[#47161D] p-3.5 space-y-3.5">
                    <div className="grid grid-cols-1 md:grid-cols-[1fr_180px] gap-4">
                      <div>
                        <label htmlFor="conv-output-path" className="block text-[13.5px] font-semibold text-[#FDF7F7]">
                          Output Path
                        </label>
                        <p className="text-[13px] text-[#C89EA4] mb-1.5">
                          Path where output audio is saved (defaults to assets/audios/output.wav).
                        </p>
                        <input
                          id="conv-output-path"
                          type="text"
                          value={convertAudio.outputPath}
                          onChange={(e) =>
                            onUpdateConvertAudio({
                              ...convertAudio,
                              outputPath: e.target.value,
                            })
                          }
                          className="w-full rounded-lg bg-[#100506] border border-[#47161D] px-3 py-2 text-xs font-mono text-[#FDF7F7] focus:outline-none focus:border-[#E51923]"
                        />
                      </div>

                      <div>
                        <label htmlFor="conv-output-format" className="block text-[13.5px] font-semibold text-[#FDF7F7]">
                          Export Format
                        </label>
                        <p className="text-[13px] text-[#C89EA4] mb-1.5">
                          Audio export container.
                        </p>
                        <select
                          id="conv-output-format"
                          value={convertAudio.outputFormat}
                          onChange={(e) =>
                            onUpdateConvertAudio({
                              ...convertAudio,
                              outputFormat: e.target.value as 'wav' | 'mp3' | 'flac',
                            })
                          }
                          className="w-full rounded-lg bg-[#100506] border border-[#47161D] px-3 py-2 text-xs font-mono text-[#FDF7F7] focus:outline-none focus:border-[#E51923]"
                        >
                          <option value="wav">WAV</option>
                          <option value="mp3">MP3</option>
                          <option value="flac">FLAC</option>
                        </select>
                      </div>
                    </div>

                    <button
                      type="button"
                      onClick={handleClearAllOutputs}
                      className="w-full inline-flex items-center justify-center gap-2 rounded-lg border border-[#47161D] bg-[#200B0E] px-4 py-2 text-xs font-semibold text-[#DFBFC3] hover:text-white hover:border-[#E51923]"
                    >
                      <Trash2 className="h-3.5 w-3.5 text-[#E51923]" />
                      <span>Clear Outputs (Deletes all audios in assets/audios)</span>
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Terms of Use */}
        <div className="rounded-lg border border-[#47161D] bg-[#100506] p-4">
          <label className="flex items-start gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={convertAudio.agreedTerms}
              onChange={(e) =>
                onUpdateConvertAudio({
                  ...convertAudio,
                  agreedTerms: e.target.checked,
                })
              }
              className="mt-0.5 h-4 w-4 rounded accent-[#E51923]"
            />
            <div>
              <span className="block text-[13.5px] font-semibold text-[#FDF7F7]">
                I agree to the terms of use
              </span>
              <span className="block text-[13px] text-[#C89EA4] mt-0.5">
                Confirm responsible usage guidelines prior to rendering voice conversion samples.
              </span>
            </div>
          </label>
        </div>

        <button
          type="button"
          disabled={isConverting}
          onClick={() => void handleConvert()}
          className="w-full inline-flex items-center justify-center gap-2 rounded-lg py-2.5 text-sm font-semibold text-white transition-colors hover:bg-[#FF2B35] disabled:opacity-60"
          style={{ backgroundColor: ACCENT_HEX }}
        >
          <Play className="h-4 w-4" />
          <span>
            {isConverting
              ? convertAudio.noiseReduction
                ? 'Cleaning & Converting...'
                : 'Converting...'
              : convertAudio.sourceMode === 'single' && sampleQueue.length > 1
                ? `Convert All (${sampleQueue.length} Samples)`
                : 'Convert'}
          </span>
        </button>

        <div>
          <p className="text-xs font-medium text-[#DFBFC3] mb-1.5">
            Sample output
          </p>
          <div className="rounded-lg border border-[#47161D] bg-[#100506] px-3.5 py-2.5 text-xs font-mono text-[#DFBFC3]">
            {convertAudio.conversionOutput}
          </div>
        </div>

        {convertAudio.savedSamples.length > 0 && (
          <div className="rounded-lg border border-[#47161D] bg-[#100506] p-4 space-y-3">
            <p className="text-xs font-semibold text-[#DFBFC3]">
              Recent Generated Samples ({convertAudio.savedSamples.length})
            </p>

            <div className="space-y-3">
              {convertAudio.savedSamples.map((s) => (
                <div
                  key={s.id}
                  className="rounded-lg border border-[#47161D] bg-[#160709] p-3 space-y-2.5"
                >
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="min-w-0">
                      <p className="font-mono text-xs text-[#FDF7F7] truncate">{s.name}</p>
                      <p className="text-[13px] text-[#C89EA4] tabular-nums truncate">
                        {s.voiceModel ? `${s.voiceModel} · ` : ''}Pitch {s.pitch >= 0 ? `+${s.pitch}` : s.pitch}
                        {typeof s.indexRate === 'number' ? ` · idx ${s.indexRate.toFixed(2)}` : ''} · {s.createdAt}
                      </p>
                    </div>
                    <div className="flex items-center gap-1.5 shrink-0">
                      <button
                        type="button"
                        onClick={() => handleSaveSample(s.url, s.name)}
                        className="inline-flex items-center gap-1.5 rounded-md border border-[#47161D] bg-[#200B0E] px-2.5 py-1 text-xs font-medium text-[#FDF7F7] hover:border-[#E51923] whitespace-nowrap shrink-0"
                      >
                        <Download className="h-3 w-3" />
                        <span>Save</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => handleDiscardSavedSample(s.id)}
                        aria-label={`Discard sample ${s.name}`}
                        className="rounded-md border border-[#47161D] bg-[#100506] p-1.5 text-[#DFBFC3] hover:text-white hover:border-[#E51923]"
                      >
                        <X className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </div>
                  <CrimsonAudioPlayer src={s.url} compact />
                </div>
              ))}
            </div>
          </div>
        )}

        <pre className="rounded-lg border border-[#47161D] bg-[#100506] p-3 text-[11px] font-mono text-[#AD8288] overflow-x-auto">
          {inferCliCommand}
        </pre>
      </section>
    </div>
  );
});
