import React, { useEffect, useMemo, useRef, useState } from 'react';
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
import { FREE_SAMPLES, FreeSampleItem } from '../data/freeSamples';
import {
  ConvertAudioState,
  EmbedderModel,
  GeneratedSampleItem,
  ImportedLocalModel,
  PitchExtractor,
  SampleClipItem,
  VoiceModelProject,
} from '../types/studio';
import {
  collectUnusedBlobUrls,
  computeCleanupParameters,
  processAudioUrlWithCleanup,
} from '../utils/audioCleanup';
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
  const [cliOpen, setCliOpen] = useState(false);
  const clipSeqRef = useRef(1);
  const refreshTimerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (refreshTimerRef.current !== null) {
        window.clearTimeout(refreshTimerRef.current);
      }
    };
  }, []);

  const cleanupCalibration = useMemo(
    () => computeCleanupParameters(convertAudio.noiseReductionStrength),
    [convertAudio.noiseReductionStrength]
  );

  const toggleSub = (key: string) =>
    setSubOpen((prev) => ({ ...prev, [key]: !prev[key] }));

  const revokeUnused = (
    candidateUrls: Array<string | undefined | null>,
    nextQueue: SampleClipItem[] = sampleQueue,
    nextSaved: GeneratedSampleItem[] = convertAudio.savedSamples
  ) => {
    const dead = collectUnusedBlobUrls(candidateUrls, nextQueue, nextSaved);
    dead.forEach((url) => revokeUrl(url));
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

  const inferCliCommand = useMemo(() => {
    if (convertAudio.sourceMode !== 'single' || sampleQueue.length === 0) {
      return buildInferCliCommand(convertAudio);
    }
    const first = sampleQueue[0];
    return buildInferCliCommand({
      ...convertAudio,
      audioFileName: first.fileName,
      voiceModel: first.voiceModel || convertAudio.voiceModel,
      indexFile: first.indexFile || convertAudio.indexFile,
      pitch: first.pitch,
      indexRate: first.indexRate,
      protect: first.protect,
      volumeEnvelope: first.volumeEnvelope,
      autotune: first.autotune,
      proposedPitch: first.proposedPitch,
      splitAudio: first.splitAudio,
      outputFormat: first.outputFormat,
    });
  }, [convertAudio, sampleQueue]);

  const buildQueueItem = (
    fileName: string,
    sourceUrl: string
  ): SampleClipItem => ({
    id: `clip-${Date.now()}-${clipSeqRef.current++}`,
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
    const addedClips: SampleClipItem[] = fileArray.map((file) => {
      const objUrl = registerUrl(URL.createObjectURL(file));
      return buildQueueItem(file.name, objUrl);
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
          ? `Added "${first.fileName}" to sample queue. Adjust Pitch & Similarity in Step 3 or click Convert.`
          : `Added ${addedClips.length} audio clips to sample queue. Step 3 controls all clips unless overridden individually.`,
    });
  };

  const handleAddFreeSample = (sample: FreeSampleItem) => {
    const newClip = buildQueueItem(sample.fileName, sample.url);
    const nextCount = sampleQueue.length + 1;
    setSampleQueue((prev) => [...prev, newClip]);
    onUpdateConvertAudio({
      ...convertAudio,
      sourceMode: 'single',
      audioFileName: nextCount === 1 ? sample.fileName : `${nextCount} audio clips loaded`,
      audioObjectUrl: sample.url,
      conversionOutput: `Added "${sample.title}" (${sample.fileName}) to queue following Step 3 Pitch (${convertAudio.pitch >= 0 ? `+${convertAudio.pitch}` : convertAudio.pitch}) & Voice Similarity (${convertAudio.indexRate.toFixed(2)}).`,
    });
  };

  const handleDiscardClip = (clipId: string) => {
    const target = sampleQueue.find((c) => c.id === clipId);
    const remaining = sampleQueue.filter((c) => c.id !== clipId);
    if (target) {
      revokeUnused([target.sourceUrl, target.convertedUrl], remaining, convertAudio.savedSamples);
    }
    setSampleQueue(remaining);
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
  };

  const handleClearAllClips = () => {
    const removedUrls = sampleQueue.flatMap((c) => [c.sourceUrl, c.convertedUrl]);
    setSampleQueue([]);
    revokeUnused(removedUrls, [], convertAudio.savedSamples);
    onUpdateConvertAudio({
      ...convertAudio,
      audioFileName: '',
      audioObjectUrl: '',
      conversionOutput: 'Cleared all queued audio recordings.',
    });
  };

  const handleDiscardSavedSample = (sampleId: string) => {
    const target = convertAudio.savedSamples.find((s) => s.id === sampleId);
    const nextSaved = convertAudio.savedSamples.filter((s) => s.id !== sampleId);
    if (target) {
      revokeUnused([target.url], sampleQueue, nextSaved);
    }
    onUpdateConvertAudio({
      ...convertAudio,
      savedSamples: nextSaved,
      conversionOutput: target
        ? `Discarded generated sample "${target.name}".`
        : convertAudio.conversionOutput,
    });
  };

  const handleClearAllOutputs = () => {
    const removedSavedUrls = convertAudio.savedSamples.map((s) => s.url);
    const removedConvertedUrls = sampleQueue.map((c) => c.convertedUrl);
    const nextQueue = sampleQueue.map((c) => ({
      ...c,
      convertedUrl: '',
      convertedName: '',
      status: 'queued' as const,
    }));
    setSampleQueue(nextQueue);
    revokeUnused([...removedSavedUrls, ...removedConvertedUrls], nextQueue, []);
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

  const updateDefaultsAndSync = (patch: Partial<ConvertAudioState>) => {
    const nextState: ConvertAudioState = { ...convertAudio, ...patch };
    onUpdateConvertAudio(nextState);

    const hasSyncableChange =
      patch.pitch !== undefined ||
      patch.indexRate !== undefined ||
      patch.protect !== undefined ||
      patch.volumeEnvelope !== undefined ||
      patch.autotune !== undefined ||
      patch.proposedPitch !== undefined ||
      patch.splitAudio !== undefined ||
      patch.outputFormat !== undefined;

    if (!hasSyncableChange) return;

    setSampleQueue((prev) => {
      if (prev.length === 0) return prev;
      const single = prev.length === 1;
      return prev.map((c) => {
        const nextClip: SampleClipItem = { ...c };
        if (patch.pitch !== undefined && (single || !c.customPitch)) {
          nextClip.pitch = patch.pitch;
          nextClip.customPitch = false;
        }
        if (patch.indexRate !== undefined && (single || !c.customIndex)) {
          nextClip.indexRate = patch.indexRate;
          nextClip.customIndex = false;
        }
        if (patch.protect !== undefined && (single || !c.customProtect)) {
          nextClip.protect = patch.protect;
          nextClip.customProtect = false;
        }
        if (patch.volumeEnvelope !== undefined && (single || !c.customVol)) {
          nextClip.volumeEnvelope = patch.volumeEnvelope;
          nextClip.customVol = false;
        }
        if (patch.autotune !== undefined && (single || !c.customAt)) {
          nextClip.autotune = patch.autotune;
          nextClip.customAt = false;
        }
        if (patch.proposedPitch !== undefined && (single || !c.customPp)) {
          nextClip.proposedPitch = patch.proposedPitch;
          nextClip.customPp = false;
        }
        if (patch.splitAudio !== undefined && (single || !c.customSp)) {
          nextClip.splitAudio = patch.splitAudio;
          nextClip.customSp = false;
        }
        if (patch.outputFormat !== undefined && (single || !c.customFmt)) {
          nextClip.outputFormat = patch.outputFormat;
          nextClip.customFmt = false;
        }
        return nextClip;
      });
    });
  };

  const handleResetClipOverrides = (clipId: string) => {
    setSampleQueue((prev) =>
      prev.map((c) =>
        c.id === clipId
          ? {
              ...c,
              pitch: convertAudio.pitch,
              indexRate: convertAudio.indexRate,
              protect: convertAudio.protect,
              volumeEnvelope: convertAudio.volumeEnvelope,
              autotune: convertAudio.autotune,
              proposedPitch: convertAudio.proposedPitch,
              splitAudio: convertAudio.splitAudio,
              outputFormat: convertAudio.outputFormat,
              voiceModel: '',
              indexFile: '',
              customPitch: false,
              customIndex: false,
              customProtect: false,
              customVol: false,
              customAt: false,
              customPp: false,
              customSp: false,
              customFmt: false,
            }
          : c
      )
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
        customPitch: false,
        customIndex: false,
        customProtect: false,
        customVol: false,
        customAt: false,
        customPp: false,
        customSp: false,
        customFmt: false,
      }))
    );
    onUpdateConvertAudio({
      ...convertAudio,
      conversionOutput: `Applied Step 3 settings (Pitch ${convertAudio.pitch >= 0 ? `+${convertAudio.pitch}` : convertAudio.pitch}, Index ${convertAudio.indexRate.toFixed(2)}) to all ${sampleQueue.length} queued clips.`,
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
    if (refreshTimerRef.current !== null) {
      window.clearTimeout(refreshTimerRef.current);
    }
    refreshTimerRef.current = window.setTimeout(() => {
      setRefreshMsg('');
      refreshTimerRef.current = null;
    }, 2200);
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
      const nextQueue = sampleQueue.map((c) => (c.id === clipId ? updatedClip : c));
      setSampleQueue(nextQueue);

      const nextSamples = [sampleRecord, ...convertAudio.savedSamples].slice(0, 12);
      const evicted = [sampleRecord, ...convertAudio.savedSamples].slice(12);
      revokeUnused(
        [clip.convertedUrl, ...evicted.map((s) => s.url)],
        nextQueue,
        nextSamples
      );

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
      const nextQueue = rendered.map((r) => r.updatedClip);
      setSampleQueue(nextQueue);

      const newRecords = rendered.map((r) => r.sampleRecord).reverse();
      const nextSamples = [...newRecords, ...convertAudio.savedSamples].slice(0, 12);
      const evicted = [...newRecords, ...convertAudio.savedSamples].slice(12);
      revokeUnused(
        [...sampleQueue.map((c) => c.convertedUrl), ...evicted.map((s) => s.url)],
        nextQueue,
        nextSamples
      );

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

  const pitchDisplay = convertAudio.pitch > 0 ? `+${convertAudio.pitch}` : `${convertAudio.pitch}`;
  const step1Ready = Boolean(convertAudio.voiceModel.trim());
  const step2Ready =
    convertAudio.sourceMode === 'batch'
      ? Boolean(convertAudio.batchFolderPath.trim())
      : sampleQueue.length > 0;

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

      {/* 4-Step Workflow Readiness Strip */}
      <div
        data-testid="samples-readiness-strip"
        className="grid grid-cols-2 lg:grid-cols-4 gap-2.5"
      >
        <div
          className={`rounded-lg border px-3.5 py-2.5 ${
            step1Ready ? 'border-[#10B981]/50 bg-[#081A14]' : 'border-[#47161D] bg-[#160709]'
          }`}
        >
          <p className="text-[11px] font-mono uppercase tracking-wider text-[#C89EA4]">
            1. Voice Model
          </p>
          <p className="text-xs font-semibold text-[#FDF7F7] truncate mt-0.5">
            {step1Ready ? convertAudio.voiceModel.split('/').pop() : 'Select .pth in Step 1'}
          </p>
        </div>

        <div
          className={`rounded-lg border px-3.5 py-2.5 ${
            step2Ready ? 'border-[#10B981]/50 bg-[#081A14]' : 'border-[#47161D] bg-[#160709]'
          }`}
        >
          <p className="text-[11px] font-mono uppercase tracking-wider text-[#C89EA4]">
            2. Audio Source
          </p>
          <p className="text-xs font-semibold text-[#FDF7F7] truncate mt-0.5">
            {convertAudio.sourceMode === 'batch'
              ? 'Batch folder ready'
              : sampleQueue.length > 0
                ? `${sampleQueue.length} clip${sampleQueue.length === 1 ? '' : 's'} queued`
                : 'Upload or quick-pick clip'}
          </p>
        </div>

        <div className="rounded-lg border border-[#47161D] bg-[#160709] px-3.5 py-2.5">
          <p className="text-[11px] font-mono uppercase tracking-wider text-[#C89EA4]">
            3. Pitch &amp; Similarity
          </p>
          <p className="text-xs font-mono font-semibold text-[#FDF7F7] tabular-nums truncate mt-0.5">
            Pitch {pitchDisplay} · Index {convertAudio.indexRate.toFixed(2)}
          </p>
        </div>

        <div
          className={`rounded-lg border px-3.5 py-2.5 ${
            convertAudio.hasConvertedSample
              ? 'border-[#10B981]/50 bg-[#081A14]'
              : 'border-[#47161D] bg-[#160709]'
          }`}
        >
          <p className="text-[11px] font-mono uppercase tracking-wider text-[#C89EA4]">
            4. Convert &amp; Output
          </p>
          <p className="text-xs font-semibold text-[#FDF7F7] truncate mt-0.5">
            {convertAudio.hasConvertedSample
              ? `${convertAudio.savedSamples.length} rendered`
              : step1Ready && step2Ready
                ? 'Ready to convert'
                : 'Complete Steps 1–2'}
          </p>
        </div>
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
                Upload or add recordings below, or switch to Batch to convert a whole folder. Pitch &amp; settings live in Step 3.
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

            {/* Inline Quick-Pick Free Test Voices */}
            <div className="rounded-lg border border-[#47161D] bg-[#100506] px-3.5 py-3 flex flex-wrap items-center justify-between gap-2.5">
              <div>
                <p className="text-xs font-semibold text-[#FDF7F7]">
                  Quick Test Voice (Free Samples)
                </p>
                <p className="text-[11.5px] text-[#AD8288]">
                  1-click add a dry mono talking sample without uploading your own file.
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-1.5">
                {FREE_SAMPLES.map((fs) => (
                  <button
                    key={fs.id}
                    type="button"
                    onClick={() => handleAddFreeSample(fs)}
                    className="rounded-md border border-[#47161D] bg-[#160709] px-2.5 py-1 text-[11.5px] font-medium text-[#DFBFC3] hover:text-white hover:border-[#E51923] whitespace-nowrap"
                  >
                    + {fs.title.replace(' Speaking ', ' ')}
                  </button>
                ))}
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
                        Apply Step 3 Settings to All
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
                  const hasCustom = Boolean(
                    clip.customPitch ||
                      clip.customIndex ||
                      clip.customProtect ||
                      clip.customVol ||
                      clip.customAt ||
                      clip.customPp ||
                      clip.customSp ||
                      clip.customFmt ||
                      clip.voiceModel ||
                      clip.indexFile
                  );
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
                            <p className="text-[13px] text-[#C89EA4] truncate tabular-nums flex flex-wrap items-center gap-1.5">
                              <span>Pitch {pitchText}</span>
                              <span
                                className={`inline-block rounded px-1.5 py-0.5 font-mono text-[10.5px] font-semibold ${
                                  clip.customPitch
                                    ? 'border border-[#F59E0B]/50 bg-[#1B1207] text-[#F59E0B]'
                                    : 'border border-[#10B981]/50 bg-[#081A14] text-[#10B981]'
                                }`}
                              >
                                {clip.customPitch ? 'Custom Override' : 'Following Step 3'}
                              </span>
                              <span>· Index {clip.indexRate.toFixed(2)} · {clip.outputFormat.toUpperCase()}</span>
                              {clip.status === 'ready' ? '· Converted' : ''}
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

                      {/* Optional per-sample pitch & parameter overrides (kept collapsed so Step 3 Pitch stays the single primary slider) */}
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
                              ? 'Hide per-clip overrides'
                              : 'Override pitch, index & model for this clip (optional)'}
                          </span>
                        </button>

                        {isAdvExpanded && (
                          <div className="mt-2.5 rounded-lg border border-[#47161D] bg-[#160709] p-3.5 space-y-3">
                            <div className="flex flex-wrap items-center justify-between gap-2">
                              <p className="text-[12px] text-[#C89EA4]">
                                Changes here only apply to this clip. Leave untouched to follow Step 3 automatically.
                              </p>
                              {hasCustom && (
                                <button
                                  type="button"
                                  onClick={() => handleResetClipOverrides(clip.id)}
                                  className="inline-flex items-center gap-1 rounded-md border border-[#47161D] bg-[#200B0E] px-2.5 py-1 text-[11px] font-medium text-[#FDF7F7] hover:border-[#E51923]"
                                >
                                  <RotateCcw className="h-3 w-3" />
                                  <span>Use Step 3 Settings</span>
                                </button>
                              )}
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                              <div>
                                <div className="flex items-center justify-between text-xs mb-1">
                                  <label htmlFor={`clip-pitch-${clip.id}`} className="text-[#FDF7F7] font-medium">
                                    Clip Pitch Override
                                  </label>
                                  <span className="font-mono text-[#FDF7F7] tabular-nums">{pitchText}</span>
                                </div>
                                <input
                                  id={`clip-pitch-${clip.id}`}
                                  type="range"
                                  min={-32}
                                  max={32}
                                  step={1}
                                  value={clip.pitch}
                                  onChange={(e) =>
                                    handleUpdateClip(clip.id, {
                                      pitch: Number(e.target.value),
                                      customPitch: true,
                                    })
                                  }
                                  className="w-full scarlet-slider"
                                />
                              </div>

                              <div>
                                <div className="flex items-center justify-between text-xs mb-1">
                                  <label htmlFor={`clip-index-${clip.id}`} className="text-[#FDF7F7] font-medium">
                                    Clip Index Rate Override
                                  </label>
                                  <span className="font-mono text-[#FDF7F7] tabular-nums">{clip.indexRate.toFixed(2)}</span>
                                </div>
                                <input
                                  id={`clip-index-${clip.id}`}
                                  type="range"
                                  min={0}
                                  max={1}
                                  step={0.05}
                                  value={clip.indexRate}
                                  onChange={(e) =>
                                    handleUpdateClip(clip.id, {
                                      indexRate: Number(e.target.value),
                                      customIndex: true,
                                    })
                                  }
                                  className="w-full scarlet-slider"
                                />
                              </div>
                            </div>

                            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
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
                                      customFmt: true,
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
                                    handleUpdateClip(clip.id, {
                                      protect: Number(e.target.value),
                                      customProtect: true,
                                    })
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
                                    handleUpdateClip(clip.id, {
                                      volumeEnvelope: Number(e.target.value),
                                      customVol: true,
                                    })
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
                                      handleUpdateClip(clip.id, {
                                        autotune: e.target.checked,
                                        customAt: true,
                                      })
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
                                      handleUpdateClip(clip.id, {
                                        proposedPitch: e.target.checked,
                                        customPp: true,
                                      })
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
                                      handleUpdateClip(clip.id, {
                                        splitAudio: e.target.checked,
                                        customSp: true,
                                      })
                                    }
                                    className="h-3.5 w-3.5 rounded accent-[#E51923]"
                                  />
                                  <span>Split</span>
                                </label>
                              </div>
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
                              const oldConvertedUrl = clip.convertedUrl;
                              const nextQueue = sampleQueue.map((c) =>
                                c.id === clip.id
                                  ? {
                                      ...c,
                                      convertedUrl: '',
                                      convertedName: '',
                                      status: 'queued' as const,
                                    }
                                  : c
                              );
                              setSampleQueue(nextQueue);
                              revokeUnused([oldConvertedUrl], nextQueue, convertAudio.savedSamples);
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

      {/* STEP 3: Pitch & Settings */}
      <section className="rounded-xl border border-[#47161D] bg-[#160709] p-6 space-y-5">
        <div className="flex items-start gap-3">
          <StepBadge num={3} />
          <div>
            <h3 className="font-display text-[17px] font-semibold text-[#FDF7F7]">
              Pitch &amp; Settings
            </h3>
            <p className="text-[13.5px] text-[#C89EA4] mt-0.5">
              Master Voice Settings — controls Pitch and Voice Similarity for all clips automatically unless overridden per clip in Step 2.
            </p>
          </div>
        </div>

        {/* Single Primary Master Controls: Pitch + Voice Similarity */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="rounded-lg border border-[#47161D] bg-[#100506] p-4 space-y-2.5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <label
                  htmlFor="convert-pitch-slider"
                  className="block text-[13.5px] font-semibold text-[#FDF7F7]"
                >
                  Master Pitch
                </label>
                <p className="text-[12.5px] text-[#C89EA4] mt-0.5">
                  Shift semitones (-32 to +32). Updates Step 2 clips automatically.
                </p>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <span className="rounded-md border border-[#47161D] bg-[#160709] px-2.5 py-1 font-mono text-xs text-[#FDF7F7] tabular-nums">
                  {convertAudio.pitch > 0 ? `+${convertAudio.pitch}` : convertAudio.pitch}
                </span>
                <button
                  type="button"
                  onClick={() => updateDefaultsAndSync({ pitch: 0 })}
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
                updateDefaultsAndSync({ pitch: Number(e.target.value) })
              }
              className="w-full scarlet-slider"
            />
            <div className="flex justify-between text-xs font-mono text-[#C89EA4] tabular-nums">
              <span>-32</span>
              <span>0</span>
              <span>+32</span>
            </div>
            <div className="flex flex-wrap gap-1.5 pt-1">
              {[
                { label: '-12 F → M', val: -12 },
                { label: '-5 Lower', val: -5 },
                { label: '0 Same', val: 0 },
                { label: '+5 Higher', val: 5 },
                { label: '+12 M → F', val: 12 },
              ].map((preset) => (
                <button
                  key={preset.val}
                  type="button"
                  onClick={() => updateDefaultsAndSync({ pitch: preset.val })}
                  className={`rounded-md border px-2 py-1 text-[11px] font-mono transition-colors whitespace-nowrap ${
                    convertAudio.pitch === preset.val
                      ? 'border-[#E51923] bg-[#E51923] text-white font-semibold'
                      : 'border-[#47161D] bg-[#160709] text-[#DFBFC3] hover:text-white hover:border-[#E51923]/60'
                  }`}
                >
                  {preset.label}
                </button>
              ))}
            </div>
          </div>

          <div className="rounded-lg border border-[#47161D] bg-[#100506] p-4 space-y-2.5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <label
                  htmlFor="conv-index-rate"
                  className="block text-[13.5px] font-semibold text-[#FDF7F7]"
                >
                  Voice Similarity (Index Rate)
                </label>
                <p className="text-[12.5px] text-[#C89EA4] mt-0.5">
                  Controls how strongly accent &amp; timbre match your .index (0.75 recommended).
                </p>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <span className="rounded-md border border-[#47161D] bg-[#160709] px-2.5 py-1 font-mono text-xs text-[#FDF7F7] tabular-nums">
                  {convertAudio.indexRate.toFixed(2)}
                </span>
                <button
                  type="button"
                  onClick={() => updateDefaultsAndSync({ indexRate: 0.75 })}
                  aria-label="Reset search feature ratio to 0.75"
                  className="rounded-md border border-[#47161D] bg-[#200B0E] p-1.5 text-[#DFBFC3] hover:text-white"
                >
                  <RotateCcw className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
            <input
              id="conv-index-rate"
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={convertAudio.indexRate}
              onChange={(e) =>
                updateDefaultsAndSync({ indexRate: Number(e.target.value) })
              }
              className="w-full scarlet-slider"
            />
            <div className="flex justify-between text-xs font-mono text-[#C89EA4] tabular-nums">
              <span>0 (Smoother)</span>
              <span>0.75 (Default)</span>
              <span>1 (Strong Accent)</span>
            </div>
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
                  <span>Voice Matching (Envelope &amp; Consonant Protect)</span>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[11px] font-normal text-[#AD8288] tabular-nums">
                      env {convertAudio.volumeEnvelope} · protect {convertAudio.protect}
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
                            onClick={() => updateDefaultsAndSync({ volumeEnvelope: 1 })}
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
                          updateDefaultsAndSync({ volumeEnvelope: Number(e.target.value) })
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
                            onClick={() => updateDefaultsAndSync({ protect: 0.5 })}
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
                          updateDefaultsAndSync({ protect: Number(e.target.value) })
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
                          updateDefaultsAndSync({ autotune: e.target.checked })
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
                          updateDefaultsAndSync({ proposedPitch: e.target.checked })
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
                          updateDefaultsAndSync({ splitAudio: e.target.checked })
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
                            updateDefaultsAndSync({
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
      </section>

      {/* STEP 4: Convert & Output */}
      <section className="rounded-xl border border-[#47161D] bg-[#160709] p-6 space-y-5">
        <div className="flex items-start gap-3">
          <StepBadge num={4} />
          <div>
            <h3 className="font-display text-[17px] font-semibold text-[#FDF7F7]">
              Convert &amp; Output
            </h3>
            <p className="text-[13.5px] text-[#C89EA4] mt-0.5">
              Render your queued recordings or batch folder, then listen and download results below.
            </p>
          </div>
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

        <div className="rounded-lg border border-[#47161D] bg-[#100506]">
          <button
            type="button"
            onClick={() => setCliOpen((v) => !v)}
            aria-expanded={cliOpen}
            className="w-full flex items-center justify-between px-4 py-2.5 text-xs font-semibold text-[#C89EA4] hover:text-[#FDF7F7]"
          >
            <span>View Generated core.py CLI Command</span>
            {cliOpen ? (
              <ChevronDown className="h-3.5 w-3.5" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5" />
            )}
          </button>
          {cliOpen && (
            <pre className="border-t border-[#47161D] p-3 text-[11px] font-mono text-[#AD8288] overflow-x-auto">
              {inferCliCommand}
            </pre>
          )}
        </div>
      </section>
    </div>
  );
});
