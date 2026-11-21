import { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { DEFAULT_PROJECTS, DEFAULT_STUDIO_STATE } from './data/defaults';
import {
  ApplioGeneralSettings,
  ApplioTrainingSettings,
  BlenderRecipe,
  ConvertAudioState,
  DownloadModelItem,
  ForkConfig,
  ImportedLocalModel,
  SampleClipItem,
  StudioSection,
  StudioState,
  TrainingAnalysis,
  TrainingPrecision,
  TtsDefaults,
  VoiceModelProject,
} from './types/studio';
import { I18nProvider, resolveLocale } from './utils/i18n';
import { TopBar } from './components/TopBar';
import { TrainWorkspaceView } from './components/TrainWorkspaceView';

const loadConvertAudioView = () =>
  import('./components/ConvertAudioView').then((m) => ({ default: m.ConvertAudioView }));
const loadWorkbenchView = () =>
  import('./components/WorkbenchView').then((m) => ({ default: m.WorkbenchView }));
const loadTensorBoardView = () =>
  import('./components/TensorBoardView').then((m) => ({ default: m.TensorBoardView }));
const loadSettingsView = () =>
  import('./components/SettingsView').then((m) => ({ default: m.SettingsView }));

const ConvertAudioView = lazy(loadConvertAudioView);
const WorkbenchView = lazy(loadWorkbenchView);
const TensorBoardView = lazy(loadTensorBoardView);
const SettingsView = lazy(loadSettingsView);

const STORAGE_KEY = 'noregrets_rvc_studio_v5';

const VALID_SECTIONS: StudioSection[] = [
  'training',
  'models',
  'samples',
  'blender',
  'tts',
  'tensorboard',
  'settings',
];

function serializeState(state: StudioState): string {
  const serializable: StudioState = {
    ...state,
    convertAudio: {
      ...state.convertAudio,
      audioObjectUrl: '',
      savedSamples: [],
    },
  };
  return JSON.stringify(serializable);
}

function loadInitialState(): StudioState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_STUDIO_STATE;
    const parsed = JSON.parse(raw) as Partial<StudioState>;
    if (!parsed.projects || !Array.isArray(parsed.projects) || parsed.projects.length === 0) {
      return DEFAULT_STUDIO_STATE;
    }
    const safeSection: StudioSection = VALID_SECTIONS.includes(
      parsed.activeSection as StudioSection
    )
      ? (parsed.activeSection as StudioSection)
      : 'training';
    const rawConvert: Partial<ConvertAudioState> = parsed.convertAudio ?? {};
    const usesLegacyProtectVol =
      rawConvert.protect === 0.33 && rawConvert.volumeEnvelope === 0.25;
    const migratedConvert: ConvertAudioState = {
      ...DEFAULT_STUDIO_STATE.convertAudio,
      ...rawConvert,
      protect: usesLegacyProtectVol
        ? DEFAULT_STUDIO_STATE.convertAudio.protect
        : (rawConvert.protect ?? DEFAULT_STUDIO_STATE.convertAudio.protect),
      volumeEnvelope: usesLegacyProtectVol
        ? DEFAULT_STUDIO_STATE.convertAudio.volumeEnvelope
        : (rawConvert.volumeEnvelope ?? DEFAULT_STUDIO_STATE.convertAudio.volumeEnvelope),
      audioObjectUrl: '',
      savedSamples: [],
    };
    const rawGeneral: Partial<ApplioGeneralSettings> = parsed.generalSettings ?? {};
    const migratedVersionStatus =
      typeof rawGeneral.versionStatus === 'string' &&
      /Beta version 0\.0(?!\.\d)/.test(rawGeneral.versionStatus)
        ? DEFAULT_STUDIO_STATE.generalSettings.versionStatus
        : (rawGeneral.versionStatus ?? DEFAULT_STUDIO_STATE.generalSettings.versionStatus);
    const rawTts: Partial<TtsDefaults> = parsed.ttsDefaults ?? {};
    const migratedTtsProtect =
      rawTts.protect === 0.33 ? DEFAULT_STUDIO_STATE.ttsDefaults.protect : (rawTts.protect ?? DEFAULT_STUDIO_STATE.ttsDefaults.protect);

    const activeSaved =
      parsed.projects.find((p) => p.id === parsed.activeProjectId) ?? parsed.projects[0];

    return {
      ...DEFAULT_STUDIO_STATE,
      ...parsed,
      projects: [activeSaved],
      activeProjectId: activeSaved.id,
      activeSection: safeSection,
      convertAudio: migratedConvert,
      generalSettings: {
        ...DEFAULT_STUDIO_STATE.generalSettings,
        ...rawGeneral,
        versionStatus: migratedVersionStatus,
        listFilterEnabled: false,
      },
      trainingSettings: {
        ...DEFAULT_STUDIO_STATE.trainingSettings,
        ...(parsed.trainingSettings ?? {}),
      },
      forkConfig: {
        ...DEFAULT_STUDIO_STATE.forkConfig,
        ...(parsed.forkConfig ?? {}),
        activeTabs: {
          ...DEFAULT_STUDIO_STATE.forkConfig.activeTabs,
          ...(parsed.forkConfig?.activeTabs ?? {}),
        },
      },
      blender: {
        ...DEFAULT_STUDIO_STATE.blender,
        ...(parsed.blender ?? {}),
      },
      ttsDefaults: {
        ...DEFAULT_STUDIO_STATE.ttsDefaults,
        ...rawTts,
        protect: migratedTtsProtect,
      },
    };
  } catch {
    return DEFAULT_STUDIO_STATE;
  }
}

const SectionSkeleton = () => (
  <div className="space-y-6" aria-busy="true" aria-label="Loading section">
    <div className="h-14 rounded-xl border border-[#47161D] bg-[#160709]" />
    <div className="h-56 rounded-xl border border-[#47161D] bg-[#160709]" />
    <div className="h-56 rounded-xl border border-[#47161D] bg-[#160709]" />
  </div>
);

export default function App() {
  const [state, setState] = useState<StudioState>(loadInitialState);
  const [sampleQueue, setSampleQueue] = useState<SampleClipItem[]>([]);
  const sampleQueueRef = useRef<SampleClipItem[]>(sampleQueue);
  sampleQueueRef.current = sampleQueue;
  const createdBlobUrlsRef = useRef<Set<string>>(new Set());
  const latestStateRef = useRef<StudioState>(state);
  latestStateRef.current = state;

  useEffect(() => {
    const urls = createdBlobUrlsRef.current;
    return () => {
      urls.forEach((u) => URL.revokeObjectURL(u));
      urls.clear();
    };
  }, []);

  const registerBlobUrl = useCallback((url: string) => {
    if (url.startsWith('blob:')) {
      createdBlobUrlsRef.current.add(url);
    }
    return url;
  }, []);

  const revokeBlobUrl = useCallback((url: string) => {
    if (!url) return;
    if (createdBlobUrlsRef.current.has(url)) {
      URL.revokeObjectURL(url);
      createdBlobUrlsRef.current.delete(url);
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      try {
        localStorage.setItem(STORAGE_KEY, serializeState(latestStateRef.current));
      } catch {
        // Ignore storage quota errors
      }
    }, 250);
    return () => window.clearTimeout(timer);
  }, [state]);

  useEffect(() => {
    const flush = () => {
      try {
        localStorage.setItem(STORAGE_KEY, serializeState(latestStateRef.current));
      } catch {
        // Ignore storage quota errors
      }
    };
    window.addEventListener('pagehide', flush);
    window.addEventListener('beforeunload', flush);
    return () => {
      window.removeEventListener('pagehide', flush);
      window.removeEventListener('beforeunload', flush);
      flush();
    };
  }, []);

  const activeProject = useMemo(
    () => state.projects.find((p) => p.id === state.activeProjectId) ?? state.projects[0],
    [state.projects, state.activeProjectId]
  );

  useEffect(() => {
    const resolved = resolveLocale(state.generalSettings.language);
    document.documentElement.lang = resolved.replace('_', '-');
    document.documentElement.dir = resolved === 'ar_SA' ? 'rtl' : 'ltr';
  }, [state.generalSettings.language]);

  const handlePrefetchSection = useCallback((section: StudioSection) => {
    switch (section) {
      case 'samples':
        void loadConvertAudioView();
        break;
      case 'models':
      case 'blender':
      case 'tts':
        void loadWorkbenchView();
        break;
      case 'tensorboard':
        void loadTensorBoardView();
        break;
      case 'settings':
        void loadSettingsView();
        break;
      default:
        break;
    }
  }, []);

  const handleSelectSection = useCallback((activeSection: StudioSection) => {
    setState((prev) =>
      prev.activeSection === activeSection ? prev : { ...prev, activeSection }
    );
  }, []);

  const handleUpdateProject = useCallback((updated: VoiceModelProject) => {
    const stamp = new Date().toISOString();
    setState((prev) => ({
      ...prev,
      projects: [{ ...updated, updatedAt: stamp }],
      activeProjectId: updated.id,
    }));
  }, []);

  const handleUpdateConvertAudio = useCallback((convertAudio: ConvertAudioState) => {
    setState((prev) => ({ ...prev, convertAudio }));
  }, []);

  const handleUpdateGeneral = useCallback((generalSettings: ApplioGeneralSettings) => {
    setState((prev) => ({ ...prev, generalSettings }));
  }, []);

  const handleUpdateTrainingSettings = useCallback(
    (trainingSettings: ApplioTrainingSettings) => {
      setState((prev) => ({ ...prev, trainingSettings }));
    },
    []
  );

  const handleUpdatePrecisionAcrossProjects = useCallback(
    (precision: TrainingPrecision) => {
      setState((prev) => ({
        ...prev,
        trainingSettings: { ...prev.trainingSettings, precision },
        projects: prev.projects.map((p) => ({
          ...p,
          fp16: precision === 'fp16',
        })),
      }));
    },
    []
  );

  const handleUpdateForkConfig = useCallback((forkConfig: ForkConfig) => {
    setState((prev) => ({ ...prev, forkConfig }));
  }, []);

  const handleUpdateBlender = useCallback((blender: BlenderRecipe) => {
    setState((prev) => ({ ...prev, blender }));
  }, []);

  const handleUpdateTts = useCallback((ttsDefaults: TtsDefaults) => {
    setState((prev) => ({ ...prev, ttsDefaults }));
  }, []);

  const handleUpdateManifest = useCallback((downloadManifest: DownloadModelItem[]) => {
    setState((prev) => ({ ...prev, downloadManifest }));
  }, []);

  const handleUpdateImportedModels = useCallback(
    (importedModels: ImportedLocalModel[]) => {
      setState((prev) => ({ ...prev, importedModels }));
    },
    []
  );

  const handleAnalysisChange = useCallback((lastAnalysis: TrainingAnalysis | null) => {
    setState((prev) => ({ ...prev, lastAnalysis }));
  }, []);

  const handleApplyCustomPretrained = useCallback((gPath: string, dPath: string) => {
    const stamp = new Date().toISOString();
    setState((prev) => ({
      ...prev,
      projects: prev.projects.map((p) =>
        p.id === prev.activeProjectId
          ? {
              ...p,
              customPretrained: true,
              customPretrainedG: gPath || p.customPretrainedG,
              customPretrainedD: dPath || p.customPretrainedD,
              updatedAt: stamp,
            }
          : p
      ),
    }));
  }, []);

  return (
    <I18nProvider language={state.generalSettings.language}>
      <div className="min-h-screen flex flex-col bg-[#0D0405] text-[#FDF7F7]">
        <TopBar
          activeSection={state.activeSection}
          onSelectSection={handleSelectSection}
          onPrefetchSection={handlePrefetchSection}
          activeProject={activeProject}
        />

      <main className="flex-1 px-4 md:px-8 py-4 pb-10 max-w-[1440px] w-full mx-auto">
          {state.activeSection === 'training' && (
            <TrainWorkspaceView
              activeProject={activeProject}
              onUpdateProject={handleUpdateProject}
            />
          )}

          {state.activeSection !== 'training' && (
            <Suspense fallback={<SectionSkeleton />}>
              {state.activeSection === 'samples' && (
                <ConvertAudioView
                  convertAudio={state.convertAudio}
                  projects={state.projects}
                  importedModels={state.importedModels}
                  sampleQueue={sampleQueue}
                  onSetSampleQueue={setSampleQueue}
                  onRegisterBlobUrl={registerBlobUrl}
                  onRevokeUrl={revokeBlobUrl}
                  onUpdateConvertAudio={handleUpdateConvertAudio}
                />
              )}

              {(state.activeSection === 'blender' ||
                state.activeSection === 'tts' ||
                state.activeSection === 'models') && (
                <WorkbenchView
                  mode={state.activeSection}
                  projects={state.projects}
                  activeProject={activeProject}
                  blender={state.blender}
                  ttsDefaults={state.ttsDefaults}
                  downloadManifest={state.downloadManifest}
                  importedModels={state.importedModels}
                  onUpdateBlender={handleUpdateBlender}
                  onUpdateTts={handleUpdateTts}
                  onUpdateManifest={handleUpdateManifest}
                  onUpdateImportedModels={handleUpdateImportedModels}
                  onApplyCustomPretrained={handleApplyCustomPretrained}
                />
              )}

              {state.activeSection === 'tensorboard' && (
                <TensorBoardView
                  activeProject={activeProject}
                  lastAnalysis={state.lastAnalysis}
                  onAnalysisChange={handleAnalysisChange}
                />
              )}

              {state.activeSection === 'settings' && (
                <SettingsView
                  generalSettings={state.generalSettings}
                  trainingSettings={state.trainingSettings}
                  forkConfig={state.forkConfig}
                  activeProject={activeProject}
                  downloadManifest={state.downloadManifest}
                  onUpdateGeneral={handleUpdateGeneral}
                  onUpdateTrainingSettings={handleUpdateTrainingSettings}
                  onUpdatePrecisionAcrossProjects={handleUpdatePrecisionAcrossProjects}
                  onUpdateForkConfig={handleUpdateForkConfig}
                />
              )}
            </Suspense>
          )}
        </main>
      </div>
    </I18nProvider>
  );
}
