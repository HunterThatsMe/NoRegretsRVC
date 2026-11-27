import type {
  ApplioTabId,
  StudioState,
  VoiceModelProject,
} from '../types/studio.ts';

export const BRAND_NAME = 'NoRegretsRVC';
export const ACCENT_HEX = '#E51923';
export const APP_VERSION = '0.0.0';
export const REPO_URL = 'https://github.com/HunterThatsMe/NoRegretsRVC';
export const REPO_README_URL = 'https://github.com/HunterThatsMe/NoRegretsRVC#readme';
export const LEGACY_CORE_PRETRAIN_NAME = 'Legacy Core 1.5 V2';
export const LEGACY_CORE_PRETRAIN_G =
  'https://huggingface.co/notiscBalbalEko/PretrainReupload/resolve/main/G_LegacyCore1.5v2.pth?download=true';
export const LEGACY_CORE_PRETRAIN_D =
  'https://huggingface.co/notiscBalbalEko/PretrainReupload/resolve/main/D_LegacyCore1.5v2.pth?download=true';

export const SUPPORTED_LANGUAGES: { code: string; label: string }[] = [
  { code: 'System Auto', label: 'System Auto' },
  { code: 'en_US', label: 'English' },
  { code: 'es_ES', label: 'Spanish (Español)' },
  { code: 'pt_BR', label: 'Portuguese (Português)' },
  { code: 'fr_FR', label: 'French (Français)' },
  { code: 'it_IT', label: 'Italian (Italiano)' },
  { code: 'nl_NL', label: 'Dutch (Nederlands)' },
  { code: 'ja_JP', label: 'Japanese (日本語)' },
  { code: 'zh_CN', label: 'Chinese (中文)' },
  { code: 'ar_SA', label: 'Arabic (العربية)' },
];

export const APPLIO_TAB_CATALOG: {
  id: ApplioTabId;
  label: string;
  description: string;
}[] = [
  {
    id: 'train',
    label: 'Training',
    description: 'Slice voice datasets, extract pitch contours, and train custom RVC checkpoints',
  },
  {
    id: 'models',
    label: 'Models',
    description: 'Fetch pretrained weights, drop local .pth/.index archives, and queue downloads',
  },
  {
    id: 'samples',
    label: 'Samples',
    description: 'Audition trained voices on reference clips and save converted sample exports',
  },
  {
    id: 'voice_blender',
    label: 'Voice Blender',
    description: 'Merge two compatible RVC checkpoints with weighted vocal blending',
  },
  {
    id: 'tts',
    label: 'TTS',
    description: 'Synthesize spoken lines with Edge-TTS and route them through your voice model',
  },
  {
    id: 'extra',
    label: 'Extra',
    description: 'Inspect checkpoint headers, run audio diagnostics, and inspect dataset stats',
  },
  {
    id: 'settings',
    label: 'Settings',
    description: 'Manage runtime precision, author tags, audio output, and Kaggle tunnels',
  },
];

export const DEFAULT_PROJECTS: VoiceModelProject[] = [
  {
    id: 'proj-default-32k',
    name: '',
    samplingRate: '32000',
    vocoder: 'HiFi-GAN',
    cpuCores: 4,
    gpuNumber: '0-1',
    gpuCount: 2,
    datasetPath: '',
    datasetCreator: false,
    audioCutting: 'Automatic',
    chunkLengthSec: 3,
    overlapLengthSec: 0.3,
    truncateSilenceEnabled: false,
    silenceDurationMs: 300,
    noiseFilter: false,
    loudnessNorm: true,
    loudnessTargetLufs: -20.0,
    pitchExtractor: 'rmvpe',
    embedder: 'contentvec',
    silentTrainingFiles: 0,
    batchSize: 2,
    saveEveryEpoch: 5,
    totalEpoch: 200,
    fp16: false,
    optimizer: 'AdamW',
    melLoss: 'Single-Scale',
    saveOnlyLatest: true,
    saveEveryWeights: true,
    pretrained: true,
    freshTraining: false,
    cacheDatasetGpu: false,
    checkpointing: false,
    shutdownAfterFinish: false,
    customPretrained: false,
    customPretrainedG: '',
    customPretrainedD: '',
    indexAlgorithm: 'Auto',
    overtrainingDetector: false,
    agreedTerms: true,
    preprocessOutput: 'The output information will be displayed here.',
    extractOutput: 'The output information will be displayed here.',
    trainingOutput: 'The output information will be displayed here.',
    exportPthFile: '',
    exportIndexFile: '',
    updatedAt: '2026-11-16T12:00:00.000Z',
  },
];

export const DEFAULT_STUDIO_STATE: StudioState = {
  activeSection: 'training',
  projects: DEFAULT_PROJECTS,
  activeProjectId: DEFAULT_PROJECTS[0].id,
  convertAudio: {
    voiceModel: '',
    indexFile: '',
    sourceMode: 'single',
    audioFileName: '',
    audioObjectUrl: '',
    batchFolderPath: '/kaggle/working/audio_inputs',
    pitch: 0,
    indexRate: 0.75,
    protect: 0.5,
    volumeEnvelope: 1,
    speakerId: 0,
    pitchExtractor: 'rmvpe',
    embedder: 'contentvec',
    splitAudio: false,
    autotune: false,
    proposedPitch: false,
    noiseReduction: false,
    noiseReductionStrength: 12,
    formantShiftEnabled: false,
    formantQuefrency: 1.0,
    formantTimbre: 1.0,
    outputPath: '/kaggle/working/program_ml/assets/audios/output.wav',
    outputFormat: 'wav',
    agreedTerms: true,
    conversionOutput: 'The output information will be displayed here.',
    hasConvertedSample: false,
    savedSamples: [],
  },
  generalSettings: {
    listFilterEnabled: false,
    rmvpeHighRegisterCorrection: false,
    rmvpeCorrectionMode: 'true_pitch',
    rmvpeF0CeilingHz: 1250,
    discordPresence: false,
    language: 'System Auto',
    versionStatus: `${BRAND_NAME} Beta version ${APP_VERSION} (newest version: ${APP_VERSION}).`,
  },
  trainingSettings: {
    modelAuthor: 'None',
    precision: 'fp32',
  },
  forkConfig: {
    customRepoUrl: '',
    activeTabs: {
      samples: true,
      train: true,
      voice_blender: true,
      tts: true,
      models: true,
      tensorboard: false,
      extra: true,
      settings: true,
    },
    tunnelMode: 'Gradio + Cloudflare',
    horizonId: '',
    randomAuthEnabled: true,
    embeddedTensorBoard: false,
    dualGpuT4: true,
    useUvVenv: true,
    verifyOnnxGpu: true,
    turboTraining: true,
  },
  blender: {
    modelAPath: '',
    modelBPath: '',
    ratio: 0.5,
    samplingRate: '32000',
    outputName: '',
    f0Method: 'rmvpe',
  },
  ttsDefaults: {
    voice: 'en-US-AndrewNeural',
    modelPthPath: '',
    indexFilePath: '',
    rate: 0,
    pitch: 0,
    protect: 0.5,
    indexRate: 0.75,
  },
  downloadManifest: [
    {
      id: 'dl-base-rmvpe',
      name: 'RMVPE Pitch Model (Default)',
      url: 'https://huggingface.co/IAHispano/Applio/resolve/main/Resources/predictors/rmvpe.pt',
      category: 'pretrained',
    },
    {
      id: 'dl-contentvec',
      name: 'ContentVec Embedder',
      url: 'https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/contentvec/pytorch_model.bin',
      category: 'embedder',
    },
  ],
  importedModels: [],
  lastAnalysis: null,
};

export const SAMPLE_TRAINING_LOG = `Starting preprocess with 4 processes...
Preprocess completed in 6.58 seconds on 00:18:01 seconds of audio.
Model StudioDemo preprocessed successfully.
Starting pitch extraction on cuda:0, cuda:1 using rmvpe...
Pitch extraction completed in 28.99 seconds (411/411 files successful).
Starting embedding extraction with 4 cores on cuda:0, cuda:1...
Embedding extraction completed in 19.52 seconds.
Saved index file '/kaggle/working/program_ml/logs/StudioDemo/StudioDemo.index'
Using HiFi-GAN vocoder
Using AdamW optimizer
Using Multi-Scale Mel (512 / 1024 / 2048) loss + Generator EMA (0.999) + Grad Clipping
Using Float32 for training.
StudioDemo | epoch=1 | step=104 | time=19:04:10 | training_speed=0:00:52
StudioDemo | epoch=2 | step=208 | time=19:04:59 | training_speed=0:00:49 | lowest_value=20.992 (epoch 1 and step 75) | ema_decay=0.999 | grad_norm=28.4
StudioDemo | epoch=7 | step=728 | time=19:09:05 | training_speed=0:00:49 | lowest_value=18.901 (epoch 7 and step 653) | ema_decay=0.999 | grad_norm=26.1
StudioDemo | epoch=13 | step=1352 | time=19:14:00 | training_speed=0:00:49 | lowest_value=16.913 (epoch 13 and step 1331) | ema_decay=0.999 | grad_norm=24.8
StudioDemo | epoch=44 | step=4576 | time=19:39:40 | training_speed=0:00:49 | lowest_value=15.394 (epoch 44 and step 4506) | ema_decay=0.999 | grad_norm=23.9
StudioDemo | epoch=51 | step=5304 | time=19:45:28 | training_speed=0:00:50 | lowest_value=14.616 (epoch 51 and step 5287) | ema_decay=0.999 | grad_norm=24.2`;

export const SAMPLE_OVERTRAIN_LOG = `Starting preprocess with 4 processes...
Preprocess completed in 4.12 seconds on 00:08:20 seconds of audio.
Model CrispVocal preprocessed successfully.
Starting pitch extraction on cuda:0, cuda:1 using rmvpe...
Pitch extraction completed in 18.40 seconds (135/135 files successful).
Using HiFi-GAN vocoder
Using AdamW optimizer
CrispVocal | epoch=20 | step=340 | training_speed=0:00:22 | lowest_value=21.400 (epoch 20 and step 340) | mel=21.40 kl=1.42 fm=6.10 d_total=2.12 g_total=28.40 grad_norm=42.5
CrispVocal | epoch=60 | step=1020 | training_speed=0:00:21 | lowest_value=16.200 (epoch 60 and step 1020) | mel=16.20 kl=1.08 fm=4.55 d_total=1.96 g_total=23.10 grad_norm=38.2
CrispVocal | epoch=120 | step=2040 | training_speed=0:00:21 | lowest_value=13.450 (epoch 120 and step 2040) | mel=13.45 kl=0.88 fm=3.65 d_total=1.89 g_total=19.80 grad_norm=34.1
CrispVocal | epoch=160 | step=2720 | training_speed=0:00:21 | lowest_value=12.680 (epoch 160 and step 2720) | mel=12.68 kl=0.82 fm=3.32 d_total=1.85 g_total=18.90 grad_norm=35.8
CrispVocal | epoch=200 | step=3400 | training_speed=0:00:21 | lowest_value=13.120 (epoch 160 and step 2720) | mel=13.12 kl=0.89 fm=3.58 d_total=1.62 g_total=21.40 grad_norm=58.4
CrispVocal | epoch=240 | step=4080 | training_speed=0:00:21 | lowest_value=13.840 (epoch 160 and step 2720) | mel=13.84 kl=1.04 fm=4.15 d_total=1.34 g_total=24.90 grad_norm=96.2`;
