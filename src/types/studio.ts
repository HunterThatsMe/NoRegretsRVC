export type SamplingRate = '32000' | '40000' | '48000';
export type Vocoder = 'HiFi-GAN' | 'MRF HiFi-GAN' | 'RefineGAN';
export type PitchExtractor = 'rmvpe' | 'fcpe' | 'swift-f0' | 'crepe';
export type EmbedderModel = 'contentvec' | 'spin' | 'chinese-hubert-base';
export type OptimizerType = 'AdamW' | 'RAdam';
export type MelLossType = 'Single-Scale' | 'Multi-Scale';
export type AudioCuttingMethod = 'Automatic' | 'Simple' | 'Skip';
export type IndexAlgorithm = 'Auto' | 'Faiss' | 'KMeans';
export type RmvpeCorrectionMode = 'true_pitch' | 'fold';
export type TrainingPrecision = 'fp32' | 'fp16';

export type ApplioTabId =
  | 'samples'
  | 'train'
  | 'voice_blender'
  | 'tts'
  | 'models'
  | 'tensorboard'
  | 'extra'
  | 'settings';

export type TunnelMode = 'Gradio + Cloudflare' | 'Horizon';

export interface VoiceModelProject {
  id: string;
  name: string;
  samplingRate: SamplingRate;
  vocoder: Vocoder;
  cpuCores: number;
  gpuNumber: string;
  gpuCount: 1 | 2;
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
  pitchExtractor: PitchExtractor;
  embedder: EmbedderModel;
  silentTrainingFiles: number;
  batchSize: number;
  saveEveryEpoch: number;
  totalEpoch: number;
  fp16: boolean;
  optimizer: OptimizerType;
  melLoss: MelLossType;
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
  overtrainingDetector: boolean;
  agreedTerms: boolean;
  preprocessOutput: string;
  extractOutput: string;
  trainingOutput: string;
  exportPthFile: string;
  exportIndexFile: string;
  updatedAt: string;
}

export interface GeneratedSampleItem {
  id: string;
  name: string;
  pitch: number;
  format: 'wav' | 'mp3' | 'flac';
  url: string;
  createdAt: string;
  voiceModel?: string;
  indexRate?: number;
}

export interface SampleClipItem {
  id: string;
  fileName: string;
  sourceUrl: string;
  pitch: number;
  indexRate: number;
  voiceModel: string;
  indexFile: string;
  protect: number;
  volumeEnvelope: number;
  speakerId: number;
  pitchExtractor: PitchExtractor;
  embedder: EmbedderModel;
  splitAudio: boolean;
  autotune: boolean;
  proposedPitch: boolean;
  outputFormat: 'wav' | 'mp3' | 'flac';
  convertedUrl: string;
  convertedName: string;
  status: 'queued' | 'ready';
}

export type ModelSoundFeedback =
  | 'untested'
  | 'undertrained'
  | 'clean'
  | 'overtrained';

export interface ConvertAudioState {
  voiceModel: string;
  indexFile: string;
  sourceMode: 'single' | 'batch';
  audioFileName: string;
  audioObjectUrl: string;
  batchFolderPath: string;
  pitch: number;
  indexRate: number;
  protect: number;
  volumeEnvelope: number;
  speakerId: number;
  pitchExtractor: PitchExtractor;
  embedder: EmbedderModel;
  splitAudio: boolean;
  autotune: boolean;
  proposedPitch: boolean;
  noiseReduction: boolean;
  noiseReductionStrength: number;
  formantShiftEnabled: boolean;
  formantQuefrency: number;
  formantTimbre: number;
  outputPath: string;
  outputFormat: 'wav' | 'mp3' | 'flac';
  agreedTerms: boolean;
  conversionOutput: string;
  hasConvertedSample: boolean;
  savedSamples: GeneratedSampleItem[];
}

export interface ApplioGeneralSettings {
  listFilterEnabled: boolean;
  rmvpeHighRegisterCorrection: boolean;
  rmvpeCorrectionMode: RmvpeCorrectionMode;
  rmvpeF0CeilingHz: number;
  discordPresence: boolean;
  language: string;
  versionStatus: string;
}

export interface ApplioTrainingSettings {
  modelAuthor: string;
  precision: TrainingPrecision;
}

export interface ForkConfig {
  customRepoUrl: string;
  activeTabs: Record<ApplioTabId, boolean>;
  tunnelMode: TunnelMode;
  horizonId: string;
  randomAuthEnabled: boolean;
  embeddedTensorBoard: boolean;
  dualGpuT4: boolean;
  useUvVenv: boolean;
  verifyOnnxGpu: boolean;
  turboTraining: boolean;
}

export interface BlenderRecipe {
  modelAPath: string;
  modelBPath: string;
  ratio: number;
  samplingRate: SamplingRate;
  outputName: string;
  f0Method: PitchExtractor;
}

export interface TtsDefaults {
  voice: string;
  modelPthPath: string;
  indexFilePath: string;
  rate: number;
  pitch: number;
  protect: number;
  indexRate: number;
}

export interface DownloadModelItem {
  id: string;
  name: string;
  url: string;
  category: 'voice' | 'pretrained' | 'embedder';
}

export interface ImportedLocalModel {
  id: string;
  name: string;
  pthPath: string;
  indexPath: string;
}

export interface TrainingEpochPoint {
  epoch: number;
  step: number;
  speedSec: number;
  lowestLoss: number | null;
}

export interface LiveEpochProgress {
  epoch: number;
  step: number;
  percent: number;
  batchDone: number;
  batchTotal: number;
  elapsedLabel: string;
  etaLabel: string;
  itPerSec: number;
}

export interface TrainingAnalysis {
  speakerName: string;
  audioDurationLabel: string;
  segmentCount: number;
  epochsCompleted: number;
  latestStep: number;
  avgSpeedSec: number;
  lowestLoss: number | null;
  lowestLossEpoch: number | null;
  lowestLossStep: number | null;
  gpuDevices: string[];
  vocoderDetected: string;
  optimizerDetected: string;
  points: TrainingEpochPoint[];
  liveProgress?: LiveEpochProgress | null;
}

export interface T4VramEstimate {
  estimatedVramGb: number;
  vramCapGb: number;
  fitStatus: 'Optimal' | 'Tight' | 'Risk of OOM';
  recommendedBatchSize: number;
  warnings: string[];
}

export interface KaggleSurvivalEstimate {
  estimatedDiskGb: number;
  diskCapGb: number;
  diskStatus: 'Safe' | 'Watch Disk' | 'Risk of Full';
  checkpointCount: number;
  estimatedHours: number;
  sessionStatus: 'Fits 12h' | 'Tight 12h' | 'May Exceed 12h';
}

export interface TrainingCompletionEstimate {
  remainingSeconds: number;
  remainingLabel: string;
  finishTimeLabel: string;
  finishDayLabel: string;
  accuracyPercent: number;
  disclaimer: string;
}

export interface TrainingZoneInsight {
  confidence: 'observed' | 'estimate';
  saveEveryEpoch: number;
  modelContext: string;
  summary: string;
  undertrainingLabel: string;
  undertrainingEndEpoch: number | null;
  sweetSpotLabel: string;
  sweetSpotEpoch: number | null;
  sweetSpotMinEpoch: number | null;
  sweetSpotMaxEpoch: number | null;
  overtrainingLabel: string;
  overtrainingStartEpoch: number | null;
}

export interface ValidationIssue {
  field: string;
  severity: 'error' | 'warning';
  message: string;
}

export type StudioSection =
  | 'training'
  | 'samples'
  | 'free_samples'
  | 'blender'
  | 'tts'
  | 'models'
  | 'tensorboard'
  | 'settings';

export interface StudioState {
  activeSection: StudioSection;
  projects: VoiceModelProject[];
  activeProjectId: string;
  convertAudio: ConvertAudioState;
  generalSettings: ApplioGeneralSettings;
  trainingSettings: ApplioTrainingSettings;
  forkConfig: ForkConfig;
  blender: BlenderRecipe;
  ttsDefaults: TtsDefaults;
  downloadManifest: DownloadModelItem[];
  importedModels: ImportedLocalModel[];
  lastAnalysis: TrainingAnalysis | null;
}
