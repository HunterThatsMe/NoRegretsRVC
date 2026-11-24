export interface CleanupCalibration {
  highpassHz: number;
  maxAttenuationDb: number;
  thresholdMultiplier: number;
  dryBlend: number;
  tierLabel: 'Subtle' | 'Balanced' | 'Strong';
  description: string;
}

export function clampCleanupStrength(strengthDb: number): number {
  if (!Number.isFinite(strengthDb)) return 12;
  return Math.max(1, Math.min(30, Math.round(strengthDb)));
}

/**
 * Maps a 1..30 dB user setting onto a conservative, natural-sounding cleanup curve
 * so light room hum is removed without hollowing vocals or chopping consonant tails.
 */
export function computeCleanupParameters(strengthDb: number): CleanupCalibration {
  const s = clampCleanupStrength(strengthDb);
  const norm = (s - 1) / 29; // 0..1

  const highpassHz = Math.round(65 + norm * 30); // 65 Hz .. 95 Hz (~76 Hz at 12 dB)
  const maxAttenuationDb = Number((3.5 + norm * 14.5).toFixed(2)); // 3.5 dB .. 18.0 dB (~9.0 dB at 12 dB)
  const thresholdMultiplier = Number((1.35 + norm * 1.15).toFixed(3)); // 1.35x .. 2.5x noise floor
  const dryBlend = Number((0.26 - norm * 0.14).toFixed(3)); // 26% .. 12% dry ambience retention

  if (s <= 8) {
    return {
      highpassHz,
      maxAttenuationDb,
      thresholdMultiplier,
      dryBlend,
      tierLabel: 'Subtle',
      description: 'Subtle cleanup — trims low rumble and light hum while keeping full room ambience.',
    };
  }
  if (s <= 18) {
    return {
      highpassHz,
      maxAttenuationDb,
      thresholdMultiplier,
      dryBlend,
      tierLabel: 'Balanced',
      description: 'Balanced cleanup (recommended) — cleans background hiss without dulling speech.',
    };
  }
  return {
    highpassHz,
    maxAttenuationDb,
    thresholdMultiplier,
    dryBlend,
    tierLabel: 'Strong',
    description: 'Strong cleanup — stronger hiss reduction for noisy mics while guarding against gating.',
  };
}

/**
 * Pure PCM cleanup pipeline:
 * 1) 1st-order high-pass filter (65–95 Hz) for DC + low rumble removal
 * 2) Percentile noise-floor estimation over 20 ms windows
 * 3) Smooth downward expander with soft knee, fast attack (~8 ms), gentle release (~120 ms),
 *    and dry ambience retention so voices never sound robotic or choppy.
 */
export function applyBalancedAudioCleanup(
  input: Float32Array,
  sampleRate: number,
  strengthDb: number
): Float32Array {
  const len = input.length;
  const out = new Float32Array(len);
  if (len === 0 || sampleRate <= 0) return out;

  const cal = computeCleanupParameters(strengthDb);

  // 1. Gentle single-pole high-pass filter to remove DC & sub-vocal rumble
  const rc = 1 / (2 * Math.PI * cal.highpassHz);
  const dt = 1 / sampleRate;
  const alphaHp = rc / (rc + dt);

  const hp = new Float32Array(len);
  let prevX = input[0];
  let prevY = 0;
  hp[0] = 0;
  for (let i = 1; i < len; i++) {
    const x = input[i];
    const y = alphaHp * (prevY + x - prevX);
    hp[i] = y;
    prevX = x;
    prevY = y;
  }

  // 2. Estimate noise floor from 20 ms frame RMS percentiles
  const frameSize = Math.max(64, Math.round(sampleRate * 0.02));
  const frameRmsList: number[] = [];
  for (let start = 0; start < len; start += frameSize) {
    const end = Math.min(len, start + frameSize);
    let sumSq = 0;
    for (let i = start; i < end; i++) {
      const v = hp[i];
      sumSq += v * v;
    }
    const rms = Math.sqrt(sumSq / (end - start));
    if (rms > 1e-6) {
      frameRmsList.push(rms);
    }
  }

  if (frameRmsList.length === 0) {
    return hp;
  }

  frameRmsList.sort((a, b) => a - b);
  const p15Index = Math.min(
    frameRmsList.length - 1,
    Math.floor(frameRmsList.length * 0.15)
  );
  const p90Index = Math.min(
    frameRmsList.length - 1,
    Math.floor(frameRmsList.length * 0.9)
  );
  const noiseFloor = frameRmsList[p15Index];
  const speechPeak = frameRmsList[p90Index];

  // Guard: if clip has virtually no dynamic contrast (e.g. uniform tone or already dead quiet),
  // keep cleanup very gentle so we never pump or crush steady signal.
  const dynamicRatio = speechPeak / Math.max(noiseFloor, 1e-6);
  const effectiveFloor =
    dynamicRatio < 1.8 ? noiseFloor * 0.45 : noiseFloor;
  const threshold = effectiveFloor * cal.thresholdMultiplier;
  const minGain = Math.pow(10, -cal.maxAttenuationDb / 20);

  // 3. Envelope follower + soft-knee downward expander
  const envAttack = Math.exp(-1 / Math.max(1, sampleRate * 0.008));
  const envRelease = Math.exp(-1 / Math.max(1, sampleRate * 0.09));
  const gainAttack = Math.exp(-1 / Math.max(1, sampleRate * 0.008)); // fast open on speech
  const gainRelease = Math.exp(-1 / Math.max(1, sampleRate * 0.12)); // smooth tail fade

  let env = effectiveFloor;
  let smoothGain = 1;

  for (let i = 0; i < len; i++) {
    const sample = hp[i];
    const absVal = Math.abs(sample);
    env = absVal > env
      ? envAttack * env + (1 - envAttack) * absVal
      : envRelease * env + (1 - envRelease) * absVal;

    let targetGain = 1;
    if (threshold > 1e-7 && env < threshold) {
      const ratio = Math.max(0, env / threshold);
      // Soft knee curve (squared ratio) clamped to maxAttenuationDb
      const curve = ratio * ratio;
      targetGain = minGain + (1 - minGain) * curve;
    }

    // Fast upward opening when speech starts, smooth downward release during pauses
    smoothGain =
      targetGain > smoothGain
        ? gainAttack * smoothGain + (1 - gainAttack) * targetGain
        : gainRelease * smoothGain + (1 - gainRelease) * targetGain;

    const wetSample = sample * smoothGain;
    let mixed = wetSample * (1 - cal.dryBlend) + sample * cal.dryBlend;

    // Gentle safety soft-clip above 0.98
    if (mixed > 0.98 || mixed < -0.98) {
      mixed = Math.tanh(mixed);
    }
    out[i] = mixed;
  }

  return out;
}

export function encodePcm16Wav(channels: Float32Array[], sampleRate: number): Blob {
  const numChannels = Math.max(1, channels.length);
  const numFrames = channels[0]?.length ?? 0;
  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = numFrames * blockAlign;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeAscii = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeAscii(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeAscii(8, 'WAVE');
  writeAscii(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeAscii(36, 'data');
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < numFrames; i++) {
    for (let ch = 0; ch < numChannels; ch++) {
      const sample = Math.max(-1, Math.min(1, channels[ch][i] ?? 0));
      view.setInt16(
        offset,
        sample < 0 ? Math.round(sample * 0x8000) : Math.round(sample * 0x7fff),
        true
      );
      offset += 2;
    }
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

export async function processAudioUrlWithCleanup(
  sourceUrl: string,
  strengthDb: number
): Promise<Blob> {
  const response = await fetch(sourceUrl);
  if (!response.ok) {
    throw new Error(`Failed to read source audio (${response.status})`);
  }
  const arrayBuf = await response.arrayBuffer();
  const AudioCtx =
    window.AudioContext ||
    (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  if (!AudioCtx) {
    throw new Error('Web Audio API unavailable');
  }

  const ctx = new AudioCtx();
  try {
    const decoded = await ctx.decodeAudioData(arrayBuf.slice(0));
    const cleanedChannels: Float32Array[] = [];
    for (let ch = 0; ch < decoded.numberOfChannels; ch++) {
      const raw = decoded.getChannelData(ch);
      cleanedChannels.push(
        applyBalancedAudioCleanup(raw, decoded.sampleRate, strengthDb)
      );
    }
    return encodePcm16Wav(cleanedChannels, decoded.sampleRate);
  } finally {
    void ctx.close().catch(() => {});
  }
}

export function collectUnusedBlobUrls(
  candidateUrls: Iterable<string | undefined | null>,
  remainingQueue: ReadonlyArray<{ sourceUrl?: string; convertedUrl?: string }>,
  remainingSavedSamples: ReadonlyArray<{ url?: string }>
): string[] {
  const activeRefs = new Set<string>();
  for (const clip of remainingQueue) {
    if (clip.sourceUrl) activeRefs.add(clip.sourceUrl);
    if (clip.convertedUrl) activeRefs.add(clip.convertedUrl);
  }
  for (const sample of remainingSavedSamples) {
    if (sample.url) activeRefs.add(sample.url);
  }

  const unused: string[] = [];
  const seen = new Set<string>();
  for (const raw of candidateUrls) {
    if (!raw || !raw.startsWith('blob:') || seen.has(raw)) continue;
    seen.add(raw);
    if (!activeRefs.has(raw)) {
      unused.push(raw);
    }
  }
  return unused;
}

