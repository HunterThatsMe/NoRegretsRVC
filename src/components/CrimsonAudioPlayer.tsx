import React, { useEffect, useRef, useState } from 'react';
import { Pause, Play, Volume2, VolumeX, X } from 'lucide-react';
import { ACCENT_HEX } from '../data/defaults';

interface CrimsonAudioPlayerProps {
  src: string;
  label?: string;
  compact?: boolean;
  onDiscard?: () => void;
  discardLabel?: string;
}

function formatAudioTime(sec: number): string {
  if (!Number.isFinite(sec) || sec < 0) return '0:00';
  const whole = Math.floor(sec);
  const mins = Math.floor(whole / 60);
  const secs = whole % 60;
  return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
}

export const CrimsonAudioPlayer: React.FC<CrimsonAudioPlayerProps> = React.memo(
  ({ src, label, compact = false, onDiscard, discardLabel = 'Discard audio' }) => {
    const audioRef = useRef<HTMLAudioElement | null>(null);
    const [playing, setPlaying] = useState(false);
    const [muted, setMuted] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);

    useEffect(() => {
      setPlaying(false);
      setCurrentTime(0);
      setDuration(0);
    }, [src]);

    const togglePlay = () => {
      const el = audioRef.current;
      if (!el) return;
      if (el.paused) {
        void el.play().catch(() => {});
      } else {
        el.pause();
      }
    };

    const toggleMute = () => {
      const el = audioRef.current;
      if (!el) return;
      el.muted = !el.muted;
      setMuted(el.muted);
    };

    const handleSeek = (next: number) => {
      const el = audioRef.current;
      if (!el || !Number.isFinite(next)) return;
      el.currentTime = next;
      setCurrentTime(next);
    };

    const pct = duration > 0 ? Math.min(100, (currentTime / duration) * 100) : 0;

    return (
      <div
        className={`flex items-center gap-2.5 rounded-lg border border-[#6D202B] bg-[#160709] ${
          compact ? 'px-2.5 py-1.5' : 'px-3 py-2'
        }`}
      >
        <audio
          ref={audioRef}
          src={src}
          preload="metadata"
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={() => {
            setPlaying(false);
            setCurrentTime(0);
          }}
          onLoadedMetadata={(e) => setDuration(e.currentTarget.duration || 0)}
          onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
        />

        <button
          type="button"
          onClick={togglePlay}
          aria-label={playing ? `Pause ${label || 'audio'}` : `Play ${label || 'audio'}`}
          className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-white transition-colors hover:bg-[#FF2B35]"
          style={{ backgroundColor: ACCENT_HEX }}
        >
          {playing ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
        </button>

        <div className="flex-1 min-w-0">
          {label && (
            <p className="text-[11px] font-medium text-[#FDF7F7] truncate mb-1">
              {label}
            </p>
          )}
          <div className="flex items-center gap-2">
            <input
              type="range"
              min={0}
              max={duration > 0 ? duration : 1}
              step={0.01}
              value={duration > 0 ? currentTime : 0}
              onChange={(e) => handleSeek(Number(e.target.value))}
              aria-label={`Seek ${label || 'audio'}`}
              style={{
                background: `linear-gradient(to right, ${ACCENT_HEX} ${pct}%, #351015 ${pct}%)`,
              }}
              className="w-full scarlet-slider"
            />
            <span className="shrink-0 font-mono text-[10px] text-[#DFBFC3] tabular-nums">
              {formatAudioTime(currentTime)} / {formatAudioTime(duration)}
            </span>
          </div>
        </div>

        <button
          type="button"
          onClick={toggleMute}
          aria-label={muted ? `Unmute ${label || 'audio'}` : `Mute ${label || 'audio'}`}
          className="rounded-md p-1.5 text-[#DFBFC3] hover:text-white hover:bg-[#200B0E] shrink-0"
        >
          {muted ? <VolumeX className="h-3.5 w-3.5 text-[#FF2B35]" /> : <Volume2 className="h-3.5 w-3.5" />}
        </button>

        {onDiscard && (
          <button
            type="button"
            onClick={onDiscard}
            aria-label={discardLabel}
            className="rounded-md border border-[#47161D] bg-[#200B0E] p-1.5 text-[#DFBFC3] hover:text-white hover:border-[#E51923] shrink-0"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        )}
      </div>
    );
  }
);
