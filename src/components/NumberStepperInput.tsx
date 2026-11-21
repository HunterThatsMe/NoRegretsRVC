import React, { useEffect, useRef, useState } from 'react';
import { Minus, Plus } from 'lucide-react';

interface NumberStepperInputProps {
  id?: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  ariaLabel: string;
  unitName: string;
  disabled?: boolean;
  inputWidthClass?: string;
  onChange: (next: number) => void;
}

export const NumberStepperInput: React.FC<NumberStepperInputProps> = React.memo(({
  id,
  value,
  min,
  max,
  step = 1,
  ariaLabel,
  unitName,
  disabled = false,
  inputWidthClass = 'w-14',
  onChange,
}) => {
  const [draft, setDraft] = useState<string>(() => String(value));
  const isFocusedRef = useRef(false);

  useEffect(() => {
    if (!isFocusedRef.current) {
      setDraft(String(value));
    }
  }, [value]);

  const clamp = (raw: number) => Math.max(min, Math.min(max, raw));

  const commitDraft = (rawText: string) => {
    const trimmed = rawText.trim();
    if (trimmed === '' || trimmed === '-' || trimmed === '.' || trimmed === '-.') {
      setDraft(String(value));
      return;
    }
    const parsed = Number(trimmed);
    if (!Number.isFinite(parsed)) {
      setDraft(String(value));
      return;
    }
    const clamped = clamp(parsed);
    setDraft(String(clamped));
    if (clamped !== value) {
      onChange(clamped);
    }
  };

  const handleStep = (delta: number) => {
    const decimals = String(step).includes('.') ? String(step).split('.')[1].length : 0;
    const stepped = Number((value + delta).toFixed(decimals));
    const next = clamp(stepped);
    setDraft(String(next));
    onChange(next);
  };

  return (
    <div
      className={`inline-flex items-center rounded-lg border border-[#47161D] bg-[#160709] p-0.5 transition-colors duration-150 focus-within:border-[#E51923] ${
        disabled ? 'opacity-45 pointer-events-none' : ''
      }`}
    >
      <button
        type="button"
        disabled={disabled || value <= min}
        onClick={() => handleStep(-step)}
        aria-label={`Decrease ${unitName}`}
        className="flex h-6 w-6 items-center justify-center rounded-md text-[#DFBFC3] hover:bg-[#E51923] hover:text-white active:bg-[#B9121B] disabled:opacity-35 disabled:hover:bg-transparent disabled:hover:text-[#DFBFC3] transition-colors duration-150"
      >
        <Minus className="h-3 w-3" />
      </button>

      <input
        id={id}
        type="text"
        inputMode={step < 1 || min < 0 ? 'decimal' : 'numeric'}
        disabled={disabled}
        aria-label={ariaLabel}
        value={draft}
        onFocus={(e) => {
          isFocusedRef.current = true;
          setDraft(String(value));
          e.currentTarget.select();
        }}
        onChange={(e) => {
          const raw = e.target.value;
          if (!/^-?\d*\.?\d*$/.test(raw)) return;
          setDraft(raw);
          const trimmed = raw.trim();
          if (trimmed === '' || trimmed === '-' || trimmed.endsWith('.')) {
            return;
          }
          const parsed = Number(trimmed);
          if (Number.isFinite(parsed) && parsed >= min && parsed <= max) {
            onChange(parsed);
          }
        }}
        onBlur={() => {
          isFocusedRef.current = false;
          commitDraft(draft);
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            commitDraft(draft);
            e.currentTarget.blur();
          } else if (e.key === 'Escape') {
            setDraft(String(value));
            isFocusedRef.current = false;
            e.currentTarget.blur();
          }
        }}
        className={`${inputWidthClass} no-native-spin bg-transparent px-1 py-0.5 text-center font-mono text-xs font-medium text-[#FDF7F7] tabular-nums focus:outline-none`}
      />

      <button
        type="button"
        disabled={disabled || value >= max}
        onClick={() => handleStep(step)}
        aria-label={`Increase ${unitName}`}
        className="flex h-6 w-6 items-center justify-center rounded-md text-[#DFBFC3] hover:bg-[#E51923] hover:text-white active:bg-[#B9121B] disabled:opacity-35 disabled:hover:bg-transparent disabled:hover:text-[#DFBFC3] transition-colors duration-150"
      >
        <Plus className="h-3 w-3" />
      </button>
    </div>
  );
});
