import React, { useMemo, useRef, useState } from 'react';
import { FolderArchive, Link2, Sparkles, Upload, X } from 'lucide-react';
import {
  extractPthFilenameFromUrl,
  inferMatchingDUrl,
  isHuggingFaceRepoUrl,
  isPretrainedZipUrl,
} from '../utils/rvcTuning';
import { useI18n } from '../utils/i18n';

interface SingleSlotProps {
  idPrefix: string;
  role: 'G' | 'D';
  label: string;
  value: string;
  placeholder: string;
  autoBadge?: string;
  acceptFiles: string;
  onChange: (nextValue: string) => void;
}

const SinglePretrainedSlot: React.FC<SingleSlotProps> = ({
  idPrefix,
  role,
  label,
  value,
  placeholder,
  autoBadge,
  acceptFiles,
  onChange,
}) => {
  const { t } = useI18n();
  const fileRef = useRef<HTMLInputElement | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const trimmed = value.trim();
  const isUrl = /^https?:\/\//i.test(trimmed);
  const isRepo = isUrl && isHuggingFaceRepoUrl(trimmed);
  const isZip = isPretrainedZipUrl(trimmed);

  const resolvedName = trimmed
    ? isRepo
      ? `HF Repo: ${trimmed.replace(/^https?:\/\/huggingface\.co\//i, '')}`
      : isZip
        ? trimmed.split('?')[0].split('/').pop() || 'custom_pretrains.zip'
        : isUrl
          ? extractPthFilenameFromUrl(trimmed, `custom_${role}.pth`)
          : trimmed.split('/').pop() || trimmed
    : '';

  const handlePickedFile = (file: File | undefined) => {
    if (!file) return;
    const lower = file.name.toLowerCase();
    if (!lower.endsWith('.pth') && !(role === 'G' && lower.endsWith('.zip'))) {
      return;
    }
    onChange(`rvc/models/pretraineds/custom/${file.name}`);
  };

  return (
    <div className="rounded-lg border border-[#47161D] bg-[#160709] p-3.5 space-y-2.5">
      <div className="flex items-center justify-between gap-2">
        <label
          htmlFor={`${idPrefix}-${role.toLowerCase()}`}
          className="text-xs font-semibold text-[#FDF7F7]"
        >
          {label}
        </label>
        {resolvedName ? (
          <span className="inline-flex items-center gap-1 rounded-md border border-[#E51923]/40 bg-[#200B0E] px-2 py-0.5 font-mono text-[10px] text-[#DFBFC3] max-w-[200px] truncate">
            {isZip ? (
              <FolderArchive className="h-2.5 w-2.5 text-[#E51923] shrink-0" />
            ) : isUrl ? (
              <Link2 className="h-2.5 w-2.5 text-[#E51923] shrink-0" />
            ) : (
              <Upload className="h-2.5 w-2.5 text-[#E51923] shrink-0" />
            )}
            <span className="truncate">{resolvedName}</span>
          </span>
        ) : autoBadge ? (
          <span className="inline-flex items-center gap-1 rounded-md border border-[#E51923]/35 bg-[#200B0E] px-2 py-0.5 font-mono text-[10px] text-[#DFBFC3] max-w-[210px] truncate">
            <Sparkles className="h-2.5 w-2.5 text-[#E51923] shrink-0" />
            <span className="truncate">{autoBadge}</span>
          </span>
        ) : null}
      </div>

      {/* URL link or filesystem path input */}
      <div className="relative">
        <input
          id={`${idPrefix}-${role.toLowerCase()}`}
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          className="w-full rounded-lg bg-[#100506] border border-[#47161D] pl-3 pr-8 py-2 text-xs font-mono text-[#FDF7F7] placeholder-[#AD8288]/45 focus:outline-none focus:border-[#E51923]"
        />
        {trimmed && (
          <button
            type="button"
            onClick={() => onChange('')}
            aria-label={`Clear ${label}`}
            className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-0.5 text-[#AD8288] hover:text-white"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        )}
      </div>

      {/* Upload or Drop .pth File */}
      <input
        ref={fileRef}
        type="file"
        accept={acceptFiles}
        className="hidden"
        onChange={(e) => {
          handlePickedFile(e.target.files?.[0]);
          e.currentTarget.value = '';
        }}
      />
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          handlePickedFile(e.dataTransfer.files?.[0]);
        }}
        className={`flex flex-wrap items-center justify-between gap-2 rounded-lg border border-dashed px-3 py-2 transition-colors ${
          dragOver
            ? 'border-[#E51923] bg-[#200B0E]'
            : 'border-[#47161D] bg-[#100506]'
        }`}
      >
        <span className="text-[11px] text-[#AD8288]">
          {t('customPretrain.dropHint')}
        </span>
        <button
          type="button"
          onClick={() => fileRef.current?.click()}
          className="inline-flex items-center gap-1.5 rounded-md border border-[#47161D] bg-[#200B0E] px-2.5 py-1 text-[11px] font-semibold text-[#FDF7F7] hover:border-[#E51923] whitespace-nowrap shrink-0"
        >
          <Upload className="h-3 w-3 text-[#E51923]" />
          <span>{t('customPretrain.uploadBtn')}</span>
        </button>
      </div>
    </div>
  );
};

interface CustomPretrainedPairProps {
  idPrefix: string;
  gValue: string;
  dValue: string;
  onChangeG: (nextG: string) => void;
  onChangeD: (nextD: string) => void;
  showGuide?: boolean;
}

export const CustomPretrainedPair: React.FC<CustomPretrainedPairProps> = React.memo(
  ({ idPrefix, gValue, dValue, onChangeG, onChangeD, showGuide = true }) => {
    const { t } = useI18n();

    const autoDiscoveryBadge = useMemo(() => {
      if (dValue.trim()) return '';
      const g = gValue.trim();
      if (!g) return '';
      if (isHuggingFaceRepoUrl(g) || isPretrainedZipUrl(g)) {
        return 'Auto-discover from source';
      }
      const inferredD = inferMatchingDUrl(g);
      if (inferredD) {
        return `Auto: ${extractPthFilenameFromUrl(inferredD, 'D.pth')}`;
      }
      return '';
    }, [gValue, dValue]);

    return (
      <div className="space-y-3 pt-1">
        {showGuide && (
          <div className="rounded-lg border border-[#47161D] bg-[#160709] px-3.5 py-2.5 text-xs text-[#DFBFC3] leading-relaxed">
            {t('customPretrain.pairGuide')}
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <SinglePretrainedSlot
            idPrefix={idPrefix}
            role="G"
            label={t('customPretrain.gLabel')}
            value={gValue}
            placeholder={t('customPretrain.gPlaceholder')}
            acceptFiles=".pth,.zip"
            onChange={onChangeG}
          />
          <SinglePretrainedSlot
            idPrefix={idPrefix}
            role="D"
            label={t('customPretrain.dLabel')}
            value={dValue}
            placeholder={t('customPretrain.dPlaceholder')}
            autoBadge={autoDiscoveryBadge}
            acceptFiles=".pth"
            onChange={onChangeD}
          />
        </div>
      </div>
    );
  }
);
