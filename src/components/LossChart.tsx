import React, { useMemo } from 'react';
import { ACCENT_HEX } from '../data/defaults';
import { TrainingEpochPoint, TrainingZoneInsight } from '../types/studio';

interface LossChartProps {
  points: TrainingEpochPoint[];
  zoneInsight: TrainingZoneInsight;
}

const W = 1000;
const H = 164;
const PAD_L = 24;
const PAD_R = 24;
const PAD_T = 22;
const PAD_B = 24;
const PLOT_W = W - PAD_L - PAD_R;
const PLOT_H = H - PAD_T - PAD_B;

function pickSpacedLabels<T extends { x: number }>(coords: T[], minGap = 46): T[] {
  if (coords.length <= 2) return coords;
  const result: T[] = [coords[0]];
  const last = coords[coords.length - 1];
  for (let i = 1; i < coords.length - 1; i++) {
    const prev = result[result.length - 1];
    if (coords[i].x - prev.x >= minGap && last.x - coords[i].x >= minGap) {
      result.push(coords[i]);
    }
  }
  result.push(last);
  return result;
}

export const LossChart: React.FC<LossChartProps> = React.memo(
  ({ points, zoneInsight }) => {
    const model = useMemo(() => {
      const realPoints = points.filter(
        (p): p is TrainingEpochPoint & { lowestLoss: number } => p.lowestLoss !== null
      );
      if (realPoints.length < 2) {
        return null;
      }

      const step = Math.max(1, zoneInsight.saveEveryEpoch || 5);
      const bestEp = zoneInsight.sweetSpotEpoch ?? realPoints[realPoints.length - 1].epoch;
      const sweetMin = zoneInsight.sweetSpotMinEpoch ?? bestEp;
      const sweetMax = zoneInsight.sweetSpotMaxEpoch ?? bestEp;
      const underEnd = zoneInsight.undertrainingEndEpoch ?? realPoints[0].epoch;
      const overStart = zoneInsight.overtrainingStartEpoch;

      const minEpoch = realPoints[0].epoch;
      const maxEpoch = Math.max(minEpoch + step, realPoints[realPoints.length - 1].epoch);
      const losses = realPoints.map((p) => p.lowestLoss);
      const minLoss = Math.floor(Math.min(...losses) * 0.92);
      const maxLoss = Math.ceil(Math.max(...losses) * 1.05);
      const span = Math.max(0.5, maxLoss - minLoss);
      const toX = (ep: number) =>
        PAD_L +
        ((Math.min(maxEpoch, Math.max(minEpoch, ep)) - minEpoch) /
          Math.max(1, maxEpoch - minEpoch)) *
          PLOT_W;

      const coords = realPoints.map((p) => ({
        x: toX(p.epoch),
        y: PAD_T + (1 - (p.lowestLoss - minLoss) / span) * PLOT_H,
        epoch: p.epoch,
      }));

      return {
        coords,
        labelCoords: pickSpacedLabels(coords),
        polyline: coords.map((c) => `${c.x.toFixed(1)},${c.y.toFixed(1)}`).join(' '),
        sweetStartX: toX(sweetMin),
        sweetEndX: toX(sweetMax),
        sweetMidX: toX(bestEp),
        bestEpoch: bestEp,
        overStartX: overStart !== null ? toX(overStart) : null,
        underEndX: toX(underEnd),
      };
    }, [points, zoneInsight]);

    if (!model) {
      return null;
    }

    return (
      <div className="rounded-lg border border-[#47161D] bg-[#160709] p-4">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-2.5 px-1">
          <p className="text-[13px] font-semibold text-[#FDF7F7]">
            Observed Mel Loss &amp; Best Epoch Zone
          </p>
          <div className="flex flex-wrap items-center gap-3.5 text-xs font-mono text-[#C89EA4]">
            <span className="inline-flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full bg-[#E51923]" />
              <span>Best Epoch (e{model.bestEpoch})</span>
            </span>
            {model.overStartX !== null && (
              <span className="inline-flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-[#6D202B]" />
                <span>Overtrain Risk</span>
              </span>
            )}
          </div>
        </div>

        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="w-full h-auto overflow-visible"
          role="img"
          aria-label="Observed model mel loss curve and recommended best checkpoint"
        >
          {model.underEndX > PAD_L && (
            <rect
              x={PAD_L}
              y={PAD_T}
              width={Math.max(0, model.underEndX - PAD_L)}
              height={PLOT_H}
              fill="#47161D"
              fillOpacity="0.22"
            />
          )}

          <rect
            x={model.sweetStartX}
            y={PAD_T}
            width={Math.max(10, model.sweetEndX - model.sweetStartX)}
            height={PLOT_H}
            fill={ACCENT_HEX}
            fillOpacity="0.18"
          />

          {model.overStartX !== null && (
            <rect
              x={model.overStartX}
              y={PAD_T}
              width={Math.max(0, W - PAD_R - model.overStartX)}
              height={PLOT_H}
              fill="#991B1B"
              fillOpacity="0.18"
            />
          )}

          {[0.25, 0.5, 0.75].map((ratio) => {
            const y = PAD_T + PLOT_H * ratio;
            return (
              <line
                key={ratio}
                x1={PAD_L}
                y1={y}
                x2={W - PAD_R}
                y2={y}
                stroke="#47161D"
                strokeDasharray="3 3"
              />
            );
          })}

          <line
            x1={model.sweetMidX}
            y1={PAD_T}
            x2={model.sweetMidX}
            y2={PAD_T + PLOT_H}
            stroke="#FF2B35"
            strokeWidth="1.5"
            strokeDasharray="4 3"
          />
          <text
            x={model.sweetMidX}
            y={PAD_T - 6}
            textAnchor="middle"
            fill="#FDF7F7"
            fontSize="10.5"
            fontFamily="JetBrains Mono, monospace"
          >
            Best e{model.bestEpoch}
          </text>

          <polyline
            fill="none"
            stroke={ACCENT_HEX}
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            points={model.polyline}
          />

          {model.coords.map((c, idx) => (
            <circle
              key={`${c.epoch}-${idx}`}
              cx={c.x}
              cy={c.y}
              r="3"
              fill="#FDF7F7"
              stroke={ACCENT_HEX}
              strokeWidth="2"
            />
          ))}
          {model.labelCoords.map((c, idx) => (
            <text
              key={`lbl-${c.epoch}-${idx}`}
              x={c.x}
              y={H - 5}
              textAnchor="middle"
              fill="#C89EA4"
              fontSize="10.5"
              fontFamily="JetBrains Mono, monospace"
            >
              e{c.epoch}
            </text>
          ))}
        </svg>
      </div>
    );
  }
);
