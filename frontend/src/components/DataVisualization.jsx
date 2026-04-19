import { useMemo } from "react";
import { BarChart3, LineChart, PieChart, Hash } from "lucide-react";

const PALETTE = ["#66d9ef", "#80ffd3", "#7bb8ff", "#f5be6b", "#ff8e7a", "#9ca8ff", "#5fe0a9", "#b3f1ff"];

export default function DataVisualization({ visualization }) {
  const type = visualization?.type;

  if (!visualization || type === "table") {
    return null;
  }

  const icon = {
    number: <Hash size={13} />,
    bar: <BarChart3 size={13} />,
    line: <LineChart size={13} />,
    pie: <PieChart size={13} />,
  }[type] || <BarChart3 size={13} />;

  return (
    <div className="mt-4 glass-panel rounded-[1.6rem] overflow-hidden">
      <div className="px-5 py-4 border-b border-[rgba(125,168,214,0.12)] bg-[rgba(255,255,255,0.015)]">
        <div className="flex items-center gap-2 text-xs text-[var(--accent)]">
          {icon}
          <span className="uppercase tracking-[0.18em]">Visualization</span>
        </div>
        {visualization.title && (
          <div className="mt-1 text-base font-semibold text-[var(--text-1)] tracking-tight">{visualization.title}</div>
        )}
        {visualization.description && (
          <div className="mt-1 text-xs text-[var(--text-3)]">{visualization.description}</div>
        )}
      </div>

      <div className="p-5">
        {type === "number" && <NumberCard visualization={visualization} />}
        {type === "bar" && <BarChartCard visualization={visualization} />}
        {type === "line" && <LineChartCard visualization={visualization} />}
        {type === "pie" && <PieChartCard visualization={visualization} />}
      </div>
    </div>
  );
}

function NumberCard({ visualization }) {
  return (
    <div className="rounded-[1.7rem] border border-[rgba(102,217,239,0.16)] bg-[linear-gradient(135deg,rgba(102,217,239,0.14)_0%,rgba(128,255,211,0.08)_100%)] px-6 py-7">
      <div className="text-[11px] uppercase tracking-[0.2em] text-[var(--accent)]">
        {visualization.number_label || "Value"}
      </div>
      <div className="mt-3 text-5xl font-semibold tracking-tight text-[var(--text-1)]">
        {formatValue(visualization.number_value)}
      </div>
      {visualization.number_context && (
        <div className="mt-3 text-xs text-[var(--text-2)]">{visualization.number_context}</div>
      )}
    </div>
  );
}

function BarChartCard({ visualization }) {
  const values = visualization.values || [];
  const labels = visualization.labels || [];
  const bars = useMemo(() => buildBarSeries(labels, values), [labels, values]);

  return (
    <div className="space-y-3">
      <svg viewBox="0 0 640 320" className="w-full h-auto">
        <rect x="0" y="0" width="640" height="320" rx="16" fill="#07111f" />
        {[0, 1, 2, 3].map((i) => (
          <line
            key={i}
            x1="48"
            x2="610"
            y1={40 + i * 56}
            y2={40 + i * 56}
            stroke="rgba(125,168,214,0.16)"
            strokeWidth="1"
          />
        ))}
        {bars.map((bar, index) => (
          <g key={`${bar.label}-${index}`}>
            <text x={bar.centerX} y={bar.y - 10} textAnchor="middle" fontSize="11" fill="#9bb0c9">
              {formatCompact(bar.value)}
            </text>
            <rect
              x={bar.x}
              y={bar.y}
              width={bar.width}
              height={bar.height}
              rx="10"
              fill={PALETTE[index % PALETTE.length]}
              opacity="0.96"
            />
            <text x={bar.centerX} y="300" textAnchor="middle" fontSize="11" fill="#6f849c">
              {truncateLabel(bar.label, 12)}
            </text>
          </g>
        ))}
      </svg>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
        {bars.map((bar, index) => (
          <div key={`${bar.label}-${index}`} className="rounded-xl border border-[rgba(125,168,214,0.12)] bg-[rgba(255,255,255,0.03)] px-3 py-2">
            <div className="text-[11px] text-[var(--text-3)] truncate" title={bar.label}>{bar.label}</div>
            <div className="mt-1 text-sm text-[var(--text-2)]">{formatCompact(bar.value)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function LineChartCard({ visualization }) {
  const labels = visualization.labels || [];
  const values = visualization.values || [];
  const points = useMemo(() => buildLinePoints(values), [values]);

  return (
    <div>
      <svg viewBox="0 0 640 280" className="w-full h-auto">
        <defs>
          <linearGradient id="line-fill" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#66d9ef" stopOpacity="0.32" />
            <stop offset="100%" stopColor="#66d9ef" stopOpacity="0.02" />
          </linearGradient>
        </defs>
        <rect x="0" y="0" width="640" height="280" rx="16" fill="#07111f" />
        {[0, 1, 2, 3].map((i) => (
          <line
            key={i}
            x1="48"
            x2="610"
            y1={40 + i * 56}
            y2={40 + i * 56}
            stroke="rgba(125,168,214,0.16)"
            strokeWidth="1"
          />
        ))}
        <path d={points.areaPath} fill="url(#line-fill)" />
        <path d={points.linePath} fill="none" stroke="#66d9ef" strokeWidth="4" strokeLinejoin="round" strokeLinecap="round" />
        {points.nodes.map((point, index) => (
          <g key={index}>
            <circle cx={point.x} cy={point.y} r="5" fill="#07111f" stroke="#b3f1ff" strokeWidth="3" />
          </g>
        ))}
      </svg>
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
        {labels.map((label, index) => (
          <div key={`${label}-${index}`} className="rounded-xl bg-[rgba(255,255,255,0.03)] border border-[rgba(125,168,214,0.12)] px-3 py-2">
            <div className="text-[11px] text-[var(--text-3)] truncate" title={label}>{label}</div>
            <div className="mt-1 text-sm text-[var(--accent)]">{formatCompact(values[index])}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function PieChartCard({ visualization }) {
  const labels = visualization.labels || [];
  const values = visualization.values || [];
  const arcs = useMemo(() => buildPieArcs(values), [values]);

  return (
    <div className="grid gap-4 md:grid-cols-[260px_1fr] md:items-center">
      <div className="flex justify-center">
        <svg viewBox="0 0 240 240" className="w-60 h-60">
          {arcs.map((arc, index) => (
            <path key={index} d={arc.path} fill={PALETTE[index % PALETTE.length]} stroke="#020617" strokeWidth="2" />
          ))}
          <circle cx="120" cy="120" r="44" fill="#020617" />
        </svg>
      </div>
      <div className="space-y-2">
        {labels.map((label, index) => (
          <div key={`${label}-${index}`} className="flex items-center gap-3 rounded-xl border border-[rgba(125,168,214,0.12)] bg-[rgba(255,255,255,0.03)] px-3 py-2">
            <span
              className="w-3 h-3 rounded-full flex-shrink-0"
              style={{ backgroundColor: PALETTE[index % PALETTE.length] }}
            />
            <div className="min-w-0 flex-1 text-sm text-[var(--text-2)] truncate" title={label}>
              {label}
            </div>
            <div className="text-xs text-[var(--text-3)]">
              {formatCompact(values[index])}
            </div>
            <div className="text-xs text-[var(--accent)]">
              {arcs[index]?.percent ?? 0}%
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function buildLinePoints(values) {
  const width = 640;
  const height = 280;
  const paddingX = 48;
  const paddingY = 28;
  const innerWidth = width - paddingX * 2;
  const innerHeight = height - paddingY * 2;
  const max = Math.max(...values, 1);
  const min = Math.min(...values, 0);
  const range = max - min || 1;

  const nodes = values.map((value, index) => ({
    x: paddingX + (index * innerWidth) / Math.max(values.length - 1, 1),
    y: height - paddingY - ((value - min) / range) * innerHeight,
  }));

  const linePath = nodes.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`).join(" ");
  const areaPath = `${linePath} L ${nodes[nodes.length - 1]?.x ?? paddingX} ${height - paddingY} L ${nodes[0]?.x ?? paddingX} ${height - paddingY} Z`;

  return { nodes, linePath, areaPath };
}

function buildBarSeries(labels, values) {
  const chartLeft = 56;
  const chartBottom = 268;
  const chartHeight = 200;
  const chartWidth = 540;
  const max = Math.max(...values, 1);
  const count = Math.max(values.length, 1);
  const slotWidth = chartWidth / count;
  const barWidth = Math.min(48, Math.max(slotWidth * 0.58, 18));

  return values.map((value, index) => {
    const height = Math.max((value / max) * chartHeight, 10);
    const centerX = chartLeft + slotWidth * index + slotWidth / 2;
    return {
      label: labels[index],
      value,
      width: barWidth,
      height,
      x: centerX - barWidth / 2,
      y: chartBottom - height,
      centerX,
    };
  });
}

function buildPieArcs(values) {
  const total = values.reduce((sum, value) => sum + Math.max(value, 0), 0) || 1;
  let angle = -Math.PI / 2;
  const center = 120;
  const radius = 92;

  return values.map((value) => {
    const slice = (Math.max(value, 0) / total) * Math.PI * 2;
    const startX = center + radius * Math.cos(angle);
    const startY = center + radius * Math.sin(angle);
    angle += slice;
    const endX = center + radius * Math.cos(angle);
    const endY = center + radius * Math.sin(angle);
    const largeArcFlag = slice > Math.PI ? 1 : 0;

    return {
      path: [
        `M ${center} ${center}`,
        `L ${startX} ${startY}`,
        `A ${radius} ${radius} 0 ${largeArcFlag} 1 ${endX} ${endY}`,
        "Z",
      ].join(" "),
      percent: Math.round((Math.max(value, 0) / total) * 100),
    };
  });
}

function formatCompact(value) {
  return new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

function formatValue(value) {
  if (typeof value === "number") {
    return new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(value);
  }
  return String(value ?? "N/A");
}

function truncateLabel(label, maxLength) {
  if (!label || label.length <= maxLength) {
    return label;
  }
  return `${label.slice(0, maxLength - 1)}…`;
}
