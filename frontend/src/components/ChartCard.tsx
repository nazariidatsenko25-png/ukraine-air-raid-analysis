"use client";

import { useRef, useState, useCallback } from "react";
import { Download, Copy, Maximize2, X } from "lucide-react";
import { Chart } from "@/components/ui/Chart";
import { ChartState } from "@/components/ui/tactical-skeleton";
import type { Variant } from "@/components/ui/tactical-skeleton";

// ─── CSV utility ─────────────────────────────────────────────────────────────

type PlotlyTrace = {
  type?: string;
  name?: string;
  x?: (string | number)[];
  y?: (string | number)[];
  z?: (string | number)[][];
};

function tracesToCsv(traces: PlotlyTrace[]): string {
  const rows: string[][] = [];

  traces.forEach((trace) => {
    const label = trace.name ?? trace.type ?? "series";

    if (trace.type === "heatmap" && Array.isArray(trace.z)) {
      // Flatten 2D z matrix row by row
      rows.push([`# ${label} (heatmap z values, row per y-tick)`]);
      (trace.z as (string | number)[][]).forEach((row, i) => {
        rows.push([`row_${i}`, ...row.map(String)]);
      });
      rows.push([]);
      return;
    }

    const xs = trace.x ?? [];
    const ys = trace.y ?? [];
    rows.push([`x_${label}`, `y_${label}`]);
    const len = Math.max(xs.length, ys.length);
    for (let i = 0; i < len; i++) {
      rows.push([String(xs[i] ?? ""), String(ys[i] ?? "")]);
    }
    rows.push([]);
  });

  return rows.map((r) => r.join(",")).join("\n");
}

// ─── Types ───────────────────────────────────────────────────────────────────

interface ChartCardProps {
  title: string;
  subtitle?: string;
  data?: Plotly.Data[];
  layout?: Partial<Plotly.Layout>;
  isLoading: boolean;
  isError: boolean;
  isEmpty?: boolean;
  variant?: Variant;
  loadingLabel?: string;
  emptyTitle?: string;
  emptyMessage?: string;
  className?: string;
  minHeight?: string;
}

// ─── Toolbar Button ───────────────────────────────────────────────────────────

function ToolbarButton({
  onClick,
  label,
  children,
}: {
  onClick: () => void;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      aria-label={label}
      className="p-1.5 rounded text-foreground/40 hover:text-foreground hover:bg-white/5 transition-colors focus:outline-none focus:ring-1 focus:ring-primary"
    >
      {children}
    </button>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

export function ChartCard({
  title,
  subtitle,
  data,
  layout,
  isLoading,
  isError,
  isEmpty = false,
  variant = "block",
  loadingLabel = "LOADING",
  emptyTitle,
  emptyMessage,
  className = "",
  minHeight = "400px",
}: ChartCardProps) {
  const graphDivRef = useRef<HTMLElement | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [copyFeedback, setCopyFeedback] = useState(false);

  const handleInit = useCallback((graphDiv: HTMLElement) => {
    graphDivRef.current = graphDiv;
  }, []);

  const handleDownload = useCallback(() => {
    const el = graphDivRef.current as HTMLElement & { layout?: Plotly.Layout } | null;
    if (!el) return;
    // @ts-expect-error No type definitions for plotly.js-dist-min
    import("plotly.js-dist-min").then((PlotlyModule) => {
      const Plotly = PlotlyModule.default || PlotlyModule;
      const slug = title.toLowerCase().replace(/\s+/g, "-").replace(/[^a-z0-9-]/g, "");
      Plotly.downloadImage(el, {
        format: "png",
        scale: 2,
        filename: slug,
        width: 1400,
        height: 700,
      });
    });
  }, [title]);

  const handleCopy = useCallback(async () => {
    const el = graphDivRef.current as (HTMLElement & { data?: Plotly.Data[] }) | null;
    if (!el?.data) return;
    const csv = tracesToCsv(el.data as PlotlyTrace[]);
    try {
      await navigator.clipboard.writeText(csv);
      setCopyFeedback(true);
      setTimeout(() => setCopyFeedback(false), 1800);
    } catch {
      // Clipboard unavailable — silently ignore
    }
  }, []);

  const chartContent = (
    <Chart
      data={data ?? []}
      layout={layout}
      onInit={handleInit}
    />
  );

  return (
    <>
      {/* Card */}
      <div
        className={`group glass-panel rounded-xl shadow-lg flex flex-col ${className}`}
        style={{ minHeight }}
      >
        {/* Card header */}
        <div className="flex items-start justify-between px-4 pt-4 pb-0">
          <div>
            <h3 className="text-sm font-semibold uppercase tracking-wider text-glow-primary">
              {title}
            </h3>
            {subtitle && (
              <p className="text-xs text-foreground/40 mt-0.5">{subtitle}</p>
            )}
          </div>

          {/* Toolbar — visible on card hover */}
          <div className="chart-toolbar flex items-center gap-0.5 -mt-0.5">
            <ToolbarButton onClick={handleDownload} label="Download PNG">
              <Download size={14} />
            </ToolbarButton>
            <ToolbarButton onClick={handleCopy} label="Copy CSV">
              {copyFeedback ? (
                <span className="text-[10px] font-mono text-green-400 px-1">✓</span>
              ) : (
                <Copy size={14} />
              )}
            </ToolbarButton>
            <ToolbarButton onClick={() => setIsFullscreen(true)} label="Fullscreen">
              <Maximize2 size={14} />
            </ToolbarButton>
          </div>
        </div>

        {/* Chart area */}
        <div className="flex-1 relative p-4">
          <ChartState
            isLoading={isLoading}
            isError={isError}
            isEmpty={isEmpty}
            variant={variant}
            loadingLabel={loadingLabel}
            emptyTitle={emptyTitle}
            emptyMessage={emptyMessage}
          >
            {chartContent}
          </ChartState>
        </div>
      </div>

      {/* Fullscreen modal (CSS-only Dialog — no shadcn dependency needed) */}
      {isFullscreen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 md:p-8"
          onClick={() => setIsFullscreen(false)}
        >
          {/* Backdrop */}
          <div className="absolute inset-0 bg-black/80 backdrop-blur-sm" />

          {/* Modal */}
          <div
            className="relative z-10 glass-panel rounded-2xl shadow-2xl flex flex-col w-full max-w-7xl"
            style={{ height: "85vh" }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-border/50 shrink-0">
              <div>
                <h2 className="text-base font-semibold text-white">{title}</h2>
                {subtitle && (
                  <p className="text-xs text-foreground/40 mt-0.5">{subtitle}</p>
                )}
              </div>
              <button
                onClick={() => setIsFullscreen(false)}
                aria-label="Close fullscreen"
                className="p-2 rounded-lg text-foreground/40 hover:text-foreground hover:bg-white/5 transition-colors"
              >
                <X size={16} />
              </button>
            </div>

            {/* Chart fills the modal — stretch to 80vh */}
            <div className="flex-1 p-4 min-h-0">
              <Chart
                data={data ?? []}
                layout={{
                  ...layout,
                  height: undefined, // let autosize take over
                }}
              />
            </div>
          </div>
        </div>
      )}
    </>
  );
}
