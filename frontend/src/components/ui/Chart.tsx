"use client";

import dynamic from "next/dynamic";
import { TacticalSkeleton } from "./tactical-skeleton";

const Plot = dynamic(() => import("react-plotly.js"), {
  ssr: false,
  loading: () => <TacticalSkeleton variant="block" label="LOADING PLOTLY" />,
});

// Frontend-owned base layout — Python sends only structural data (titles, traces, axis labels).
// All aesthetic properties (colors, fonts, hover style, margins) live here.
const baseLayout: Partial<Plotly.Layout> = {
  autosize: true,
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: {
    family: "Fira Sans, ui-sans-serif, system-ui, sans-serif",
    size: 12,
    color: "rgba(237,237,237,0.6)",
  },
  title: {
    font: { size: 14, color: "#ededed" },
    x: 0,
    xanchor: "left",
    pad: { l: 0, t: 0 },
  },
  margin: { l: 48, r: 16, t: 36, b: 40 },
  xaxis: {
    gridcolor: "rgba(237,237,237,0.06)",
    zeroline: false,
    showline: false,
    tickfont: { size: 11, color: "rgba(237,237,237,0.4)" },
    title: { font: { size: 12, color: "rgba(237,237,237,0.4)" } },
  },
  yaxis: {
    gridcolor: "rgba(237,237,237,0.06)",
    zeroline: false,
    showline: false,
    automargin: true,
    tickfont: { size: 11, color: "rgba(237,237,237,0.4)" },
    title: { font: { size: 12, color: "rgba(237,237,237,0.4)" } },
  },
  legend: {
    orientation: "h",
    x: 0,
    y: 1.08,
    xanchor: "left",
    yanchor: "bottom",
    bgcolor: "transparent",
    borderwidth: 0,
    font: { size: 11 },
  },
  hoverlabel: {
    bgcolor: "#141414",
    bordercolor: "#262626",
    font: { size: 12, color: "#ededed", family: "Fira Code, monospace" },
  },
  hovermode: "closest",
  colorway: [
    "#FF3333",
    "#F4A261",
    "#457B9D",
    "#E9C46A",
    "#2A9D8F",
    "#E76F51",
    "#264653",
  ],
};

interface ChartProps {
  data: Plotly.Data[];
  layout?: Partial<Plotly.Layout>;
  config?: Partial<Plotly.Config>;
  className?: string;
  onInit?: (graphDiv: HTMLElement) => void;
}

export function Chart({ data, layout = {}, config = {}, className, onInit }: ChartProps) {
  // Deep-merge: baseLayout provides aesthetics; layout from Python provides structure.
  // Title is handled specially: if Python sends a string title, wrap it so it merges cleanly.
  const pythonTitle = layout?.title;
  const mergedTitle =
    typeof pythonTitle === "string"
      ? { ...baseLayout.title, text: pythonTitle }
      : { ...baseLayout.title, ...(pythonTitle as object | undefined) };

  const mergedLayout: Partial<Plotly.Layout> = {
    ...baseLayout,
    ...layout,
    title: mergedTitle,
    xaxis: { ...baseLayout.xaxis, ...(layout.xaxis ?? {}) },
    yaxis: { ...baseLayout.yaxis, ...(layout.yaxis ?? {}) },
    // Strip backend-injected color fields so they don't override our palette
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
  };

  return (
    <div className={`w-full h-full ${className ?? ""}`}>
      <Plot
        data={data}
        layout={mergedLayout}
        config={{
          responsive: true,
          displayModeBar: false,
          displaylogo: false,
          ...config,
        }}
        style={{ width: "100%", height: "100%" }}
        useResizeHandler
        onInitialized={onInit ? (_, graphDiv) => onInit(graphDiv) : undefined}
        onUpdate={onInit ? (_, graphDiv) => onInit(graphDiv) : undefined}
      />
    </div>
  );
}
