"use client";

import dynamic from "next/dynamic";
import { Skeleton } from "./Skeleton";

// Dynamically import Plotly with no SSR because it relies on window
const Plot = dynamic(() => import("react-plotly.js"), {
  ssr: false,
  loading: () => <Skeleton className="w-full h-full min-h-[400px]" />,
});

interface ChartProps {
  data: any;
  layout?: any;
  config?: any;
  className?: string;
}

export function Chart({ data, layout = {}, config = {}, className }: ChartProps) {
  return (
    <div className={`w-full h-full ${className || ""}`}>
      <Plot
        data={data}
        layout={{
          ...layout,
          autosize: true,
          paper_bgcolor: "transparent",
          plot_bgcolor: "transparent",
          font: { family: "Fira Sans, sans-serif", color: "#ededed" },
          margin: { t: 40, r: 20, b: 40, l: 40 },
        }}
        config={{ responsive: true, displayModeBar: false, ...config }}
        style={{ width: "100%", height: "100%" }}
        useResizeHandler={true}
      />
    </div>
  );
}
