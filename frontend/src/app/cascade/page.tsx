"use client";

import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import { Chart } from "@/components/ui/Chart";
import { Skeleton } from "@/components/ui/Skeleton";
import { fetchChartData, fetchJsonData } from "@/lib/api";

const pageVariants = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0, transition: { staggerChildren: 0.1 } },
};

const cardVariants = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
};

export default function CascadeAnalysis() {
  const [triggerRegion, setTriggerRegion] = useState("");

  const { data: regionsData } = useQuery({
    queryKey: ["/cascade/regions"],
    queryFn: () => fetchJsonData("/cascade/regions"),
  });

  // Auto-select first region when data arrives
  useEffect(() => {
    if (regionsData?.regions?.length && !regionsData.regions.includes(triggerRegion)) {
      setTriggerRegion(regionsData.regions[0]);
    }
  }, [regionsData, triggerRegion]);

  const { data: heatmapData, isLoading: isHeatmapLoading, isError: isHeatmapError } = useQuery({
    queryKey: ["/cascade/heatmap"],
    queryFn: () => fetchChartData("/cascade/heatmap"),
    retry: 1,
  });

  const { data: curveData, isLoading: isCurveLoading, isError: isCurveError } = useQuery({
    queryKey: ["/cascade/secondary-curve", triggerRegion],
    queryFn: () => fetchChartData(`/cascade/secondary-curve?trigger_region=${encodeURIComponent(triggerRegion)}`),
    enabled: !!triggerRegion,
    retry: 1,
  });

  return (
    <motion.div
      className="max-w-7xl mx-auto flex flex-col gap-6"
      variants={pageVariants}
      initial="initial"
      animate="animate"
    >
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-white mb-2">Cascade Analysis</h1>
        <p className="text-foreground/60 text-sm">Analyze sequential strikes and alert propagation.</p>
      </div>

      <motion.div variants={cardVariants} className="glass-panel p-4 rounded-xl shadow-lg min-h-[500px] flex flex-col">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-glow-primary mb-4">Secondary Strike Cascade Matrix</h3>
        <div className="flex-1 relative">
          {isHeatmapLoading && <Skeleton className="absolute inset-0" />}
          {isHeatmapError && (
            <div className="absolute inset-0 flex items-center justify-center text-foreground/40 text-sm">
              Failed to load cascade heatmap. The computation may have timed out — try refreshing.
            </div>
          )}
          {heatmapData && <Chart data={heatmapData.data} layout={heatmapData.layout} />}
        </div>
      </motion.div>

      <motion.div variants={cardVariants} className="glass-panel p-6 rounded-xl shadow-lg flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-glow-primary">Trigger Region Analysis</h3>
          <select
            value={triggerRegion}
            onChange={(e) => setTriggerRegion(e.target.value)}
            className="bg-black/50 border border-border text-foreground text-sm rounded-md px-3 py-1.5 outline-none focus:border-primary transition-colors"
          >
            {regionsData?.regions?.map((region: string) => (
              <option key={region} value={region}>{region}</option>
            ))}
          </select>
        </div>
        <div className="relative min-h-[400px]">
          {isCurveLoading && <Skeleton className="absolute inset-0" />}
          {isCurveError && (
            <div className="absolute inset-0 flex items-center justify-center text-foreground/40 text-sm">
              Failed to load secondary strike curve for {triggerRegion}. Try another region.
            </div>
          )}
          {curveData && <Chart data={curveData.data} layout={curveData.layout} />}
        </div>
      </motion.div>
    </motion.div>
  );
}
