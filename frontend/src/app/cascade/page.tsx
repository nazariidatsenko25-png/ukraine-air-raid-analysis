"use client";

import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import { ChartCard } from "@/components/ChartCard";
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

  const regionShort = triggerRegion.replace(" oblast", "");

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

      <motion.div variants={cardVariants}>
        <ChartCard
          title="Secondary Strike Cascade Matrix"
          subtitle="Probability that an alert in region A is followed by an alert in region B within 1 hour"
          data={heatmapData?.data}
          layout={heatmapData?.layout}
          isLoading={isHeatmapLoading}
          isError={isHeatmapError}
          isEmpty={!isHeatmapLoading && !isHeatmapError && !heatmapData?.data?.length}
          variant="heatmap"
          loadingLabel="LOADING CASCADE MATRIX"
          emptyTitle="Cascade matrix unavailable"
          emptyMessage="The computation may have timed out — try refreshing."
          minHeight="500px"
        />
      </motion.div>

      {/* Trigger Region Analysis */}
      <motion.div variants={cardVariants} className="glass-panel rounded-xl shadow-lg flex flex-col">
        <div className="flex items-center justify-between px-4 pt-4 pb-2">
          <div>
            <h3 className="text-sm font-semibold uppercase tracking-wider text-glow-primary">
              Trigger Region Analysis
            </h3>
            <p className="text-xs text-foreground/40 mt-0.5">
              Expected secondary regions on alert after a strike in the selected oblast
            </p>
          </div>
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

        <div className="px-4 pb-4 flex-1 relative min-h-[400px]">
          <ChartCard
            title={`Secondary Cascade — ${regionShort}`}
            data={curveData?.data}
            layout={curveData?.layout}
            isLoading={isCurveLoading || !triggerRegion}
            isError={isCurveError}
            isEmpty={!isCurveLoading && !isCurveError && !curveData?.data?.length}
            variant="line"
            loadingLabel={regionShort ? `LOADING ${regionShort.toUpperCase()} DATA` : "LOADING"}
            emptyTitle="No cascade data"
            emptyMessage={`No cascade data available for ${regionShort}. Try another region.`}
            minHeight="400px"
            className="border-0 shadow-none bg-transparent backdrop-filter-none"
          />
        </div>
      </motion.div>
    </motion.div>
  );
}
