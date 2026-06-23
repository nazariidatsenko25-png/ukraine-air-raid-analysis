"use client";

import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { useState } from "react";
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

export default function ModelingPage() {
  const [region, setRegion] = useState("Kyiv");

  const { data: regionsData } = useQuery({
    queryKey: ["/models/regions"],
    queryFn: () => fetchJsonData("/models/regions"),
  });

  const { data: regimesData, isLoading: isRegimesLoading } = useQuery({
    queryKey: ["/models/regimes", region],
    queryFn: () => fetchChartData(`/models/${encodeURIComponent(region)}/regimes`),
    enabled: !!region,
  });

  const { data: forecastData, isLoading: isForecastLoading } = useQuery({
    queryKey: ["/models/forecast", region],
    queryFn: () => fetchChartData(`/models/${encodeURIComponent(region)}/forecast`),
    enabled: !!region,
  });

  return (
    <motion.div
      className="max-w-7xl mx-auto flex flex-col gap-6"
      variants={pageVariants}
      initial="initial"
      animate="animate"
    >
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white mb-2">Predictive Modeling</h1>
          <p className="text-foreground/60 text-sm">HMM Regime Detection and Prophet Forecasting.</p>
        </div>
        <select
          value={region}
          onChange={(e) => setRegion(e.target.value)}
          className="bg-black/50 border border-border text-foreground text-sm rounded-md px-4 py-2 outline-none focus:border-primary transition-colors min-w-[200px]"
        >
          {regionsData?.regions?.map((r: string) => (
            <option key={r} value={r}>{r}</option>
          ))}
        </select>
      </div>

      <motion.div variants={cardVariants} className="glass-panel p-4 rounded-xl shadow-lg flex flex-col min-h-[450px]">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-glow-primary mb-4">Hidden Markov Model: Attack Regimes</h3>
        <div className="flex-1 relative">
          {isRegimesLoading && <Skeleton className="absolute inset-0" />}
          {regimesData && <Chart data={regimesData.data} layout={regimesData.layout} />}
        </div>
      </motion.div>

      <motion.div variants={cardVariants} className="glass-panel p-4 rounded-xl shadow-lg flex flex-col min-h-[450px]">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-glow-primary mb-4">Prophet Forecast (14 Days)</h3>
        <div className="flex-1 relative">
          {isForecastLoading && <Skeleton className="absolute inset-0" />}
          {forecastData && <Chart data={forecastData.data} layout={forecastData.layout} />}
        </div>
      </motion.div>
    </motion.div>
  );
}
