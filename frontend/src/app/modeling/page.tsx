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

export default function ModelingPage() {
  const [region, setRegion] = useState("");

  const { data: regionsData } = useQuery({
    queryKey: ["/models/regions"],
    queryFn: () => fetchJsonData("/models/regions"),
  });

  useEffect(() => {
    if (regionsData?.regions?.length && !regionsData.regions.includes(region)) {
      setRegion(regionsData.regions[0]);
    }
  }, [regionsData, region]);

  const { data: regimesData, isLoading: isRegimesLoading, isError: isRegimesError } = useQuery({
    queryKey: ["/models/regimes", region],
    queryFn: () => fetchChartData(`/models/${encodeURIComponent(region)}/regimes`),
    enabled: !!region,
    retry: 1,
  });

  const { data: forecastData, isLoading: isForecastLoading, isError: isForecastError } = useQuery({
    queryKey: ["/models/forecast", region],
    queryFn: () => fetchChartData(`/models/${encodeURIComponent(region)}/forecast`),
    enabled: !!region,
    retry: 1,
  });

  const regionShort = region.replace(" oblast", "");

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

      <motion.div variants={cardVariants}>
        <ChartCard
          title="Hidden Markov Model: Attack Regimes"
          subtitle={regionShort ? `Threat regime classification for ${regionShort}` : undefined}
          data={regimesData?.data}
          layout={regimesData?.layout}
          isLoading={isRegimesLoading || !region}
          isError={isRegimesError}
          isEmpty={!isRegimesLoading && !isRegimesError && !regimesData?.data?.length}
          variant="line"
          loadingLabel={regionShort ? `LOADING ${regionShort.toUpperCase()} REGIMES` : "LOADING"}
          emptyTitle="No regime data"
          emptyMessage={`No regime data available for ${regionShort}. Try another region.`}
          minHeight="450px"
        />
      </motion.div>

      <motion.div variants={cardVariants}>
        <ChartCard
          title="Prophet Forecast (14 Days)"
          subtitle={regionShort ? `Alert count forecast for ${regionShort}` : undefined}
          data={forecastData?.data}
          layout={forecastData?.layout}
          isLoading={isForecastLoading || !region}
          isError={isForecastError}
          isEmpty={!isForecastLoading && !isForecastError && !forecastData?.data?.length}
          variant="line"
          loadingLabel={regionShort ? `LOADING ${regionShort.toUpperCase()} FORECAST` : "LOADING"}
          emptyTitle="No forecast data"
          emptyMessage={`No forecast available for ${regionShort}. Try another region.`}
          minHeight="450px"
        />
      </motion.div>
    </motion.div>
  );
}
