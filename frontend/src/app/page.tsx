"use client";

import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { ChartCard } from "@/components/ChartCard";
import { fetchChartData } from "@/lib/api";

const pageVariants = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0, transition: { staggerChildren: 0.1 } },
};

const cardVariants = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
};

function EDAChartCard({ title, endpoint, variant, className }: {
  title: string;
  endpoint: string;
  variant: "line" | "bars" | "heatmap" | "scatter" | "block";
  className?: string;
}) {
  const { data, isLoading, isError } = useQuery({
    queryKey: [endpoint],
    queryFn: () => fetchChartData(endpoint),
    retry: 1,
  });

  return (
    <motion.div variants={cardVariants} className={className}>
      <ChartCard
        title={title}
        data={data?.data}
        layout={data?.layout}
        isLoading={isLoading}
        isError={isError}
        isEmpty={!isLoading && !isError && !data?.data?.length}
        variant={variant}
        loadingLabel={`LOADING ${title.toUpperCase()}`}
        emptyTitle="No data available"
        emptyMessage="This chart requires data to be present in the dataset."
        minHeight="400px"
      />
    </motion.div>
  );
}

export default function EDADashboard() {
  return (
    <motion.div
      className="max-w-7xl mx-auto flex flex-col gap-6"
      variants={pageVariants}
      initial="initial"
      animate="animate"
    >
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-white mb-2">Exploratory Data Analysis</h1>
        <p className="text-foreground/60 text-sm">Temporal and spatial distribution of air raid alerts.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <EDAChartCard
          title="Daily Alert Volume"
          endpoint="/eda/daily-volume"
          variant="line"
          className="lg:col-span-2"
        />
        <EDAChartCard
          title="Regional Share"
          endpoint="/eda/regional-treemap"
          variant="block"
          className="lg:col-span-1"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <EDAChartCard
          title="Hourly Pattern (Heatmap)"
          endpoint="/eda/heatmap"
          variant="heatmap"
        />
        <EDAChartCard
          title="Alert Duration Distribution"
          endpoint="/eda/regional-duration"
          variant="bars"
        />
      </div>

      <EDAChartCard
        title="Most Frequently Targeted Regions"
        endpoint="/eda/regional-ranking"
        variant="bars"
      />
    </motion.div>
  );
}
