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

export default function ThreatsPage() {
  const { data: scatterData, isLoading: isScatterLoading, isError: isScatterError } = useQuery({
    queryKey: ["/threats/scatter"],
    queryFn: () => fetchChartData("/threats/scatter"),
    retry: 1,
  });

  const { data: timelineData, isLoading: isTimelineLoading, isError: isTimelineError } = useQuery({
    queryKey: ["/threats/timeline"],
    queryFn: () => fetchChartData("/threats/timeline"),
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
        <h1 className="text-3xl font-bold tracking-tight text-white mb-2">Threat Profiles (GMM Clustering)</h1>
        <p className="text-foreground/60 text-sm">
          Unsupervised Gaussian Mixture Models to categorize attack waves into distinct weapon signatures.
        </p>
      </div>

      <motion.div variants={cardVariants}>
        <ChartCard
          title="Cluster Scatter: Duration vs Regions"
          subtitle="Each point is an individual alert wave, colored by inferred weapon cluster"
          data={scatterData?.data}
          layout={scatterData?.layout}
          isLoading={isScatterLoading}
          isError={isScatterError}
          isEmpty={!isScatterLoading && !isScatterError && !scatterData?.data?.length}
          variant="scatter"
          loadingLabel="LOADING THREAT CLUSTERS"
          emptyTitle="No threat cluster data"
          emptyMessage="GMM clustering requires sufficient data volume."
          minHeight="500px"
        />
      </motion.div>

      <motion.div variants={cardVariants}>
        <ChartCard
          title="Cluster Timeline"
          subtitle="Threat profile distribution over time — how weapon mixes shift across the war"
          data={timelineData?.data}
          layout={timelineData?.layout}
          isLoading={isTimelineLoading}
          isError={isTimelineError}
          isEmpty={!isTimelineLoading && !isTimelineError && !timelineData?.data?.length}
          variant="line"
          loadingLabel="LOADING THREAT TIMELINE"
          emptyTitle="No timeline data"
          emptyMessage="Timeline requires at minimum 30 days of classified threat data."
          minHeight="500px"
        />
      </motion.div>
    </motion.div>
  );
}
