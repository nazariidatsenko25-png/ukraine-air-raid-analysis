"use client";

import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { Chart } from "@/components/ui/Chart";
import { Skeleton } from "@/components/ui/Skeleton";
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
  const { data: scatterData, isLoading: isScatterLoading } = useQuery({
    queryKey: ["/threats/scatter"],
    queryFn: () => fetchChartData("/threats/scatter"),
  });

  const { data: timelineData, isLoading: isTimelineLoading } = useQuery({
    queryKey: ["/threats/timeline"],
    queryFn: () => fetchChartData("/threats/timeline"),
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
        <p className="text-foreground/60 text-sm">Unsupervised Gaussian Mixture Models to categorize attack waves into distinct weapon signatures.</p>
      </div>

      <motion.div variants={cardVariants} className="glass-panel p-4 rounded-xl shadow-lg flex flex-col min-h-[500px]">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-glow-primary mb-4">Cluster Scatter: Duration vs Regions</h3>
        <div className="flex-1 relative">
          {isScatterLoading && <Skeleton className="absolute inset-0" />}
          {scatterData && <Chart data={scatterData.data} layout={scatterData.layout} />}
        </div>
      </motion.div>

      <motion.div variants={cardVariants} className="glass-panel p-4 rounded-xl shadow-lg flex flex-col min-h-[500px]">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-glow-primary mb-4">Cluster Timeline</h3>
        <div className="flex-1 relative">
          {isTimelineLoading && <Skeleton className="absolute inset-0" />}
          {timelineData && <Chart data={timelineData.data} layout={timelineData.layout} />}
        </div>
      </motion.div>
    </motion.div>
  );
}
