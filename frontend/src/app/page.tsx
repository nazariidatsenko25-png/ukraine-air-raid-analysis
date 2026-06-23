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

function ChartCard({ title, endpoint, className }: { title: string; endpoint: string; className?: string }) {
  const { data, isLoading, error } = useQuery({
    queryKey: [endpoint],
    queryFn: () => fetchChartData(endpoint),
  });

  return (
    <motion.div variants={cardVariants} className={`glass-panel p-4 flex flex-col rounded-xl shadow-lg ${className || ""}`}>
      <h3 className="text-sm font-semibold uppercase tracking-wider text-glow-primary mb-4">{title}</h3>
      <div className="flex-1 relative min-h-[300px]">
        {isLoading && <Skeleton className="absolute inset-0" />}
        {error && <div className="text-primary text-sm font-mono flex items-center justify-center h-full">Error loading chart</div>}
        {data && <Chart data={data.data} layout={data.layout} />}
      </div>
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
        <ChartCard title="Daily Alert Volume" endpoint="/eda/daily-volume" className="lg:col-span-2 min-h-[400px]" />
        <ChartCard title="Regional Share" endpoint="/eda/regional-treemap" className="lg:col-span-1 min-h-[400px]" />
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartCard title="Hourly Pattern (Heatmap)" endpoint="/eda/heatmap" className="min-h-[400px]" />
        <ChartCard title="Alert Duration Distribution" endpoint="/eda/regional-duration" className="min-h-[400px]" />
      </div>

      <ChartCard title="Most Frequently Targeted Regions" endpoint="/eda/regional-ranking" className="min-h-[400px]" />
    </motion.div>
  );
}
