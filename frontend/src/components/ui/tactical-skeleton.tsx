"use client";

export type Variant = "line" | "bars" | "heatmap" | "scatter" | "block";

interface TacticalSkeletonProps {
  variant?: Variant;
  label?: string;
  className?: string;
}

export function TacticalSkeleton({
  variant = "block",
  label = "LOADING",
  className = "",
}: TacticalSkeletonProps) {
  return (
    <div className={`absolute inset-0 flex flex-col ${className}`} aria-busy="true" aria-label={label}>
      {/* Status label */}
      <span className="absolute top-2 left-2 text-[10px] font-mono text-foreground/25 tracking-widest uppercase z-10">
        {label}
      </span>

      {/* Chart-shaped skeleton body */}
      <div className="flex-1 flex items-end gap-1.5 p-6 pt-8">
        {variant === "bars" && (
          <>
            {[55, 80, 40, 90, 65, 75, 45, 85, 60, 70, 50, 88].map((h, i) => (
              <div
                key={i}
                className="shimmer-base flex-1 rounded-sm"
                style={{ height: `${h}%` }}
              />
            ))}
          </>
        )}

        {variant === "line" && (
          <div className="w-full h-full relative shimmer-base rounded">
            <svg
              viewBox="0 0 400 150"
              className="absolute inset-0 w-full h-full opacity-20"
              preserveAspectRatio="none"
            >
              <polyline
                points="0,120 40,90 80,100 120,60 160,70 200,40 240,55 280,30 320,45 360,20 400,35"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              />
            </svg>
          </div>
        )}

        {variant === "heatmap" && (
          <div className="w-full h-full grid gap-0.5" style={{ gridTemplateColumns: "repeat(12, 1fr)", gridTemplateRows: "repeat(7, 1fr)" }}>
            {Array.from({ length: 84 }).map((_, i) => (
              <div
                key={i}
                className="shimmer-base rounded-sm"
                style={{ opacity: 0.3 + (Math.sin(i * 0.7) + 1) * 0.3 }}
              />
            ))}
          </div>
        )}

        {variant === "scatter" && (
          <div className="w-full h-full relative shimmer-base rounded">
            {Array.from({ length: 20 }).map((_, i) => (
              <div
                key={i}
                className="absolute w-2 h-2 rounded-full bg-foreground/20"
                style={{
                  left: `${10 + ((i * 37) % 80)}%`,
                  bottom: `${10 + ((i * 53) % 75)}%`,
                }}
              />
            ))}
          </div>
        )}

        {variant === "block" && (
          <div className="w-full h-full shimmer-base rounded" />
        )}
      </div>
    </div>
  );
}

interface ChartEmptyProps {
  title?: string;
  message?: string;
}

export function ChartEmpty({
  title = "No data for this view",
  message = "The dataset returned no results for the selected parameters.",
}: ChartEmptyProps) {
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-center">
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="36"
        height="36"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1"
        className="text-foreground/20"
      >
        <path d="M3 3l18 18M10.5 10.677A2 2 0 0 0 10 12a2 2 0 1 0 4 0c0-.668-.3-1.263-.779-1.663" />
        <path d="M21 16.418A9 9 0 0 0 3.757 7.75" />
        <path d="M12 3a9 9 0 0 1 8.244 12.626" />
      </svg>
      <p className="text-sm font-medium text-foreground/40">{title}</p>
      <p className="text-xs text-foreground/25 max-w-xs">{message}</p>
    </div>
  );
}

interface ChartStateProps {
  isLoading: boolean;
  isError: boolean;
  isEmpty?: boolean;
  variant?: Variant;
  loadingLabel?: string;
  emptyTitle?: string;
  emptyMessage?: string;
  children: React.ReactNode;
}

export function ChartState({
  isLoading,
  isError,
  isEmpty = false,
  variant = "block",
  loadingLabel = "LOADING",
  emptyTitle,
  emptyMessage,
  children,
}: ChartStateProps) {
  if (isLoading) {
    return <TacticalSkeleton variant={variant} label={loadingLabel} />;
  }
  if (isError) {
    return (
      <ChartEmpty
        title="Failed to load chart"
        message={emptyMessage ?? "An error occurred. Try refreshing the page."}
      />
    );
  }
  if (isEmpty) {
    return <ChartEmpty title={emptyTitle} message={emptyMessage} />;
  }
  return <>{children}</>;
}
