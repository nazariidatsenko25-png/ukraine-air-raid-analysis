"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, LayoutDashboard, ShieldAlert, Zap } from "lucide-react";
import { motion } from "framer-motion";

const navItems = [
  { name: "EDA", path: "/", icon: LayoutDashboard },
  { name: "Cascade Analysis", path: "/cascade", icon: Zap },
  { name: "Modeling", path: "/modeling", icon: Activity },
  { name: "Threat Profiles", path: "/threats", icon: ShieldAlert },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <div className="w-64 h-screen border-r border-border bg-card/50 flex flex-col pt-8 pb-4">
      <div className="px-6 mb-8">
        <h1 className="text-xl font-mono font-bold tracking-tight text-glow-primary text-primary uppercase">
          Ukraine Alerts
        </h1>
        <p className="text-xs text-foreground/50 mt-1 uppercase tracking-widest">Tactical Dashboard</p>
      </div>

      <nav className="flex-1 px-4 space-y-2">
        {navItems.map((item) => {
          const isActive = pathname === item.path;
          return (
            <Link
              key={item.path}
              href={item.path}
              className={`relative flex items-center gap-3 px-4 py-3 rounded-md transition-colors duration-300 ${
                isActive ? "text-primary font-medium" : "text-foreground/70 hover:text-foreground hover:bg-white/5"
              }`}
            >
              {isActive && (
                <motion.div
                  layoutId="activeTab"
                  className="absolute inset-0 bg-primary/10 border border-primary/20 rounded-md"
                  initial={false}
                  transition={{ type: "spring", stiffness: 400, damping: 30 }}
                />
              )}
              <item.icon className="w-5 h-5 relative z-10" />
              <span className="relative z-10">{item.name}</span>
            </Link>
          );
        })}
      </nav>

      <div className="px-6">
        <div className="text-xs text-foreground/40 font-mono">
          STATUS: <span className="text-primary text-glow-primary">LIVE</span>
        </div>
      </div>
    </div>
  );
}
