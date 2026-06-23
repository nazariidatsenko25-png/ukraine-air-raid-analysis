import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "./providers";
import { Sidebar } from "@/components/layout/Sidebar";

export const metadata: Metadata = {
  title: "Ukraine Alerts | Tactical Dashboard",
  description: "Real-time analysis and forecasting of air raid alerts",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark h-full antialiased">
      <body className="min-h-full flex bg-background text-foreground overflow-hidden">
        <Providers>
          <Sidebar />
          <main className="flex-1 overflow-y-auto bg-background/50 relative">
            <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-primary/5 via-background to-background pointer-events-none" />
            <div className="relative z-10 p-8 min-h-full">
              {children}
            </div>
          </main>
        </Providers>
      </body>
    </html>
  );
}
