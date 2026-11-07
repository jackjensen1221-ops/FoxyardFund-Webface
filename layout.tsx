import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Apex | Investment Intelligence",
  description: "Professional investment research and portfolio analytics.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
