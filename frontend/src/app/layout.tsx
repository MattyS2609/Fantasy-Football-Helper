import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FPL Field Notes",
  description: "A sharper shortlist for your next FPL transfer.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
