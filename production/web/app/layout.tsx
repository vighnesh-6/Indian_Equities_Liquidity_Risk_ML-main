import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Liquidity | Indian Equities Risk Intelligence",
  description:
    "Historical liquidity risk, frozen-model HRE-5D forecasts and anomaly surveillance for 300 Indian equities.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
