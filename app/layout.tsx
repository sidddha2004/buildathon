import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SwapShield AI",
  description: "A defense-only return authenticity verifier with calibrated risk, independent evidence audit, and human review.",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased">{children}</body>
    </html>
  );
}
