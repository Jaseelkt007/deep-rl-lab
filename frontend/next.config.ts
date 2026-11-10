import type { NextConfig } from "next";

// Only what this app needs: every panel is a client component fed by a WebSocket, so the
// scaffold's cacheComponents / partialPrefetching flags are left out on purpose.
const nextConfig: NextConfig = {
  // On Vercel there is no backend: read the saved runs from public/data (see src/lib/api.ts).
  env: { NEXT_PUBLIC_STATIC: process.env.NEXT_PUBLIC_STATIC ?? (process.env.VERCEL ? "1" : "") },
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
