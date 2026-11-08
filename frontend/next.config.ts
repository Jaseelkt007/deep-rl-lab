import type { NextConfig } from "next";

// Only what this app needs: every panel is a client component fed by a WebSocket, so the
// scaffold's cacheComponents / partialPrefetching flags are left out on purpose.
const nextConfig: NextConfig = {
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
