"use client";

import dynamic from "next/dynamic";

// Everything draws in the browser (canvas, charts): render client-side only.
const CartPolePage = dynamic(() => import("@/components/CartPolePage"), { ssr: false });

export default function Home() {
  return <CartPolePage />;
}
