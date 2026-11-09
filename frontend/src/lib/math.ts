// Number formatting for the UI.

export function fmt(x: number | null | undefined, digits = 3): string {
  if (x == null || Number.isNaN(x)) return "–";
  const a = Math.abs(x);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return x.toExponential(2);
  return x.toFixed(a >= 100 ? 1 : digits);
}
