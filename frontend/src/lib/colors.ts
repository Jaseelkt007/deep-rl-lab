// One palette for the whole app (validated with the dataviz skill's checker).
// Actor = blue, critic = orange everywhere; δ sign = aqua (+) / red (−), always with a sign or arrow.

export const C = {
  ink: "#0b0b0b",
  ink2: "#52514e",
  muted: "#8a8984",
  grid: "#ecebe8",
  axis: "#b5b3ab",
  surface: "#ffffff",
  actor: "#2a78d6",
  critic: "#eb6834",
  pos: "#1baf7a",
  neg: "#e34948",
  neutral: "#f0efec",
  marker1: "#6250d6",
  marker2: "#eda100",
  // categorical order for compare groups / multi-series
  series: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
};

// Actions: left / none / right (CartPole: left / right)
export const ACTION_COLORS: Record<string, string> = { left: "#6250d6", none: "#9a9890", right: "#eda100" };
export const ACTION_GLYPH: Record<string, string> = { left: "←", none: "·", right: "→" };

type RGB = [number, number, number];
const hex = (h: string): RGB => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)) as RGB;

function ramp(stops: string[]) {
  const rgb = stops.map(hex);
  return (t: number): RGB => {
    const x = Math.min(1, Math.max(0, t)) * (rgb.length - 1);
    const i = Math.min(rgb.length - 2, Math.floor(x));
    const f = x - i;
    return [0, 1, 2].map((k) => Math.round(rgb[i][k] + (rgb[i + 1][k] - rgb[i][k]) * f)) as RGB;
  };
}

// V(s): sequential, one hue (the critic's orange), light = low, dark = high.
export const seqOrange = ramp(["#fdf0e8", "#f9c7a8", "#f19463", "#eb6834", "#b8461c", "#6e2a0e"]);
// δ: diverging, red (−) · gray (0) · aqua (+).
export const divDelta = ramp(["#b52b2a", "#e34948", "#f3b3ad", "#f0efec", "#a6e2c9", "#1baf7a", "#0b7a52"]);
// P(push right): diverging violet (left) · gray (0.5) · amber (right).
export const divAction = ramp(["#3f2f9e", "#6250d6", "#c6bff0", "#f0efec", "#f7d98c", "#eda100", "#a86f00"]);

export const rgbCss = ([r, g, b]: RGB) => `rgb(${r},${g},${b})`;
export { hex };
