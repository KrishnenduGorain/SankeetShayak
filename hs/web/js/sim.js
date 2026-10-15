// Synthetic gestures (same idea as GestureDatasetBuilder.generate_synthetic_gestures in the Python plan).
// Identical hand shape and path; only the speed of the strike differs.
const T2 = [
  [0, 0],
  [-0.03, -0.015], [-0.058, -0.04], [-0.082, -0.062], [-0.1, -0.082],
  [-0.03, -0.09], [-0.033, -0.125], [-0.035, -0.148], [-0.036, -0.168],
  [-0.005, -0.095], [-0.005, -0.135], [-0.005, -0.16], [-0.005, -0.182],
  [0.02, -0.09], [0.022, -0.125], [0.023, -0.147], [0.024, -0.165],
  [0.04, -0.08], [0.044, -0.108], [0.046, -0.126], [0.047, -0.142],
];
export const BASE_Y = -0.05;
export const SIM_W = 640, SIM_H = 480, SIM_PX_PER_M = 1200;

export function templateWorld() { return T2.map(([x, y]) => ({ x, y, z: 0 })); }

export function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const gauss = rng => Math.sqrt(-2 * Math.log(1 - rng())) * Math.cos(2 * Math.PI * rng());
const minJerk = tau => { const t = Math.min(1, Math.max(0, tau)); return t * t * t * (10 - 15 * t + 6 * t * t); };

/** Strike down by `amplitude` metres with a min-jerk profile whose peak speed is `peakSpeed` m/s. */
export function makeTapSequence({ peakSpeed, fps = 30, amplitude = 0.10, noise = 0.0005, seed = 7 }) {
  const rng = mulberry32(seed);
  const T = 1.875 * amplitude / peakSpeed;      // min-jerk peak speed = 1.875 A / T
  const pre = 0.5, hold = 0.5, ret = 2.0, post = 0.4;
  const total = pre + T + hold + ret + post;
  const N = Math.floor(total * fps) + 1;
  const frames = [];
  for (let k = 0; k < N; k++) {
    const t = k / fps;
    let disp;
    if (t < pre) disp = 0;
    else if (t < pre + T) disp = amplitude * minJerk((t - pre) / T);
    else if (t < pre + T + hold) disp = amplitude;
    else if (t < pre + T + hold + ret) disp = amplitude * (1 - minJerk((t - pre - T - hold) / ret));
    else disp = 0;
    const P = new Float64Array(63);
    for (let i = 0; i < 21; i++) {
      P[i * 3] = T2[i][0] + noise * gauss(rng);
      P[i * 3 + 1] = T2[i][1] + BASE_Y + disp + noise * gauss(rng);
      P[i * 3 + 2] = noise * gauss(rng);
    }
    frames.push({ t, P });
  }
  return frames;
}

/** Map metre positions to normalised canvas coordinates for the shared skeleton drawer. */
export function toNormalized(P) {
  const out = [];
  for (let i = 0; i < 21; i++) out.push({ x: (SIM_W / 2 + P[i * 3] * SIM_PX_PER_M) / SIM_W, y: (330 + P[i * 3 + 1] * SIM_PX_PER_M) / SIM_H, z: 0 });
  return out;
}
