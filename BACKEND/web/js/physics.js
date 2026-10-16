// HandShayak physics engine (browser + Node). JS port of kinematics.py / dynamics.py,
// extended with rotational KE, torque, impulse, time-uniform resampling and stroke detection.

export const HAND_MASS_KG = 0.40;   // Dempster: hand ~0.6 % of body mass
export const REF_LEN_M = 0.095;     // wrist -> middle-finger MCP, used for pixel->metre scale
export const NUM_LM = 21;
export const PALM_IDX = [0, 5, 9, 13, 17];

// [proximal, distal, mass fraction] (same linkage as dynamics.py; fractions are re-normalised)
export const SEGMENTS = [
  [0, 1, 0.08], [0, 5, 0.12], [0, 9, 0.16], [0, 13, 0.12], [0, 17, 0.12],
  [1, 2, 0.03], [2, 3, 0.03], [3, 4, 0.02],
  [5, 6, 0.03], [6, 7, 0.03], [7, 8, 0.02],
  [9, 10, 0.04], [10, 11, 0.03], [11, 12, 0.02],
  [13, 14, 0.03], [14, 15, 0.03], [15, 16, 0.02],
  [17, 18, 0.02], [18, 19, 0.02], [19, 20, 0.02],
];

/* ---------------- Savitzky–Golay ---------------- */
function solveLinear(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (Math.abs(M[p][c]) < 1e-12) throw new Error('Singular matrix in Savitzky-Golay design');
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = M[i][n];
    for (let k = i + 1; k < n; k++) s -= M[i][k] * x[k];
    x[i] = s / M[i][i];
  }
  return x;
}

const wCache = new Map();
/** Centre-point Savitzky–Golay weights; result is per-sample (divide by dt^deriv for SI units). */
export function savgolWeights(window, order, deriv = 0) {
  if (!Number.isInteger(window) || window < 3 || window % 2 === 0) throw new RangeError('window must be an odd integer >= 3');
  if (!Number.isInteger(order) || order < 0 || order >= window) throw new RangeError('order must satisfy 0 <= order < window');
  if (!Number.isInteger(deriv) || deriv < 0) throw new RangeError('deriv must be a non-negative integer');
  const key = `${window}|${order}|${deriv}`;
  if (wCache.has(key)) return wCache.get(key);
  let w;
  if (deriv > order) {
    w = new Array(window).fill(0);
  } else {
    const half = (window - 1) / 2, n = order + 1;
    const ATA = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let j = 0; j < window; j++) {
      const x = j - half;
      for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) ATA[a][b] += Math.pow(x, a + b);
    }
    const e = new Array(n).fill(0); e[deriv] = 1;
    const u = solveLinear(ATA, e);
    let fact = 1; for (let i = 2; i <= deriv; i++) fact *= i;
    w = [];
    for (let j = 0; j < window; j++) {
      const x = j - half; let s = 0;
      for (let k = 0; k < n; k++) s += u[k] * Math.pow(x, k);
      w.push(fact * s);
    }
  }
  wCache.set(key, w);
  return w;
}

/* ---------------- Dynamics ---------------- */
/** Rigid-rod dynamics for all segments. pos/vel/acc are flat arrays length 63. */
export function computeDynamics(pos, vel, acc, handMass = HAND_MASS_KG, segments = SEGMENTS) {
  const fsum = segments.reduce((s, x) => s + x[2], 0);
  let keTrans = 0, keRot = 0, maxTau = 0;
  const mom = [0, 0, 0], F = [0, 0, 0];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = v => Math.hypot(v[0], v[1], v[2]);
  for (const [p, d, frac] of segments) {
    const m = handMass * frac / fsum;
    const ip = p * 3, id = d * 3;
    const r = [pos[id] - pos[ip], pos[id + 1] - pos[ip + 1], pos[id + 2] - pos[ip + 2]];
    const L2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    if (L2 < 1e-10) continue;
    const vc = [0, 1, 2].map(k => 0.5 * (vel[ip + k] + vel[id + k]));
    const ac = [0, 1, 2].map(k => 0.5 * (acc[ip + k] + acc[id + k]));
    const vr = [0, 1, 2].map(k => vel[id + k] - vel[ip + k]);
    const ar = [0, 1, 2].map(k => acc[id + k] - acc[ip + k]);
    const omega = norm(cross(r, vr)) / L2;   // rad/s  (rigid rod: |r x v_rel| / L^2)
    const alpha = norm(cross(r, ar)) / L2;   // rad/s^2
    const Ic = m * L2 / 12;                  // thin rod about centre, eq. (6)
    keTrans += 0.5 * m * (vc[0] ** 2 + vc[1] ** 2 + vc[2] ** 2);
    keRot += 0.5 * Ic * omega * omega;       // eq. (9)
    for (let k = 0; k < 3; k++) { mom[k] += m * vc[k]; F[k] += m * ac[k]; }
    maxTau = Math.max(maxTau, Ic * alpha);   // eq. (14)
  }
  return {
    ke: keTrans + keRot, keTrans, keRot,
    momentum: mom, momentumMag: Math.hypot(...mom),
    force: F, forceMag: Math.hypot(...F),
    peakTorque: maxTau,
  };
}

/* ---------------- Effort classifier (dynamics-only model) ---------------- */
// Geometric midpoints of the 0.3 / 0.6 / 1.2 m/s reference taps from Table 2 of the report.
export function effortBoundaries(sensitivity = 1) { return [0.42 / sensitivity, 0.85 / sensitivity]; }
export function classifyEffort(peakSpeed, sensitivity = 1) {
  const [b1, b2] = effortBoundaries(sensitivity);
  const v = Math.max(peakSpeed, 1e-6);
  let label, d;
  if (v < b1) { label = 'Gentle'; d = Math.log(b1 / v); }
  else if (v < b2) { label = 'Moderate'; d = Math.min(Math.log(v / b1), Math.log(b2 / v)); }
  else { label = 'Forceful'; d = Math.log(v / b2); }
  return { label, confidence: 0.5 + 0.5 * Math.tanh(d / 0.35) };
}

/* ---------------- Streaming analyser ---------------- */
function resample(frames, tEnd, dt, n) {
  const out = []; let j = 0;
  for (let i = 0; i < n; i++) {
    const tg = tEnd - (n - 1 - i) * dt;
    if (tg < frames[0].t - 1e-9) return null;
    while (j < frames.length - 2 && frames[j + 1].t < tg) j++;
    const a = frames[j], b = frames[j + 1];
    const span = b.t - a.t;
    const u = span > 0 ? Math.min(1, Math.max(0, (tg - a.t) / span)) : 0;
    const s = new Float64Array(a.P.length);
    for (let k = 0; k < s.length; k++) s[k] = a.P[k] + u * (b.P[k] - a.P[k]);
    out.push(s);
  }
  return out;
}

export class Analyzer {
  constructor({ window = 7, order = 3, handMass = HAND_MASS_KG, sensitivity = 1 } = {}) {
    this.window = window; this.order = order;
    this.handMass = handMass; this.sensitivity = sensitivity;
    this.params = { startSpeed: 0.15, endSpeed: 0.08, endHold: 0.12, minDuration: 0.08, minPeak: 0.2, maxDuration: 2.5 };
    this.keepSeconds = 1.0;
    this.reset();
  }
  reset() { this.frames = []; this.stroke = null; }
  get strokeActive() { return !!this.stroke; }

  _dt() {
    const f = this.frames, d = [];
    for (let i = Math.max(1, f.length - 9); i < f.length; i++) d.push(f[i].t - f[i - 1].t);
    d.sort((a, b) => a - b);
    const med = d[Math.floor(d.length / 2)];
    return Math.min(1 / 10, Math.max(1 / 120, med));
  }

  /** @returns {{telemetry, stroke}|null}  P = Float64Array(63) positions in metres */
  push(t, P, tag = null) {
    const last = this.frames[this.frames.length - 1];
    if (last) {
      if (t <= last.t) { if (t < last.t - 0.5) this.reset(); else return null; }
      else if (t - last.t > 0.3) this.reset();            // tracking gap: never differentiate across it
    }
    this.frames.push({ t, P });
    while (this.frames.length > 2 && this.frames[0].t < t - this.keepSeconds) this.frames.shift();
    if (this.frames.length < this.window) return null;
    const dt = this._dt();
    const n = this.window;
    const S = resample(this.frames, t, dt, n);
    if (!S) return null;

    const w0 = savgolWeights(n, this.order, 0), w1 = savgolWeights(n, this.order, 1);
    const w2 = savgolWeights(n, this.order, 2), w3 = savgolWeights(n, this.order, 3);
    const len = S[0].length;
    const pos = new Float64Array(len), vel = new Float64Array(len), acc = new Float64Array(len), jerk = new Float64Array(len);
    for (let k = 0; k < len; k++) {
      let a = 0, b = 0, c = 0, d = 0;
      for (let j = 0; j < n; j++) { const y = S[j][k]; a += w0[j] * y; b += w1[j] * y; c += w2[j] * y; d += w3[j] * y; }
      pos[k] = a; vel[k] = b / dt; acc[k] = c / (dt * dt); jerk[k] = d / (dt * dt * dt);
    }
    const palm = arr => {
      const v = [0, 0, 0];
      for (const i of PALM_IDX) for (let k = 0; k < 3; k++) v[k] += arr[i * 3 + k] / PALM_IDX.length;
      return v;
    };
    const pv = palm(vel);
    const dyn = computeDynamics(pos, vel, acc, this.handMass);
    const tel = {
      t: t - ((n - 1) / 2) * dt,        // estimates are centred, so they lag by half a window
      dt,
      speed: Math.hypot(...pv), accel: Math.hypot(...palm(acc)), jerk: Math.hypot(...palm(jerk)),
      vel: pv, ke: dyn.ke, keRot: dyn.keRot, force: dyn.forceMag, F: dyn.force,
      momentum: dyn.momentumMag, torque: dyn.peakTorque,
    };
    return { telemetry: tel, stroke: this._stroke(tel, tag) };
  }

  _stroke(tel, tag) {
    const P = this.params;
    let s = this.stroke;
    if (!s) {
      if (tel.speed <= P.startSpeed) return null;
      s = this.stroke = { t0: tel.t, lastT: tel.t, tPeak: tel.t, peakSpeed: 0, peakAccel: 0, peakJerk: 0, peakKE: 0,
        peakForce: 0, peakTorque: 0, imp: [0, 0, 0], belowSince: null, tags: {} };
    }
    const dt = Math.min(0.1, Math.max(0, tel.t - s.lastT)); s.lastT = tel.t;
    if (tel.speed > s.peakSpeed) { s.peakSpeed = tel.speed; s.tPeak = tel.t; }
    s.peakAccel = Math.max(s.peakAccel, tel.accel); s.peakJerk = Math.max(s.peakJerk, tel.jerk);
    s.peakKE = Math.max(s.peakKE, tel.ke); s.peakForce = Math.max(s.peakForce, tel.force);
    s.peakTorque = Math.max(s.peakTorque, tel.torque);
    for (let k = 0; k < 3; k++) s.imp[k] += tel.F[k] * dt;
    if (tag) s.tags[tag] = (s.tags[tag] || 0) + 1;
    if (tel.speed < P.endSpeed) { if (s.belowSince === null) s.belowSince = tel.t; } else s.belowSince = null;
    const ended = (s.belowSince !== null && tel.t - s.belowSince >= P.endHold) || tel.t - s.t0 > P.maxDuration;
    if (!ended) return null;
    this.stroke = null;
    const t1 = s.belowSince ?? tel.t, duration = t1 - s.t0;
    if (duration < P.minDuration || s.peakSpeed < P.minPeak) return null;
    const shape = Object.entries(s.tags).sort((a, b) => b[1] - a[1])[0]?.[0] ?? 'Unknown';
    return {
      t0: s.t0, duration, timeToPeak: s.tPeak - s.t0,
      peakSpeed: s.peakSpeed, peakAccel: s.peakAccel, peakJerk: s.peakJerk, peakKE: s.peakKE,
      peakForce: s.peakForce, peakTorque: s.peakTorque, impulse: Math.hypot(...s.imp),
      shape, effort: classifyEffort(s.peakSpeed, this.sensitivity),
    };
  }
}
