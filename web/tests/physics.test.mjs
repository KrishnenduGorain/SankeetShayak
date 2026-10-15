import test from 'node:test';
import assert from 'node:assert/strict';
import { savgolWeights, computeDynamics, Analyzer, classifyEffort, HAND_MASS_KG } from '../js/physics.js';
import { classifyShape } from '../js/shape.js';
import { makeTapSequence, templateWorld } from '../js/sim.js';

const close = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) <= tol, `${a} !~ ${b}`);

test('savgol weights match published tables', () => {
  const w1 = savgolWeights(7, 3, 1);
  [22, -67, -58, 0, 58, 67, -22].forEach((v, i) => close(w1[i], v / 252));
  const w0 = savgolWeights(5, 2, 0);
  [-3, 12, 17, 12, -3].forEach((v, i) => close(w0[i], v / 35));
});

test('savgol recovers exact derivatives of a cubic', () => {
  const dt = 1 / 30, f = t => 1 + 2 * t - 3 * t * t + 4 * t ** 3;
  const y = Array.from({ length: 7 }, (_, j) => f((j - 3) * dt));
  const d = k => savgolWeights(7, 3, k).reduce((s, w, j) => s + w * y[j], 0) / dt ** k;
  close(d(0), 1, 1e-9); close(d(1), 2, 1e-9); close(d(2), -6, 1e-7); close(d(3), 24, 1e-6);
});

test('savgol rejects bad arguments', () => {
  assert.throws(() => savgolWeights(6, 2, 1)); assert.throws(() => savgolWeights(5, 5, 1));
});

test('rigid translation: KE = 1/2 M v^2 and F = M a', () => {
  const pos = new Float64Array(63).map((_, i) => (i * 0.013) % 0.2), vel = new Float64Array(63), acc = new Float64Array(63);
  for (let i = 0; i < 21; i++) { vel[i * 3] = 1; acc[i * 3] = 2; }
  const d = computeDynamics(pos, vel, acc, 0.4);
  close(d.ke, 0.2, 1e-12); close(d.forceMag, 0.8, 1e-12); close(d.keRot, 0, 1e-12);
});

test('thin rod pivoting about its end: KE = 1/2 (mL^2/3) w^2', () => {
  const L = 0.1, w = 10, m = 0.05;
  const pos = new Float64Array(63), vel = new Float64Array(63), acc = new Float64Array(63);
  pos[3] = L; vel[4] = w * L;
  const d = computeDynamics(pos, vel, acc, m, [[0, 1, 1]]);
  close(d.ke, 0.5 * (m * L * L / 3) * w * w, 1e-12);
});

for (const fps of [30, 60]) {
  for (const [v, label] of [[0.3, 'Gentle'], [0.6, 'Moderate'], [1.2, 'Forceful']]) {
    test(`simulated ${v} m/s strike @${fps}fps -> ${label}`, () => {
      const an = new Analyzer(); const strokes = [];
      for (const f of makeTapSequence({ peakSpeed: v, fps, seed: 3 })) {
        const r = an.push(f.t, f.P, 'Open Palm'); if (r?.stroke) strokes.push(r.stroke);
      }
      assert.equal(strokes.length, 1, 'exactly one stroke');
      assert.equal(strokes[0].effort.label, label);
      assert.equal(strokes[0].shape, 'Open Palm');
      assert.ok(Math.abs(strokes[0].peakSpeed - v) / v < 0.25, `peak ${strokes[0].peakSpeed}`);
    });
  }
}

test('still hand with jitter produces no strokes', () => {
  const an = new Analyzer(); let n = 0;
  const frames = makeTapSequence({ peakSpeed: 0.3, amplitude: 0, fps: 30, seed: 9 });
  for (const f of frames) if (an.push(f.t, f.P)?.stroke) n++;
  assert.equal(n, 0);
});

test('KE ratio between strikes is ~ (v ratio)^2', () => {
  const peak = v => { const an = new Analyzer(); let m = 0; for (const f of makeTapSequence({ peakSpeed: v, fps: 60, noise: 0 })) { const r = an.push(f.t, f.P); if (r) m = Math.max(m, r.telemetry.ke); } return m; };
  const ratio = peak(1.2) / peak(0.3);
  assert.ok(ratio > 12 && ratio < 20, `ratio ${ratio}`);
});

test('tracking gap resets the buffer and duplicate timestamps are ignored', () => {
  const an = new Analyzer(); const fr = makeTapSequence({ peakSpeed: 0.6, fps: 30, noise: 0 });
  fr.slice(0, 20).forEach(f => an.push(f.t, f.P));
  an.push(10, fr[0].P); assert.equal(an.frames.length, 1);
  assert.equal(an.push(10, fr[0].P), null);
});

test('shape classifier', () => {
  assert.equal(classifyShape(templateWorld()).label, 'Open Palm');
  const w = templateWorld();
  for (const [mcp, a, b, c] of [[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]) {
    const base = w[mcp]; [a, b, c].forEach((i, k) => { w[i] = { x: base.x, y: base.y + 0.01 * (k + 1), z: 0.01 }; });
  }
  w[4] = { x: -0.01, y: -0.07, z: 0 };
  assert.equal(classifyShape(w).label, 'Fist');
  assert.equal(classifyShape(null).label, 'Unknown');
});

test('effort classifier boundaries and sensitivity', () => {
  assert.equal(classifyEffort(0.3).label, 'Gentle'); assert.equal(classifyEffort(0.6).label, 'Moderate');
  assert.equal(classifyEffort(1.2).label, 'Forceful'); assert.equal(classifyEffort(0.3, 2).label, 'Moderate');
  assert.equal(HAND_MASS_KG, 0.4);
});
