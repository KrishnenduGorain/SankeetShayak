/* =========================================================
   Sankeet Shayak — frontend behaviour
   Sections: intro · nav · reveal · background · demo ·
             hand preview · pipeline · physics · compare
   ========================================================= */
(() => {
  'use strict';

  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const NS = 'http://www.w3.org/2000/svg';
  const svgEl = (tag, attrs = {}) => {
    const el = document.createElementNS(NS, tag);
    for (const k in attrs) el.setAttribute(k, attrs[k]);
    return el;
  };
  const safe = (fn, fallback) => { try { return fn(); } catch { return fallback; } };

  const body = document.body;
  const app = $('#app');
  const intro = $('#intro');
  const heroVideo = $('.hero-video');

  /* ---------------- INTRO ----------------
     One timeline (seconds) drives everything — canvas, title, status label and
     progress bar — so they can never disagree. 5 scenes:
       1 signal awakening 0–.7 · 2 hand reconstruction .7–1.7 · 3 motion burst 1.7–2.5
       4 brand reveal 2.5–3.6 · 5 final lock-in 3.6–5.0
     The hand is an ILLUSTRATION of the 21 landmark layout — not live detection. */
  /* ---------------- INTRO ----------------
     One timeline (seconds) drives everything — canvas, title, status label and
     progress bar — so they can never disagree. 5 scenes:
       1 signal awakening 0–.7 · 2 hand reconstruction .7–1.8 · 3 motion burst 1.8–2.8
       4 brand reveal 2.8–3.8 · 5 final lock-in 3.8–5.0
     The hand is an ILLUSTRATION of the 21 landmark layout — not live detection. */
  const INTRO_MS = 5000;
  let introDone = false;
  let introTimer = 0, introRaf = 0;
  let introCleanup = () => {};

  function finishIntro(fast = false) {
    if (introDone) return;
    introDone = true;
    clearTimeout(introTimer);
    cancelAnimationFrame(introRaf);
    introCleanup();
    safe(() => sessionStorage.setItem('ss-intro-seen', '1'));
    const hadFocus = intro.contains(document.activeElement);
    const fill = $('.intro-bar span');
    if (fill) fill.style.transform = 'scaleX(1)';
    intro.classList.add(fast ? 'fast-leave' : 'leaving');
    app.inert = false;
    app.removeAttribute('aria-hidden');
    body.classList.remove('intro-active');
    body.classList.add('ready');
    if (hadFocus) $('.hero-copy .button-primary')?.focus({ preventScroll: true });
    setTimeout(() => { intro.hidden = true; }, fast ? 420 : 1050);
  }

  function runIntro() {
    const cv = $('#intro-canvas');
    const ctx = cv && cv.getContext && cv.getContext('2d');
    const stageEl = $('#intro-stage'), pctEl = $('#intro-pct'), fill = $('.intro-bar span');
    const D = INTRO_MS / 1000;
    const LABELS = [
      'OPTICAL CALIBRATION',
      'SKELETAL RECONSTRUCTION',
      'KINEMATICS & MOTION',
      'SPATIAL SYNCHRONIZATION',
      'SYSTEM READY'
    ];
    const small = innerWidth < 640;

    const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
    let t = 0, last = 0, scene = 0, pct = -1, titleOn = false, tagOn = false;
    const S = (a, b, tt = t) => clamp((tt - a) / (b - a));              // segment progress 0..1
    const eo = x => 1 - Math.pow(1 - x, 3);                               // ease out
    const ei = x => x * x * x;                                            // ease in
    const eio = x => x < .5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
    const bump = x => Math.sin(Math.PI * clamp(x));                       // 0→1→0

    // Illustrative landmark layout of MediaPipe's 21 landmarks with estimated volumetric Z-depth
    const P = [
      [.50,.92],[.40,.80],[.31,.68],[.25,.55],[.20,.44],
      [.42,.55],[.40,.38],[.39,.25],[.385,.14],
      [.50,.52],[.50,.33],[.50,.19],[.50,.07],
      [.58,.55],[.60,.38],[.61,.26],[.615,.16],
      [.65,.62],[.69,.48],[.715,.38],[.73,.29]
    ];
    const Z_DEPTH = [
      0.0,
      -0.03, -0.06, -0.09, -0.13,
      0.02, -0.02, -0.06, -0.11,
      0.03, -0.01, -0.05, -0.10,
      0.02, -0.02, -0.06, -0.09,
      -0.01, -0.04, -0.07, -0.09
    ];
    const E = [
      [0,1],[1,2],[2,3],[3,4],
      [0,5],[5,6],[6,7],[7,8],
      [5,9],[9,10],[10,11],[11,12],
      [9,13],[13,14],[14,15],[15,16],
      [13,17],[17,18],[18,19],[19,20],
      [0,17]
    ];
    const TIPS = [4, 8, 12, 16, 20];
    const KEY_NAMES = { 0: 'WRIST', 4: 'THUMB_TIP', 8: 'INDEX_TIP', 12: 'MIDDLE_TIP', 16: 'RING_TIP', 20: 'PINKY_TIP' };

    // deterministic particles
    let seed = 11; const rnd = () => (seed = seed * 16807 % 2147483647, (seed - 1) / 2147483646);
    const amb = Array.from({ length: small ? 32 : 64 }, () => ({
      a: rnd() * 6.283, r: .12 + rnd() * .88, wv: (rnd() - .5) * .45,
      sz: .8 + rnd() * 1.4, hue: rnd() < .32 ? 1 : 0, ph: rnd() * 6
    }));
    const bur = Array.from({ length: small ? 38 : 72 }, () => {
      const j = rnd() * 21 | 0, ox = P[j][0] - .5, oy = P[j][1] - .5;
      return {
        j, ang: Math.atan2(oy, ox) + (rnd() - .5) * 1.5, dist: .25 + rnd() * .65,
        curve: (rnd() - .5) * 1.6, t0: 1.8 + rnd() * .15, tc: 2.6 + rnd() * .25,
        sz: 1 + rnd() * 2, hue: rnd() < .35 ? 1 : 0
      };
    });
    const RINGS = [
      { r: .50, dir: 1, sp: .55, col: '49,208,226', arcs: [[0, 1.1], [2.1, .8], [3.9, 1.3]], w: 1.6 },
      { r: .62, dir: -1, sp: .35, col: '139,123,255', arcs: [[.4, .7], [1.9, 1.5], [4.2, .9]], w: 1.3 },
      { r: .75, dir: 1, sp: .2, col: '49,208,226', arcs: [[1, .5], [2.6, 1.2], [5, .8]], w: 1.1 }
    ];

    let w = 0, h = 0, dpr = 1, U = 0, cx = 0, cy = 0;
    function size() {
      if (!cv) return;
      dpr = Math.min(devicePixelRatio || 1, 1.75);
      w = cv.width = Math.round(innerWidth * dpr);
      h = cv.height = Math.round(innerHeight * dpr);
      U = Math.min(w * .88, h * .62);
      cx = w / 2;
      cy = h * .47;
    }
    size();
    addEventListener('resize', size);
    introCleanup = () => { removeEventListener('resize', size); if (ctx) ctx.clearRect(0, 0, w, h); };

    // 3D hand pose with volumetric yaw, pitch, and wrist kinematics
    const wrist = [(P[0][0] - .5), (P[0][1] - .5)];
    function pose3D(i, tt) {
      const bx = (P[i][0] - .5) * U, by = (P[i][1] - .5) * U, wx = wrist[0] * U, wy = wrist[1] * U;
      const rx = bx - wx, ry = by - wy;
      // Wave gesture during scene 3
      const wave = .45 * Math.sin(S(1.75, 2.7, tt) * 6.2832);
      const cW = Math.cos(wave), sW = Math.sin(wave);
      const pulse = 1 + .07 * bump(S(1.75, 2.45, tt));
      const shrink = 1 - .42 * eio(S(2.5, 3.2, tt));

      // 3D rotation angles (subtle inspection orbit during reconstruction)
      const yaw = Math.sin(S(0.7, 1.8, tt) * Math.PI) * .28 + Math.sin(tt * 1.2) * .04;
      const pitch = -0.12 + Math.cos(S(0.7, 1.8, tt) * Math.PI) * .08;
      const cyaw = Math.cos(yaw), syaw = Math.sin(yaw), cpit = Math.cos(pitch), spit = Math.sin(pitch);

      let x3 = wx + (rx * cW - ry * sW) * pulse;
      let y3 = wy + (rx * sW + ry * cW) * pulse;
      let z3 = Z_DEPTH[i] * U * 1.5;

      // Rotate around Y (yaw)
      const xRot = x3 * cyaw + z3 * syaw;
      const zRot1 = -x3 * syaw + z3 * cyaw;
      // Rotate around X (pitch)
      const yRot = y3 * cpit - zRot1 * spit;
      const zRot = y3 * spit + zRot1 * cpit;

      // Perspective projection
      const fov = 3.2, factor = fov / (fov - (zRot / U) * .85);
      return [
        cx + xRot * factor * shrink,
        cy + yRot * factor * shrink + Math.sin(tt * 1.6 + i * .35) * U * .003,
        factor * shrink,
        zRot
      ];
    }

    function glow(x, y, r, rgb, a) {
      if (a <= .003) return;
      const g = ctx.createRadialGradient(x, y, 0, x, y, r);
      g.addColorStop(0, `rgba(${rgb},${a})`);
      g.addColorStop(1, `rgba(${rgb},0)`);
      ctx.fillStyle = g;
      ctx.fillRect(x - r, y - r, r * 2, r * 2);
    }

    // Anamorphic horizontal lens flare
    function anamorphicFlare(x, y, wFlare, hFlare, rgb, a) {
      if (a <= .003) return;
      ctx.save();
      ctx.translate(x, y);
      ctx.scale(1, hFlare / wFlare);
      const g = ctx.createRadialGradient(0, 0, 0, 0, 0, wFlare);
      g.addColorStop(0, `rgba(255,255,255,${a * 1.2})`);
      g.addColorStop(0.18, `rgba(${rgb},${a * .9})`);
      g.addColorStop(0.55, `rgba(${rgb},${a * .25})`);
      g.addColorStop(1, `rgba(${rgb},0)`);
      ctx.fillStyle = g;
      ctx.fillRect(-wFlare, -wFlare, wFlare * 2, wFlare * 2);
      ctx.restore();
    }

    const mono = (px, weight = 500) => {
      ctx.font = `${weight} ${px * dpr}px "JetBrains Mono", ui-monospace, monospace`;
    };

    function draw() {
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, w, h);
      ctx.globalCompositeOperation = 'source-over';
      ctx.globalAlpha = 1;
      ctx.lineCap = 'round';

      // Camera punch-in on kinematic burst, easing back smoothly
      const z = 1 + .09 * (eo(S(1.72, 1.84)) - eio(S(1.84, 2.75)));
      ctx.translate(cx, cy);
      ctx.scale(z, z);
      ctx.translate(-cx, -cy);

      /* ---- Volumetric ambience & soft depth lighting ---- */
      glow(cx, cy, U * .9, '49,208,226', .11 * eo(S(.2, 1.2)));
      glow(cx, cy, U * .72, '139,123,255', .18 * bump(S(2.8, 4.0)));
      glow(cx, cy, U * .58, '49,208,226', .32 * bump(S(1.7, 2.3)));

      /* ---- Holographic perspective grid floor (Scene 1-2) ---- */
      const gridA = eo(S(.1, .6)) * (1 - S(1.9, 2.4));
      if (gridA > .01) {
        ctx.save();
        const gy = cy + U * .44;
        ctx.strokeStyle = `rgba(49,208,226,${.16 * gridA})`;
        ctx.lineWidth = dpr * .8;
        // Radial perspective lines
        const rays = 18;
        for (let i = 0; i <= rays; i++) {
          const rx = (i / rays - .5) * w * 1.1;
          ctx.beginPath();
          ctx.moveTo(cx, gy - U * .1);
          ctx.lineTo(cx + rx, h);
          ctx.stroke();
        }
        // Concentric depth rings
        for (let k = 1; k <= 5; k++) {
          const depth = k / 5;
          const py = gy + depth * (h - gy) * .85;
          ctx.strokeStyle = `rgba(49,208,226,${.22 * (1 - depth * .5) * gridA})`;
          ctx.beginPath();
          ctx.ellipse(cx, py, U * (depth * .9 + .3), U * .12 * depth, 0, 0, 6.283);
          ctx.stroke();
        }
        ctx.restore();
      }

      /* ---- Scene 1: Calibration beacon & Anamorphic Awakening ---- */
      const dotA = eo(S(0, .28)) * (1 - S(.9, 1.25));
      if (dotA > .01) {
        ctx.fillStyle = `rgba(220,252,255,${dotA})`;
        ctx.shadowColor = '#31d0e2';
        ctx.shadowBlur = 24 * dpr;
        ctx.beginPath();
        ctx.arc(cx, cy, dpr * (2.2 + 2.8 * bump(S(.2, .8))), 0, 6.283);
        ctx.fill();
        ctx.shadowBlur = 0;
        // Anamorphic horizontal flare burst
        const flareA = bump(S(.1, .75));
        anamorphicFlare(cx, cy, U * 1.1, U * .06, '49,208,226', flareA * .75);
      }
      const pl = S(.22, .85);
      if (pl > 0 && pl < 1) {
        ctx.strokeStyle = `rgba(128,241,251,${(1 - pl) * .75})`;
        ctx.lineWidth = 2.2 * dpr * (1 - pl) + dpr * .4;
        ctx.beginPath();
        ctx.arc(cx, cy, eo(pl) * U * .65, 0, 6.283);
        ctx.stroke();
      }

      /* ---- HUD rings + Angle ticks (expands with burst, stabilizes at end) ---- */
      const hudA = eo(S(.42, 1.25)), lock = bump(S(3.6, 4.3));
      const tEff = Math.min(t, 3.7) + Math.max(0, t - 3.7) * .25;
      if (hudA > .01) {
        RINGS.forEach((rg, k) => {
          const R = U * rg.r * (.86 + .14 * hudA) * (1 + .20 * bump(S(1.7 + .1 * k, 2.7 + .1 * k)));
          ctx.strokeStyle = `rgba(${rg.col},${.15 * hudA})`;
          ctx.lineWidth = dpr;
          ctx.beginPath();
          ctx.arc(cx, cy, R, 0, 6.283);
          ctx.stroke();

          ctx.strokeStyle = `rgba(${rg.col},${(.62 + .32 * lock) * hudA})`;
          ctx.lineWidth = rg.w * dpr;
          rg.arcs.forEach(([a0, len]) => {
            const a = rg.dir * tEff * rg.sp + a0;
            ctx.beginPath();
            ctx.arc(cx, cy, R, a, a + len * hudA);
            ctx.stroke();
          });
        });
        const TR = U * .56 * (1 + .20 * bump(S(1.75, 2.75)));
        ctx.strokeStyle = `rgba(128,241,251,${.4 * hudA})`;
        ctx.lineWidth = dpr;
        ctx.beginPath();
        for (let i = 0; i < 72; i++) {
          const a = i * 6.2832 / 72 + tEff * .08, len = (i % 6 === 0 ? 11 : 5) * dpr;
          ctx.moveTo(cx + Math.cos(a) * TR, cy + Math.sin(a) * TR);
          ctx.lineTo(cx + Math.cos(a) * (TR + len), cy + Math.sin(a) * (TR + len));
        }
        ctx.stroke();
      }

      /* ---- Ambient particle field ---- */
      ctx.globalCompositeOperation = 'lighter';
      const ambA = eo(S(.2, .75)) * (1 - S(1.5, 1.95));
      if (ambA > .01) {
        amb.forEach(p => {
          const a = p.a + p.wv * t, r = p.r * U * (1.15 - .15 * eo(S(.1, .9)));
          ctx.fillStyle = p.hue
            ? `rgba(139,123,255,${ambA * .85})`
            : `rgba(128,241,251,${ambA * (.65 + .35 * Math.sin(t * 3.2 + p.ph))})`;
          ctx.beginPath();
          ctx.arc(cx + Math.cos(a) * r, cy + Math.sin(a) * r * .7, p.sz * dpr, 0, 6.283);
          ctx.fill();
        });
      }
      ctx.globalCompositeOperation = 'source-over';

      /* ---- Scene 2: Volumetric Hand Reconstruction & Laser Telemetry Scan ---- */
      const ha = 1 - eio(S(2.5, 3.1));
      if (t > .68 && ha > .01) {
        const pts = P.map((_, i) => pose3D(i, t));
        ctx.globalAlpha = ha;

        // Palm geometry fill
        const pf = eo(S(1.25, 1.8));
        if (pf > 0) {
          ctx.beginPath();
          [0, 1, 5, 9, 13, 17].forEach((i, k) => k ? ctx.lineTo(pts[i][0], pts[i][1]) : ctx.moveTo(pts[i][0], pts[i][1]));
          ctx.closePath();
          ctx.fillStyle = `rgba(49,208,226,${.10 * pf})`;
          ctx.fill();
        }

        // Skeletal linkages draw in staggered
        E.forEach(([a, b], k) => {
          const p = eo(S(.72 + k * .03, 1.12 + k * .03));
          if (p <= 0) return;
          const A = pts[a], B = pts[b];
          const X = A[0] + (B[0] - A[0]) * p, Y = A[1] + (B[1] - A[1]) * p;
          const g = ctx.createLinearGradient(A[0], A[1], B[0], B[1]);
          g.addColorStop(0, 'rgba(49,208,226,.95)');
          g.addColorStop(1, 'rgba(139,123,255,.95)');
          ctx.strokeStyle = g;
          ctx.lineWidth = 1.8 * dpr * ((A[2] + B[2]) / 2);
          ctx.beginPath();
          ctx.moveTo(A[0], A[1]);
          ctx.lineTo(X, Y);
          ctx.stroke();

          // Electric neural pulse packet traveling along bone
          if (p >= 1 && t > 1.1 && t < 2.5) {
            const beadP = (t * 2.2 + k * .14) % 1;
            const bx = A[0] + (B[0] - A[0]) * beadP, by = A[1] + (B[1] - A[1]) * beadP;
            ctx.fillStyle = '#fff';
            ctx.shadowColor = '#31d0e2';
            ctx.shadowBlur = 8 * dpr;
            ctx.beginPath();
            ctx.arc(bx, by, 1.6 * dpr, 0, 6.283);
            ctx.fill();
            ctx.shadowBlur = 0;
          }
        });

        // Laser scan plane sweep
        const sb = eio(S(.95, 1.78)), x0 = cx - U * .35, x1 = cx + U * .35, bx = x0 + (x1 - x0) * sb;
        const ba = Math.min(S(.95, 1.08), 1 - S(1.7, 1.82));

        // Joints light up with depth perspective & telemetry
        let lit = 0;
        pts.forEach((p, i) => {
          const tj = .82 + i * .034, k = S(tj, tj + .3);
          if (k <= 0) return;
          lit++;
          const flash = 1 - k, near = ba > 0 ? clamp(1 - Math.abs(p[0] - bx) / (U * .07)) * ba : 0;
          const r = dpr * (TIPS.includes(i) ? 3.6 : 2.5) * p[2] * (1 + flash * 1.25 + near * .65);

          if (flash > 0) {
            ctx.strokeStyle = `rgba(128,241,251,${flash * .55})`;
            ctx.lineWidth = dpr;
            ctx.beginPath();
            ctx.arc(p[0], p[1], r * (1 + 3.2 * k), 0, 6.283);
            ctx.stroke();
          }

          ctx.fillStyle = (flash > .3 || near > .5) ? '#fff' : i === 0 ? '#8b7bff' : '#80f1fb';
          ctx.shadowColor = '#31d0e2';
          ctx.shadowBlur = (10 + 20 * flash + 15 * near) * dpr;
          ctx.beginPath();
          ctx.arc(p[0], p[1], r, 0, 6.283);
          ctx.fill();
          ctx.shadowBlur = 0;

          // Technical telemetry tag for fingertips during scan
          if (TIPS.includes(i) && ba > .2 && Math.abs(p[0] - bx) < U * .12) {
            mono(9);
            ctx.fillStyle = 'rgba(214,246,251,.9)';
            const right = p[0] >= cx;
            const tx = p[0] + (right ? 14 : -14) * dpr, ty = p[1] - 8 * dpr;
            ctx.textAlign = right ? 'left' : 'right';
            ctx.fillText(KEY_NAMES[i], tx, ty);
            ctx.strokeStyle = 'rgba(49,208,226,.6)';
            ctx.lineWidth = dpr * .8;
            ctx.beginPath();
            ctx.moveTo(p[0], p[1]);
            ctx.lineTo(tx, ty + 3 * dpr);
            ctx.stroke();
          }
        });

        // Scan beam visual optics
        if (ba > .01) {
          ctx.globalCompositeOperation = 'lighter';
          const bw = U * .18, g = ctx.createLinearGradient(bx - bw, 0, bx, 0);
          g.addColorStop(0, 'rgba(49,208,226,0)');
          g.addColorStop(1, `rgba(49,208,226,${.35 * ba})`);
          ctx.fillStyle = g;
          ctx.fillRect(bx - bw, cy - U * .54, bw, U * 1.08);

          ctx.strokeStyle = `rgba(210,252,255,${.92 * ba})`;
          ctx.lineWidth = 1.6 * dpr;
          ctx.shadowColor = '#31d0e2';
          ctx.shadowBlur = 16 * dpr;
          ctx.beginPath();
          ctx.moveTo(bx, cy - U * .56);
          ctx.lineTo(bx, cy + U * .56);
          ctx.stroke();
          ctx.shadowBlur = 0;
          ctx.globalCompositeOperation = 'source-over';

          mono(10);
          ctx.fillStyle = `rgba(214,246,251,${.88 * ba})`;
          ctx.textAlign = 'center';
          ctx.fillText(`X ${((bx - cx) / U + .5).toFixed(3)}  ·  VOLUMETRIC_SCAN`, bx, cy - U * .60);
        }

        // Bounding reticle brackets
        const bbA = eo(S(1.0, 1.3));
        if (bbA > 0) {
          let mnx = 1e9, mny = 1e9, mxx = -1e9, mxy = -1e9;
          pts.forEach(p => {
            mnx = Math.min(mnx, p[0]); mny = Math.min(mny, p[1]);
            mxx = Math.max(mxx, p[0]); mxy = Math.max(mxy, p[1]);
          });
          const pad = U * .05, L = U * .05;
          mnx -= pad; mny -= pad; mxx += pad; mxy += pad;
          ctx.strokeStyle = `rgba(128,241,251,${.82 * bbA})`;
          ctx.lineWidth = 1.4 * dpr;
          ctx.beginPath();
          [[mnx, mny, 1, 1], [mxx, mny, -1, 1], [mnx, mxy, 1, -1], [mxx, mxy, -1, -1]].forEach(([x, y, sx, sy]) => {
            ctx.moveTo(x, y + sy * L); ctx.lineTo(x, y); ctx.lineTo(x + sx * L, y);
          });
          ctx.stroke();

          mono(10);
          ctx.fillStyle = `rgba(214,246,251,${.85 * bbA})`;
          ctx.textAlign = 'left';
          ctx.fillText(`HAND · MEDIAPIPE_3D · NODES ${String(lit).padStart(2, '0')}/21`, mnx, mny - 9 * dpr);
        }
        ctx.globalAlpha = 1;
      }

      /* ---- Scene 3: Kinematic Fluid Velocity Trails & Dynamic Vectors ---- */
      if (t > 1.75 && t < 3.0) {
        const ea = S(1.75, 1.9) * (1 - S(2.55, 2.95));
        ctx.globalCompositeOperation = 'lighter';
        TIPS.forEach((ti, n) => {
          let prev = pose3D(ti, t);
          for (let k = 1; k <= 16; k++) {
            const q = pose3D(ti, t - k * .024), f = 1 - k / 17;
            ctx.strokeStyle = n % 2
              ? `rgba(139,123,255,${ea * f * .9})`
              : `rgba(128,241,251,${ea * f * .9})`;
            ctx.lineWidth = dpr * (3.6 * f + .5);
            ctx.beginPath();
            ctx.moveTo(prev[0], prev[1]);
            ctx.lineTo(q[0], q[1]);
            ctx.stroke();
            prev = q;
          }
        });
        ctx.globalCompositeOperation = 'source-over';
      }

      /* ---- Scene 3→4: Dynamic burst & Inward Particle Implosion ---- */
      if (t >= 1.75 && t < 3.6) {
        const org = P.map((_, i) => pose3D(i, 1.75));
        const bpos = (p, tt) => {
          const o = org[p.j], k = eo(S(p.t0, p.t0 + .85, tt)), d = p.dist * U * k, an = p.ang + p.curve * k;
          let x = o[0] + Math.cos(an) * d - cx, y = o[1] + Math.sin(an) * d - cy;
          const m = eio(S(p.tc, 3.35, tt));
          if (m > 0) {
            const c = Math.cos(m * 1.15), s = Math.sin(m * 1.15);
            const nx = (x * c - y * s) * (1 - m), ny = (x * s + y * c) * (1 - m);
            x = nx; y = ny;
          }
          return [cx + x, cy + y, m];
        };

        ctx.globalCompositeOperation = 'lighter';
        bur.forEach(p => {
          const al = S(p.t0, p.t0 + .1) * (1 - S(3.25, 3.55));
          if (al < .02) return;
          const rgb = p.hue ? '139,123,255' : '128,241,251', head = bpos(p, t);
          ctx.strokeStyle = `rgba(${rgb},${al * .6})`;
          ctx.lineWidth = p.sz * dpr * .75;
          ctx.beginPath();
          ctx.moveTo(head[0], head[1]);
          for (let k = 1; k <= 5; k++) {
            const q = bpos(p, t - k * .016);
            ctx.lineTo(q[0], q[1]);
          }
          ctx.stroke();

          ctx.fillStyle = `rgba(${rgb},${al})`;
          ctx.beginPath();
          ctx.arc(head[0], head[1], p.sz * dpr * (1 - .6 * head[2]), 0, 6.283);
          ctx.fill();
        });
        ctx.globalCompositeOperation = 'source-over';
      }

      /* ---- Scene 4: Anamorphic Light Sweep behind Brand Reveal ---- */
      if (t >= 2.8 && t < 3.9) {
        const brandFlare = bump(S(2.85, 3.75));
        anamorphicFlare(cx, cy, U * 1.25, U * .08, '49,208,226', brandFlare * .6);
      }

      /* ---- Telemetry Frame HUD Coordinates ---- */
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      const ta = eo(S(.8, 1.3)) * .65;
      if (ta > .01) {
        mono(10);
        ctx.fillStyle = `rgba(128,241,251,${ta})`;
        ctx.textAlign = 'left';
        const m = 24 * dpr;
        ctx.fillText('SANKEET.SHAYAK // CALIBRATION ENGINE', m, m + 4 * dpr);
        ctx.textAlign = small ? 'left' : 'right';
        ctx.fillText('ILLUSTRATIVE · 21 LANDMARKS · ZERO SENSORS', small ? m : w - m, m + (small ? 20 : 4) * dpr);
        ctx.textAlign = 'left';
        ctx.fillText('T+' + t.toFixed(2) + 's · FRAME ' + Math.round(t * 60), m, h - 28 * dpr);
      }
    }

    function frame(now) {
      introRaf = requestAnimationFrame(frame);
      if (introDone) return;
      t += last ? Math.min(.05, (now - last) / 1000) : 0;
      last = now;
      const sc = t < .7 ? 1 : t < 1.8 ? 2 : t < 2.8 ? 3 : t < 3.8 ? 4 : 5;
      if (sc !== scene) {
        scene = sc;
        intro.dataset.scene = sc;
        if (stageEl) stageEl.textContent = LABELS[sc - 1];
      }
      if (!titleOn && t >= 2.85) { titleOn = true; intro.classList.add('t-title'); }
      if (!tagOn && t >= 3.65) { tagOn = true; intro.classList.add('t-tag'); }
      const p = clamp(t / D);
      if (fill) fill.style.transform = `scaleX(${p})`;
      const pc = Math.round(p * 100);
      if (pc !== pct && pctEl) { pct = pc; pctEl.textContent = pc + '%'; }
      if (ctx) { try { draw(); } catch (e) { /* Canvas is decoration — never block intro */ } }
      if (t >= D) finishIntro(false);
    }
    introRaf = requestAnimationFrame(frame);
  }

  function startIntro() {
    const forced = new URLSearchParams(location.search).has('intro');
    const seen = safe(() => sessionStorage.getItem('ss-intro-seen') === '1', false);
    if (reduceMotion || (seen && !forced)) {
      intro.hidden = true;
      body.classList.remove('intro-active');
      body.classList.add('ready');
      return;
    }
    app.inert = true;
    $('#skip-intro').focus({ preventScroll: true });
    introTimer = setTimeout(() => finishIntro(false), 12000);   // safety net
    document.addEventListener('keydown', function esc(e) {
      if (e.key === 'Escape' && !introDone) finishIntro(true);
      if (introDone) document.removeEventListener('keydown', esc);
    });
    // keep focus inside intro while active
    intro.addEventListener('keydown', e => {
      if (e.key === 'Tab') { e.preventDefault(); $('#skip-intro').focus(); }
    });
    runIntro();
  }
  $('#skip-intro').addEventListener('click', () => finishIntro(true));
  $('#replay-intro').addEventListener('click', () => { location.href = location.pathname + '?intro=1'; });

  // Hero video: play from the start; fall back to gradient if it cannot load or play.
  if (heroVideo) {
    heroVideo.muted = true;
    heroVideo.currentTime = 0;
    heroVideo.play().catch(() => {});
    $$('source', heroVideo).pop()?.addEventListener('error', () => heroVideo.style.display = 'none');
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden) heroVideo.play().catch(() => {});
    });
    // Save battery: pause loop when hero is off-screen.
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(([e]) => {
        e.isIntersecting ? heroVideo.play().catch(() => {}) : heroVideo.pause();
      }).observe($('#home'));
    }
  }
  startIntro();

  /* ---------------- NAVIGATION ---------------- */
  const navbar = $('#navbar');
  const nav = $('#site-nav');
  const toggle = $('.menu-toggle');
  const progress = $('.scroll-progress');

  function setMenu(open) {
    nav.classList.toggle('open', open);
    toggle.setAttribute('aria-expanded', String(open));
    toggle.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
  }
  toggle.addEventListener('click', () => setMenu(!nav.classList.contains('open')));
  $$('a', nav).forEach(a => a.addEventListener('click', () => setMenu(false)));
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && nav.classList.contains('open')) { setMenu(false); toggle.focus(); }
  });

  let ticking = false;
  function onScroll() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => {
      const y = window.scrollY;
      const max = document.documentElement.scrollHeight - innerHeight;
      navbar.classList.toggle('scrolled', y > 40);
      progress.style.transform = `scaleX(${max > 0 ? y / max : 0})`;
      ticking = false;
    });
  }
  addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  // Active link while scrolling
  if ('IntersectionObserver' in window) {
    const links = new Map($$('a', nav).map(a => [a.getAttribute('href').slice(1), a]));
    const spy = new IntersectionObserver(entries => {
      entries.forEach(e => {
        if (!e.isIntersecting) return;
        links.forEach(l => { l.classList.remove('active'); l.removeAttribute('aria-current'); });
        const l = links.get(e.target.id);
        if (l) { l.classList.add('active'); l.setAttribute('aria-current', 'true'); }
      });
    }, { rootMargin: '-45% 0px -50% 0px' });
    $$('main#home, section.section').forEach(s => spy.observe(s));
  }

  /* ---------------- REVEAL + CARD GLOW ---------------- */
  if ('IntersectionObserver' in window && !reduceMotion) {
    const io = new IntersectionObserver(es => es.forEach(e => {
      if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
    }), { threshold: 0.12, rootMargin: '0px 0px -6% 0px' });
    $$('.reveal').forEach(el => io.observe(el));
  } else {
    $$('.reveal').forEach(el => el.classList.add('in'));
  }
  $$('.panel, .card').forEach(el => el.addEventListener('pointermove', e => {
    const r = el.getBoundingClientRect();
    el.style.setProperty('--mx', (e.clientX - r.left) + 'px');
    el.style.setProperty('--my', (e.clientY - r.top) + 'px');
  }));

  /* ---------------- BACKGROUND PARTICLES ---------------- */
  (function particles() {
    const c = $('#bg-canvas');
    if (!c || reduceMotion) return;
    const ctx = c.getContext('2d');
    let w, h, dpr, pts = [], raf;
    const mouse = { x: -999, y: -999 };
    function size() {
      dpr = Math.min(devicePixelRatio || 1, 1.5);
      w = c.width = innerWidth * dpr; h = c.height = innerHeight * dpr;
      const n = Math.max(36, Math.min(80, Math.round(innerWidth * innerHeight / 22000)));
      pts = Array.from({ length: n }, () => ({
        x: Math.random() * w, y: Math.random() * h,
        vx: (Math.random() - .5) * .35 * dpr, vy: (Math.random() - .5) * .35 * dpr, r: (Math.random() * 1.4 + .6) * dpr
      }));
    }
    function frame() {
      ctx.clearRect(0, 0, w, h);
      const link = 130 * dpr;
      for (let i = 0; i < pts.length; i++) {
        const p = pts[i];
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0 || p.x > w) p.vx *= -1;
        if (p.y < 0 || p.y > h) p.vy *= -1;
        const dx = p.x - mouse.x, dy = p.y - mouse.y, d2 = dx * dx + dy * dy;
        if (d2 < 14000 * dpr) { p.x += dx * .002 * dpr * 8; p.y += dy * .002 * dpr * 8; }
        ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, 6.283); ctx.fillStyle = 'rgba(128,241,251,.85)'; ctx.fill();
        for (let j = i + 1; j < pts.length; j++) {
          const q = pts[j], ax = p.x - q.x, ay = p.y - q.y, d = Math.hypot(ax, ay);
          if (d < link) {
            ctx.strokeStyle = `rgba(49,208,226,${(1 - d / link) * .35})`;
            ctx.lineWidth = dpr * .8;
            ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
          }
        }
      }
      raf = document.body.classList.contains('fx-off') ? 0 : requestAnimationFrame(frame);
    }
    size(); frame();
    addEventListener('resize', size);
    addEventListener('fxchange', () => { cancelAnimationFrame(raf); if (document.body.classList.contains('fx-off')) ctx.clearRect(0, 0, w, h); else frame(); });
    addEventListener('pointermove', e => { mouse.x = e.clientX * dpr; mouse.y = e.clientY * dpr; }, { passive: true });
    document.addEventListener('visibilitychange', () => {
      cancelAnimationFrame(raf);
      if (!document.hidden) frame();
    });
  })();

  /* ---------------- DEMO ---------------- */
  const stage = $('#stage');
  const cam = $('#cam');
  const statusEl = $('#status');
  const errBox = $('#error-box');
  const errText = $('#error-text');
  const busy = $('#stage-busy');
  const badge = $('#source-badge');
  const btnStart = $('#btn-start');
  const btnStop = $('#btn-stop');
  const fileInput = $('#file-input');
  let stream = null;
  let objectUrl = null;

  const say = msg => { statusEl.textContent = msg; };
  function showError(msg, kind = 'error') { errText.textContent = msg; errBox.dataset.kind = kind; errBox.hidden = false; say('Something went wrong.'); }
  function clearError() { errBox.hidden = true; }

  function releaseMedia() {
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
    cam.pause();
    cam.srcObject = null;
    cam.removeAttribute('src');
    cam.controls = false;
    cam.load();
  }
  function setIdle(msg = 'Camera is off.') {
    releaseMedia();
    if (window.SS_FrameSender) window.SS_FrameSender.stop();
    stage.dataset.state = 'idle';
    badge.hidden = true;
    btnStart.disabled = false;
    btnStop.disabled = true;
    busy.hidden = true;
    say(msg);
  }

  async function startCamera() {
    clearError();
    if (!navigator.mediaDevices?.getUserMedia) {
      return showError('This browser cannot access the camera here. Open the site over HTTPS or localhost, or upload a sample video instead.', 'unsupported');
    }
    releaseMedia();
    busy.hidden = false;
    btnStart.disabled = true;
    say('Requesting camera…');
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
      cam.srcObject = stream;
      cam.loop = false;
      await cam.play();
      stage.dataset.state = 'live';
      badge.textContent = 'LIVE'; badge.classList.remove('file'); badge.hidden = false;
      btnStop.disabled = false;
      if (window.SS_FrameSender) window.SS_FrameSender.start(cam);
      say('Camera is live — recognition and motion analytics active.');
    } catch (err) {
      releaseMedia();
      stage.dataset.state = 'idle';
      btnStart.disabled = false;
      const msgs = {
        NotAllowedError: 'Camera permission was denied. Allow camera access in your browser settings, then retry.',
        NotFoundError: 'No camera was found on this device.',
        NotReadableError: 'The camera is being used by another app. Close it and retry.',
        OverconstrainedError: 'The camera does not support the requested settings.'
      };
      const kinds = { NotAllowedError: 'denied', NotFoundError: 'unavailable', NotReadableError: 'unavailable', OverconstrainedError: 'unavailable' };
      showError(msgs[err?.name] || 'The camera could not be started.', kinds[err?.name] || 'error');
    } finally {
      busy.hidden = true;
    }
  }

  function loadSample(file) {
    clearError();
    if (!file) return;
    if (!file.type.startsWith('video/')) return showError('Please choose a video file (MP4, WebM or MOV).', 'file');
    releaseMedia();
    objectUrl = URL.createObjectURL(file);
    cam.src = objectUrl;
    cam.loop = true;
    cam.controls = true;
    cam.onerror = () => { setIdle(); showError('That video could not be played. Try an MP4 or WebM file.', 'file'); };
    cam.onloadeddata = () => {
      cam.play().catch(() => {});
      stage.dataset.state = 'file';
      badge.textContent = 'SAMPLE'; badge.classList.add('file'); badge.hidden = false;
      btnStart.disabled = false; btnStop.disabled = false;
      btnStop.textContent = 'Stop video';
      if (window.SS_FrameSender) window.SS_FrameSender.start(cam);
      say(`Loaded “${file.name}” — recognition and motion analytics active.`);
    };
  }

  btnStart.addEventListener('click', () => { btnStop.textContent = 'Stop camera'; startCamera(); });
  btnStop.addEventListener('click', () => { setIdle(stage.dataset.state === 'file' ? 'Video stopped.' : 'Camera stopped.'); btnStop.textContent = 'Stop camera'; });
  fileInput.addEventListener('change', () => { loadSample(fileInput.files[0]); fileInput.value = ''; });
  $('#btn-retry').addEventListener('click', () => { btnStop.textContent = 'Stop camera'; startCamera(); });
  $('#btn-reset').addEventListener('click', () => {
    clearError(); setIdle('Reset. Camera is off.'); btnStop.textContent = 'Stop camera';
    API.setPrediction(null);
  });
  addEventListener('pagehide', releaseMedia);

  /* Prediction panel — hooks for the backend integration phase.
     Nothing here invents values: without a model it stays on "—". */
  const predictEl = $('#predict');
  const API = window.SanketSahayak = {
    /** 'disconnected' | 'loading' | 'ready' | 'error' */
    setModelState(state, note) {
      const text = { disconnected: 'Model not connected', loading: 'Loading…', ready: 'Model ready', error: 'Model error' }[state] || state;
      predictEl.dataset.state = state;
      $('#model-pill').textContent = text;
      if (note) $('#pred-note').textContent = note;
    },
    /** Pass { label: 'HELLO', confidence: 0.92 } or null to clear. */
    setPrediction(result) {
      const has = result && result.label != null;
      $('#pred-label').textContent = has ? result.label : '—';
      const c = has && typeof result.confidence === 'number' ? Math.max(0, Math.min(1, result.confidence)) : null;
      $('#pred-conf').textContent = c == null ? '—' : Math.round(c * 100) + '%';
      $('#meter-fill').style.width = c == null ? '0' : (c * 100) + '%';
      if (has) {
        this.setModelState('ready', 'Live sign recognized with high confidence.');
      } else {
        this.setModelState(predictEl.dataset.state === 'ready' ? 'ready' : 'disconnected',
          predictEl.dataset.state === 'ready' ? 'Waiting for a sign…' : 'Recognition model not connected. Real predictions and confidence will appear here once the classifier is integrated.');
      }
    },
    video: cam
  };

  /* ---------------- HAND LANDMARK PREVIEW ---------------- */
  (function hand() {
    const cv = $('#hand-canvas');
    const ctx = cv.getContext('2d');
    const W = cv.width, H = cv.height;
    // Illustrative layout of MediaPipe's 21 landmarks (normalised x, y).
    const P = [[.50,.92],[.40,.80],[.31,.68],[.25,.55],[.20,.44],[.42,.55],[.40,.38],[.39,.25],[.385,.14],[.50,.52],[.50,.33],[.50,.19],[.50,.07],[.58,.55],[.60,.38],[.61,.26],[.615,.16],[.65,.62],[.69,.48],[.715,.38],[.73,.29]];
    const E = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
    const bx = W * .18, bw = W * .64;
    const X = p => bx + p[0] * bw, Y = p => 14 + p[1] * (H - 28);
    function draw(t) {
      ctx.clearRect(0, 0, W, H);
      ctx.lineWidth = 2; ctx.lineCap = 'round';
      const sway = reduceMotion ? 0 : Math.sin(t / 900) * 4;
      const pos = P.map((p, i) => [X(p) + (i ? sway * (1 - p[1]) * 1.2 : 0), Y(p)]);
      E.forEach(([a, b], k) => {
        const g = ctx.createLinearGradient(pos[a][0], pos[a][1], pos[b][0], pos[b][1]);
        g.addColorStop(0, 'rgba(49,208,226,.9)'); g.addColorStop(1, 'rgba(139,123,255,.9)');
        ctx.strokeStyle = g; ctx.beginPath(); ctx.moveTo(...pos[a]); ctx.lineTo(...pos[b]); ctx.stroke();
      });
      const wave = (t / 140) % 28;
      pos.forEach((p, i) => {
        const on = !reduceMotion && Math.abs(wave - i) < 1.2;
        ctx.beginPath(); ctx.arc(p[0], p[1], on ? 7 : (i % 4 === 0 && i ? 5 : 4), 0, 6.283);
        ctx.fillStyle = on ? '#fff' : (i === 0 ? '#8b7bff' : '#80f1fb'); ctx.shadowColor = '#31d0e2'; ctx.shadowBlur = on ? 18 : 6; ctx.fill(); ctx.shadowBlur = 0;
      });
      if (!reduceMotion) { // scan band sweep
        const sy = (t / 14) % (H + 80) - 40, g = ctx.createLinearGradient(0, sy - 30, 0, sy + 30);
        g.addColorStop(0, 'rgba(49,208,226,0)'); g.addColorStop(.5, 'rgba(49,208,226,.22)'); g.addColorStop(1, 'rgba(49,208,226,0)');
        ctx.fillStyle = g; ctx.fillRect(0, sy - 30, W, 60);
        ctx.fillStyle = 'rgba(128,241,251,.7)'; ctx.fillRect(0, sy, W, 1);
      }
      ctx.fillStyle = 'rgba(185,203,224,.7)'; ctx.font = '11px JetBrains Mono, monospace';
      ctx.fillText('21 landmarks · illustrative', 12, H - 10);
    }
    let raf, vis = true;
    const loop = t => { draw(t); if (vis && !reduceMotion) raf = requestAnimationFrame(loop); };
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(([e]) => { vis = e.isIntersecting; cancelAnimationFrame(raf); if (vis) raf = requestAnimationFrame(loop); }).observe(cv);
    }
    draw(0);
  })();

  /* ---------------- tab helper (roving tabindex + arrows) ---------------- */
  function bindTabs(list, onSelect) {
    const tabs = $$('[role=tab]', list);
    function select(i, focus) {
      tabs.forEach((t, k) => { t.setAttribute('aria-selected', String(k === i)); t.tabIndex = k === i ? 0 : -1; });
      if (focus) tabs[i].focus();
      onSelect(i);
    }
    tabs.forEach((t, i) => {
      t.addEventListener('click', () => select(i, false));
      t.addEventListener('keydown', e => {
        const n = tabs.length; let j = null;
        if (e.key === 'ArrowRight' || e.key === 'ArrowDown') j = (i + 1) % n;
        if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') j = (i - 1 + n) % n;
        if (e.key === 'Home') j = 0; if (e.key === 'End') j = n - 1;
        if (j !== null) { e.preventDefault(); select(j, true); }
      });
    });
    return select;
  }

  /* ---------------- PIPELINE ---------------- */
  (function pipeline() {
    const icons = [
      '<path d="M4 7h11a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2H4zM17 11l4-2v6l-4-2" />',
      '<circle cx="12" cy="6" r="1.6"/><circle cx="7" cy="12" r="1.6"/><circle cx="17" cy="12" r="1.6"/><circle cx="9" cy="19" r="1.6"/><circle cx="15" cy="19" r="1.6"/><path d="M12 6 7 12l2 7M12 6l5 6-2 7M7 12h10"/>',
      '<path d="M5 19 12 5l7 14M8 14h8"/>',
      '<path d="M3 17c3-9 5-9 8 0s5 9 10-10"/>',
      '<path d="M13 3 5 14h6l-1 7 8-11h-6z"/>',
      '<path d="M4 6h5M4 12h5M4 18h5M15 8l5 4-5 4zM9 6l6 2M9 12h6M9 18l6-2"/>'
    ];
    const stages = [
      { t: 'Video input', d: 'A standard camera or a short clip supplies frames. No special sensors or gloves are needed — just a clear view of the hands.', o: 'frames: RGB, ~30 per second' },
      { t: 'MediaPipe hand landmarks', d: 'A hand-tracking model finds each hand and marks 21 key points — wrist, knuckles and fingertips — in every frame.', o: 'landmarks: 21 × (x, y, z)\nper hand, per frame' },
      { t: 'Hand geometry & joint angles', d: 'Landmarks become shape descriptors: distances between fingertips, finger curl, and the angle at every joint. This captures the static hand shape.', o: 'joint_angles: [θ₁ … θₙ]\nfinger_spread, palm_orientation' },
      { t: 'Kinematic analysis', d: 'Tracking landmarks across frames gives the trajectory of the hand: how far it travelled, how fast, and how its speed changes.', o: 'velocity v(t)\nacceleration a(t)\njerk j(t)' },
      { t: 'Dynamic feature estimation', d: 'Physics-inspired quantities summarise the movement — peak speed, smoothness, and kinetic-energy estimates that grow with speed squared.', o: 'peak_speed, smoothness\nKE ∝ v²' },
      { t: 'Machine-learning classification', d: 'Shape features and dynamic features are combined into one vector, and a classifier maps it to a sign. Results await experimental validation.', o: 'sign: ?\nconfidence: ? (pending model)' }
    ];
    const track = $('#pipe-tabs'), panel = $('#pipe-panel'), packet = $('.packet');
    stages.forEach((s, i) => {
      const b = document.createElement('button');
      b.className = 'pipe-step'; b.type = 'button'; b.id = 'pipe-tab-' + i;
      b.setAttribute('role', 'tab'); b.setAttribute('aria-controls', 'pipe-panel');
      b.innerHTML = `<span class="num">0${i + 1}</span><svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[i]}</svg><span class="ttl">${s.t}</span>`;
      track.appendChild(b);
    });
    panel.setAttribute('aria-labelledby', 'pipe-tab-0');
    let userTouched = false, cur = 0, inView = false;
    const select = bindTabs(track, i => {
      cur = i;
      const s = stages[i];
      panel.setAttribute('aria-labelledby', 'pipe-tab-' + i);
      panel.innerHTML = `<div><p class="kicker">Stage ${i + 1} of 6</p><h3>${s.t}</h3><p>${s.d}</p></div><div class="out"><small>Expected output</small>${s.o.replace(/&/g, '&amp;').replace(/</g, '&lt;')}</div>`;
      packet.style.left = (i * 100 / 6) + '%';
    });
    track.addEventListener('pointerdown', () => userTouched = true);
    track.addEventListener('keydown', () => userTouched = true);
    select(0, false);
    if ('IntersectionObserver' in window) new IntersectionObserver(([e]) => inView = e.isIntersecting).observe($('#pipeline-ui'));
    if (!reduceMotion) setInterval(() => { if (!userTouched && inView && !document.hidden) select((cur + 1) % stages.length, false); }, 4200);
  })();

  /* ---------------- PHYSICS ---------------- */
  (function physics() {
    const host = $('#charts');
    const slider = $('#speed-slider');
    const tSlider = $('#time-slider');
    const playBtn = $('#btn-play');
    const W = 600, H = 96, L = 44, R = 10, T = 8, B = 14;
    const PEAK_MAX = 3, D_MAX = PEAK_MAX / 1.875;
    // Minimum-jerk reach over 1 s: distance D, normalised time τ.
    const fns = {
      v: (t, D) => D * (30 * t * t - 60 * t ** 3 + 30 * t ** 4),
      a: (t, D) => D * (60 * t - 180 * t * t + 120 * t ** 3),
      j: (t, D) => D * (60 - 360 * t + 360 * t * t)
    };
    const defs = [
      { k: 'v', name: 'Velocity', unit: 'm/s', color: '#31d0e2', lo: 0, hi: PEAK_MAX * 1.1, fmt: 2 },
      { k: 'a', name: 'Acceleration', unit: 'm/s²', color: '#8b7bff', lo: -D_MAX * 5.78 * 1.1, hi: D_MAX * 5.78 * 1.1, fmt: 1 },
      { k: 'j', name: 'Jerk', unit: 'm/s³', color: '#ffb454', lo: -D_MAX * 60 * 1.1, hi: D_MAX * 60 * 1.1, fmt: 0 }
    ];
    const N = 120;
    const x = t => L + t * (W - L - R);
    defs.forEach(d => {
      const wrap = document.createElement('div'); wrap.className = 'chart';
      wrap.innerHTML = `<h4>${d.name} <span id="rd-${d.k}"></span></h4>`;
      const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `${d.name} over time (illustrative)` });
      d.y = v => T + (1 - (v - d.lo) / (d.hi - d.lo)) * (H - T - B);
      // grid + zero line
      [d.hi, (d.hi + d.lo) / 2, d.lo].forEach((v, i) => {
        svg.appendChild(svgEl('line', { x1: L, x2: W - R, y1: d.y(v), y2: d.y(v), stroke: i === 1 && d.lo < 0 ? 'rgba(255,255,255,.28)' : 'rgba(120,190,230,.14)', 'stroke-dasharray': i === 1 && d.lo < 0 ? '' : '3 4' }));
      });
      const lab = (txt, yy) => { const t = svgEl('text', { x: L - 6, y: yy + 3, 'text-anchor': 'end', fill: '#7f95ad', 'font-size': 10, 'font-family': 'JetBrains Mono, monospace' }); t.textContent = txt; svg.appendChild(t); };
      lab(d.hi.toFixed(0), d.y(d.hi)); lab(d.lo.toFixed(0), d.y(d.lo));
      d.area = svgEl('path', { fill: d.color, opacity: .12 });
      d.path = svgEl('path', { fill: 'none', stroke: d.color, 'stroke-width': 2.4, 'stroke-linejoin': 'round' });
      d.cur = svgEl('line', { y1: T, y2: H - B, stroke: '#fff', 'stroke-width': 1, opacity: .7 });
      d.dot = svgEl('circle', { r: 5, fill: '#fff', stroke: d.color, 'stroke-width': 3 });
      svg.append(d.area, d.path, d.cur, d.dot);
      if (d.k === 'j') {
        [['0 s', L], ['0.5 s', x(.5)], ['1 s', W - R]].forEach(([s, xx], i) => {
          const t = svgEl('text', { x: xx, y: H + 8, 'text-anchor': i === 0 ? 'start' : i === 2 ? 'end' : 'middle', fill: '#7f95ad', 'font-size': 10, 'font-family': 'JetBrains Mono, monospace' }); t.textContent = s; svg.appendChild(t);
        });
        svg.setAttribute('viewBox', `0 0 ${W} ${H + 14}`);
      }
      wrap.appendChild(svg); host.appendChild(wrap);
    });

    let tNow = 0, playing = !reduceMotion, last = 0, D = (+slider.value) / 1.875, raf = 0, inV = !('IntersectionObserver' in window);
    const kick = () => { if (!raf && playing && inV && !document.hidden) raf = requestAnimationFrame(tick); };
    function curves() {
      D = (+slider.value) / 1.875;
      defs.forEach(d => {
        let p = '';
        for (let i = 0; i <= N; i++) { const t = i / N; p += (i ? 'L' : 'M') + x(t).toFixed(1) + ' ' + d.y(fns[d.k](t, D)).toFixed(1); }
        d.path.setAttribute('d', p);
        d.area.setAttribute('d', p + `L${x(1)} ${d.y(0)}L${x(0)} ${d.y(0)}Z`);
      });
      cursor();
    }
    function cursor() {
      defs.forEach(d => {
        const v = fns[d.k](tNow, D), xx = x(tNow);
        d.cur.setAttribute('x1', xx); d.cur.setAttribute('x2', xx);
        d.dot.setAttribute('cx', xx); d.dot.setAttribute('cy', d.y(v));
        $('#rd-' + d.k).textContent = `${v.toFixed(d.fmt)} ${d.unit}`;
      });
      tSlider.value = Math.round(tNow * 1000);
      tSlider.style.setProperty('--fill', Math.min(100, tNow * 100).toFixed(1) + '%');
    }
    function tick(ts) {
      raf = 0;
      if (!playing || !inV || document.hidden) { last = 0; return; }
      if (last) tNow = (tNow + (ts - last) / 3600) % 1.12;
      last = ts;
      cursor2();
      raf = requestAnimationFrame(tick);
    }
    const cursor2 = () => { const was = tNow; tNow = Math.min(tNow, 1); cursor(); tNow = was; };
    tSlider.addEventListener('input', () => { playing = false; setPlay(); tNow = tSlider.value / 1000; cursor(); });
    function setPlay() { playBtn.textContent = playing ? 'Pause' : 'Play'; playBtn.setAttribute('aria-pressed', String(playing)); last = 0; kick(); }
    playBtn.addEventListener('click', () => { playing = !playing; setPlay(); });
    slider.addEventListener('input', () => { curves(); ke(); });
    setPlay(); curves();
    if (!inV) new IntersectionObserver(([e]) => { inV = e.isIntersecting; kick(); }).observe(host);
    document.addEventListener('visibilitychange', kick);
    kick();

    /* Kinetic energy: KE = ½ m v², m = 0.4 kg (illustrative) */
    const M = 0.4, ks = $('#ke-chart');
    const kw = 320, kh = 190, kl = 36, kb = 26, kt = 12, kr = 12, KE_MAX = .5 * M * PEAK_MAX ** 2;
    const kx = v => kl + v / PEAK_MAX * (kw - kl - kr), ky = e => kt + (1 - e / KE_MAX) * (kh - kt - kb);
    let cp = '';
    for (let i = 0; i <= 60; i++) { const v = i / 60 * PEAK_MAX; cp += (i ? 'L' : 'M') + kx(v).toFixed(1) + ' ' + ky(.5 * M * v * v).toFixed(1); }
    ks.append(
      svgEl('line', { x1: kl, x2: kl, y1: kt, y2: kh - kb, stroke: 'rgba(120,190,230,.3)' }),
      svgEl('line', { x1: kl, x2: kw - kr, y1: kh - kb, y2: kh - kb, stroke: 'rgba(120,190,230,.3)' }),
      svgEl('path', { d: cp, fill: 'none', stroke: '#31d0e2', 'stroke-width': 2.5 }),
      svgEl('path', { d: cp + `L${kx(PEAK_MAX)} ${ky(0)}L${kx(0)} ${ky(0)}Z`, fill: '#31d0e2', opacity: .1 })
    );
    const ref = svgEl('circle', { cx: kx(1), cy: ky(.5 * M), r: 4, fill: 'none', stroke: '#8b7bff', 'stroke-width': 2 });
    const dropV = svgEl('line', { stroke: '#fff', 'stroke-dasharray': '3 3', opacity: .5 });
    const dropH = svgEl('line', { stroke: '#fff', 'stroke-dasharray': '3 3', opacity: .5 });
    const mk = svgEl('circle', { r: 6, fill: '#fff', stroke: '#31d0e2', 'stroke-width': 3 });
    const tx = (txt, xx, yy, anchor = 'middle') => { const t = svgEl('text', { x: xx, y: yy, 'text-anchor': anchor, fill: '#7f95ad', 'font-size': 10, 'font-family': 'JetBrains Mono, monospace' }); t.textContent = txt; return t; };
    ks.append(ref, dropV, dropH, mk, tx('speed v (m/s) →', kw - kr, kh - 6, 'end'), tx('KE (J)', kl + 4, kt + 8, 'start'), tx('0', kl, kh - 12), tx('3', kx(3), kh - 12));
    function ke() {
      const v = +slider.value, e = .5 * M * v * v;
      mk.setAttribute('cx', kx(v)); mk.setAttribute('cy', ky(e));
      dropV.setAttribute('x1', kx(v)); dropV.setAttribute('x2', kx(v)); dropV.setAttribute('y1', ky(e)); dropV.setAttribute('y2', ky(0));
      dropH.setAttribute('x1', kx(0)); dropH.setAttribute('x2', kx(v)); dropH.setAttribute('y1', ky(e)); dropH.setAttribute('y2', ky(e));
      $('#speed-out').textContent = v.toFixed(2) + ' m/s';
      $('#ke-out').textContent = e.toFixed(2) + ' J';
      $('#ke-ratio').textContent = (v * v).toFixed(2) + '×';
    }
    ke();
  })();

  /* ---------------- COMPARE ---------------- */
  (function compare() {
    const levels = { Strong: 90, Partial: 50, Weak: 14 };
    const rows = ['Tells different hand shapes apart', 'Tells same shape, different motion apart', 'Captures speed and rhythm', 'Easy to explain and inspect'];
    const items = [
      { n: 'Shape only', h: 'Static geometry', d: 'Uses landmark positions and joint angles from each frame — the hand’s pose.', good: ['Clear for signs defined by pose', 'Simple, fast features'], bad: ['Blind to motion', 'Confuses signs that share a hand shape'], m: ['Strong', 'Weak', 'Weak', 'Strong'] },
      { n: 'Dynamics only', h: 'Motion signature', d: 'Uses velocity, acceleration, jerk and energy of the moving hand, without its detailed pose.', good: ['Separates signs by how they move', 'Captures speed and smoothness'], bad: ['Cannot tell different hand shapes apart', 'Needs reliable tracking over time'], m: ['Weak', 'Strong', 'Strong', 'Partial'] },
      { n: 'Combined', h: 'Shape + motion', d: 'Joins geometry and dynamics into a single feature vector for the classifier — the Sankeet Shayak approach.', good: ['Uses both cues together', 'Each feature group can be inspected'], bad: ['Larger feature set to validate', 'Results pending experimental validation'], m: ['Strong', 'Strong', 'Strong', 'Strong'] }
    ];
    const seg = $('#cmp-tabs'), panel = $('#cmp-panel');
    items.forEach((it, i) => {
      const b = document.createElement('button'); b.type = 'button'; b.id = 'cmp-tab-' + i;
      b.setAttribute('role', 'tab'); b.setAttribute('aria-controls', 'cmp-panel'); b.textContent = it.n; seg.appendChild(b);
    });
    const select = bindTabs(seg, i => {
      const it = items[i];
      panel.setAttribute('aria-labelledby', 'cmp-tab-' + i);
      panel.innerHTML = `<div><h4>${it.h}</h4><h3>${it.n}</h3><p>${it.d}</p>
        <ul>${it.good.map(g => `<li>${g}</li>`).join('')}</ul>
        <ul style="color:#ffd9a8">${it.bad.map(g => `<li>${g}</li>`).join('')}</ul></div>
        <div class="meters"><h4>Conceptual strength — not measured</h4>${rows.map((r, k) => `<div class="mrow"><div><span>${r}</span><b>${it.m[k]}</b></div><div class="mbar"><i data-w="${levels[it.m[k]]}"></i></div></div>`).join('')}</div>`;
      requestAnimationFrame(() => requestAnimationFrame(() => $$('.mbar i', panel).forEach(el => el.style.width = el.dataset.w + '%')));
    });
    select(2, false);
  })();
  /* =========================================================
     UPGRADE LAYER — futuristic FX (additive)
     ========================================================= */
  const fxStored = safe(() => localStorage.getItem('ss-fx'), null);
  const fx = { on: !reduceMotion && fxStored !== 'off' };
  const fxBtn = $('#fx-toggle');
  function applyFx() {
    body.classList.toggle('fx-off', !fx.on);
    fxBtn.setAttribute('aria-pressed', String(fx.on));
    dispatchEvent(new Event('fxchange'));
  }
  fxBtn.addEventListener('click', () => { fx.on = !fx.on; safe(() => localStorage.setItem('ss-fx', fx.on ? 'on' : 'off')); applyFx(); });
  if (reduceMotion) fxBtn.hidden = true;
  applyFx();

  const finePointer = matchMedia('(hover: hover) and (pointer: fine)').matches;

  /* ---- custom cursor + spotlight + click shockwave ---- */
  (function cursor() {
    if (!finePointer || reduceMotion) return;
    const ring = $('.cursor-ring'), dot = $('.cursor-dot'), glow = $('.cursor-glow');
    let mx = innerWidth / 2, my = innerHeight / 2, rx = mx, ry = my, gx = mx, gy = my, raf = 0;
    function loop() {
      rx += (mx - rx) * .2; ry += (my - ry) * .2; gx += (mx - gx) * .08; gy += (my - gy) * .08;
      ring.style.transform = `translate3d(${rx}px,${ry}px,0)`;
      glow.style.transform = `translate3d(${gx}px,${gy}px,0)`;
      raf = (Math.abs(mx - rx) + Math.abs(my - ry) + Math.abs(mx - gx) + Math.abs(my - gy) > .5) ? requestAnimationFrame(loop) : 0;
    }
    addEventListener('pointermove', e => {
      if (e.pointerType !== 'mouse') return;
      mx = e.clientX; my = e.clientY;
      dot.style.transform = `translate3d(${mx}px,${my}px,0)`;
      body.classList.add('cursor-on');
      ring.classList.toggle('hot', !!e.target.closest?.('a, button, input, label, [role=tab]'));
      if (!raf) raf = requestAnimationFrame(loop);
    }, { passive: true });
    document.addEventListener('mouseleave', () => body.classList.remove('cursor-on'));
    addEventListener('pointerdown', e => {
      if (!fx.on || e.pointerType !== 'mouse') return;
      const s = document.createElement('div'); s.className = 'shock';
      s.style.left = e.clientX + 'px'; s.style.top = e.clientY + 'px';
      document.body.appendChild(s); s.addEventListener('animationend', () => s.remove());
    });
  })();

  /* ---- magnetic buttons + 3D tilt cards ---- */
  (function interactions() {
    if (!finePointer || reduceMotion) return;
    $$('.button').forEach(b => {
      b.addEventListener('pointermove', e => {
        if (!fx.on || b.disabled) return;
        const r = b.getBoundingClientRect();
        b.style.transform = `translate(${(e.clientX - r.left - r.width / 2) * .22}px, ${(e.clientY - r.top - r.height / 2) * .32 - 2}px)`;
      });
      b.addEventListener('pointerleave', () => { b.style.transform = ''; });
    });
    $$('.card, .person').forEach(el => {
      el.addEventListener('pointermove', e => {
        if (!fx.on) return;
        const r = el.getBoundingClientRect(), px = (e.clientX - r.left) / r.width - .5, py = (e.clientY - r.top) / r.height - .5;
        el.style.transitionDelay = '0s';
        el.style.transform = `perspective(900px) rotateX(${(-py * 3).toFixed(2)}deg) rotateY(${(px * 4).toFixed(2)}deg) translateY(-6px)`;
      });
      el.addEventListener('pointerleave', () => { el.style.transform = ''; });
    });
  })();

  /* ---- text scramble (kickers + hero accents) ---- */
  const GLYPHS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<>/[]{}=+*#';
  function scramble(el, ms = 800) {
    const final = el.dataset.final || el.textContent; el.dataset.final = final;
    if (reduceMotion || !fx.on) { el.textContent = final; return; }
    const t0 = performance.now();
    (function step(now) {
      const p = Math.min(1, (now - t0) / ms); let out = '';
      for (let i = 0; i < final.length; i++) out += (final[i] === ' ' || i < p * final.length) ? final[i] : GLYPHS[Math.random() * GLYPHS.length | 0];
      el.textContent = p < 1 ? out : final;
      if (p < 1) requestAnimationFrame(step);
    })(t0);
  }
  if ('IntersectionObserver' in window) {
    const so = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { scramble(e.target); so.unobserve(e.target); } }), { threshold: .8 });
    $$('.section-head .kicker').forEach(k => so.observe(k));
  }

  /* ---- hero: counters, HUD clock, scroll parallax ---- */
  function onReady(fn) {
    if (body.classList.contains('ready')) return fn();
    const mo = new MutationObserver(() => { if (body.classList.contains('ready')) { mo.disconnect(); fn(); } });
    mo.observe(body, { attributes: true, attributeFilter: ['class'] });
  }
  onReady(() => {
    $$('[data-count]').forEach(el => {
      const end = +el.dataset.count; if (reduceMotion) return;
      const t0 = performance.now() + 900;
      (function tick(now) {
        const p = Math.max(0, Math.min(1, (now - t0) / 1200));
        el.textContent = Math.round(end * (1 - Math.pow(1 - p, 3)));
        if (p < 1) requestAnimationFrame(tick);
      })(performance.now());
    });
  });
  (function hudClock() {
    const el = $('#hud-time'); if (!el) return;
    const t0 = Date.now(), pad = n => String(n).padStart(2, '0');
    setInterval(() => {
      if (document.hidden) return;
      const s = (Date.now() - t0) / 1000 | 0;
      el.textContent = `${pad(s / 3600 | 0)}:${pad(s / 60 % 60 | 0)}:${pad(s % 60)}`;
    }, 1000);
  })();
  (function heroParallax() {
    const hero = $('#home'); let tk = false;
    addEventListener('scroll', () => {
      if (tk) return; tk = true;
      requestAnimationFrame(() => { hero.style.setProperty('--hp', Math.min(1, Math.max(0, scrollY / (innerHeight * .8))).toFixed(3)); tk = false; });
    }, { passive: true });
  })();

  /* ---- stage HUD mirrors the real source state ---- */
  (function stageHud() {
    const hud = $('#stage-hud'), names = { idle: 'NONE', live: 'WEBCAM', file: 'FILE' };
    const upd = () => { hud.textContent = 'SRC · ' + (names[stage.dataset.state] || 'NONE'); };
    new MutationObserver(upd).observe(stage, { attributes: true, attributeFilter: ['data-state'] });
    upd();
  })();

  /* ---- hero: rotating 3D hand-landmark model (illustrative, NOT detection) ---- */
  (function heroHand() {
    const cv = $('#hero-hand'); if (!cv) return;
    const ctx = cv.getContext('2d');
    const P = [[.50,.92],[.40,.80],[.31,.68],[.25,.55],[.20,.44],[.42,.55],[.40,.38],[.39,.25],[.385,.14],[.50,.52],[.50,.33],[.50,.19],[.50,.07],[.58,.55],[.60,.38],[.61,.26],[.615,.16],[.65,.62],[.69,.48],[.715,.38],[.73,.29]];
    const E = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
    const TIPS = [4, 8, 12, 16, 20];
    const NAMES = { 0: 'WRIST', 4: 'THUMB_TIP', 8: 'INDEX_TIP', 12: 'MIDDLE_TIP', 16: 'RING_TIP', 20: 'PINKY_TIP' };
    const V = P.map((p, i) => ({ x: (p[0] - .5) * 1.5, y: (p[1] - .5) * 1.7, z: (TIPS.includes(i) ? -.16 : 0) + Math.sin(i * 1.7) * .07 + (p[0] - .5) * .5 }));
    let w, h, dpr, mx = 0, my = 0, sx = 0, sy = 0, raf = 0, vis = true;
    function size() { dpr = Math.min(devicePixelRatio || 1, 2); const r = cv.getBoundingClientRect(); w = cv.width = Math.max(1, r.width * dpr); h = cv.height = Math.max(1, r.height * dpr); }
    addEventListener('pointermove', e => { mx = e.clientX / innerWidth * 2 - 1; my = e.clientY / innerHeight * 2 - 1; }, { passive: true });
    function draw(t) {
      sx += (mx - sx) * .05; sy += (my - sy) * .05;
      const yaw = (fx.on ? Math.sin(t / 2600) * .6 : 0) + sx * .55, pitch = sy * .25 - .08;
      const cy = Math.cos(yaw), sY = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
      const s = Math.min(w, h) * .4, cx = w / 2, cyy = h / 2;
      ctx.clearRect(0, 0, w, h);
      // orbit rings
      ctx.lineWidth = dpr; ctx.setLineDash([4 * dpr, 7 * dpr]);
      for (let k = 0; k < 3; k++) {
        const rr = s * (1.15 + k * .2), rot = (fx.on ? t / 5000 : 0) * (k % 2 ? -1 : 1) + k;
        ctx.strokeStyle = k === 1 ? 'rgba(139,123,255,.28)' : 'rgba(49,208,226,.24)';
        ctx.beginPath(); ctx.ellipse(cx, cyy, rr, rr * .3, rot, 0, 6.283); ctx.stroke();
        if (k === 0) {
          const a = t / 900, ox = Math.cos(a) * rr, oy = Math.sin(a) * rr * .3, c = Math.cos(rot), sn = Math.sin(rot);
          ctx.setLineDash([]); ctx.fillStyle = '#fff'; ctx.shadowColor = '#31d0e2'; ctx.shadowBlur = 14 * dpr;
          ctx.beginPath(); ctx.arc(cx + ox * c - oy * sn, cyy + ox * sn + oy * c, 3 * dpr, 0, 6.283); ctx.fill(); ctx.shadowBlur = 0; ctx.setLineDash([4 * dpr, 7 * dpr]);
        }
      }
      ctx.setLineDash([]);
      // project
      const pr = V.map(v => {
        const x1 = v.x * cy + v.z * sY, z1 = -v.x * sY + v.z * cy;
        const y2 = v.y * cp - z1 * sp, z2 = v.y * sp + z1 * cp, f = 2.6 / (2.6 - z2);
        return { x: cx + x1 * f * s, y: cyy + y2 * f * s, f, z: z2 };
      });
      // palm fill
      ctx.beginPath(); [0, 1, 5, 9, 13, 17].forEach((i, k) => k ? ctx.lineTo(pr[i].x, pr[i].y) : ctx.moveTo(pr[i].x, pr[i].y)); ctx.closePath();
      ctx.fillStyle = 'rgba(49,208,226,.07)'; ctx.fill();
      // bones
      ctx.lineCap = 'round';
      E.forEach(([a, b]) => {
        const A = pr[a], B = pr[b], al = Math.max(.25, Math.min(1, .55 + (A.f + B.f - 2) * 1.3));
        const g = ctx.createLinearGradient(A.x, A.y, B.x, B.y);
        g.addColorStop(0, `rgba(49,208,226,${al})`); g.addColorStop(1, `rgba(139,123,255,${al})`);
        ctx.strokeStyle = g; ctx.lineWidth = 2.2 * dpr * ((A.f + B.f) / 2);
        ctx.beginPath(); ctx.moveTo(A.x, A.y); ctx.lineTo(B.x, B.y); ctx.stroke();
      });
      // joints + signal
      const wave = (t / 130) % 30;
      pr.forEach((p, i) => {
        const on = fx.on && Math.abs(wave - i) < 1.3, r = (TIPS.includes(i) ? 5 : 3.4) * dpr * p.f * (on ? 1.6 : 1);
        ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, 6.283);
        ctx.fillStyle = on ? '#fff' : i === 0 ? '#8b7bff' : '#80f1fb'; ctx.shadowColor = '#31d0e2'; ctx.shadowBlur = (on ? 22 : 9) * dpr; ctx.fill(); ctx.shadowBlur = 0;
      });
      // labels
      ctx.font = `${10 * dpr}px JetBrains Mono, monospace`; ctx.textBaseline = 'middle';
      Object.keys(NAMES).forEach(k => {
        const p = pr[k], right = p.x >= cx, dx = (right ? 1 : -1) * 34 * dpr, dy = (p.y < cyy ? -1 : 1) * 14 * dpr;
        ctx.strokeStyle = 'rgba(128,241,251,.4)'; ctx.lineWidth = dpr;
        ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(p.x + dx * .5, p.y + dy); ctx.lineTo(p.x + dx, p.y + dy); ctx.stroke();
        ctx.fillStyle = 'rgba(214,246,251,.85)'; ctx.textAlign = right ? 'left' : 'right';
        ctx.fillText(NAMES[k], p.x + dx + (right ? 5 : -5) * dpr, p.y + dy);
      });
      ctx.textAlign = 'left'; ctx.fillStyle = 'rgba(185,203,224,.6)';
      ctx.fillText('ILLUSTRATIVE 3D MODEL · NOT LIVE DETECTION', 6 * dpr, h - 8 * dpr);
    }
    function loop(t) { draw(t); raf = (vis && fx.on && !document.hidden) ? requestAnimationFrame(loop) : 0; }
    const kick = () => { if (!raf) raf = requestAnimationFrame(loop); };
    size(); draw(0);
    addEventListener('resize', () => { size(); draw(performance.now()); });
    addEventListener('fxchange', () => { if (fx.on) kick(); else { cancelAnimationFrame(raf); raf = 0; draw(performance.now()); } });
    document.addEventListener('visibilitychange', () => { if (!document.hidden) kick(); });
    if ('IntersectionObserver' in window) new IntersectionObserver(([e]) => { vis = e.isIntersecting; if (vis) kick(); }).observe($('#home'));
    kick();
  })();

  /* =========================================================
     UPGRADE v2 — cinematic depth, slider fill, chart draw-in
     (additive; every effect respects the FX toggle / reduced motion)
     ========================================================= */
  (function heroDepth() {            // mouse parallax: video, glow, grid and hand move at different depths
    const hero = $('#home'); if (!hero || reduceMotion || !finePointer) return;
    let tx = 0, ty = 0, x = 0, y = 0, raf = 0;
    const step = () => {
      x += (tx - x) * .08; y += (ty - y) * .08;
      hero.style.setProperty('--px', x.toFixed(3)); hero.style.setProperty('--py', y.toFixed(3));
      raf = Math.abs(tx - x) + Math.abs(ty - y) > .002 ? requestAnimationFrame(step) : 0;
    };
    hero.addEventListener('pointermove', e => {
      if (!fx.on || e.pointerType !== 'mouse') return;
      const r = hero.getBoundingClientRect();
      tx = (e.clientX - r.left) / r.width * 2 - 1; ty = (e.clientY - r.top) / r.height * 2 - 1;
      if (!raf) raf = requestAnimationFrame(step);
    }, { passive: true });
    hero.addEventListener('pointerleave', () => { tx = ty = 0; if (!raf) raf = requestAnimationFrame(step); });
  })();

  (function sectionDepth() {         // gentle parallax on each section's coloured light
    if (reduceMotion) return;
    const secs = $$('.section'); let tk = false;
    addEventListener('scroll', () => {
      if (tk) return; tk = true;
      requestAnimationFrame(() => {
        secs.forEach(s => { const r = s.getBoundingClientRect(); if (r.bottom < 0 || r.top > innerHeight) return; s.style.setProperty('--sy', ((r.top + r.height / 2 - innerHeight / 2) / innerHeight).toFixed(3)); });
        tk = false;
      });
    }, { passive: true });
  })();

  (function sliderFill() {           // filled track on every range input
    $$('input[type=range]').forEach(r => {
      const set = () => r.style.setProperty('--fill', ((r.value - r.min) / (r.max - r.min) * 100).toFixed(1) + '%');
      r.addEventListener('input', set); set();
    });
  })();

  (function chartDrawIn() {          // motion curves draw themselves once when first seen
    const host = $('#charts'); if (!host || reduceMotion || !('IntersectionObserver' in window)) return;
    new IntersectionObserver(([e], o) => {
      if (!e.isIntersecting) return; o.disconnect();
      $$('path[fill="none"]', host).forEach((p, i) => {
        const L = p.getTotalLength();
        p.style.cssText = `stroke-dasharray:${L};stroke-dashoffset:${L};transition:stroke-dashoffset 1.6s ${i * .15}s cubic-bezier(.2,.7,.2,1)`;
        requestAnimationFrame(() => requestAnimationFrame(() => { p.style.strokeDashoffset = 0; }));
        setTimeout(() => { p.style.cssText = ''; }, 2300);
      });
    }, { threshold: .3 }).observe(host);
  })();
})();

(() => {
  /* =========================================================
     PROJECT CREDITS MODAL (frontend only — no backend calls)
     Fade/scale open + close, Escape / backdrop / button to close,
     focus trap, focus restore, background made inert.
     ========================================================= */
  'use strict';
  const overlay = document.getElementById('credits-modal');
  if (!overlay) return;
  const dialog = overlay.querySelector('.credits-dialog');
  const app = document.getElementById('app');
  const openers = [...document.querySelectorAll('[data-credits-open]')];
  const closers = [...overlay.querySelectorAll('[data-credits-close]')];
  const closeBtn = overlay.querySelector('.credits-close');
  const menuToggle = document.querySelector('.menu-toggle');
  const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select, textarea, [tabindex]:not([tabindex="-1"])';
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let lastFocus = null, closing = false;

  function open(trigger) {
    // never open while the intro is still playing
    if (document.body.classList.contains('intro-active') || !overlay.hidden) return;
    lastFocus = trigger || document.activeElement;
    // close the mobile menu if it is open
    if (menuToggle && menuToggle.getAttribute('aria-expanded') === 'true') menuToggle.click();
    overlay.hidden = false;
    document.body.classList.add('modal-open');
    if (app) app.inert = true;
    openers.forEach(b => b.setAttribute('aria-expanded', 'true'));
    void overlay.offsetWidth;                       // flush so the transition runs
    overlay.classList.add('open');
    dialog.scrollTop = 0;
    (closeBtn || dialog).focus({ preventScroll: true });
  }

  function close() {
    if (overlay.hidden || closing) return;
    closing = true;
    overlay.classList.remove('open');
    const finish = () => {
      overlay.hidden = true;
      document.body.classList.remove('modal-open');
      if (app) app.inert = false;
      openers.forEach(b => b.setAttribute('aria-expanded', 'false'));
      closing = false;
      const inMenu = lastFocus && lastFocus.closest && lastFocus.closest('#site-nav');
      if (inMenu && menuToggle) menuToggle.focus({ preventScroll: true });   // the dropdown menu is closed again
      else if (lastFocus && document.contains(lastFocus) && lastFocus.offsetParent !== null) lastFocus.focus({ preventScroll: true });
      else (openers.find(b => b.offsetParent !== null) || document.body).focus({ preventScroll: true });
    };
    if (reduce) finish(); else setTimeout(finish, 380);
  }

  openers.forEach(b => b.addEventListener('click', () => open(b)));
  closers.forEach(b => b.addEventListener('click', close));
  // click on the dimmed backdrop (not inside the dialog) closes it
  overlay.addEventListener('mousedown', e => { if (e.target === overlay) close(); });

  document.addEventListener('keydown', e => {
    if (overlay.hidden) return;
    if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); close(); return; }
    if (e.key !== 'Tab') return;
    const items = [...dialog.querySelectorAll(FOCUSABLE)].filter(el => el.offsetParent !== null);
    if (!items.length) { e.preventDefault(); dialog.focus(); return; }
    const first = items[0], last = items[items.length - 1];
    if (e.shiftKey && (document.activeElement === first || document.activeElement === dialog)) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  }, true);

  // optional deep-link: index.html#credits opens the modal once the intro is done
  if (location.hash === '#credits') {
    const t = setInterval(() => { if (!document.body.classList.contains('intro-active')) { clearInterval(t); open(); } }, 400);
    setTimeout(() => clearInterval(t), 30000);
  }
})();