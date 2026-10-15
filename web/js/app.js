import { Analyzer, REF_LEN_M, effortBoundaries } from './physics.js';
import { classifyShape } from './shape.js';
import { makeTapSequence, templateWorld, toNormalized, SIM_W, SIM_H } from './sim.js';

/* ------------------------------------------------------------------ setup */
const TASKS_VERSION = '0.10.14';
const CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VERSION}`;
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const CONNECTIONS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
const HIST_SECONDS = 6;

const $ = id => document.getElementById(id);
const el = {};
['video','overlay','stage','stageMsg','badge','status','alert','btnCamera','btnStop','fileInput','cameraSelect','simAll',
 'mirror','sens','sensVal','mass','delegate','t-shape','t-speed','t-accel','t-jerk','t-ke','t-force','t-torque',
 'chartSpeed','chartKE','sLabel','sBar','sMeta','logBody','btnCsv','btnClear'].forEach(id => { el[id] = $(id); });
const ctx = el.overlay.getContext('2d');

const analyzer = new Analyzer();
let mode = 'idle';            // idle | camera | video | sim
let session = 0;              // bumped on every start/stop so stale async work can bail out
let stream = null, objectUrl = null, vfcId = 0, rafId = 0, simRaf = 0;
let visionMod = null, fileset = null, landmarker = null, landmarkerDelegate = null, loading = null, loadingDelegate = null;
let lastTs = 0, lastMediaT = -1, scaleEMA = null, lastSeenT = -Infinity, fallingBack = false;
let hist = [], log = [], strokeNo = 0, frameCount = 0, fpsStamp = performance.now(), simShapeTag = null;

/* ------------------------------------------------------------------ helpers */
const fmt = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '—');
function setStatus(text, kind = '') { el.status.textContent = text; el.status.className = 'pill ' + kind; }
function showAlert(msg) { el.alert.textContent = msg || ''; el.alert.hidden = !msg; }
function setBusy(running) { el.btnStop.disabled = !running; }

function friendlyError(e) {
  const n = e?.name || '';
  if (n === 'InsecureContext') return 'The camera needs a secure page. Open this app via https:// (e.g. your Vercel URL) or http://localhost — not file:// or a LAN IP address.';
  if (n === 'NotSupported') return 'This browser does not support camera access (getUserMedia). Use a recent Chrome, Edge, Firefox or Safari.';
  if (n === 'NotAllowedError' || n === 'PermissionDeniedError' || n === 'SecurityError')
    return 'Camera permission was denied.\n• Click the camera/lock icon in the address bar and choose Allow, then press Start again.\n• macOS: System Settings → Privacy & Security → Camera → enable your browser.\n• Windows: Settings → Privacy & security → Camera → allow desktop apps / your browser.\n• Linux: check the browser permission and that your user can read /dev/video0 (group "video").';
  if (n === 'NotFoundError' || n === 'OverconstrainedError' || n === 'DevicesNotFoundError') return 'No camera was found. Plug in a webcam, pick it in the dropdown, or use "Video file" / "Simulate".';
  if (n === 'NotReadableError' || n === 'TrackStartError') return 'The camera is busy or blocked. Close other apps using it (Zoom, Teams, OBS, another tab) and try again.';
  if (n === 'ModelLoadError') return `Could not download the hand-tracking model/runtime (needs internet on first load).\n${e.message}`;
  return `${n ? n + ': ' : ''}${e?.message || e}`;
}
const err = (name, message) => Object.assign(new Error(message), { name });

/* ------------------------------------------------------------------ MediaPipe */
async function getLandmarker(delegate) {
  if (landmarker && landmarkerDelegate === delegate) return landmarker;
  if (loading && loadingDelegate === delegate) return loading;
  loadingDelegate = delegate;
  loading = (async () => {
    try {
      setStatus('Loading hand model…', 'warn');
      visionMod ??= await import(`${CDN}/vision_bundle.mjs`);
      fileset ??= await visionMod.FilesetResolver.forVisionTasks(`${CDN}/wasm`);
      const lm = await visionMod.HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate },
        runningMode: 'VIDEO', numHands: 1,
        minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.5, minTrackingConfidence: 0.5,
      });
      if (landmarker) { try { landmarker.close(); } catch { /* ignore */ } }
      landmarker = lm; landmarkerDelegate = delegate;
      return lm;
    } catch (e) {
      throw err('ModelLoadError', e?.message || String(e));
    } finally { loading = null; }
  })();
  return loading;
}

/* ------------------------------------------------------------------ sources */
function stopSource() {
  session++;
  if (vfcId && el.video.cancelVideoFrameCallback) { try { el.video.cancelVideoFrameCallback(vfcId); } catch { /* ignore */ } }
  vfcId = 0;
  cancelAnimationFrame(rafId); cancelAnimationFrame(simRaf); rafId = simRaf = 0;
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  el.video.pause(); el.video.srcObject = null; el.video.removeAttribute('src'); el.video.load();
  if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
  mode = 'idle'; scaleEMA = null; lastSeenT = -Infinity; lastMediaT = -1; simShapeTag = null;
  analyzer.reset();
  ctx.clearRect(0, 0, el.overlay.width, el.overlay.height);
  el.badge.hidden = true; el.stageMsg.hidden = false; el.video.classList.remove('hidden');
  setBusy(false);
}

function prepareStage(m) {
  mode = m;
  el.stage.classList.toggle('mirror', m === 'camera' && el.mirror.checked);
  el.stageMsg.hidden = true; el.badge.hidden = false;
  el.badge.textContent = { camera: '● LIVE', video: '▶ VIDEO FILE', sim: '⚙ SIMULATION' }[m];
  setBusy(true);
}

function resizeToVideo() {
  const w = el.video.videoWidth, h = el.video.videoHeight;
  if (w && h) { el.overlay.width = w; el.overlay.height = h; el.stage.style.aspectRatio = `${w} / ${h}`; }
}

async function openCamera(deviceId) {
  if (!window.isSecureContext) throw err('InsecureContext', 'insecure');
  if (!navigator.mediaDevices?.getUserMedia) throw err('NotSupported', 'unsupported');
  const base = { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60 } };
  const tries = [];
  if (deviceId) tries.push({ video: { ...base, deviceId: { exact: deviceId } }, audio: false });
  tries.push({ video: { ...base, facingMode: 'user' }, audio: false }, { video: true, audio: false });
  let last;
  for (const c of tries) {
    try { return await navigator.mediaDevices.getUserMedia(c); }
    catch (e) { last = e; if (e.name === 'NotAllowedError' || e.name === 'SecurityError') break; }
  }
  throw last;
}

async function refreshDevices(activeId) {
  try {
    const devs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
    el.cameraSelect.innerHTML = '<option value="">Default camera</option>' +
      devs.map((d, i) => `<option value="${d.deviceId}">${(d.label || `Camera ${i + 1}`).replace(/</g, '&lt;')}</option>`).join('');
    if (activeId && devs.some(d => d.deviceId === activeId)) el.cameraSelect.value = activeId;
  } catch { /* non-fatal */ }
}

async function startCamera() {
  stopSource(); showAlert('');
  const my = session;
  try {
    setStatus('Requesting camera…', 'warn');
    let camStream = null;
    const camP = openCamera(el.cameraSelect.value).then(s => { camStream = s; return s; });
    const lmP = getLandmarker(el.delegate.value);
    const [c, l] = await Promise.allSettled([camP, lmP]);
    if (my !== session) { camStream?.getTracks().forEach(t => t.stop()); return; }
    if (c.status === 'rejected' || l.status === 'rejected') {
      camStream?.getTracks().forEach(t => t.stop());
      throw c.status === 'rejected' ? c.reason : l.reason;
    }
    stream = camStream;
    el.video.srcObject = stream; el.video.muted = true;
    await el.video.play();
    if (my !== session) return;
    const track = stream.getVideoTracks()[0];
    track.addEventListener('ended', () => { if (my === session) { stopSource(); setStatus('Camera disconnected', 'err'); showAlert('The camera was disconnected or stopped by the system.'); } });
    resizeToVideo();
    await refreshDevices(track.getSettings().deviceId);
    prepareStage('camera'); resetRun();
    const s = track.getSettings();
    setStatus(`Camera live ${s.width || ''}×${s.height || ''}`, 'ok');
    scheduleNext();
  } catch (e) {
    if (my !== session) return;
    stopSource(); setStatus('Camera unavailable', 'err'); showAlert(friendlyError(e));
  }
}

async function startVideoFile(file) {
  stopSource(); showAlert('');
  const my = session;
  try {
    objectUrl = URL.createObjectURL(file);
    el.video.srcObject = null; el.video.src = objectUrl; el.video.muted = true; el.video.loop = false;
    await new Promise((res, rej) => {
      el.video.onloadedmetadata = res;
      el.video.onerror = () => rej(err('VideoError', 'This video format is not supported by your browser (try MP4/H.264 or WebM).'));
    });
    await getLandmarker(el.delegate.value);
    if (my !== session) return;
    resizeToVideo(); prepareStage('video'); resetRun();
    el.video.onended = () => { if (my === session) setStatus('Video finished', ''); };
    await el.video.play();
    setStatus('Analysing video…', 'ok');
    scheduleNext();
  } catch (e) {
    if (my !== session) return;
    stopSource(); setStatus('Video error', 'err'); showAlert(friendlyError(e));
  }
}

function resetRun() { analyzer.reset(); hist = []; scaleEMA = null; lastSeenT = -Infinity; lastMediaT = -1; frameCount = 0; fpsStamp = performance.now(); }

/* ------------------------------------------------------------------ frame loop */
function scheduleNext() {
  if (mode !== 'camera' && mode !== 'video') return;
  if (el.video.requestVideoFrameCallback) vfcId = el.video.requestVideoFrameCallback((now, meta) => onFrame(now, meta));
  else rafId = requestAnimationFrame(now => onFrame(now, null));
}

function onFrame(now, meta) {
  if (mode !== 'camera' && mode !== 'video') return;
  scheduleNext();
  try { processFrame(now, meta); }
  catch (e) { console.error(e); setStatus('Processing error (see console)', 'err'); }
}

function processFrame(now, meta) {
  const v = el.video;
  if (!landmarker || fallingBack || v.readyState < 2 || !v.videoWidth) return;
  if (!meta) { if (v.currentTime === lastMediaT) return; lastMediaT = v.currentTime; }
  let ts = performance.now(); if (ts <= lastTs) ts = lastTs + 1; lastTs = ts;

  let res;
  try { res = landmarker.detectForVideo(v, ts); }
  catch (e) {
    if (landmarkerDelegate === 'GPU') { fallBackToCpu(e); return; }
    throw e;
  }

  // Physics time base: capture time for webcams (least jitter), media time for files.
  let t;
  if (mode === 'camera') t = (meta && Number.isFinite(meta.captureTime) ? meta.captureTime : (meta?.expectedDisplayTime ?? now)) / 1000;
  else t = meta && Number.isFinite(meta.mediaTime) ? meta.mediaTime : v.currentTime;

  tickFps();
  const lm = res.landmarks?.[0], wl = res.worldLandmarks?.[0];
  if (!lm || lm.length < 21) { handLost(t); return; }
  lastSeenT = t;

  const W = v.videoWidth, H = v.videoHeight;
  // metres-per-pixel from hand size, using 3D length so rotation/foreshortening does not change the scale
  const len3 = Math.hypot((lm[0].x - lm[9].x) * W, (lm[0].y - lm[9].y) * H, (lm[0].z - lm[9].z) * W);
  if (len3 < 8) { handLost(t); return; }
  const raw = REF_LEN_M / len3;
  scaleEMA = scaleEMA === null ? raw : scaleEMA + 0.08 * (raw - scaleEMA);
  const P = new Float64Array(63);
  for (let i = 0; i < 21; i++) {
    P[i * 3] = (lm[i].x - 0.5) * W * scaleEMA;
    P[i * 3 + 1] = (lm[i].y - 0.5) * H * scaleEMA;
    P[i * 3 + 2] = lm[i].z * W * scaleEMA;
  }
  const shape = classifyShape(wl);
  ingest(t, P, shape.label, lm, W, H);
}

async function fallBackToCpu(e) {
  console.warn('GPU delegate failed, switching to CPU:', e);
  fallingBack = true; setStatus('GPU failed — switching to CPU…', 'warn');
  try { el.delegate.value = 'CPU'; await getLandmarker('CPU'); setStatus(mode === 'camera' ? 'Camera live (CPU)' : 'Analysing video (CPU)', 'ok'); }
  catch (e2) { stopSource(); setStatus('Model error', 'err'); showAlert(friendlyError(e2)); }
  finally { fallingBack = false; }
}

function handLost(t) {
  if (t - lastSeenT > 0.25) {
    if (analyzer.frames.length) { hist.push({ t, speed: NaN, ke: NaN }); }
    analyzer.reset(); scaleEMA = null;
    el.badge.textContent = mode === 'camera' ? '● LIVE — no hand' : '▶ VIDEO — no hand';
    el['t-shape'].textContent = '—';
  }
  ctx.clearRect(0, 0, el.overlay.width, el.overlay.height);
}

function tickFps() {
  frameCount++;
  const n = performance.now();
  if (n - fpsStamp >= 1000) {
    const fps = frameCount * 1000 / (n - fpsStamp);
    frameCount = 0; fpsStamp = n;
    if (mode === 'camera') setStatus(`Camera live · ${fps.toFixed(0)} fps`, 'ok');
  }
}

/* ------------------------------------------------------------------ shared pipeline */
function ingest(t, P, shapeLabel, lm, W, H) {
  el.badge.textContent = { camera: '● LIVE', video: '▶ VIDEO FILE', sim: '⚙ SIMULATION' }[mode];
  el['t-shape'].textContent = shapeLabel;
  const out = analyzer.push(t, P, shapeLabel);
  const tel = out?.telemetry;
  if (tel) {
    el['t-speed'].textContent = fmt(tel.speed, 2);
    el['t-accel'].textContent = fmt(tel.accel, 1);
    el['t-jerk'].textContent = fmt(tel.jerk, 0);
    el['t-ke'].textContent = fmt(tel.ke * 1000, 1);
    el['t-force'].textContent = fmt(tel.force, 2);
    el['t-torque'].textContent = fmt(tel.torque * 1000, 3);
    hist.push({ t: tel.t, speed: tel.speed, ke: tel.ke * 1000 });
    const cut = tel.t - HIST_SECONDS; while (hist.length && hist[0].t < cut) hist.shift();
  }
  drawHand(lm, W, H, tel, scaleEMA);
  drawCharts();
  if (out?.stroke) onStroke(out.stroke);
}

function speedColor(s) {
  const [b1, b2] = effortBoundaries(analyzer.sensitivity);
  return s < b1 ? '#34d399' : s < b2 ? '#fbbf24' : '#f87171';
}

function drawHand(lm, W, H, tel, scale) {
  const c = el.overlay.width, h = el.overlay.height;
  ctx.clearRect(0, 0, c, h);
  const lw = Math.max(2, c / 320);
  ctx.lineWidth = lw; ctx.strokeStyle = 'rgba(106,166,255,.9)'; ctx.lineCap = 'round';
  ctx.beginPath();
  for (const [a, b] of CONNECTIONS) { ctx.moveTo(lm[a].x * c, lm[a].y * h); ctx.lineTo(lm[b].x * c, lm[b].y * h); }
  ctx.stroke();
  ctx.fillStyle = '#fff';
  for (const p of lm) { ctx.beginPath(); ctx.arc(p.x * c, p.y * h, lw * 1.2, 0, 6.2832); ctx.fill(); }
  if (tel && scale) {                                    // velocity arrow from palm centre
    const cx = [0, 5, 9, 13, 17].reduce((s, i) => s + lm[i].x, 0) / 5 * c;
    const cy = [0, 5, 9, 13, 17].reduce((s, i) => s + lm[i].y, 0) / 5 * h;
    const k = 0.25 / scale * (c / W);                    // metres/s -> canvas px over 0.25 s
    let dx = tel.vel[0] * k, dy = tel.vel[1] * k;
    const mag = Math.hypot(dx, dy), maxLen = c * 0.4;
    if (mag > maxLen) { dx *= maxLen / mag; dy *= maxLen / mag; }
    if (mag > 4) {
      const col = speedColor(tel.speed);
      ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = lw * 1.8;
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + dx, cy + dy); ctx.stroke();
      const ang = Math.atan2(dy, dx), hl = lw * 5;
      ctx.beginPath(); ctx.moveTo(cx + dx, cy + dy);
      ctx.lineTo(cx + dx - hl * Math.cos(ang - 0.45), cy + dy - hl * Math.sin(ang - 0.45));
      ctx.lineTo(cx + dx - hl * Math.cos(ang + 0.45), cy + dy - hl * Math.sin(ang + 0.45));
      ctx.closePath(); ctx.fill();
    }
  }
}

/* ------------------------------------------------------------------ charts */
function drawChart(canvas, key, color, floor, lines) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
  const g = canvas.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  const pts = hist.filter(p => p.t !== undefined);
  const tEnd = pts.length ? pts[pts.length - 1].t : 0, tStart = tEnd - HIST_SECONDS;
  let ymax = floor;
  for (const p of pts) if (Number.isFinite(p[key])) ymax = Math.max(ymax, p[key] * 1.15);
  const X = t => ((t - tStart) / HIST_SECONDS) * w, Y = v => h - 4 - (v / ymax) * (h - 10);
  g.font = '10px system-ui'; g.lineWidth = 1;
  for (const [v, label, col] of lines) {
    if (v > ymax) continue;
    g.strokeStyle = col; g.globalAlpha = .45; g.setLineDash([4, 4]);
    g.beginPath(); g.moveTo(0, Y(v)); g.lineTo(w, Y(v)); g.stroke();
    g.setLineDash([]); g.globalAlpha = .8; g.fillStyle = col; g.fillText(label, 4, Y(v) - 3);
  }
  g.globalAlpha = 1; g.strokeStyle = color; g.lineWidth = 2; g.lineJoin = 'round';
  g.beginPath(); let pen = false;
  for (const p of pts) {
    if (!Number.isFinite(p[key])) { pen = false; continue; }
    if (!pen) { g.moveTo(X(p.t), Y(p[key])); pen = true; } else g.lineTo(X(p.t), Y(p[key]));
  }
  g.stroke();
  g.fillStyle = '#8d9ab3'; g.fillText(ymax.toFixed(ymax < 10 ? 2 : 0), w - 34, 11);
}

function drawCharts() {
  const [b1, b2] = effortBoundaries(analyzer.sensitivity);
  drawChart(el.chartSpeed, 'speed', '#6aa6ff', 1.3, [[b1, 'gentle|moderate', '#34d399'], [b2, 'moderate|forceful', '#f87171']]);
  drawChart(el.chartKE, 'ke', '#c084fc', 50, []);
}

/* ------------------------------------------------------------------ strokes & log */
function onStroke(s) {
  strokeNo++;
  const row = {
    n: strokeNo, src: mode, shape: s.shape, effort: s.effort.label, conf: s.effort.confidence,
    v: s.peakSpeed, a: s.peakAccel, jerk: s.peakJerk, ke: s.peakKE * 1000, force: s.peakForce,
    torque: s.peakTorque * 1000, impulse: s.impulse, dur: s.duration, ttp: s.timeToPeak,
  };
  log.unshift(row); if (log.length > 200) log.pop();
  el.sLabel.textContent = `${s.shape} · ${s.effort.label.toUpperCase()}`;
  el.sBar.style.width = `${Math.round(s.effort.confidence * 100)}%`;
  el.sMeta.textContent = `confidence ${(s.effort.confidence * 100).toFixed(0)}% · peak ${fmt(s.peakSpeed)} m/s · KE ${fmt(row.ke, 1)} mJ · ` +
    `force ${fmt(s.peakForce)} N · impulse ${fmt(s.impulse, 3)} N·s · ${fmt(s.duration * 1000, 0)} ms (peak at ${fmt(s.timeToPeak * 1000, 0)} ms)`;
  renderLog();
}

const cls = e => (e === 'Gentle' ? 'g' : e === 'Moderate' ? 'a' : 'r');
function renderLog() {
  if (!log.length) { el.logBody.innerHTML = '<tr class="empty"><td colspan="7">No movements yet</td></tr>'; return; }
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  el.logBody.innerHTML = log.slice(0, 12).map(r =>
    `<tr><td>${r.n}</td><td>${esc(r.shape)} <small>(effort ?)</small></td><td class="${cls(r.effort)}">${r.effort} ${(r.conf * 100).toFixed(0)}%</td>` +
    `<td><b class="${cls(r.effort)}">${esc(r.shape)} · ${r.effort}</b></td><td>${fmt(r.v)}</td><td>${fmt(r.ke, 1)}</td><td>${fmt(r.jerk, 0)}</td></tr>`).join('');
}

function exportCsv() {
  if (!log.length) return;
  const head = ['n', 'source', 'shape', 'effort', 'confidence', 'peak_speed_m_s', 'peak_accel_m_s2', 'peak_jerk_m_s3', 'peak_ke_mJ', 'peak_force_N', 'peak_torque_mNm', 'impulse_Ns', 'duration_s', 'time_to_peak_s'];
  const rows = [...log].reverse().map(r => [r.n, r.src, r.shape, r.effort, r.conf, r.v, r.a, r.jerk, r.ke, r.force, r.torque, r.impulse, r.dur, r.ttp]
    .map(x => (typeof x === 'number' ? +x.toFixed(5) : `"${String(x).replace(/"/g, '""')}"`)).join(','));
  const blob = new Blob([head.join(',') + '\n' + rows.join('\n')], { type: 'text/csv' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'handshayak-log.csv';
  document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

/* ------------------------------------------------------------------ simulation */
function runSim(peakSpeed) {
  stopSource(); showAlert('');
  const my = session;
  prepareStage('sim'); resetRun();
  el.video.classList.add('hidden');
  el.overlay.width = SIM_W; el.overlay.height = SIM_H; el.stage.style.aspectRatio = `${SIM_W} / ${SIM_H}`;
  simShapeTag = classifyShape(templateWorld()).label;
  setStatus(`Simulating ${peakSpeed} m/s strike…`, 'ok');
  const frames = makeTapSequence({ peakSpeed, fps: 30, seed: Math.round(peakSpeed * 100) });
  return new Promise(resolve => {
    let i = 0; const t0 = performance.now();
    const tick = () => {
      if (my !== session) { resolve(false); return; }
      const elapsed = (performance.now() - t0) / 1000;
      while (i < frames.length && frames[i].t <= elapsed) {
        const f = frames[i++];
        ingest(f.t, f.P, simShapeTag, toNormalized(f.P), SIM_W, SIM_H);
      }
      if (i < frames.length) simRaf = requestAnimationFrame(tick);
      else { setStatus('Simulation complete', ''); resolve(true); }
    };
    simRaf = requestAnimationFrame(tick);
  });
}

async function runAll() {
  for (const v of [0.3, 0.6, 1.2]) { if (!(await runSim(v))) return; await new Promise(r => setTimeout(r, 250)); }
}

/* ------------------------------------------------------------------ events */
el.btnCamera.addEventListener('click', startCamera);
el.btnStop.addEventListener('click', () => { stopSource(); setStatus('Stopped'); });
el.cameraSelect.addEventListener('change', () => { if (mode === 'camera') startCamera(); });
el.fileInput.addEventListener('change', () => { const f = el.fileInput.files?.[0]; if (f) startVideoFile(f); el.fileInput.value = ''; });
document.querySelectorAll('[data-sim]').forEach(b => b.addEventListener('click', () => runSim(parseFloat(b.dataset.sim))));
el.simAll.addEventListener('click', runAll);
el.mirror.addEventListener('change', () => el.stage.classList.toggle('mirror', mode === 'camera' && el.mirror.checked));
el.sens.addEventListener('input', () => { analyzer.sensitivity = parseFloat(el.sens.value) || 1; el.sensVal.textContent = analyzer.sensitivity.toFixed(1) + '×'; drawCharts(); });
el.mass.addEventListener('change', () => {
  const m = parseFloat(el.mass.value);
  if (Number.isFinite(m) && m >= 0.1 && m <= 1.5) analyzer.handMass = m; else el.mass.value = analyzer.handMass.toFixed(2);
});
el.delegate.addEventListener('change', async () => {
  if (mode !== 'camera' && mode !== 'video') return;
  fallingBack = true;
  try { setStatus('Switching engine…', 'warn'); await getLandmarker(el.delegate.value); setStatus(`Engine: ${el.delegate.value}`, 'ok'); }
  catch (e) { showAlert(friendlyError(e)); } finally { fallingBack = false; }
});
el.btnCsv.addEventListener('click', exportCsv);
el.btnClear.addEventListener('click', () => { log = []; strokeNo = 0; renderLog(); el.sLabel.textContent = 'Waiting for a movement…'; el.sBar.style.width = '0'; el.sMeta.textContent = 'Move your hand: a slow tap, then a sharp strike.'; });
window.addEventListener('resize', drawCharts);
window.addEventListener('pagehide', stopSource);
window.addEventListener('error', e => { setStatus('Script error', 'err'); showAlert(`Unexpected error: ${e.message}`); });
window.addEventListener('unhandledrejection', e => { console.error(e.reason); });
if (!window.isSecureContext) showAlert(friendlyError(err('InsecureContext', '')) + '\n(Simulation still works.)');
if (!navigator.mediaDevices?.getUserMedia) el.btnCamera.disabled = true;
drawCharts();
