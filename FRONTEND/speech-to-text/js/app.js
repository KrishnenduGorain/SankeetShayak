(() => {
  'use strict';

  /* ------------------------------------------------------------------ constants */
  // The worklet file sits next to this script; resolve it from the script's own URL so the app also works from a sub-path.
  const SCRIPT_BASE = document.currentScript && document.currentScript.src ? new URL('.', document.currentScript.src).href : 'js/';
  const WORKLET_URL = SCRIPT_BASE + 'gate-processor.js';

  /* ------------------------------------------------------------------ helpers */
  const $ = (id) => document.getElementById(id);
  const el = {};
  ['banner','meter','levelText','gateText','micBtn','calibrateBtn','status','lang','mic','ns','ec','agc','hp','gate','feed',
   'thr','thrOut','output','interim','copyBtn','saveBtn','clearBtn','count'].forEach((id) => { el[id] = $(id); });

  const store = {
    get(k, d) { try { const v = localStorage.getItem('stt.' + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
    set(k, v) { try { localStorage.setItem('stt.' + k, JSON.stringify(v)); } catch { /* storage unavailable */ } },
  };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const dbOf = (x) => 20 * Math.log10(Math.max(x, 1e-6));

  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;

  /* ------------------------------------------------------------------ state */
  let recognition = null;
  let wantListening = false;      // user intent: keep listening until they press stop
  let usingTrack = false;         // recognizer is reading our filtered track
  let audio = null;               // { stream, ctx, track, hp, gateNode, analyser, timer }
  let restartTimes = [];
  let lastFinal = { norm: '', t: 0 };
  let calib = null;               // { samples: [] } while calibrating
  let starting = false;
  const history = [];             // rolling level history for the meter
  const HISTORY = 140;

  /* ------------------------------------------------------------------ noise gate (runs on the audio thread) */


  /* ------------------------------------------------------------------ ui helpers */
  function setStatus(text, isError = false) {
    el.status.textContent = text;
    el.status.classList.toggle('err', isError);
  }
  function showBanner(text) { el.banner.textContent = text; el.banner.hidden = !text; }
  function setUi(listening) {
    el.micBtn.textContent = listening ? 'Stop listening' : 'Start listening';
    el.micBtn.classList.toggle('live', listening);
    el.micBtn.setAttribute('aria-pressed', String(listening));
    el.calibrateBtn.disabled = !listening;
  }
  function updateCount() {
    const n = el.output.value.trim().split(/\s+/).filter(Boolean).length;
    el.count.textContent = n + (n === 1 ? ' word' : ' words');
  }
  const thresholdLinear = () => Math.pow(10, Number(el.thr.value) / 20);

  /* ------------------------------------------------------------------ level meter */
  function pushLevel(rms, open) {
    history.push({ db: dbOf(rms), open });
    if (history.length > HISTORY) history.shift();
    if (calib) calib.samples.push(rms);
    el.levelText.textContent = 'Level ' + Math.round(dbOf(rms)) + ' dB';
    el.gateText.textContent = el.gate.checked && audio && audio.gateNode ? (open ? 'Speech passing' : 'Background reduced') : '';
  }
  function drawMeter() {
    const c = el.meter, g = c.getContext && c.getContext('2d');
    if (!g) return;
    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const css = getComputedStyle(document.documentElement);
    const quiet = css.getPropertyValue('--bar-quiet').trim(), loud = css.getPropertyValue('--bar-loud').trim();
    const yOf = (db) => h - Math.max(0, Math.min(1, (db + 70) / 70)) * (h - 4) - 2;
    const bw = w / HISTORY;
    for (let i = 0; i < history.length; i++) {
      const p = history[i], x = (HISTORY - history.length + i) * bw, y = yOf(p.db);
      g.fillStyle = p.open ? loud : quiet;
      g.fillRect(x + 0.5, y, Math.max(1, bw - 1.5), h - y);
    }
    if (el.gate.checked) {                         // threshold line
      const ty = yOf(Number(el.thr.value));
      g.strokeStyle = css.getPropertyValue('--live').trim(); g.globalAlpha = .8; g.setLineDash([5, 4]);
      g.beginPath(); g.moveTo(0, ty); g.lineTo(w, ty); g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
    }
  }
  let raf = 0;
  function meterLoop() { drawMeter(); raf = requestAnimationFrame(meterLoop); }

  /* ------------------------------------------------------------------ audio pipeline */
  async function openMicrophone() {
    const base = { echoCancellation: el.ec.checked, noiseSuppression: el.ns.checked, autoGainControl: el.agc.checked, channelCount: 1 };
    const id = el.mic.value;
    try {
      return await navigator.mediaDevices.getUserMedia({ audio: id ? { ...base, deviceId: { exact: id } } : base });
    } catch (e) {
      if (id && (e.name === 'OverconstrainedError' || e.name === 'NotFoundError')) {   // saved microphone was unplugged
        el.mic.value = '';
        return navigator.mediaDevices.getUserMedia({ audio: base });
      }
      throw e;
    }
  }

  async function startAudio() {
    const stream = await openMicrophone();
    const Ctx = window.AudioContext || window.webkitAudioContext;
    const ctx = new Ctx({ latencyHint: 'interactive' });
    if (ctx.state === 'suspended') await ctx.resume();

    const source = ctx.createMediaStreamSource(stream);
    const hp = ctx.createBiquadFilter(); hp.type = 'highpass'; hp.Q.value = 0.707; hp.frequency.value = el.hp.checked ? 90 : 10;
    const lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.Q.value = 0.707; lp.frequency.value = 7600;        // speech band
    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -38; comp.knee.value = 30; comp.ratio.value = 3; comp.attack.value = 0.005; comp.release.value = 0.25;
    const dest = ctx.createMediaStreamDestination();
    source.connect(hp); hp.connect(lp); lp.connect(comp);

    let gateNode = null, analyser = null, timer = 0;
    if (ctx.audioWorklet && window.AudioWorkletNode) {
      try {
        await ctx.audioWorklet.addModule(WORKLET_URL);
        gateNode = new AudioWorkletNode(ctx, 'speech-gate', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] });
        gateNode.port.onmessage = (e) => pushLevel(e.data.rms, e.data.open);
        comp.connect(gateNode); gateNode.connect(dest);
      } catch (e) { console.warn('Noise gate unavailable, continuing without it:', e); gateNode = null; }
    }
    if (!gateNode) {                                  // no AudioWorklet: still filter, and drive the meter from an analyser
      comp.connect(dest);
      analyser = ctx.createAnalyser(); analyser.fftSize = 1024; comp.connect(analyser);
      const buf = new Float32Array(analyser.fftSize);
      timer = setInterval(() => {
        analyser.getFloatTimeDomainData(buf);
        let s = 0; for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i];
        pushLevel(Math.sqrt(s / buf.length), true);
      }, 45);
    }

    audio = { stream, ctx, hp, gateNode, analyser, timer, track: dest.stream.getAudioTracks()[0] };
    sendGateConfig();
    stream.getAudioTracks()[0].addEventListener('ended', () => {
      if (wantListening) { setStatus('The microphone was disconnected.', true); stopListening(); }
    });

    const s = stream.getAudioTracks()[0].getSettings ? stream.getAudioTracks()[0].getSettings() : {};
    const on = (v) => (v === undefined ? 'n/a' : v ? 'on' : 'off');
    el.levelText.textContent = 'Listening';
    console.info('Browser audio processing — noise suppression:', on(s.noiseSuppression), 'echo cancellation:', on(s.echoCancellation), 'auto gain:', on(s.autoGainControl));
    await refreshMicList(s.deviceId);
    cancelAnimationFrame(raf); meterLoop();
    return s;
  }

  function sendGateConfig() {
    if (audio && audio.gateNode) audio.gateNode.port.postMessage({ threshold: thresholdLinear(), enabled: el.gate.checked });
  }

  function stopAudio() {
    releaseScreen();
    if (!audio) return;
    const a = audio; audio = null;
    clearInterval(a.timer);
    try { a.stream.getTracks().forEach((t) => t.stop()); } catch { /* already stopped */ }
    try { a.ctx.close(); } catch { /* already closed */ }
    cancelAnimationFrame(raf); raf = 0;
    history.length = 0; drawMeter();
    el.levelText.textContent = 'Microphone is off'; el.gateText.textContent = '';
  }

  async function refreshMicList(activeId) {
    try {
      const devs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'audioinput');
      const keep = el.mic.value || activeId || '';
      el.mic.innerHTML = '';
      el.mic.add(new Option('Default microphone', ''));
      devs.forEach((d, i) => { if (d.deviceId && d.deviceId !== 'default') el.mic.add(new Option(d.label || 'Microphone ' + (i + 1), d.deviceId)); });
      el.mic.value = [...el.mic.options].some((o) => o.value === keep) ? keep : '';
    } catch { /* listing is optional */ }
  }

  async function calibrate() {
    if (!audio) return;
    calib = { samples: [] };
    el.calibrateBtn.disabled = true;
    setStatus('Learning room noise — stay quiet for 2 seconds…');
    await sleep(2000);
    const s = calib.samples.sort((a, b) => a - b);
    calib = null;
    if (s.length >= 5) {
      const p90 = s[Math.floor(s.length * 0.9)];
      const db = Math.round(Math.min(-20, Math.max(-70, dbOf(p90) + 8)));       // about 2.5x the noise floor
      el.thr.value = db; syncThreshold();
      setStatus('Done. Gate threshold set to ' + db + ' dB for this room.');
    } else {
      setStatus('Could not measure the room. Try again.', true);
    }
    el.calibrateBtn.disabled = !wantListening;
  }

  /* ------------------------------------------------------------------ speech recognition */
  function appendFinal(raw) {
    const text = raw.trim();
    if (!text) return;
    const norm = text.toLowerCase(), now = Date.now();
    if (norm === lastFinal.norm && now - lastFinal.t < 1500) return;         // Android Chrome sometimes repeats a final result
    lastFinal = { norm, t: now };
    const sentence = text.charAt(0).toUpperCase() + text.slice(1);
    const cur = el.output.value;
    el.output.value = cur + (cur && !/\s$/.test(cur) ? ' ' : '') + sentence;
    el.output.scrollTop = el.output.scrollHeight;
    updateCount();
  }

  function createRecognition() {
    const r = new SR();
    r.continuous = true;
    r.interimResults = true;
    r.maxAlternatives = 1;
    r.onresult = (e) => {
      let interim = '';
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const res = e.results[i];
        if (res.isFinal) appendFinal(res[0].transcript); else interim += res[0].transcript;
      }
      el.interim.textContent = interim;
    };
    r.onerror = (e) => {
      switch (e.error) {
        case 'no-speech': case 'aborted': return;                             // normal: onend restarts
        case 'audio-capture':
          if (usingTrack) {                                                   // filtered track rejected: fall back to raw microphone
            el.feed.checked = false; usingTrack = false;
            setStatus('Filtered sound could not be used here. Switched to the raw microphone.', true);
            return;
          }
          fatal('No working microphone was found.'); return;
        case 'not-allowed': case 'service-not-allowed':
          fatal('Microphone permission is blocked. Allow it in the address bar, then press Start listening.'); return;
        case 'network':
          setStatus('Network problem: the speech service is unreachable. Retrying…', true); return;
        case 'language-not-supported':
          fatal('This browser does not support the selected language.'); return;
        default: setStatus('Recognition error: ' + e.error, true);
      }
    };
    r.onend = () => {
      el.interim.textContent = '';
      if (!wantListening) { setTimeout(stopAudio, 150); setUi(false); return; }
      const now = Date.now();
      restartTimes = restartTimes.filter((t) => now - t < 8000); restartTimes.push(now);
      if (restartTimes.length > 8) { fatal('Recognition keeps stopping. Check your connection and microphone, then start again.'); return; }
      setTimeout(beginRecognition, restartTimes.length > 3 ? 800 : 150);       // Chrome ends sessions after silence or ~60 s
    };
    return r;
  }

  function fatal(message) {
    setStatus(message, true);
    stopListening(true);
  }

  function beginRecognition() {
    if (!wantListening || !recognition) return;
    recognition.lang = el.lang.value;
    try {
      if (el.feed.checked && audio && audio.track) { usingTrack = true; recognition.start(audio.track); }
      else { usingTrack = false; recognition.start(); }
    } catch (e) {
      if (e.name !== 'InvalidStateError') { console.error(e); fatal('Could not start recognition: ' + e.message); }
    }
  }

  async function startListening() {
    if (wantListening || starting) return;
    starting = true;
    setStatus('Opening the microphone…');
    try {
      const settings = await startAudio();
      wantListening = true; restartTimes = [];
      setUi(true); holdScreen();
      const ns = settings.noiseSuppression;
      setStatus('Listening' + (ns === false && el.ns.checked ? ' (this microphone does not support browser noise suppression)' : '') + '…');
      beginRecognition();
    } catch (e) {
      stopAudio(); setUi(false);
      const msg = {
        NotAllowedError: 'Microphone permission was denied. Allow it in the address bar, then try again.',
        SecurityError: 'Microphone permission was denied. Allow it in the address bar, then try again.',
        NotFoundError: 'No microphone was found. Connect one and try again.',
        NotReadableError: 'The microphone is in use by another app. Close it and try again.',
      }[e.name] || ('Could not open the microphone: ' + (e.message || e.name));
      setStatus(msg, true);
    } finally { starting = false; }
  }

  function stopListening(silent = false) {
    wantListening = false;
    setUi(false);
    if (recognition) { try { recognition.stop(); } catch { /* not running */ } }
    setTimeout(stopAudio, 900);                       // let the recognizer deliver its last words first
    if (!silent) setStatus('Stopped.');
  }

  async function restartIfListening() {
    if (!wantListening) return;
    stopListening(true); setStatus('Applying settings…');
    await sleep(1000);
    startListening();
  }

  /* ------------------------------------------------------------------ keep the screen on while listening (phones) */
  let wakeLock = null;
  async function holdScreen() {
    try { if ('wakeLock' in navigator && !wakeLock) { wakeLock = await navigator.wakeLock.request('screen'); wakeLock.addEventListener('release', () => { wakeLock = null; }); } }
    catch { /* not allowed (battery saver) - listening still works */ }
  }
  function releaseScreen() { try { if (wakeLock) wakeLock.release(); } catch { /* ignore */ } wakeLock = null; }
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) return;
    if (audio && audio.ctx.state === 'suspended') audio.ctx.resume().catch(() => {});
    if (wantListening) holdScreen();                                          // the lock is dropped when the tab is hidden
  });

  /* ------------------------------------------------------------------ settings + events */
  function syncThreshold() {
    el.thrOut.textContent = el.thr.value + ' dB';
    store.set('thr', Number(el.thr.value));
    sendGateConfig(); drawMeter();
  }
  const PERSIST = ['ns', 'ec', 'agc', 'hp', 'gate', 'feed'];
  function loadSettings() {
    const savedLang = store.get('lang', 'en-IN');
    el.lang.value = savedLang;
    if (el.lang.value !== savedLang) el.lang.selectedIndex = 0;
    PERSIST.forEach((k) => { el[k].checked = store.get(k, true); });
    el.thr.value = store.get('thr', -52); el.thrOut.textContent = el.thr.value + ' dB';
    el.output.value = store.get('text', '');
    updateCount();
  }

  el.micBtn.addEventListener('click', () => (wantListening ? stopListening() : startListening()));
  el.calibrateBtn.addEventListener('click', calibrate);
  el.lang.addEventListener('change', () => {
    store.set('lang', el.lang.value);
    if (wantListening && recognition) { try { recognition.stop(); } catch { /* ignore */ } }     // onend restarts with the new language
  });
  el.mic.addEventListener('change', restartIfListening);
  ['ns', 'ec', 'agc', 'feed'].forEach((k) => el[k].addEventListener('change', () => { store.set(k, el[k].checked); restartIfListening(); }));
  el.hp.addEventListener('change', () => { store.set('hp', el.hp.checked); if (audio) audio.hp.frequency.value = el.hp.checked ? 90 : 10; });
  el.gate.addEventListener('change', () => { store.set('gate', el.gate.checked); sendGateConfig(); drawMeter(); });
  el.thr.addEventListener('input', syncThreshold);

  el.output.addEventListener('input', () => { store.set('text', el.output.value); updateCount(); });
  el.clearBtn.addEventListener('click', () => { el.output.value = ''; el.interim.textContent = ''; store.set('text', ''); updateCount(); el.output.focus(); });
  el.copyBtn.addEventListener('click', async () => {
    const text = el.output.value;
    if (!text) { setStatus('Nothing to copy yet.'); return; }
    try { await navigator.clipboard.writeText(text); setStatus('Copied to clipboard.'); }
    catch { el.output.select(); setStatus(document.execCommand && document.execCommand('copy') ? 'Copied to clipboard.' : 'Copy failed. Select the text and copy it manually.', false); }
  });
  el.saveBtn.addEventListener('click', () => {
    if (!el.output.value) { setStatus('Nothing to save yet.'); return; }
    const stamp = new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-');
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([el.output.value], { type: 'text/plain;charset=utf-8' }));
    a.download = 'transcript-' + stamp + '.txt';
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    setStatus('Saved ' + a.download);
  });
  window.addEventListener('resize', drawMeter);
  window.addEventListener('pagehide', () => { wantListening = false; try { recognition && recognition.abort(); } catch { /* ignore */ } stopAudio(); });
  // Persist text written by the recognizer too (programmatic changes do not fire "input")
  setInterval(() => { if (wantListening) store.set('text', el.output.value); }, 2000);

  /* ------------------------------------------------------------------ boot */
  loadSettings();
  drawMeter();
  if (!window.isSecureContext) {
    showBanner('The microphone only works on https:// pages or http://localhost. Open this page from your deployed https address or a local server.');
    el.micBtn.disabled = true;
  } else if (!SR) {
    showBanner('Live transcription is not supported in this browser. Use Google Chrome, Microsoft Edge or Safari.');
    el.micBtn.disabled = true;
  } else if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
    showBanner('This browser cannot access the microphone.');
    el.micBtn.disabled = true;
  } else {
    recognition = createRecognition();
    setStatus('Press Start listening and allow the microphone.');
  }

  if (window.__STT_TEST__) window.__stt = { appendFinal, el, get state() { return { wantListening, usingTrack, audio: !!audio }; } };   // test hook (never set in production)
})();
