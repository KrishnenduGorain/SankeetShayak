/**
 * Sankeet Shayak - UI Controller
 * Bridges BackendAdapter, Prediction Panel, History, and Translation UI
 */
(function() {
  document.addEventListener('DOMContentLoaded', () => {
    const adapter = window.SS_BackendAdapter;
    if (!adapter) return;

    // Elements
    const bePanel = document.getElementById('backend-panel');
    const bePill = document.getElementById('be-pill');
    const bePillText = document.getElementById('be-pill-text');
    const beMsg = document.getElementById('be-msg');
    const beMode = document.getElementById('be-mode');
    const beHost = document.getElementById('be-host');
    const beTransport = document.getElementById('be-transport');
    const beChecked = document.getElementById('be-checked');
    const beCaps = document.getElementById('be-caps');
    const beFrames = document.getElementById('be-frames');
    const beRetry = document.getElementById('be-retry');

    // Translation Elements
    const langSelect = document.getElementById('lang-select');
    const trOrig = document.getElementById('tr-orig');
    const trText = document.getElementById('tr-text');
    const trChip = document.getElementById('tr-chip');
    const ttsOrig = document.getElementById('tts-orig');
    const ttsTrans = document.getElementById('tts-trans');
    const ttsStop = document.getElementById('tts-stop');

    // History Elements
    const histList = document.getElementById('hist-list');
    const histEmpty = document.getElementById('hist-empty');
    const histClear = document.getElementById('hist-clear');

    // Frame Sender checkbox
    const sendFramesCheck = document.getElementById('send-frames');
    if (sendFramesCheck) {
      sendFramesCheck.addEventListener('change', (e) => {
        if (window.SS_FrameSender) window.SS_FrameSender.setEnabled(e.target.checked);
      });
    }

    // Backend status updates
    adapter.on('status', ({ state, message }) => {
      if (bePanel) bePanel.dataset.state = state;
      if (bePill) bePill.dataset.state = state;
      if (bePillText) bePillText.textContent = state === 'connected' ? 'Connected' : (state === 'checking' ? 'Connecting…' : 'Disconnected');
      if (beMsg) {
        beMsg.textContent = message;
        beMsg.classList.remove('skeleton');
      }
      if (beMode) beMode.textContent = 'Production';
      if (beHost) beHost.textContent = adapter.config.baseUrl;
      if (beTransport) beTransport.textContent = adapter.config.transport;
      if (beChecked) beChecked.textContent = new Date().toLocaleTimeString();
      if (beCaps && adapter.serverCapabilities) {
        const cls = adapter.serverCapabilities.classes || [];
        beCaps.textContent = `${cls.length} signs: ${cls.join(', ')}`;
      }

      // Update main prediction panel state
      if (window.SanketSahayak && window.SanketSahayak.setModelState) {
        window.SanketSahayak.setModelState(
          state === 'connected' ? 'ready' : (state === 'checking' ? 'loading' : 'disconnected'),
          message
        );
      }
    });

    // Frame counter updates
    adapter.on('frameCount', ({ sent, skipped }) => {
      if (beFrames) beFrames.textContent = `${sent} / ${skipped}`;
    });

    // Prediction events
    let currentPrediction = null;
    adapter.on('prediction', (res) => {
      currentPrediction = res;
      if (window.SanketSahayak && window.SanketSahayak.setPrediction) {
        window.SanketSahayak.setPrediction(res);
      }

      // Translation
      const curLang = langSelect ? langSelect.value : 'en';
      const translated = window.SS_Speech ? window.SS_Speech.translate(res.label, curLang) : res.label;
      if (trOrig) trOrig.textContent = res.label;
      if (trText) trText.textContent = translated;
      if (trChip) {
        trChip.dataset.kind = 'ready';
        trChip.textContent = `Translated to ${curLang.toUpperCase()}`;
      }
      if (ttsOrig) ttsOrig.disabled = false;
      if (ttsTrans) ttsTrans.disabled = false;
      if (ttsStop) ttsStop.disabled = false;

      // History
      if (histEmpty) histEmpty.hidden = true;
      if (histClear) histClear.disabled = false;
      if (histList) {
        const li = document.createElement('li');
        li.className = 'history-item';
        li.innerHTML = `<span><strong>${res.label}</strong> (${Math.round(res.confidence * 100)}%)</span> <time>${new Date().toLocaleTimeString()}</time>`;
        histList.prepend(li);
        while (histList.children.length > 20) histList.removeChild(histList.lastChild);
      }
    });

    // Speech handlers
    if (ttsOrig) {
      ttsOrig.addEventListener('click', () => {
        if (currentPrediction && window.SS_Speech) window.SS_Speech.speak(currentPrediction.label, 'en');
      });
    }
    if (ttsTrans) {
      ttsTrans.addEventListener('click', () => {
        if (currentPrediction && window.SS_Speech) {
          const curLang = langSelect ? langSelect.value : 'en';
          const text = window.SS_Speech.translate(currentPrediction.label, curLang);
          window.SS_Speech.speak(text, curLang);
        }
      });
    }
    if (ttsStop) {
      ttsStop.addEventListener('click', () => {
        if (window.SS_Speech) window.SS_Speech.stop();
      });
    }

    // Clear history
    if (histClear) {
      histClear.addEventListener('click', () => {
        if (histList) histList.innerHTML = '';
        if (histEmpty) histEmpty.hidden = false;
        histClear.disabled = true;
      });
    }

    // Retry button
    if (beRetry) {
      beRetry.addEventListener('click', () => {
        adapter.init();
      });
    }
  });
})();
