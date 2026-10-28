/**
 * Sankeet Shayak - UI Controller
 * Bridges BackendAdapter, SignRecognizer, Prediction Panel, History,
 * Interactive Sign Guide, and Translation UI
 */
(function() {
  function initController() {
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
    const predStatusEl = document.getElementById('pred-status');
    const predTimeEl = document.getElementById('pred-time');
    const predTextEl = document.getElementById('pred-text');

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

    // Guide Elements
    const guideGrid = document.getElementById('guide-grid');
    const guideTabs = document.querySelectorAll('.guide-tab');

    // Render Sign Guide
    function renderSignGuide(filterCat = 'all') {
      if (!guideGrid || !window.SS_SignRecognizer) return;
      const signs = window.SS_SignRecognizer.getAllSigns();
      guideGrid.innerHTML = '';

      Object.entries(signs).forEach(([label, info]) => {
        if (filterCat !== 'all' && info.category !== filterCat) return;

        const card = document.createElement('div');
        card.className = 'guide-card';
        card.dataset.label = label;
        card.dataset.category = info.category;
        card.setAttribute('tabindex', '0');
        card.setAttribute('role', 'button');
        card.setAttribute('aria-label', `Sign ${label}: ${info.desc}`);

        card.innerHTML = `
          <div class="gc-top">
            <span class="gc-label">${label}</span>
            <span class="gc-cat">${info.category}</span>
          </div>
          <div class="gc-tr">${info.hi || ''} · ${info.bn || ''}</div>
          <p class="gc-desc">${info.desc}</p>
        `;

        card.addEventListener('click', () => {
          if (window.SS_Speech) {
            const curLang = langSelect ? langSelect.value : 'en';
            const translated = window.SS_Speech.translate(label, curLang);
            window.SS_Speech.speak(translated, curLang);
          }
          highlightGuideCard(label, true);
        });

        guideGrid.appendChild(card);
      });
    }

    // Guide category tab switching
    guideTabs.forEach(tab => {
      tab.addEventListener('click', () => {
        guideTabs.forEach(t => {
          t.classList.remove('active');
          t.setAttribute('aria-selected', 'false');
        });
        tab.classList.add('active');
        tab.setAttribute('aria-selected', 'true');
        renderSignGuide(tab.dataset.cat);
      });
    });

    renderSignGuide('all');

    let lastHighlightedCard = null;
    function highlightGuideCard(label, scroll = false) {
      if (!guideGrid) return;
      if (lastHighlightedCard) lastHighlightedCard.classList.remove('active-detected');
      const card = guideGrid.querySelector(`.guide-card[data-label="${label}"]`);
      if (card) {
        card.classList.add('active-detected');
        lastHighlightedCard = card;
        if (scroll) {
          card.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'nearest' });
        }
      }
    }

    function updateStatusUI(state, message) {
      if (bePanel) bePanel.dataset.state = state;
      if (bePill) bePill.dataset.state = state;
      if (bePillText) bePillText.textContent = state === 'connected' ? 'Connected' : (state === 'checking' ? 'Connecting…' : 'Disconnected');
      if (beMsg) {
        beMsg.textContent = message || (state === 'connected' ? 'Connected to recognition service.' : 'Connecting…');
        beMsg.classList.remove('skeleton');
      }
      if (beMode) beMode.textContent = 'Production';
      if (beHost) beHost.textContent = adapter.config.baseUrl;
      if (beTransport) beTransport.textContent = (adapter.activeTransport || adapter.config.transport).toUpperCase();
      if (beChecked) beChecked.textContent = new Date().toLocaleTimeString();
      if (beCaps && adapter.serverCapabilities) {
        const cls = adapter.serverCapabilities.classes || [];
        beCaps.textContent = cls.length ? `${cls.length} signs: ${cls.join(', ')}` : 'Live Model Ready';
      }

      // Update prediction panel status
      if (predStatusEl) {
        predStatusEl.textContent = state === 'connected' 
          ? 'Recognition service connected' 
          : (state === 'checking' ? 'Connecting to recognition service…' : 'Instant client recognition active');
      }

      // Update main prediction panel state
      if (window.SanketSahayak && window.SanketSahayak.setModelState) {
        window.SanketSahayak.setModelState(
          state === 'connected' ? 'ready' : (state === 'checking' ? 'loading' : 'ready'),
          message || 'Ready for signs'
        );
      }
    }

    // Backend status listener
    adapter.on('status', ({ state, message }) => {
      updateStatusUI(state, message);
    });

    // Immediately sync current status
    updateStatusUI(adapter.state, adapter.lastMessage);

    // Frame counter updates
    adapter.on('frameCount', ({ sent, skipped }) => {
      if (beFrames) beFrames.textContent = `${sent} / ${skipped}`;
    });

    // Prediction events (handles both instant client-side and backend predictions)
    let currentPrediction = null;
    let lastHistoryLabel = null;
    let lastHistoryTime = 0;

    adapter.on('prediction', (res) => {
      currentPrediction = res;
      if (window.SanketSahayak && window.SanketSahayak.setPrediction) {
        window.SanketSahayak.setPrediction(res);
      }

      if (predTimeEl) {
        predTimeEl.textContent = new Date().toLocaleTimeString();
      }

      if (predStatusEl) {
        predStatusEl.textContent = res.isReliable !== false ? 'Live prediction received' : 'Low confidence inference';
      }

      // Translation
      const curLang = langSelect ? langSelect.value : 'en';
      const translated = window.SS_Speech ? window.SS_Speech.translate(res.label, curLang) : res.label;
      if (trOrig) trOrig.textContent = res.label;
      if (trText) trText.textContent = translated;
      if (predTextEl) predTextEl.textContent = translated;
      if (trChip) {
        trChip.dataset.kind = 'ready';
        trChip.textContent = `Translated to ${curLang.toUpperCase()}`;
      }
      if (ttsOrig) ttsOrig.disabled = false;
      if (ttsTrans) ttsTrans.disabled = false;
      if (ttsStop) ttsStop.disabled = false;

      // Alternative predictions
      const altsWrap = document.getElementById('pred-alts-wrap');
      const altsList = document.getElementById('pred-alts');
      if (altsWrap && altsList && res.probabilities) {
        const sorted = Object.entries(res.probabilities)
          .sort((a, b) => b[1] - a[1])
          .filter(([label, prob]) => label !== res.label && prob > 0.05);
        if (sorted.length > 0) {
          altsList.innerHTML = sorted.map(([l, p]) => `<li>${l} <b>${Math.round(p * 100)}%</b></li>`).join('');
          altsWrap.hidden = false;
        } else {
          altsWrap.hidden = true;
        }
      }

      // Highlight in sign guide
      highlightGuideCard(res.label);

      // Session History (debounced so it logs meaningful transitions, not 60 items/sec)
      const now = Date.now();
      if (res.label && (res.label !== lastHistoryLabel || (now - lastHistoryTime > 2500))) {
        lastHistoryLabel = res.label;
        lastHistoryTime = now;
        if (histEmpty) histEmpty.hidden = true;
        if (histClear) histClear.disabled = false;
        if (histList) {
          const li = document.createElement('li');
          li.className = 'history-item';
          li.innerHTML = `<span><strong>${res.label}</strong> (${Math.round(res.confidence * 100)}%)</span> <time>${new Date().toLocaleTimeString()}</time>`;
          histList.prepend(li);
          while (histList.children.length > 25) histList.removeChild(histList.lastChild);
        }
      }

      // Live hand panel sign display
      const signDisplay = document.getElementById('sign-display');
      const signLabel = document.getElementById('sign-label');
      const signConf = document.getElementById('sign-conf');
      if (signDisplay && signLabel && signConf && res.label) {
        signDisplay.hidden = false;
        signLabel.textContent = res.label;
        signConf.textContent = Math.round(res.confidence * 100) + '%';
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
        lastHistoryLabel = null;
      });
    }

    // Retry button
    if (beRetry) {
      beRetry.addEventListener('click', () => {
        adapter.init();
      });
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initController);
  } else {
    initController();
  }
})();
