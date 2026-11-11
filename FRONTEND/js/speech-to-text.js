/**
 * Sankeet Shayak - Google-Style Live Speech to Text Module
 * Integrated directly on the main dashboard under the sign language recognition output.
 */
(function() {
  'use strict';

  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;

  const state = {
    listening: false,
    recognition: null,
    audioCtx: null,
    analyser: null,
    mediaStream: null,
    animFrame: null,
    restartTimer: null,
    lang: 'en-IN'
  };

  function init() {
    const micBtn = document.getElementById('stt-mic-btn');
    const langSelect = document.getElementById('stt-lang');
    const outputArea = document.getElementById('stt-output');
    const interimBox = document.getElementById('stt-interim');
    const copyBtn = document.getElementById('stt-copy-btn');
    const clearBtn = document.getElementById('stt-clear-btn');
    const speakBtn = document.getElementById('stt-speak-btn');
    const statusText = document.getElementById('stt-status-text');
    const dotsWrap = document.getElementById('stt-dots');
    const waveRings = document.getElementById('stt-wave-rings');

    if (!micBtn) return;

    if (!SR) {
      if (statusText) statusText.textContent = 'Voice recognition not supported in this browser. Use Chrome, Edge or Safari.';
      micBtn.disabled = true;
      micBtn.classList.add('disabled');
      return;
    }

    // Load saved language if available
    try {
      const savedLang = localStorage.getItem('sankeet_stt_lang');
      if (savedLang && langSelect) {
        langSelect.value = savedLang;
        state.lang = savedLang;
      }
    } catch (_) {}

    function updateUi(isListening) {
      state.listening = isListening;
      micBtn.setAttribute('aria-pressed', String(isListening));
      micBtn.classList.toggle('is-listening', isListening);

      if (dotsWrap) {
        dotsWrap.classList.toggle('active', isListening);
      }
      if (waveRings) {
        waveRings.classList.toggle('active', isListening);
      }

      if (isListening) {
        if (statusText) statusText.textContent = 'Listening… Speak clearly into your mic';
      } else {
        if (statusText) statusText.textContent = 'Microphone off. Tap to speak.';
        if (interimBox) interimBox.textContent = '';
      }
    }

    function createRecognizer() {
      const r = new SR();
      r.continuous = true;
      r.interimResults = true;
      r.maxAlternatives = 1;
      r.lang = langSelect ? langSelect.value : state.lang;

      r.onresult = function(event) {
        let interim = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const res = event.results[i];
          const transcript = res[0].transcript;
          if (res.isFinal) {
            const trimmed = transcript.trim();
            if (trimmed) {
              const current = outputArea ? outputArea.value : '';
              const separator = (current && !/\s$/.test(current)) ? ' ' : '';
              const formatted = trimmed.charAt(0).toUpperCase() + trimmed.slice(1);
              if (outputArea) {
                outputArea.value = current + separator + formatted;
                outputArea.scrollTop = outputArea.scrollHeight;
              }
            }
          } else {
            interim += transcript;
          }
        }
        if (interimBox) {
          interimBox.textContent = interim;
        }
      };

      r.onerror = function(event) {
        console.warn('STT Error:', event.error);
        if (event.error === 'no-speech' || event.error === 'aborted') {
          return;
        }
        if (event.error === 'not-allowed') {
          if (statusText) statusText.textContent = 'Microphone permission blocked. Please allow mic access.';
          stopListening();
        } else if (event.error === 'network') {
          if (statusText) statusText.textContent = 'Network issue connecting to speech service.';
        } else {
          if (statusText) statusText.textContent = 'Speech error: ' + event.error;
        }
      };

      r.onend = function() {
        if (state.listening) {
          // Keep continuous listening alive like Google voice search
          state.restartTimer = setTimeout(() => {
            if (state.listening && state.recognition) {
              try {
                state.recognition.start();
              } catch (_) {}
            }
          }, 200);
        } else {
          updateUi(false);
          stopAudioAnalyser();
        }
      };

      return r;
    }

    async function startAudioAnalyser() {
      try {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
          state.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
          const AudioContext = window.AudioContext || window.webkitAudioContext;
          state.audioCtx = new AudioContext();
          if (state.audioCtx.state === 'suspended') {
            await state.audioCtx.resume();
          }
          state.analyser = state.audioCtx.createAnalyser();
          state.analyser.fftSize = 64;
          const source = state.audioCtx.createMediaStreamSource(state.mediaStream);
          source.connect(state.analyser);

          const dataArray = new Uint8Array(state.analyser.frequencyBinCount);
          const dotEls = document.querySelectorAll('.stt-dot');

          function animateDots() {
            if (!state.listening || !state.analyser) return;
            state.analyser.getByteFrequencyData(dataArray);

            let sum = 0;
            for (let i = 0; i < dataArray.length; i++) {
              sum += dataArray[i];
            }
            const avg = sum / dataArray.length;
            const boost = Math.min(1.8, Math.max(0.2, avg / 30));

            dotEls.forEach((dot, idx) => {
              const freqVal = dataArray[(idx * 3) % dataArray.length] || avg;
              const scaleY = Math.max(0.6, Math.min(2.4, (freqVal / 45) * boost));
              dot.style.transform = `scaleY(${scaleY.toFixed(2)})`;
            });

            state.animFrame = requestAnimationFrame(animateDots);
          }
          animateDots();
        }
      } catch (err) {
        // Fallback: pure CSS pulsing if audio visualizer fails
        console.info('Audio visualizer fallback to CSS animation');
      }
    }

    function stopAudioAnalyser() {
      if (state.animFrame) {
        cancelAnimationFrame(state.animFrame);
        state.animFrame = null;
      }
      if (state.mediaStream) {
        try {
          state.mediaStream.getTracks().forEach(t => t.stop());
        } catch (_) {}
        state.mediaStream = null;
      }
      if (state.audioCtx) {
        try {
          state.audioCtx.close();
        } catch (_) {}
        state.audioCtx = null;
      }
      const dotEls = document.querySelectorAll('.stt-dot');
      dotEls.forEach(dot => { dot.style.transform = ''; });
    }

    function startListening() {
      if (state.listening) return;
      if (!state.recognition) {
        state.recognition = createRecognizer();
      }
      state.recognition.lang = langSelect ? langSelect.value : state.lang;
      try {
        state.recognition.start();
        updateUi(true);
        startAudioAnalyser();
      } catch (err) {
        if (err.name === 'InvalidStateError') {
          try { state.recognition.stop(); } catch (_) {}
          setTimeout(() => {
            try {
              state.recognition.start();
              updateUi(true);
              startAudioAnalyser();
            } catch (_) {}
          }, 150);
        } else {
          console.error(err);
          if (statusText) statusText.textContent = 'Could not start mic: ' + (err.message || err.name);
        }
      }
    }

    function stopListening() {
      state.listening = false;
      if (state.restartTimer) {
        clearTimeout(state.restartTimer);
        state.restartTimer = null;
      }
      if (state.recognition) {
        try { state.recognition.stop(); } catch (_) {}
      }
      stopAudioAnalyser();
      updateUi(false);
    }

    micBtn.addEventListener('click', function() {
      if (state.listening) {
        stopListening();
      } else {
        startListening();
      }
    });

    if (langSelect) {
      langSelect.addEventListener('change', function() {
        state.lang = langSelect.value;
        try { localStorage.setItem('sankeet_stt_lang', state.lang); } catch (_) {}
        if (state.listening) {
          stopListening();
          setTimeout(startListening, 300);
        }
      });
    }

    if (copyBtn && outputArea) {
      copyBtn.addEventListener('click', async function() {
        const text = outputArea.value.trim();
        if (!text) return;
        try {
          await navigator.clipboard.writeText(text);
          const orig = copyBtn.textContent;
          copyBtn.textContent = '✓ Copied!';
          setTimeout(() => { copyBtn.textContent = orig; }, 1800);
        } catch (_) {
          outputArea.select();
          document.execCommand('copy');
        }
      });
    }

    if (clearBtn && outputArea) {
      clearBtn.addEventListener('click', function() {
        outputArea.value = '';
        if (interimBox) interimBox.textContent = '';
        outputArea.focus();
      });
    }

    if (speakBtn && outputArea) {
      speakBtn.addEventListener('click', function() {
        const text = outputArea.value.trim();
        if (!text) return;
        if (window.SS_Speech && window.SS_Speech.speak) {
          const l = (langSelect && langSelect.value.startsWith('hi')) ? 'hi' :
                    (langSelect && langSelect.value.startsWith('bn')) ? 'bn' : 'en';
          window.SS_Speech.speak(text, l);
        } else if (window.speechSynthesis) {
          window.speechSynthesis.cancel();
          const u = new SpeechSynthesisUtterance(text);
          if (langSelect) u.lang = langSelect.value;
          window.speechSynthesis.speak(u);
        }
      });
    }
  }

  window.SS_SpeechToText = {
    init: init,
    start: () => { const b = document.getElementById('stt-mic-btn'); if (b && !state.listening) b.click(); },
    stop: () => { const b = document.getElementById('stt-mic-btn'); if (b && state.listening) b.click(); }
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
