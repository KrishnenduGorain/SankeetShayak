/**
 * Sankeet Shayak - Google-Style Live Speech to Text Module
 * Bulletproof Web Speech API implementation with zero audio device conflicts.
 */
(function () {
  'use strict';

  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;

  let recognition = null;
  let isListening = false;
  let finalTranscript = '';
  let restartTimer = null;
  let currentLang = 'en-IN';

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
      if (statusText) statusText.textContent = 'Voice recognition not supported in this browser. Please use Chrome, Edge or Safari.';
      micBtn.disabled = true;
      micBtn.style.opacity = '0.5';
      return;
    }

    // Restore saved language preference or default to English (India)
    try {
      const savedLang = localStorage.getItem('sankeet_stt_lang');
      if (savedLang && langSelect) {
        langSelect.value = savedLang;
        currentLang = savedLang;
      } else if (langSelect) {
        currentLang = langSelect.value;
      }
    } catch (_) {}

    function updateUiState(listening) {
      isListening = listening;
      micBtn.setAttribute('aria-pressed', String(listening));
      micBtn.classList.toggle('is-listening', listening);

      if (dotsWrap) dotsWrap.classList.toggle('active', listening);
      if (waveRings) waveRings.classList.toggle('active', listening);

      if (listening) {
        if (statusText) statusText.textContent = 'Listening… Speak now';
        statusText.style.color = '#5eead4';
      } else {
        if (statusText) statusText.textContent = 'Microphone off. Tap to speak.';
        statusText.style.color = '';
        if (interimBox) interimBox.textContent = '';
      }
    }

    function createRecognizer() {
      const r = new SR();
      r.continuous = true;
      r.interimResults = true;
      r.maxAlternatives = 1;
      r.lang = langSelect ? langSelect.value : currentLang;

      r.onstart = function () {
        updateUiState(true);
      };

      r.onresult = function (event) {
        let interimText = '';

        for (let i = event.resultIndex; i < event.results.length; ++i) {
          const res = event.results[i];
          const transcript = res[0].transcript;
          if (res.isFinal) {
            finalTranscript += transcript + ' ';
          } else {
            interimText += transcript;
          }
        }

        const fullDisplay = (finalTranscript + interimText).trim();
        if (outputArea && fullDisplay) {
          outputArea.value = fullDisplay;
          outputArea.scrollTop = outputArea.scrollHeight;
        }

        if (interimBox) {
          interimBox.textContent = interimText ? 'Speaking: ' + interimText : '';
        }

        if (statusText && isListening) {
          statusText.textContent = 'Hearing speech…';
        }
      };

      r.onerror = function (event) {
        console.warn('SpeechRecognition error:', event.error);
        if (event.error === 'no-speech') {
          if (statusText && isListening) statusText.textContent = 'Listening… Speak clearly into your mic';
          return;
        }
        if (event.error === 'aborted') {
          return;
        }
        if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
          if (statusText) statusText.textContent = 'Microphone access denied. Please click the camera/mic icon in your address bar to allow.';
          stopListening();
        } else if (event.error === 'network') {
          if (statusText) statusText.textContent = 'Network error connecting to speech recognition server.';
        } else {
          if (statusText) statusText.textContent = 'Speech error: ' + event.error;
        }
      };

      r.onend = function () {
        if (isListening) {
          // Keep listening continuously across pauses
          restartTimer = setTimeout(() => {
            if (isListening && recognition) {
              try {
                recognition.start();
              } catch (_) {}
            }
          }, 150);
        } else {
          updateUiState(false);
        }
      };

      return r;
    }

    function startListening() {
      if (isListening) return;
      if (restartTimer) {
        clearTimeout(restartTimer);
        restartTimer = null;
      }

      // Preserve any text the user manually typed or previously spoke
      if (outputArea) {
        finalTranscript = outputArea.value ? outputArea.value.trim() + ' ' : '';
      }

      try {
        if (!recognition) {
          recognition = createRecognizer();
        }
        recognition.lang = langSelect ? langSelect.value : currentLang;
        recognition.start();
        updateUiState(true);
      } catch (err) {
        if (err.name === 'InvalidStateError') {
          // If already starting or stopping, reset recognizer instance
          try { recognition.abort(); } catch (_) {}
          recognition = createRecognizer();
          setTimeout(() => {
            try {
              recognition.start();
              updateUiState(true);
            } catch (e) {
              console.error(e);
            }
          }, 100);
        } else {
          console.error('startListening error:', err);
          if (statusText) statusText.textContent = 'Could not start mic: ' + (err.message || err.name);
          updateUiState(false);
        }
      }
    }

    function stopListening() {
      isListening = false;
      if (restartTimer) {
        clearTimeout(restartTimer);
        restartTimer = null;
      }
      if (recognition) {
        try {
          recognition.stop();
        } catch (_) {}
      }
      updateUiState(false);
    }

    micBtn.addEventListener('click', function (e) {
      e.preventDefault();
      if (isListening) {
        stopListening();
      } else {
        startListening();
      }
    });

    if (langSelect) {
      langSelect.addEventListener('change', function () {
        currentLang = langSelect.value;
        try { localStorage.setItem('sankeet_stt_lang', currentLang); } catch (_) {}
        if (isListening) {
          stopListening();
          setTimeout(startListening, 250);
        }
      });
    }

    if (outputArea) {
      outputArea.addEventListener('input', function () {
        finalTranscript = outputArea.value;
      });
    }

    if (copyBtn && outputArea) {
      copyBtn.addEventListener('click', async function () {
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
      clearBtn.addEventListener('click', function () {
        outputArea.value = '';
        finalTranscript = '';
        if (interimBox) interimBox.textContent = '';
        outputArea.focus();
      });
    }

    if (speakBtn && outputArea) {
      speakBtn.addEventListener('click', function () {
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
    start: () => { const b = document.getElementById('stt-mic-btn'); if (b && !isListening) b.click(); },
    stop: () => { const b = document.getElementById('stt-mic-btn'); if (b && isListening) b.click(); }
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
