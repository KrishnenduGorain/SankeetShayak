/**
 * Sankeet Shayak - Speech & Translation
 */
(function() {
  const DICTIONARY = {
    'swipe_slow': { en: 'Slow Swipe', hi: 'धीमी स्वाइप', bn: 'ধীর সোয়াইপ' },
    'swipe_rapid': { en: 'Rapid Swipe', hi: 'तेज़ स्वाइप', bn: 'দ্রুত সোয়াইপ' },
    'tap_forceful': { en: 'Forceful Tap', hi: 'ज़ोरदार टैप', bn: 'জোরদার ট্যাপ' },
    'tap_gentle': { en: 'Gentle Tap', hi: 'हल्का टैप', bn: 'হালকা ট্যাপ' },
    'wave_rapid': { en: 'Rapid Wave', hi: 'तेज़ हाथ हिलाना', bn: 'দ্রুত হাত নাড়া' },
    'pinch_gentle': { en: 'Gentle Pinch', hi: 'हल्की चुटकी', bn: 'হালকা চিমটি' }
  };

  window.SS_Speech = {
    translate: function(label, targetLang) {
      if (!label) return '—';
      const clean = label.toLowerCase().trim();
      const entry = DICTIONARY[clean];
      if (entry && entry[targetLang]) return entry[targetLang];
      return label;
    },
    speak: function(text, lang = 'en') {
      if (!window.speechSynthesis || !text) return;
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.lang = lang === 'hi' ? 'hi-IN' : (lang === 'bn' ? 'bn-IN' : 'en-US');
      u.rate = 0.95;
      window.speechSynthesis.speak(u);
    },
    stop: function() {
      if (window.speechSynthesis) window.speechSynthesis.cancel();
    }
  };
})();
