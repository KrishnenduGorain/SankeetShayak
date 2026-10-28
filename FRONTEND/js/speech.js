/**
 * Sankeet Shayak - Speech & Translation
 */
(function() {
  const DICTIONARY = {
    // ASL Alphabet
    'a': { en: 'A', hi: 'ए', bn: 'এ' },
    'b': { en: 'B', hi: 'बी', bn: 'বি' },
    'c': { en: 'C', hi: 'सी', bn: 'সি' },
    'd': { en: 'D', hi: 'डी', bn: 'ডি' },
    'e': { en: 'E', hi: 'ई', bn: 'ই' },
    'f': { en: 'F', hi: 'एफ', bn: 'এফ' },
    'g': { en: 'G', hi: 'जी', bn: 'জি' },
    'h': { en: 'H', hi: 'एच', bn: 'এইচ' },
    'i': { en: 'I', hi: 'आई', bn: 'আই' },
    'j': { en: 'J', hi: 'जे', bn: 'জে' },
    'k': { en: 'K', hi: 'के', bn: 'কে' },
    'l': { en: 'L', hi: 'एल', bn: 'এল' },
    'm': { en: 'M', hi: 'एम', bn: 'এম' },
    'n': { en: 'N', hi: 'एन', bn: 'এন' },
    'o': { en: 'O', hi: 'ओ', bn: 'ও' },
    'p': { en: 'P', hi: 'पी', bn: 'পি' },
    'q': { en: 'Q', hi: 'क्यू', bn: 'কিউ' },
    'r': { en: 'R', hi: 'आर', bn: 'আর' },
    's': { en: 'S', hi: 'एस', bn: 'এস' },
    't': { en: 'T', hi: 'टी', bn: 'টি' },
    'u': { en: 'U', hi: 'यू', bn: 'ইউ' },
    'v': { en: 'V', hi: 'वी', bn: 'ভি' },
    'w': { en: 'W', hi: 'डब्ल्यू', bn: 'ডাব্লিউ' },
    'x': { en: 'X', hi: 'एक्स', bn: 'এক্স' },
    'y': { en: 'Y', hi: 'वाई', bn: 'ওয়াই' },
    'z': { en: 'Z', hi: 'जेड', bn: 'জেড' },
    // Numbers
    '0': { en: 'Zero', hi: 'शून्य', bn: 'শূন্য' },
    '1': { en: 'One', hi: 'एक', bn: 'এক' },
    '2': { en: 'Two', hi: 'दो', bn: 'দুই' },
    '3': { en: 'Three', hi: 'तीन', bn: 'তিন' },
    '4': { en: 'Four', hi: 'चार', bn: 'চার' },
    '5': { en: 'Five', hi: 'पांच', bn: 'পাঁচ' },
    '6': { en: 'Six', hi: 'छह', bn: 'ছয়' },
    '7': { en: 'Seven', hi: 'सात', bn: 'সাত' },
    '8': { en: 'Eight', hi: 'आठ', bn: 'আট' },
    '9': { en: 'Nine', hi: 'नौ', bn: 'নয়' },
    // Common Signs
    'hello': { en: 'Hello', hi: 'नमस्ते', bn: 'নমস্কার' },
    'yes': { en: 'Yes', hi: 'हाँ', bn: 'হ্যাঁ' },
    'no': { en: 'No', hi: 'नहीं', bn: 'না' },
    'thank_you': { en: 'Thank You', hi: 'धन्यवाद', bn: 'ধন্যবাদ' },
    'please': { en: 'Please', hi: 'कृपया', bn: 'দয়া করে' },
    'sorry': { en: 'Sorry', hi: 'माफ़ करना', bn: 'দুঃখিত' },
    'i_love_you': { en: 'I Love You', hi: 'मैं तुमसे प्यार करता हूँ', bn: 'আমি তোমাকে ভালোবাসি' },
    'help': { en: 'Help', hi: 'मदद', bn: 'সাহায্য' },
    'stop': { en: 'Stop', hi: 'रुको', bn: 'থামো' },
    'more': { en: 'More', hi: 'और', bn: 'আরও' },
    'eat': { en: 'Eat', hi: 'खाना', bn: 'খাও' },
    'drink': { en: 'Drink', hi: 'पीना', bn: 'খাও' },
    // Legacy (for backward compatibility)
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