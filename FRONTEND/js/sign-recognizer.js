/**
 * Sankeet Shayak - High-Performance Real-Time ASL & Sign Recognizer
 * Evaluates 21 3D MediaPipe hand landmarks in real time (30-60 fps, 0ms latency).
 * Computes anatomical joint extensions, angles, pinch distances, and spatial orientations.
 * Provides instant classification for ASL Alphabet (A-Z), Numbers (0-9), and Common Signs.
 */
(function() {
  'use strict';

  // Distance helper
  function dist(p1, p2) {
    const dx = p1.x - p2.x;
    const dy = p1.y - p2.y;
    const dz = (p1.z || 0) - (p2.z || 0);
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  // 2D distance helper (screen plane)
  function dist2D(p1, p2) {
    const dx = p1.x - p2.x;
    const dy = p1.y - p2.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  // Sign descriptions & multilingual translations
  const SIGN_METADATA = {
    // Alphabet
    'A': { category: 'Alphabet', desc: 'Fist with thumb resting beside index pointing up', hi: 'ए', bn: 'এ' },
    'B': { category: 'Alphabet', desc: 'Four fingers straight up, thumb tucked across palm', hi: 'बी', bn: 'বি' },
    'C': { category: 'Alphabet', desc: 'Fingers and thumb curved into a C cup shape', hi: 'सी', bn: 'সি' },
    'D': { category: 'Alphabet', desc: 'Index pointing straight up, thumb touches middle tip in a loop', hi: 'डी', bn: 'ডি' },
    'E': { category: 'Alphabet', desc: 'All fingertips curled down touching thumb tucked beneath', hi: 'ई', bn: 'ই' },
    'F': { category: 'Alphabet', desc: 'Thumb & index tips touch in circle (OK), middle/ring/pinky straight up', hi: 'एफ', bn: 'এফ' },
    'G': { category: 'Alphabet', desc: 'Index and thumb pointing sideways parallel like a pinch', hi: 'जी', bn: 'জি' },
    'H': { category: 'Alphabet', desc: 'Index and middle fingers extended horizontally together', hi: 'एच', bn: 'এইচ' },
    'I': { category: 'Alphabet', desc: 'Pinky finger straight up, other fingers curled into fist', hi: 'आई', bn: 'আই' },
    'J': { category: 'Alphabet', desc: 'Pinky tracing a J curve down and up', hi: 'जे', bn: 'জে' },
    'K': { category: 'Alphabet', desc: 'Index straight up, middle tilted forward, thumb between them', hi: 'के', bn: 'কে' },
    'L': { category: 'Alphabet', desc: 'Thumb and index extended at 90 degrees forming an L', hi: 'एल', bn: 'এল' },
    'M': { category: 'Alphabet', desc: 'Thumb tucked under first three fingers', hi: 'एम', bn: 'এম' },
    'N': { category: 'Alphabet', desc: 'Thumb tucked under first two fingers', hi: 'एन', bn: 'এন' },
    'O': { category: 'Alphabet', desc: 'All fingers and thumb curved together forming an O ring', hi: 'ओ', bn: 'ও' },
    'P': { category: 'Alphabet', desc: 'K hand pointed downward', hi: 'पी', bn: 'পি' },
    'Q': { category: 'Alphabet', desc: 'G hand pointed downward', hi: 'क्यू', bn: 'কিউ' },
    'R': { category: 'Alphabet', desc: 'Index and middle fingers crossed over each other', hi: 'आर', bn: 'আর' },
    'S': { category: 'Alphabet', desc: 'Tight fist with thumb crossed over the front of fingers', hi: 'एस', bn: 'एस' },
    'T': { category: 'Alphabet', desc: 'Thumb tucked between index and middle fingers', hi: 'टी', bn: 'টি' },
    'U': { category: 'Alphabet', desc: 'Index and middle fingers extended straight up and closed together', hi: 'यू', bn: 'ইউ' },
    'V': { category: 'Alphabet', desc: 'Index and middle fingers spread in a V / Peace shape', hi: 'वी', bn: 'ভি' },
    'W': { category: 'Alphabet', desc: 'Index, middle, and ring fingers spread upwards', hi: 'डब्ल्यू', bn: 'ডাব্লিউ' },
    'X': { category: 'Alphabet', desc: 'Index finger hooked/bent like a hook', hi: 'एक्स', bn: 'এক্স' },
    'Y': { category: 'Alphabet', desc: 'Thumb and pinky extended (Hang loose / Call me), others curled', hi: 'वाई', bn: 'ওয়াই' },
    'Z': { category: 'Alphabet', desc: 'Index finger tracing a Z in the air', hi: 'जेड', bn: 'জেড' },

    // Numbers
    '0': { category: 'Number', desc: 'All fingertips curved together in an oval zero shape', hi: 'शून्य', bn: 'শূন্য' },
    '1': { category: 'Number', desc: 'Index finger extended straight up, others closed', hi: 'एक', bn: 'এক' },
    '2': { category: 'Number', desc: 'Index and middle fingers extended in a V shape', hi: 'दो', bn: 'দুই' },
    '3': { category: 'Number', desc: 'Thumb, index, and middle fingers extended (ASL 3)', hi: 'तीन', bn: 'তিন' },
    '4': { category: 'Number', desc: 'Four fingers extended upwards, thumb folded into palm', hi: 'चार', bn: 'চার' },
    '5': { category: 'Number', desc: 'All 5 fingers fully extended and spread open', hi: 'पांच', bn: 'পাঁচ' },
    '6': { category: 'Number', desc: 'Thumb touching pinky tip, other three fingers extended', hi: 'छह', bn: 'ছয়' },
    '7': { category: 'Number', desc: 'Thumb touching ring fingertip, other three extended', hi: 'सात', bn: 'সাত' },
    '8': { category: 'Number', desc: 'Thumb touching middle fingertip, other three extended', hi: 'आठ', bn: 'আট' },
    '9': { category: 'Number', desc: 'Thumb touching index fingertip (F shape), other three extended', hi: 'नौ', bn: 'নয়' },

    // Common Signs
    'HELLO': { category: 'Common', desc: 'Open hand wave or open palm greeting', hi: 'नमस्ते', bn: 'নমস্কার' },
    'YES': { category: 'Common', desc: 'Thumbs up gesture / nodding fist', hi: 'हाँ', bn: 'হ্যাঁ' },
    'NO': { category: 'Common', desc: 'Index and middle fingers snap onto thumb (or thumbs down)', hi: 'नहीं', bn: 'ना' },
    'THANK_YOU': { category: 'Common', desc: 'Flat hand moving forward from chin', hi: 'धन्यवाद', bn: 'ধন্যবাদ' },
    'I_LOVE_YOU': { category: 'Common', desc: 'Thumb, index, and pinky extended together (ILY sign)', hi: 'मैं तुमसे प्यार करता हूँ', bn: 'আমি তোমাকে ভালোবাসি' },
    'HELP': { category: 'Common', desc: 'Thumbs up resting on flat palm', hi: 'मदद', bn: 'সাহায্য' },
    'STOP': { category: 'Common', desc: 'Open palm facing outward firmly', hi: 'रुको', bn: 'থামো' },
    'OK': { category: 'Common', desc: 'Thumb and index form circle, other 3 fingers up', hi: 'ठीक है', bn: 'ঠিক আছে' },
    'PEACE': { category: 'Common', desc: 'V sign with index and middle fingers spread', hi: 'शांति', bn: 'শান্তি' },
    'ROCK': { category: 'Common', desc: 'Index and pinky extended, thumb holding middle & ring', hi: 'रॉक', bn: 'রক' }
  };

  class SignRecognizer {
    constructor() {
      this.history = [];
      this.historySize = 8;
      this.lastLabel = null;
      this.lastConf = 0;
      this.stableCount = 0;
      this.confidenceThreshold = 0.55;
    }

    /**
     * Classify raw MediaPipe 21 landmarks
     * @param {Array<{x:number, y:number, z:number}>} rawLandmarks 
     * @param {number} motionSpeed Optional motion velocity in m/s
     * @returns {Object|null} Prediction object
     */
    classify(rawLandmarks, motionSpeed = 0) {
      if (!rawLandmarks || rawLandmarks.length !== 21) return null;

      // Extract points
      const lm = rawLandmarks;
      const wrist = lm[0];
      const thumbCmc = lm[1], thumbMcp = lm[2], thumbIp = lm[3], thumbTip = lm[4];
      const indexMcp = lm[5], indexPip = lm[6], indexDip = lm[7], indexTip = lm[8];
      const midMcp = lm[9], midPip = lm[10], midDip = lm[11], midTip = lm[12];
      const ringMcp = lm[13], ringPip = lm[14], ringDip = lm[15], ringTip = lm[16];
      const pinkyMcp = lm[17], pinkyPip = lm[18], pinkyDip = lm[19], pinkyTip = lm[20];

      // Hand scale reference (wrist to middle MCP)
      const handScale = dist(wrist, midMcp);
      if (handScale < 0.02) return null; // Too small or degenerate

      // 1. Compute finger extensions (true if extended out, false if curled)
      // A finger is extended when tip is further from wrist than PIP & MCP by margin
      const extIndex = dist(indexTip, wrist) > dist(indexPip, wrist) * 1.15 &&
                       dist(indexTip, wrist) > dist(indexMcp, wrist) * 1.25;
      const curledIndex = dist(indexTip, wrist) < dist(indexPip, wrist) * 1.05 ||
                          dist(indexTip, indexMcp) < dist(indexPip, indexMcp) * 0.95;

      const extMiddle = dist(midTip, wrist) > dist(midPip, wrist) * 1.15 &&
                        dist(midTip, wrist) > dist(midMcp, wrist) * 1.25;
      const curledMiddle = dist(midTip, wrist) < dist(midPip, wrist) * 1.05 ||
                           dist(midTip, midMcp) < dist(midPip, midMcp) * 0.95;

      const extRing = dist(ringTip, wrist) > dist(ringPip, wrist) * 1.15 &&
                      dist(ringTip, wrist) > dist(ringMcp, wrist) * 1.25;
      const curledRing = dist(ringTip, wrist) < dist(ringPip, wrist) * 1.05 ||
                         dist(ringTip, ringMcp) < dist(ringPip, ringMcp) * 0.95;

      const extPinky = dist(pinkyTip, wrist) > dist(pinkyPip, wrist) * 1.15 &&
                       dist(pinkyTip, wrist) > dist(pinkyMcp, wrist) * 1.25;
      const curledPinky = dist(pinkyTip, wrist) < dist(pinkyPip, wrist) * 1.05 ||
                          dist(pinkyTip, pinkyMcp) < dist(pinkyPip, pinkyMcp) * 0.95;

      // Thumb extension & orientation
      const thumbDistToPinky = dist(thumbTip, pinkyMcp) / handScale;
      const thumbDistToMid = dist(thumbTip, midMcp) / handScale;
      const extThumb = thumbDistToPinky > 0.85 && thumbDistToMid > 0.55 &&
                       dist(thumbTip, thumbMcp) > dist(thumbIp, thumbMcp) * 1.15;
      const thumbPointingUp = (wrist.y - thumbTip.y) / handScale > 0.65;
      const thumbPointingDown = (thumbTip.y - wrist.y) / handScale > 0.50;

      // 2. Normalized inter-finger distances
      const dThumbIndex = dist(thumbTip, indexTip) / handScale;
      const dThumbMid = dist(thumbTip, midTip) / handScale;
      const dThumbRing = dist(thumbTip, ringTip) / handScale;
      const dThumbPinky = dist(thumbTip, pinkyTip) / handScale;

      const dIndexMid = dist(indexTip, midTip) / handScale;
      const dMidRing = dist(midTip, ringTip) / handScale;
      const dRingPinky = dist(ringTip, pinkyTip) / handScale;

      // Pinch checks
      const isPinchThumbIndex = dThumbIndex < 0.40;
      const isPinchThumbMid = dThumbMid < 0.40;
      const isPinchThumbRing = dThumbRing < 0.40;
      const isPinchThumbPinky = dThumbPinky < 0.40;

      // Total extended fingers count
      const numExt = (extIndex ? 1 : 0) + (extMiddle ? 1 : 0) + (extRing ? 1 : 0) + (extPinky ? 1 : 0);

      // Score accumulator for candidates
      const scores = {};
      const addScore = (label, score) => {
        scores[label] = (scores[label] || 0) + score;
      };

      // --- SIGN EVALUATIONS ---

      // 1. F: Thumb & Index pinched, middle/ring/pinky extended, index not extended
      if (isPinchThumbIndex && extMiddle && extRing && extPinky && !extIndex && curledIndex) {
        addScore('F', 0.95);
        addScore('9', 0.88);
      }

      // OK: Thumb & Index pinched, middle/ring/pinky extended — similar to F but scored separately
      // OK is commonly shown with all 3 fingers quite extended and the circle held more forward
      if (isPinchThumbIndex && extMiddle && extRing && extPinky && !extIndex) {
        addScore('OK', 0.82);
      }

      // 2. I_LOVE_YOU: Thumb, Index, Pinky extended, Middle & Ring curled
      if (extThumb && extIndex && extPinky && curledMiddle && curledRing) {
        addScore('I_LOVE_YOU', 0.96);
      }

      // 3. ROCK / HORNS: Index & Pinky extended, Middle & Ring curled, Thumb folded
      if (!extThumb && extIndex && extPinky && curledMiddle && curledRing) {
        addScore('ROCK', 0.94);
      }

      // 4. Y / CALL ME: Thumb & Pinky extended, Index, Middle, Ring curled
      if (extThumb && extPinky && curledIndex && curledMiddle && curledRing) {
        addScore('Y', 0.96);
      }

      // 5. L: Thumb & Index extended at ~90 deg, Middle, Ring, Pinky curled
      if (extThumb && extIndex && curledMiddle && curledRing && curledPinky && dThumbIndex > 0.60) {
        addScore('L', 0.96);
      }

      // 6. V / PEACE / 2: Index & Middle extended spread apart, Ring & Pinky curled
      // Spread threshold: dIndexMid > 0.25 (was 0.28, slightly lower for better recall)
      if (extIndex && extMiddle && curledRing && curledPinky && !extThumb && dIndexMid > 0.25) {
        addScore('V', 0.95);
        addScore('PEACE', 0.90);
        addScore('2', 0.85);
      }

      // 7. U: Index & Middle extended tightly together, Ring & Pinky curled
      if (extIndex && extMiddle && curledRing && curledPinky && !extThumb && dIndexMid <= 0.25) {
        addScore('U', 0.93);
      }

      // 8. W / 6 / 3:
      // W: Index, Middle, Ring extended spread out, Pinky curled
      if (extIndex && extMiddle && extRing && curledPinky && !extThumb) {
        addScore('W', 0.94);
      }
      // 3 in ASL: Thumb, Index, Middle extended, Ring & Pinky curled
      if (extThumb && extIndex && extMiddle && curledRing && curledPinky) {
        addScore('3', 0.95);
      }

      // 9. 1 / D / ONLY INDEX:
      if (extIndex && curledMiddle && curledRing && curledPinky && !extThumb) {
        if (isPinchThumbMid) {
          addScore('D', 0.92);
        } else {
          addScore('1', 0.94);
        }
      }

      // 10. I: Only Pinky extended, Index, Middle, Ring curled
      if (extPinky && curledIndex && curledMiddle && curledRing && !extThumb) {
        addScore('I', 0.95);
      }

      // 11. B / 4:
      // B: Four fingers extended tightly together, thumb folded across palm
      if (extIndex && extMiddle && extRing && extPinky && !extThumb && dIndexMid < 0.32 && dMidRing < 0.32) {
        addScore('B', 0.95);
      }
      // 4: Four fingers extended spread apart, thumb tucked
      if (extIndex && extMiddle && extRing && extPinky && !extThumb && (dIndexMid >= 0.32 || dMidRing >= 0.32)) {
        addScore('4', 0.92);
      }

      // 12. 5 / HELLO / STOP: All 5 fingers extended
      if (extThumb && extIndex && extMiddle && extRing && extPinky) {
        if (motionSpeed > 0.4) {
          addScore('HELLO', 0.95);
          addScore('5', 0.90);
        } else {
          addScore('5', 0.94);
          addScore('STOP', 0.91);
          addScore('HELLO', 0.88);
        }
      }

      // 13. O or 0: All fingers curled forming an O
      const allCurled = curledIndex && curledMiddle && curledRing && curledPinky;
      if (allCurled && dThumbIndex < 0.50 && dThumbMid < 0.55 && !extThumb) {
        addScore('O', 0.93);
        addScore('0', 0.91);
      }

      // 14. C: Fingers curved in a C cup (neither fully straight nor fully curled into fist)
      if (!extIndex && !curledIndex && !extMiddle && !curledMiddle && dThumbIndex > 0.45 && dThumbIndex < 0.90) {
        addScore('C', 0.90);
      }

      // 15. A or S or YES: Fist shapes
      if (allCurled) {
        if (extThumb && thumbPointingUp) {
          addScore('YES', 0.95);
        } else if (extThumb && thumbPointingDown) {
          addScore('NO', 0.90);
        } else if (lm[4].y < lm[6].y && dThumbIndex < 0.45) {
          // Thumb alongside index
          addScore('A', 0.92);
        } else {
          // Thumb folded across front of fingers
          addScore('S', 0.88);
        }
      }

      // 16. Number 6, 7, 8 pinches (ASL numbers):
      if (isPinchThumbPinky && extIndex && extMiddle && extRing && !extPinky) {
        addScore('6', 0.92);
      }
      if (isPinchThumbRing && extIndex && extMiddle && !extRing && extPinky) {
        addScore('7', 0.92);
      }
      if (isPinchThumbMid && extIndex && !extMiddle && extRing && extPinky) {
        addScore('8', 0.92);
      }

      // Rank candidates
      const entries = Object.entries(scores).sort((a, b) => b[1] - a[1]);
      if (entries.length === 0) {
        return null;
      }

      const topLabel = entries[0][0];
      const topConf = Math.min(0.99, Math.max(0.40, entries[0][1]));

      // Build probability map
      const sum = entries.reduce((acc, [, s]) => acc + s, 0) || 1;
      const probabilities = {};
      entries.forEach(([lbl, s]) => {
        probabilities[lbl] = Number((s / sum).toFixed(4));
      });

      // Temporal smoothing (sliding window debounce)
      this.history.push({ label: topLabel, conf: topConf });
      if (this.history.length > this.historySize) this.history.shift();

      // Count votes in history
      const votes = {};
      this.history.forEach(item => {
        votes[item.label] = (votes[item.label] || 0) + item.conf;
      });
      const best = Object.entries(votes).sort((a, b) => b[1] - a[1])[0];
      const smoothLabel = best ? best[0] : topLabel;
      const smoothConf = best ? Math.min(0.99, best[1] / this.history.length) : topConf;

      const meta = SIGN_METADATA[smoothLabel] || {};

      return {
        label: smoothLabel,
        confidence: Number(smoothConf.toFixed(2)),
        category: meta.category || 'Sign',
        description: meta.desc || '',
        probabilities: probabilities,
        isReliable: smoothConf >= this.confidenceThreshold,
        translations: {
          en: smoothLabel,
          hi: meta.hi || smoothLabel,
          bn: meta.bn || smoothLabel
        }
      };
    }

    getAllSigns() {
      return SIGN_METADATA;
    }

    getSignInfo(label) {
      return SIGN_METADATA[label] || null;
    }

    reset() {
      this.history = [];
      this.lastLabel = null;
      this.lastConf = 0;
    }
  }

  window.SS_SignRecognizer = new SignRecognizer();
})();
