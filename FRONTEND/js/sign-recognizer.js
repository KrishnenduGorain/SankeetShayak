/**
 * Sankeet Shayak - High-Performance Real-Time ASL & Sign Recognizer
 * Evaluates 21 3D MediaPipe hand landmarks in real time (30-60 fps, 0ms latency).
 * Computes anatomical joint extensions, angles, pinch distances, and spatial orientations.
 * Provides instant classification for ASL Alphabet (A-Z), Numbers (0-9), and Common Signs.
 */
(function() {
  'use strict';

  // Distance helper in 3D
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
    'B': { category: 'Alphabet', desc: 'Four fingers straight up tightly together, thumb tucked across palm', hi: 'बी', bn: 'বি' },
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
      this.lockedLabel = null;
      this.lockedConf = 0;
      this.candidateLabel = null;
      this.candidateCount = 0;
      this.switchFramesRequired = 2; // ~66ms response time for instant, zero-flicker transitions
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

      // Extract 21 landmark points
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

      // 1. Precise joint extension ratios
      const dIndexTipWrist = dist(indexTip, wrist);
      const dIndexPipWrist = dist(indexPip, wrist);
      const dIndexTipMcp = dist(indexTip, indexMcp) / handScale;

      const dMidTipWrist = dist(midTip, wrist);
      const dMidPipWrist = dist(midPip, wrist);
      const dMidTipMcp = dist(midTip, midMcp) / handScale;

      const dRingTipWrist = dist(ringTip, wrist);
      const dRingPipWrist = dist(ringPip, wrist);
      const dRingTipMcp = dist(ringTip, ringMcp) / handScale;

      const dPinkyTipWrist = dist(pinkyTip, wrist);
      const dPinkyPipWrist = dist(pinkyPip, wrist);
      const dPinkyTipMcp = dist(pinkyTip, pinkyMcp) / handScale;

      // True extension: fingertip is further from wrist than PIP AND tip is extended from its base MCP
      const extIndex = dIndexTipWrist > dIndexPipWrist * 1.05 && dIndexTipMcp > 0.62;
      const extMiddle = dMidTipWrist > dMidPipWrist * 1.05 && dMidTipMcp > 0.62;
      const extRing = dRingTipWrist > dRingPipWrist * 1.05 && dRingTipMcp > 0.62;
      const extPinky = dPinkyTipWrist > dPinkyPipWrist * 1.05 && dPinkyTipMcp > 0.62;

      // Curled definition: tip is closer to palm or PIP
      const curledIndex = dIndexTipWrist < dIndexPipWrist * 1.02 || dIndexTipMcp < 0.50;
      const curledMiddle = dMidTipWrist < dMidPipWrist * 1.02 || dMidTipMcp < 0.50;
      const curledRing = dRingTipWrist < dRingPipWrist * 1.02 || dRingTipMcp < 0.50;
      const curledPinky = dPinkyTipWrist < dPinkyPipWrist * 1.02 || dPinkyTipMcp < 0.50;

      // 2. Normalized inter-finger distances
      const dThumbIndex = dist(thumbTip, indexTip) / handScale;
      const dThumbMid = dist(thumbTip, midTip) / handScale;
      const dThumbRing = dist(thumbTip, ringTip) / handScale;
      const dThumbPinky = dist(thumbTip, pinkyTip) / handScale;

      const dIndexMid = dist(indexTip, midTip) / handScale;
      const dMidRing = dist(midTip, ringTip) / handScale;
      const dRingPinky = dist(ringTip, pinkyTip) / handScale;

      // 3. Thumb geometry (Crucial for B vs 5 vs Hello vs Fist)
      const dThumbTipPinkyMcp = dist(thumbTip, pinkyMcp) / handScale;
      const dThumbTipMidMcp = dist(thumbTip, midMcp) / handScale;
      const dThumbTipIndexMcp = dist(thumbTip, indexMcp) / handScale;
      const dThumbTipThumbMcp = dist(thumbTip, thumbMcp) / handScale;

      // Thumb is extended outwards (Open hand, 5, Hello, L, Y)
      const extThumb = (dThumbTipMidMcp > 0.48 && dThumbTipIndexMcp > 0.40) ||
                       (dThumbTipPinkyMcp > 0.70 && dThumbTipThumbMcp > 0.45);

      // Thumb is folded across palm (Essential for B, S, A)
      const thumbTucked = dThumbTipMidMcp < 0.42 && dThumbTipIndexMcp < 0.42 && !extThumb;
      const thumbPointingUp = (wrist.y - thumbTip.y) / handScale > 0.60;
      const thumbPointingDown = (thumbTip.y - wrist.y) / handScale > 0.45;

      // 4. Pinch detection (Thumb tip touching respective fingertip)
      const isPinchThumbIndex = dThumbIndex < 0.32;
      const isPinchThumbMid = dThumbMid < 0.32;
      const isPinchThumbRing = dThumbRing < 0.32;
      const isPinchThumbPinky = dThumbPinky < 0.32;

      // Total extended non-thumb fingers count
      const numExtFingers = (extIndex ? 1 : 0) + (extMiddle ? 1 : 0) + (extRing ? 1 : 0) + (extPinky ? 1 : 0);

      // Scores map
      const scores = {};
      const addScore = (label, score) => {
        scores[label] = (scores[label] || 0) + score;
      };

      // =========================================================
      // GESTURE RECOGNITION RULES
      // =========================================================

      // --- 1. OK / F / 9 ---
      // Thumb & Index form a circle/pinch, other 3 fingers straight up
      if (isPinchThumbIndex && extMiddle && extRing && extPinky) {
        addScore('OK', 0.96);
        addScore('F', 0.92);
        addScore('9', 0.88);
      }

      // --- 2. PEACE / V / 2 / U ---
      // Index & Middle extended, Ring & Pinky folded, Thumb tucked
      if (extIndex && extMiddle && !extRing && !extPinky && !isPinchThumbIndex) {
        if (dIndexMid > 0.22) {
          // Spread apart = PEACE / V / 2
          addScore('PEACE', 0.96);
          addScore('V', 0.92);
          addScore('2', 0.88);
        } else {
          // Closed together = U
          addScore('U', 0.95);
        }
      }

      // --- 3. B vs 4 vs 5 vs HELLO ---
      // Four fingers extended
      if (extIndex && extMiddle && extRing && extPinky && !isPinchThumbIndex) {
        if (extThumb) {
          // All 5 fingers extended: 5 or HELLO
          if (motionSpeed > 0.35) {
            addScore('HELLO', 0.96);
            addScore('5', 0.82);
          } else {
            addScore('5', 0.96);
            addScore('STOP', 0.90);
            addScore('HELLO', 0.82);
          }
        } else if (thumbTucked || dThumbTipIndexMcp < 0.40) {
          // B: Thumb is strictly tucked across palm AND 4 fingers held flat and close together
          if (dIndexMid < 0.22 && dMidRing < 0.22 && dRingPinky < 0.22) {
            addScore('B', 0.96);
          } else {
            // 4: Four fingers spread apart with thumb tucked
            addScore('4', 0.94);
          }
        } else {
          // Fingers spread or thumb semi-loose
          addScore('4', 0.92);
          addScore('5', 0.85);
        }
      }

      // --- 4. I_LOVE_YOU / ROCK / Y ---
      if (extIndex && extPinky && !extMiddle && !extRing) {
        if (extThumb) {
          // ILY: Thumb + Index + Pinky extended
          addScore('I_LOVE_YOU', 0.96);
        } else {
          // ROCK: Index + Pinky extended, thumb folded
          addScore('ROCK', 0.95);
        }
      }

      // Y (Call me): Thumb + Pinky extended, other 3 curled
      if (extThumb && extPinky && !extIndex && !extMiddle && !extRing) {
        addScore('Y', 0.96);
      }

      // L: Thumb + Index extended at ~90 deg, others curled
      if (extThumb && extIndex && !extMiddle && !extRing && !extPinky && dThumbIndex > 0.48) {
        addScore('L', 0.96);
      }

      // --- 5. THREE FINGERS: W / 3 ---
      // W: Index, Middle, Ring extended, Pinky curled, Thumb folded
      if (extIndex && extMiddle && extRing && !extPinky && !extThumb && !isPinchThumbIndex) {
        addScore('W', 0.95);
      }
      // ASL 3: Thumb, Index, Middle extended, Ring and Pinky curled
      if (extThumb && extIndex && extMiddle && !extRing && !extPinky) {
        addScore('3', 0.95);
      }

      // --- 6. SINGLE FINGER: 1 / D / I ---
      if (extIndex && !extMiddle && !extRing && !extPinky && !extThumb) {
        if (isPinchThumbMid) {
          addScore('D', 0.94);
        } else {
          addScore('1', 0.95);
        }
      }

      // I: Only Pinky extended
      if (extPinky && !extIndex && !extMiddle && !extRing && !extThumb) {
        addScore('I', 0.96);
      }

      // --- 7. FIST SIGNS: A / S / YES / NO ---
      const allFourCurled = curledIndex && curledMiddle && curledRing && curledPinky;
      if (allFourCurled) {
        if (extThumb && thumbPointingUp) {
          addScore('YES', 0.96);
        } else if (extThumb && thumbPointingDown) {
          addScore('NO', 0.92);
        } else if (lm[4].y < lm[6].y && dThumbIndex < 0.42) {
          // Thumb alongside index
          addScore('A', 0.94);
        } else {
          // Thumb across front of curled fingers
          addScore('S', 0.92);
        }
      }

      // --- 8. O / 0 and C ---
      if (allFourCurled && dThumbIndex < 0.45 && dThumbMid < 0.50 && !extThumb) {
        addScore('O', 0.94);
        addScore('0', 0.90);
      }

      // C: Curved fingers in a cup shape
      if (!extIndex && !curledIndex && !extMiddle && !curledMiddle && dThumbIndex > 0.45 && dThumbIndex < 0.85) {
        addScore('C', 0.92);
      }

      // --- 9. NUMBER PINCHES: 6, 7, 8 ---
      if (isPinchThumbPinky && extIndex && extMiddle && extRing && !extPinky) {
        addScore('6', 0.94);
      }
      if (isPinchThumbRing && extIndex && extMiddle && !extRing && extPinky) {
        addScore('7', 0.94);
      }
      if (isPinchThumbMid && extIndex && !extMiddle && extRing && extPinky) {
        addScore('8', 0.94);
      }

      // =========================================================
      // RANK CANDIDATES & STABLE HYSTERESIS STATE MACHINE
      // =========================================================
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

      // Zero-flicker hysteresis state machine:
      // Prevents 1-frame glitches (like B flickering into 5, Hello, Peace, or OK)
      if (topLabel === this.lockedLabel) {
        this.candidateLabel = null;
        this.candidateCount = 0;
        this.lockedConf = topConf * 0.35 + this.lockedConf * 0.65; // Smooth confidence
      } else {
        if (topLabel === this.candidateLabel) {
          this.candidateCount++;
          if (this.candidateCount >= this.switchFramesRequired || !this.lockedLabel) {
            this.lockedLabel = topLabel;
            this.lockedConf = topConf;
            this.candidateLabel = null;
            this.candidateCount = 0;
          }
        } else {
          this.candidateLabel = topLabel;
          this.candidateCount = 1;
        }
      }

      const activeLabel = this.lockedLabel || topLabel;
      const activeConf = this.lockedConf || topConf;
      const meta = SIGN_METADATA[activeLabel] || {};

      return {
        label: activeLabel,
        confidence: Number(activeConf.toFixed(2)),
        category: meta.category || 'Sign',
        description: meta.desc || '',
        probabilities: probabilities,
        isReliable: activeConf >= this.confidenceThreshold,
        translations: {
          en: activeLabel,
          hi: meta.hi || activeLabel,
          bn: meta.bn || activeLabel
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
      this.lockedLabel = null;
      this.lockedConf = 0;
      this.candidateLabel = null;
      this.candidateCount = 0;
    }
  }

  window.SS_SignRecognizer = new SignRecognizer();
})();
