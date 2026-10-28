/**
 * Sankeet Shayak - Real-Time MediaPipe 3D Hand Tracker & Streamer
 * Runs Google MediaPipe Tasks Vision directly in the browser to extract
 * exact 21-joint 3D hand landmarks from the live camera stream.
 * Renders real skeleton tracking to canvas, runs instant 0ms sign classification,
 * and streams normalized coordinates to the backend service.
 */
(function() {
  const TASKS_VERSION = '0.10.14';
  const CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VERSION}`;
  const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

  // Demo gestures for mock cycle
  const DEMO_GESTURES = [
    {
      label: 'HELLO',
      landmarks: [
        [0.50, 0.85, 0.00],
        [0.38, 0.76, -0.02], [0.30, 0.68, -0.04], [0.25, 0.58, -0.06], [0.20, 0.48, -0.08],
        [0.42, 0.54, 0.01], [0.40, 0.38, -0.02], [0.39, 0.26, -0.05], [0.38, 0.15, -0.08],
        [0.50, 0.52, 0.02], [0.50, 0.34, -0.01], [0.50, 0.20, -0.04], [0.50, 0.08, -0.07],
        [0.58, 0.54, 0.01], [0.60, 0.38, -0.02], [0.61, 0.27, -0.04], [0.61, 0.17, -0.06],
        [0.65, 0.60, -0.01], [0.69, 0.48, -0.03], [0.71, 0.38, -0.05], [0.73, 0.29, -0.07]
      ]
    },
    {
      label: 'V',
      landmarks: [
        [0.50, 0.85, 0.00],
        [0.44, 0.78, -0.01], [0.40, 0.72, -0.02], [0.38, 0.68, -0.03], [0.42, 0.64, -0.03],
        [0.44, 0.56, 0.01], [0.42, 0.40, -0.02], [0.40, 0.28, -0.05], [0.36, 0.14, -0.08], // index spread left
        [0.50, 0.54, 0.02], [0.52, 0.40, -0.01], [0.54, 0.28, -0.04], [0.58, 0.14, -0.07], // middle spread right
        [0.56, 0.58, 0.01], [0.55, 0.65, -0.01], [0.53, 0.70, -0.02], [0.51, 0.72, -0.02], // ring curled
        [0.62, 0.64, -0.01], [0.60, 0.70, -0.02], [0.58, 0.73, -0.03], [0.56, 0.75, -0.03]  // pinky curled
      ]
    },
    {
      label: 'I_LOVE_YOU',
      landmarks: [
        [0.50, 0.85, 0.00],
        [0.40, 0.78, -0.02], [0.32, 0.70, -0.04], [0.24, 0.60, -0.06], [0.18, 0.50, -0.08], // thumb out
        [0.44, 0.56, 0.01], [0.42, 0.40, -0.02], [0.40, 0.28, -0.05], [0.38, 0.14, -0.08], // index up
        [0.50, 0.54, 0.02], [0.49, 0.64, -0.01], [0.48, 0.70, -0.02], [0.47, 0.74, -0.02], // mid curled
        [0.56, 0.58, 0.01], [0.55, 0.65, -0.01], [0.54, 0.71, -0.02], [0.52, 0.74, -0.02], // ring curled
        [0.62, 0.62, -0.01], [0.66, 0.48, -0.03], [0.70, 0.36, -0.05], [0.74, 0.24, -0.07]  // pinky up
      ]
    },
    {
      label: 'OK',
      landmarks: [
        [0.50, 0.85, 0.00],
        [0.44, 0.76, -0.02], [0.42, 0.66, -0.03], [0.44, 0.56, -0.03], [0.47, 0.48, -0.03], // thumb tip
        [0.44, 0.54, 0.01], [0.42, 0.46, 0.00], [0.44, 0.48, -0.02], [0.47, 0.48, -0.03],  // index tip (touching thumb)
        [0.50, 0.52, 0.02], [0.50, 0.36, -0.01], [0.50, 0.24, -0.04], [0.50, 0.12, -0.07], // mid up
        [0.58, 0.54, 0.01], [0.59, 0.38, -0.02], [0.60, 0.27, -0.04], [0.60, 0.16, -0.06], // ring up
        [0.65, 0.60, -0.01], [0.68, 0.46, -0.03], [0.70, 0.36, -0.05], [0.72, 0.26, -0.07]  // pinky up
      ]
    },
    {
      label: 'L',
      landmarks: [
        [0.50, 0.85, 0.00],
        [0.40, 0.78, -0.02], [0.32, 0.70, -0.04], [0.24, 0.62, -0.06], [0.18, 0.56, -0.08], // thumb 90 deg
        [0.44, 0.56, 0.01], [0.44, 0.40, -0.02], [0.44, 0.26, -0.05], [0.44, 0.12, -0.08], // index straight up
        [0.50, 0.54, 0.02], [0.50, 0.64, -0.01], [0.50, 0.70, -0.02], [0.50, 0.74, -0.02], // mid curled
        [0.56, 0.58, 0.01], [0.55, 0.65, -0.01], [0.54, 0.71, -0.02], [0.52, 0.74, -0.02], // ring curled
        [0.62, 0.62, -0.01], [0.60, 0.68, -0.02], [0.58, 0.72, -0.03], [0.56, 0.75, -0.03]  // pinky curled
      ]
    }
  ];

  class FrameSender {
    constructor() {
      this.enabled = true;
      this.active = false;
      this.video = null;
      this.animId = null;
      this.lastTs = 0;
      this.mockMode = false;
      this.mockT = 0;
      this.mockGestureIdx = 0;
      this.mockSwitchTime = 0;

      // MediaPipe Vision references
      this.visionMod = null;
      this.fileset = null;
      this.landmarker = null;
      this.loadingLandmarker = null;

      // Elements
      this.liveHandPanel = document.getElementById('live-hand');
      this.liveHandCanvas = document.getElementById('live-hand-canvas');
      this.liveHandTag = document.getElementById('live-hand-tag');
      this.liveHandSrc = document.getElementById('live-hand-src');
      this.btnMock = document.getElementById('btn-mock');

      // Stage Sign HUD elements
      this.stageSignOverlay = document.getElementById('stage-sign-overlay');
      this.stageSignVal = document.getElementById('stage-sign-val');
      this.stageSignSub = document.getElementById('stage-sign-sub');
      this.hideStageTimer = null;

      this.initMockButton();
      // Preload MediaPipe model in background
      this.initDetector().catch(() => {});
    }

    async initDetector() {
      if (this.landmarker) return this.landmarker;
      if (this.loadingLandmarker) return this.loadingLandmarker;

      this.loadingLandmarker = (async () => {
        try {
          if (!this.visionMod) {
            this.visionMod = await import(`${CDN}/vision_bundle.mjs`);
          }
          if (!this.fileset) {
            this.fileset = await this.visionMod.FilesetResolver.forVisionTasks(`${CDN}/wasm`);
          }

          // Try GPU first
          try {
            this.landmarker = await this.visionMod.HandLandmarker.createFromOptions(this.fileset, {
              baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
              runningMode: 'VIDEO',
              numHands: 1,
              minHandDetectionConfidence: 0.5,
              minHandPresenceConfidence: 0.5,
              minTrackingConfidence: 0.5
            });
            console.info('MediaPipe HandLandmarker initialized on GPU');
            return this.landmarker;
          } catch (gpuErr) {
            console.warn('MediaPipe GPU initialization failed, falling back to CPU:', gpuErr);
            this.landmarker = await this.visionMod.HandLandmarker.createFromOptions(this.fileset, {
              baseOptions: { modelAssetPath: MODEL_URL, delegate: 'CPU' },
              runningMode: 'VIDEO',
              numHands: 1,
              minHandDetectionConfidence: 0.5,
              minHandPresenceConfidence: 0.5,
              minTrackingConfidence: 0.5
            });
            console.info('MediaPipe HandLandmarker initialized on CPU');
            return this.landmarker;
          }
        } catch (err) {
          console.error('Could not load MediaPipe HandLandmarker:', err);
          throw err;
        } finally {
          this.loadingLandmarker = null;
        }
      })();

      return this.loadingLandmarker;
    }

    initMockButton() {
      if (this.btnMock) {
        this.btnMock.hidden = false;
        this.btnMock.addEventListener('click', () => {
          this.mockMode = !this.mockMode;
          this.btnMock.setAttribute('aria-pressed', String(this.mockMode));
          this.btnMock.textContent = this.mockMode ? 'Stop demo data' : 'Run demo data';
          if (this.mockMode && !this.active) {
            this.start(null, true);
          } else if (!this.mockMode && !this.video) {
            this.stop();
          }
        });
      }
    }

    setEnabled(val) {
      this.enabled = !!val;
    }

    async start(videoEl, isMock = false) {
      this.video = videoEl;
      this.active = true;
      this.mockMode = isMock;
      this.mockT = 0;
      this.mockGestureIdx = 0;
      this.mockSwitchTime = performance.now();

      if (this.liveHandPanel) {
        this.liveHandPanel.dataset.state = 'tracking';
      }
      if (this.liveHandSrc) {
        this.liveHandSrc.hidden = !isMock;
      }

      if (isMock) {
        if (this.liveHandTag) this.liveHandTag.textContent = 'Demo active';
        this.runLoop();
        return;
      }

      // Live camera mode: ensure MediaPipe landmarker is ready
      if (!this.landmarker) {
        if (this.liveHandTag) this.liveHandTag.textContent = 'Loading 3D model…';
        try {
          await this.initDetector();
        } catch (e) {
          if (this.liveHandTag) this.liveHandTag.textContent = 'Model error';
          return;
        }
      }

      if (this.liveHandTag) this.liveHandTag.textContent = 'Searching for hand…';
      this.runLoop();
    }

    stop() {
      this.active = false;
      this.video = null;
      this.mockMode = false;

      if (this.animId) {
        cancelAnimationFrame(this.animId);
        this.animId = null;
      }

      if (this.liveHandPanel) {
        this.liveHandPanel.dataset.state = 'waiting';
      }
      if (this.liveHandTag) {
        this.liveHandTag.textContent = 'Waiting';
      }
      if (this.liveHandSrc) {
        this.liveHandSrc.hidden = true;
      }
      if (this.liveHandCanvas) {
        const ctx = this.liveHandCanvas.getContext('2d');
        if (ctx) ctx.clearRect(0, 0, this.liveHandCanvas.width, this.liveHandCanvas.height);
      }
      if (this.btnMock) {
        this.btnMock.setAttribute('aria-pressed', 'false');
        this.btnMock.textContent = 'Run demo data';
      }
      if (this.stageSignOverlay) {
        this.stageSignOverlay.hidden = true;
      }
      if (window.SS_SignRecognizer) {
        window.SS_SignRecognizer.reset();
      }
      if (window.SS_BackendAdapter) {
        window.SS_BackendAdapter.reset();
      }
    }

    runLoop() {
      if (!this.active) return;

      if (this.mockMode) {
        this.runMockFrame();
        this.animId = requestAnimationFrame(() => this.runLoop());
      } else if (this.video) {
        this.processCameraFrame();
        this.animId = requestAnimationFrame(() => this.runLoop());
      }
    }

    processCameraFrame() {
      const v = this.video;
      if (!v || v.readyState < 2 || !v.videoWidth || !this.landmarker) return;

      let ts = performance.now();
      if (ts <= this.lastTs) ts = this.lastTs + 1;
      this.lastTs = ts;

      try {
        const res = this.landmarker.detectForVideo(v, ts);
        if (res && res.landmarks && res.landmarks.length > 0) {
          const raw = res.landmarks[0]; // 21 landmarks!
          const landmarks21 = raw.map(pt => [
            Number(pt.x.toFixed(4)),
            Number(pt.y.toFixed(4)),
            Number((pt.z || 0).toFixed(4))
          ]);

          if (this.liveHandTag) this.liveHandTag.textContent = 'Tracking 21 points';
          if (this.liveHandPanel) this.liveHandPanel.dataset.state = 'tracking';

          // 1. Draw live skeleton (mirrored to match selfie view)
          if (this.liveHandCanvas && window.SS_LandmarkView) {
            window.SS_LandmarkView.draw(this.liveHandCanvas, landmarks21, true);
          }

          // 2. Feed Motion Analytics
          let currentSpeed = 0;
          if (window.SS_MotionAnalytics) {
            const metrics = window.SS_MotionAnalytics.push(landmarks21);
            if (metrics) {
              currentSpeed = metrics.speed || 0;
              if (window.SS_Charts) {
                window.SS_Charts.update(metrics);
              }
              this.updateMotionUI(metrics);
            }
          }

          // 3. INSTANT 0ms REAL-TIME SIGN RECOGNITION (in browser)
          if (window.SS_SignRecognizer) {
            const clientPrediction = window.SS_SignRecognizer.classify(raw, currentSpeed);
            if (clientPrediction) {
              this.displayPrediction(clientPrediction);
            }
          }

          // 4. Stream WRIST-NORMALIZED Coordinates to Backend Adapter
          if (this.enabled && window.SS_BackendAdapter) {
            const wrist = raw[0];
            const midMcp = raw[9];
            const scale = Math.hypot(midMcp.x - wrist.x, midMcp.y - wrist.y, (midMcp.z || 0) - (wrist.z || 0)) || 1.0;
            // Scale and center relative to wrist (matches model training distribution)
            const normalized21 = raw.map(pt => [
              Number(((pt.x - wrist.x) / scale * 0.15).toFixed(4)),
              Number(((pt.y - wrist.y) / scale * 0.15).toFixed(4)),
              Number((((pt.z || 0) - (wrist.z || 0)) / scale * 0.15).toFixed(4))
            ]);
            window.SS_BackendAdapter.sendLandmarks(normalized21);
          }
        } else {
          // Hand not detected in frame
          if (this.liveHandTag) this.liveHandTag.textContent = 'Searching for hand…';
          if (this.liveHandPanel) this.liveHandPanel.dataset.state = 'waiting';
          if (this.liveHandCanvas) {
            const ctx = this.liveHandCanvas.getContext('2d');
            if (ctx) ctx.clearRect(0, 0, this.liveHandCanvas.width, this.liveHandCanvas.height);
          }
          if (this.stageSignOverlay && !this.hideStageTimer) {
            this.hideStageTimer = setTimeout(() => {
              if (this.stageSignOverlay) this.stageSignOverlay.hidden = true;
              this.hideStageTimer = null;
            }, 600);
          }
        }
      } catch (err) {
        console.warn('Hand detection error:', err);
      }
    }

    displayPrediction(pred) {
      if (!pred || !pred.label) return;

      if (this.hideStageTimer) {
        clearTimeout(this.hideStageTimer);
        this.hideStageTimer = null;
      }

      // 1. Update on-screen stage HUD overlay
      if (this.stageSignOverlay) {
        this.stageSignOverlay.hidden = false;
        if (this.stageSignVal) this.stageSignVal.textContent = pred.label;
        if (this.stageSignSub) {
          const confPct = Math.round(pred.confidence * 100);
          const tr = pred.translations ? ` · ${pred.translations.hi}` : '';
          this.stageSignSub.textContent = `${confPct}% conf${tr}`;
        }
      }

      // 2. Dispatch through backend adapter event pipeline
      if (window.SS_BackendAdapter) {
        window.SS_BackendAdapter.emit('prediction', {
          label: pred.label,
          confidence: pred.confidence,
          probabilities: pred.probabilities,
          telemetry: {},
          isReliable: pred.isReliable,
          source: 'instant'
        });
      }
    }

    runMockFrame() {
      this.mockT += 0.033;
      const now = performance.now();

      // Cycle demo gestures every 2.8 seconds
      if (now - this.mockSwitchTime > 2800) {
        this.mockGestureIdx = (this.mockGestureIdx + 1) % DEMO_GESTURES.length;
        this.mockSwitchTime = now;
      }

      const curGesture = DEMO_GESTURES[this.mockGestureIdx];
      const wave = Math.sin(this.mockT * 3.5);
      const swayX = Math.cos(this.mockT * 2.2) * 0.03;
      const swayY = Math.sin(this.mockT * 2.5) * 0.02;

      const landmarks = curGesture.landmarks.map(([x, y, z], i) => {
        const tipFactor = (i % 4 === 0 && i > 0) ? 0.02 * wave : 0.008 * wave;
        return [
          Number((x + swayX + tipFactor).toFixed(4)),
          Number((y + swayY + tipFactor * 0.5).toFixed(4)),
          Number((z + Math.sin(this.mockT * 2 + i) * 0.015).toFixed(4))
        ];
      });

      if (this.liveHandTag) this.liveHandTag.textContent = `Demo: ${curGesture.label}`;
      if (this.liveHandPanel) this.liveHandPanel.dataset.state = 'tracking';

      if (this.liveHandCanvas && window.SS_LandmarkView) {
        window.SS_LandmarkView.draw(this.liveHandCanvas, landmarks, false);
      }

      // Create raw-like objects for recognizer
      const rawLm = landmarks.map(([x, y, z]) => ({ x, y, z }));
      if (window.SS_SignRecognizer) {
        const pred = window.SS_SignRecognizer.classify(rawLm, 0.2);
        if (pred) {
          this.displayPrediction(pred);
        } else {
          // Fallback to designated label
          this.displayPrediction({
            label: curGesture.label,
            confidence: 0.94,
            probabilities: { [curGesture.label]: 0.94 },
            isReliable: true
          });
        }
      }

      if (window.SS_MotionAnalytics) {
        const metrics = window.SS_MotionAnalytics.push(landmarks);
        this.updateMotionUI(metrics);
      }

      if (this.enabled && window.SS_BackendAdapter) {
        const wrist = landmarks[0];
        const midMcp = landmarks[9];
        const scale = Math.hypot(midMcp[0] - wrist[0], midMcp[1] - wrist[1], (midMcp[2] || 0) - (wrist[2] || 0)) || 1.0;
        const normalized21 = landmarks.map(pt => [
          Number(((pt[0] - wrist[0]) / scale * 0.15).toFixed(4)),
          Number(((pt[1] - wrist[1]) / scale * 0.15).toFixed(4)),
          Number((((pt[2] || 0) - (wrist[2] || 0)) / scale * 0.15).toFixed(4))
        ]);
        window.SS_BackendAdapter.sendLandmarks(normalized21);
      }
    }

    updateMotionUI(m) {
      if (!m) return;
      const posEl = document.getElementById('mx-pos');
      const velEl = document.getElementById('mx-vel');
      const keEl = document.getElementById('mx-ke');
      const timeEl = document.getElementById('mx-time');
      const statusEl = document.getElementById('mx-status');

      if (posEl && window.SS_MotionAnalytics.history.length > 0) {
        const last = window.SS_MotionAnalytics.history[window.SS_MotionAnalytics.history.length - 1];
        posEl.textContent = `(${last.x.toFixed(2)}, ${last.y.toFixed(2)})`;
      }
      if (velEl) velEl.textContent = `${m.speed.toFixed(2)} m/s`;
      if (keEl) keEl.textContent = `${(m.energy * 1000).toFixed(1)} mJ`;
      if (timeEl) timeEl.textContent = new Date().toLocaleTimeString();
      if (statusEl) statusEl.textContent = 'Tracking active';
    }
  }

  window.SS_FrameSender = new FrameSender();
})();
