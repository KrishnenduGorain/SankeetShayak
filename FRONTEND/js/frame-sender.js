/**
 * Sankeet Shayak - Real-Time MediaPipe 3D Hand Tracker & Streamer
 * Runs Google MediaPipe Tasks Vision directly in the browser to extract
 * exact 21-joint 3D hand landmarks from the live camera stream.
 * Renders real skeleton tracking to canvas and streams coordinates to backend.
 */
(function() {
  const TASKS_VERSION = '0.10.14';
  const CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VERSION}`;
  const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

  // Base template landmarks for demo mode
  const BASE_LANDMARKS = [
    [0.50, 0.85, 0.00],
    [0.42, 0.76, -0.02], [0.35, 0.66, -0.04], [0.29, 0.54, -0.06], [0.24, 0.44, -0.08],
    [0.42, 0.54, 0.01], [0.40, 0.38, -0.02], [0.39, 0.26, -0.05], [0.38, 0.15, -0.08],
    [0.50, 0.52, 0.02], [0.50, 0.34, -0.01], [0.50, 0.20, -0.04], [0.50, 0.08, -0.07],
    [0.58, 0.54, 0.01], [0.60, 0.38, -0.02], [0.61, 0.27, -0.04], [0.61, 0.17, -0.06],
    [0.65, 0.60, -0.01], [0.69, 0.48, -0.03], [0.71, 0.38, -0.05], [0.73, 0.29, -0.07]
  ];

  class FrameSender {
    constructor() {
      this.enabled = true;
      this.active = false;
      this.video = null;
      this.animId = null;
      this.vfcId = null;
      this.lastTs = 0;
      this.mockMode = false;
      this.mockT = 0;

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
          if (window.SS_MotionAnalytics) {
            const metrics = window.SS_MotionAnalytics.push(landmarks21);
            if (metrics && window.SS_Charts) {
              window.SS_Charts.update(metrics);
            }
            this.updateMotionUI(metrics);
          }

          // 3. Stream to Backend Adapter
          if (this.enabled && window.SS_BackendAdapter) {
            window.SS_BackendAdapter.sendLandmarks(landmarks21);
          }
        } else {
          // Hand not detected in frame
          if (this.liveHandTag) this.liveHandTag.textContent = 'Searching for hand…';
          if (this.liveHandPanel) this.liveHandPanel.dataset.state = 'waiting';
          if (this.liveHandCanvas) {
            const ctx = this.liveHandCanvas.getContext('2d');
            if (ctx) ctx.clearRect(0, 0, this.liveHandCanvas.width, this.liveHandCanvas.height);
          }
        }
      } catch (err) {
        console.warn('Hand detection error:', err);
      }
    }

    runMockFrame() {
      this.mockT += 0.033;
      const wave = Math.sin(this.mockT * 3.5);
      const swayX = Math.cos(this.mockT * 2.8) * 0.08;
      const swayY = Math.sin(this.mockT * 3.2) * 0.05;

      const landmarks = BASE_LANDMARKS.map(([x, y, z], i) => {
        const tipFactor = (i % 4 === 0 && i > 0) ? 0.04 * wave : 0.015 * wave;
        return [
          Number((x + swayX + tipFactor).toFixed(4)),
          Number((y + swayY + tipFactor * 0.6).toFixed(4)),
          Number((z + Math.sin(this.mockT * 2 + i) * 0.02).toFixed(4))
        ];
      });

      if (this.liveHandTag) this.liveHandTag.textContent = 'Demo 21 points';
      if (this.liveHandPanel) this.liveHandPanel.dataset.state = 'tracking';

      if (this.liveHandCanvas && window.SS_LandmarkView) {
        window.SS_LandmarkView.draw(this.liveHandCanvas, landmarks, false);
      }

      if (window.SS_MotionAnalytics) {
        const metrics = window.SS_MotionAnalytics.push(landmarks);
        this.updateMotionUI(metrics);
      }

      if (this.enabled && window.SS_BackendAdapter) {
        window.SS_BackendAdapter.sendLandmarks(landmarks);
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
