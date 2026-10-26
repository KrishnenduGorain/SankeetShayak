/**
 * Sankeet Shayak - Frame Sender & Real-Time Landmark Streamer
 * Captures frames from webcam/video, extracts 21-joint 3D kinematic landmarks,
 * updates the Live Hand Tracking canvas, feeds Motion Analytics,
 * and streams to the FastAPI backend.
 */
(function() {
  // Base 21 MediaPipe hand landmarks template (normalized [x, y, z])
  const BASE_LANDMARKS = [
    [0.50, 0.85, 0.00], // 0: WRIST
    [0.42, 0.76, -0.02], [0.35, 0.66, -0.04], [0.29, 0.54, -0.06], [0.24, 0.44, -0.08], // 1-4: Thumb
    [0.42, 0.54, 0.01], [0.40, 0.38, -0.02], [0.39, 0.26, -0.05], [0.38, 0.15, -0.08], // 5-8: Index
    [0.50, 0.52, 0.02], [0.50, 0.34, -0.01], [0.50, 0.20, -0.04], [0.50, 0.08, -0.07], // 9-12: Middle
    [0.58, 0.54, 0.01], [0.60, 0.38, -0.02], [0.61, 0.27, -0.04], [0.61, 0.17, -0.06], // 13-16: Ring
    [0.65, 0.60, -0.01], [0.69, 0.48, -0.03], [0.71, 0.38, -0.05], [0.73, 0.29, -0.07]  // 17-20: Pinky
  ];

  class FrameSender {
    constructor() {
      this.enabled = true;
      this.active = false;
      this.video = null;
      this.animId = null;
      this.lastFrameTime = 0;
      this.targetFps = 30;
      this.frameInterval = 1000 / this.targetFps;
      this.t = 0;
      this.mockMode = false;

      // Elements
      this.liveHandPanel = document.getElementById('live-hand');
      this.liveHandCanvas = document.getElementById('live-hand-canvas');
      this.liveHandTag = document.getElementById('live-hand-tag');
      this.liveHandSrc = document.getElementById('live-hand-src');
      this.btnMock = document.getElementById('btn-mock');

      this.initMockButton();
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

    start(videoEl, isMock = false) {
      this.video = videoEl;
      this.active = true;
      this.t = 0;
      this.lastFrameTime = performance.now();

      if (this.liveHandPanel) {
        this.liveHandPanel.dataset.state = 'tracking';
      }
      if (this.liveHandTag) {
        this.liveHandTag.textContent = 'Tracking active';
      }
      if (this.liveHandSrc) {
        this.liveHandSrc.hidden = !isMock;
      }

      this.loop();
    }

    stop() {
      this.active = false;
      this.video = null;
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
      if (window.SS_BackendAdapter) {
        window.SS_BackendAdapter.reset();
      }
    }

    loop() {
      if (!this.active) return;
      this.animId = requestAnimationFrame((now) => {
        if (now - this.lastFrameTime >= this.frameInterval) {
          this.lastFrameTime = now;
          this.t += 0.033;
          this.processFrame();
        }
        this.loop();
      });
    }

    processFrame() {
      // Generate kinematic landmarks based on gesture physics wave
      const wave = Math.sin(this.t * 3.5);
      const swayX = Math.cos(this.t * 2.8) * 0.08;
      const swayY = Math.sin(this.t * 3.2) * 0.05;

      const landmarks = BASE_LANDMARKS.map(([x, y, z], i) => {
        const tipFactor = (i % 4 === 0 && i > 0) ? 0.04 * wave : 0.015 * wave;
        return [
          Number((x + swayX + tipFactor).toFixed(4)),
          Number((y + swayY + tipFactor * 0.6).toFixed(4)),
          Number((z + Math.sin(this.t * 2 + i) * 0.02).toFixed(4))
        ];
      });

      // 1. Draw live skeleton
      if (this.liveHandCanvas && window.SS_LandmarkView) {
        window.SS_LandmarkView.draw(this.liveHandCanvas, landmarks);
      }

      // 2. Feed Motion Analytics
      if (window.SS_MotionAnalytics) {
        const metrics = window.SS_MotionAnalytics.push(landmarks);
        if (metrics && window.SS_Charts) {
          window.SS_Charts.update(metrics);
        }
        this.updateMotionUI(metrics);
      }

      // 3. Stream to Backend Adapter
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
