/**
 * Sankeet Shayak - Resilient Backend Adapter
 * Features dual-transport (WebSocket with seamless HTTP REST auto-fallback)
 * Supports live streaming, health telemetry, and REST /predict requests.
 */
(function() {
  class BackendAdapter {
    constructor(config) {
      this.config = config.backend;
      this.activeTransport = this.config.transport; // 'auto' | 'websocket' | 'http'
      this.ws = null;
      this.state = 'checking'; // 'checking' | 'connected' | 'disconnected'
      this.lastMessage = 'Connecting to backend...';
      this.listeners = { status: [], prediction: [], frameCount: [] };
      this.framesSent = 0;
      this.framesSkipped = 0;
      this.serverCapabilities = null;
      this.reconnectTimer = null;
      
      // REST streaming buffer
      this.restBuffer = [];
      this.minRestFrames = 8;
      this.isPredicting = false;
      this.lastPredictTime = 0;
      this.predictThrottleMs = 350; // minimum ms between REST predictions
    }

    on(event, fn) {
      if (this.listeners[event]) this.listeners[event].push(fn);
    }

    emit(event, data) {
      if (this.listeners[event]) {
        this.listeners[event].forEach(fn => fn(data));
      }
    }

    setState(newState, message) {
      this.state = newState;
      this.emit('status', { state: newState, message: message });
    }

    async checkHealth() {
      const url = `${this.config.baseUrl}${this.config.healthEndpoint}`;
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 12000);
        
        const resp = await fetch(url, {
          method: 'GET',
          headers: { 'Accept': 'application/json' },
          signal: controller.signal
        });
        clearTimeout(timeoutId);

        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        this.serverCapabilities = data;

        // Fetch supported classes
        try {
          const signsResp = await fetch(`${this.config.baseUrl}${this.config.signsEndpoint}`, {
            method: 'GET',
            headers: { 'Accept': 'application/json' }
          });
          if (signsResp.ok) {
            const signsData = await signsResp.json();
            this.serverCapabilities.classes = signsData.classes || [];
          }
        } catch (e) {
          console.warn('Could not load sign classes list:', e);
        }
        return true;
      } catch (err) {
        console.warn('Health check failed:', err.message);
        return false;
      }
    }

    async init() {
      if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
      this.setState('checking', 'Connecting to backend service...');
      
      const healthy = await this.checkHealth();
      if (!healthy) {
        this.setState('disconnected', `Cannot reach backend at ${this.config.baseUrl} (may be waking up)`);
        this.scheduleReconnect();
        return;
      }

      // If user specifically requested pure HTTP
      if (this.config.transport === 'http') {
        this.activeTransport = 'http';
        this.setState('connected', 'Connected via HTTP REST (Production)');
        return;
      }

      // Otherwise attempt WebSocket (auto or websocket)
      this.connectWebSocket();
    }

    connectWebSocket() {
      const wsProto = this.config.baseUrl.startsWith('https') ? 'wss:' : 'ws:';
      const host = this.config.baseUrl.replace(/^https?:\/\//, '');
      const wsUrl = `${wsProto}//${host}${this.config.wsEndpoint}`;

      let wsConnected = false;

      try {
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
          wsConnected = true;
          this.activeTransport = 'websocket';
          this.setState('connected', 'Live WebSocket connected');
        };

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.type === 'inference_update' || data.prediction || data.label) {
              const normalized = window.SS_DataContract.normalizePrediction(data);
              this.emit('prediction', normalized);
            }
          } catch (e) {
            console.error('Error parsing WebSocket message:', e);
          }
        };

        this.ws.onclose = (event) => {
          if (!wsConnected && (this.config.transport === 'auto' || this.config.transport === 'websocket')) {
            // WebSocket upgrade rejected by proxy (e.g. Cloudflare / Render free tier 403)
            console.info('WebSocket upgrade unavailable, activating seamless HTTP REST fallback...');
            this.activeTransport = 'http';
            this.setState('connected', 'Connected via HTTP REST (Auto-fallback)');
            return;
          }
          
          if (this.activeTransport === 'websocket') {
            this.setState('disconnected', 'WebSocket disconnected, reconnecting...');
            this.scheduleReconnect();
          }
        };

        this.ws.onerror = (err) => {
          if (!wsConnected && this.config.transport === 'auto') {
            // Gracefully switch to HTTP without breaking the UI
            this.activeTransport = 'http';
            this.setState('connected', 'Connected via HTTP REST (Auto-fallback)');
          }
        };
      } catch (err) {
        if (this.config.transport === 'auto') {
          this.activeTransport = 'http';
          this.setState('connected', 'Connected via HTTP REST (Auto-fallback)');
        } else {
          this.setState('disconnected', err.message);
          this.scheduleReconnect();
        }
      }
    }

    scheduleReconnect() {
      if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
      this.reconnectTimer = setTimeout(() => {
        this.init();
      }, this.config.reconnectIntervalMs || 5000);
    }

    /**
     * Send landmarks frame (21x3 coordinates)
     */
    sendLandmarks(landmarks) {
      if (!landmarks || landmarks.length !== 21) {
        this.framesSkipped++;
        return;
      }

      // 1. WebSocket Path
      if (this.activeTransport === 'websocket' && this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ landmarks: landmarks }));
        this.framesSent++;
        this.emit('frameCount', { sent: this.framesSent, skipped: this.framesSkipped });
        return;
      }

      // 2. HTTP REST Path
      this.framesSent++;
      this.emit('frameCount', { sent: this.framesSent, skipped: this.framesSkipped });
      
      this.restBuffer.push(landmarks);
      if (this.restBuffer.length > 30) this.restBuffer.shift();

      const now = performance.now();
      if (this.restBuffer.length >= this.minRestFrames && !this.isPredicting && (now - this.lastPredictTime > this.predictThrottleMs)) {
        this.dispatchRestPredict();
      }
    }

    /**
     * Dispatch buffered frames to POST /predict
     */
    async dispatchRestPredict() {
      if (this.isPredicting || this.restBuffer.length < this.minRestFrames) return;
      this.isPredicting = true;
      this.lastPredictTime = performance.now();

      const trajectory = this.restBuffer.slice(-this.minRestFrames);
      const url = `${this.config.baseUrl}${this.config.predictEndpoint}`;

      try {
        const resp = await fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          body: JSON.stringify({
            trajectory: trajectory,
            fps: this.config.camera ? this.config.camera.fps : 30.0
          })
        });

        if (resp.ok) {
          const data = await resp.json();
          const normalized = window.SS_DataContract.normalizePrediction(data);
          this.emit('prediction', normalized);
        }
      } catch (err) {
        console.warn('REST prediction error:', err.message);
      } finally {
        this.isPredicting = false;
      }
    }

    /**
     * Predict from a complete trajectory array
     */
    async predictTrajectory(trajectory, fps = 30.0) {
      const url = `${this.config.baseUrl}${this.config.predictEndpoint}`;
      try {
        const resp = await fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          body: JSON.stringify({ trajectory: trajectory, fps: fps })
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        const normalized = window.SS_DataContract.normalizePrediction(data);
        this.emit('prediction', normalized);
        return normalized;
      } catch (e) {
        console.error('Trajectory prediction failed:', e);
        throw e;
      }
    }

    reset() {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ reset: true }));
      }
      this.restBuffer = [];
      this.framesSent = 0;
      this.framesSkipped = 0;
      this.emit('frameCount', { sent: 0, skipped: 0 });
    }
  }

  window.SS_BackendAdapter = new BackendAdapter(window.SS_CONFIG);
  window.SS_BackendAdapter.init();
})();
