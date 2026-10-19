/**
 * Sankeet Shayak - Backend Adapter
 * Connects to FastAPI backend via WebSocket (/ws/stream) and REST (/health, /predict)
 */
(function() {
  class BackendAdapter {
    constructor(config) {
      this.config = config.backend;
      this.ws = null;
      this.state = 'checking'; // 'checking' | 'connected' | 'disconnected'
      this.listeners = { status: [], prediction: [], frameCount: [] };
      this.framesSent = 0;
      this.framesSkipped = 0;
      this.serverCapabilities = null;
      this.reconnectTimer = null;
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
        const resp = await fetch(url, { method: 'GET', headers: { 'Accept': 'application/json' } });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        this.serverCapabilities = data;
        try {
          const signsResp = await fetch(`${this.config.baseUrl}${this.config.signsEndpoint}`, { method: 'GET', headers: { 'Accept': 'application/json' } });
          if (signsResp.ok) {
            const signsData = await signsResp.json();
            this.serverCapabilities.classes = signsData.classes;
          }
        } catch (e) {}
        return true;
      } catch (err) {
        return false;
      }
    }

    async init() {
      this.setState('checking', 'Connecting to backend...');
      const healthy = await this.checkHealth();
      if (!healthy) {
        this.setState('disconnected', `Cannot reach backend at ${this.config.baseUrl}`);
        this.scheduleReconnect();
        return;
      }

      if (this.config.transport === 'websocket') {
        this.connectWebSocket();
      } else {
        this.setState('connected', 'Connected via HTTP REST');
      }
    }

    connectWebSocket() {
      const wsProto = this.config.baseUrl.startsWith('https') ? 'wss:' : 'ws:';
      const host = this.config.baseUrl.replace(/^https?:\/\//, '');
      const wsUrl = `${wsProto}//${host}${this.config.wsEndpoint}`;

      try {
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
          this.setState('connected', 'Live WebSocket connected');
        };

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.type === 'inference_update' || data.prediction || data.label) {
              const normalized = window.SS_DataContract.normalizePrediction(data);
              this.emit('prediction', normalized);
            }
          } catch (e) {}
        };

        this.ws.onclose = () => {
          this.setState('disconnected', 'WebSocket disconnected');
          this.scheduleReconnect();
        };

        this.ws.onerror = () => {
          this.setState('disconnected', 'WebSocket error');
        };
      } catch (err) {
        this.setState('disconnected', err.message);
        this.scheduleReconnect();
      }
    }

    scheduleReconnect() {
      if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
      this.reconnectTimer = setTimeout(() => {
        this.init();
      }, this.config.reconnectIntervalMs || 5000);
    }

    sendLandmarks(landmarks) {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ landmarks: landmarks }));
        this.framesSent++;
        this.emit('frameCount', { sent: this.framesSent, skipped: this.framesSkipped });
      } else {
        this.framesSkipped++;
        this.emit('frameCount', { sent: this.framesSent, skipped: this.framesSkipped });
      }
    }

    reset() {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ reset: true }));
      }
      this.framesSent = 0;
      this.framesSkipped = 0;
      this.emit('frameCount', { sent: 0, skipped: 0 });
    }
  }

  window.SS_BackendAdapter = new BackendAdapter(window.SS_CONFIG);
  window.SS_BackendAdapter.init();
})();
