/**
 * Sankeet Shayak Frontend Configuration
 */
(function() {
  const urlParams = new URLSearchParams(window.location.search);
  const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
  
  const savedUrl = localStorage.getItem('ss_backend_url');
  const paramUrl = urlParams.get('backend');
  const defaultUrl = isLocal ? 'http://localhost:8000' : 'https://sankeetshayak-backend.onrender.com';

  const baseUrl = (paramUrl || savedUrl || defaultUrl).replace(/\/+$/, '');

  window.SS_CONFIG = {
    backend: {
      baseUrl: baseUrl,
      transport: urlParams.get('transport') || 'websocket', // 'websocket' | 'http'
      healthEndpoint: '/health',
      signsEndpoint: '/signs',
      predictEndpoint: '/predict',
      wsEndpoint: '/ws/stream',
      reconnectIntervalMs: 3000,
      pollIntervalMs: 5000,
    },
    camera: {
      fps: 30,
      windowSize: 30,
      stride: 3,
    },
    setBackendUrl: function(url) {
      if (!url) return;
      localStorage.setItem('ss_backend_url', url.replace(/\/+$/, ''));
      window.location.reload();
    }
  };
})();
