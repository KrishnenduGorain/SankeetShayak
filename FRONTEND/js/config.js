/**
 * Sankeet Shayak Frontend Configuration
 */
(function() {
  const urlParams = new URLSearchParams(window.location.search);
  const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
  
  const savedUrl = localStorage.getItem('ss_backend_url');
  const paramUrl = urlParams.get('backend');
  const defaultUrl = 'https://sankeetshayak.onrender.com';

  const baseUrl = (paramUrl || savedUrl || defaultUrl).replace(/\/+$/, '');

  window.SS_CONFIG = {
    backend: {
      baseUrl: baseUrl,
      transport: urlParams.get('transport') || 'auto', // 'auto' | 'http' | 'websocket'
      healthEndpoint: '/health',
      signsEndpoint: '/signs',
      predictEndpoint: '/predict',
      wsEndpoint: '/ws/stream',
      reconnectIntervalMs: 5000,
      pollIntervalMs: 1500,
    },
    camera: {
      fps: 30,
      windowSize: 20,
      stride: 3,
    },
    setBackendUrl: function(url) {
      if (!url) return;
      localStorage.setItem('ss_backend_url', url.replace(/\/+$/, ''));
      window.location.reload();
    }
  };
})();
