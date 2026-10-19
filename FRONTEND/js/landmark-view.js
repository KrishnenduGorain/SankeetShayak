/**
 * Sankeet Shayak - Landmark View (Canvas Rendering)
 */
(function() {
  const EDGES = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [5,9],[9,10],[10,11],[11,12],
    [9,13],[13,14],[14,15],[15,16],
    [13,17],[17,18],[18,19],[19,20],
    [0,17]
  ];

  window.SS_LandmarkView = {
    draw: function(canvas, landmarks) {
      if (!canvas || !landmarks || landmarks.length !== 21) return;
      const ctx = canvas.getContext('2d');
      const w = canvas.width = canvas.clientWidth || 320;
      const h = canvas.height = canvas.clientHeight || 240;

      ctx.clearRect(0, 0, w, h);
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#31d0e2';

      // Connect edges
      for (const [i, j] of EDGES) {
        const p1 = landmarks[i], p2 = landmarks[j];
        ctx.beginPath();
        ctx.moveTo(p1[0] * w, p1[1] * h);
        ctx.lineTo(p2[0] * w, p2[1] * h);
        ctx.stroke();
      }

      // Draw nodes
      for (let i = 0; i < landmarks.length; i++) {
        const p = landmarks[i];
        ctx.beginPath();
        ctx.arc(p[0] * w, p[1] * h, i === 0 ? 5 : 3.5, 0, 2 * Math.PI);
        ctx.fillStyle = i === 0 ? '#8b7bff' : '#80f1fb';
        ctx.fill();
      }
    }
  };
})();
