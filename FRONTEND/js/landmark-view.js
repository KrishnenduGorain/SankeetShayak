/**
 * Sankeet Shayak - 3D Landmark View (Canvas Rendering)
 * Draws real-time 21 MediaPipe hand landmarks with biomechanical linkage rendering
 */
(function() {
  const EDGES = [
    [0,1],[1,2],[2,3],[3,4],        // Thumb
    [0,5],[5,6],[6,7],[7,8],        // Index
    [5,9],[9,10],[10,11],[11,12],   // Middle
    [9,13],[13,14],[14,15],[15,16], // Ring
    [13,17],[17,18],[18,19],[19,20],// Pinky
    [0,17]                          // Palm base
  ];

  const FINGERTIPS = [4, 8, 12, 16, 20];

  window.SS_LandmarkView = {
    draw: function(canvas, landmarks, mirror = true) {
      if (!canvas || !landmarks || landmarks.length !== 21) return;
      const ctx = canvas.getContext('2d');
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const cw = canvas.clientWidth || 320;
      const ch = canvas.clientHeight || 240;

      if (canvas.width !== cw * dpr || canvas.height !== ch * dpr) {
        canvas.width = cw * dpr;
        canvas.height = ch * dpr;
      }

      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, cw, ch);

      const getX = (val) => (mirror ? (1 - val) : val) * cw;
      const getY = (val) => val * ch;

      // Draw palm polygon background
      ctx.beginPath();
      const palmIndices = [0, 1, 5, 9, 13, 17];
      palmIndices.forEach((idx, k) => {
        const x = getX(landmarks[idx][0]), y = getY(landmarks[idx][1]);
        if (k === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.fillStyle = 'rgba(49, 208, 226, 0.12)';
      ctx.fill();

      // Connect linkages
      for (const [i, j] of EDGES) {
        const p1 = landmarks[i], p2 = landmarks[j];
        const x1 = getX(p1[0]), y1 = getY(p1[1]);
        const x2 = getX(p2[0]), y2 = getY(p2[1]);

        const grad = ctx.createLinearGradient(x1, y1, x2, y2);
        grad.addColorStop(0, '#31d0e2');
        grad.addColorStop(1, '#8b7bff');

        ctx.strokeStyle = grad;
        ctx.lineWidth = 2.4;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }

      // Draw 21 joint nodes
      for (let i = 0; i < landmarks.length; i++) {
        const p = landmarks[i];
        const x = getX(p[0]), y = getY(p[1]);
        const isTip = FINGERTIPS.includes(i);
        const isWrist = (i === 0);

        ctx.beginPath();
        const r = isTip ? 5.5 : (isWrist ? 6 : 3.8);
        ctx.arc(x, y, r, 0, 2 * Math.PI);

        if (isWrist) {
          ctx.fillStyle = '#8b7bff';
          ctx.shadowColor = '#8b7bff';
          ctx.shadowBlur = 10;
        } else if (isTip) {
          ctx.fillStyle = '#ffffff';
          ctx.shadowColor = '#31d0e2';
          ctx.shadowBlur = 12;
        } else {
          ctx.fillStyle = '#80f1fb';
          ctx.shadowColor = '#31d0e2';
          ctx.shadowBlur = 6;
        }

        ctx.fill();
        ctx.shadowBlur = 0;
      }

      ctx.restore();
    }
  };
})();
