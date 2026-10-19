/**
 * Sankeet Shayak - Biomechanical Motion Analytics
 */
(function() {
  class MotionAnalytics {
    constructor() {
      this.history = [];
      this.maxHistory = 60;
    }

    push(landmarks) {
      if (!landmarks || landmarks.length !== 21) return null;
      // Use wrist (landmark 0) or palm center for trajectory velocity
      const wrist = landmarks[0];
      const now = performance.now() / 1000;
      this.history.push({ t: now, x: wrist[0], y: wrist[1], z: wrist[2] || 0 });
      if (this.history.length > this.maxHistory) this.history.shift();

      return this.computeMetrics();
    }

    computeMetrics() {
      const n = this.history.length;
      if (n < 2) return { speed: 0, accel: 0, jerk: 0, energy: 0 };
      const p1 = this.history[n - 2];
      const p2 = this.history[n - 1];
      const dt = Math.max(0.001, p2.t - p1.t);
      const dx = p2.x - p1.x, dy = p2.y - p1.y, dz = p2.z - p1.z;
      const speed = Math.sqrt(dx*dx + dy*dy + dz*dz) / dt;
      const mass = 0.4; // kg
      const energy = 0.5 * mass * speed * speed;

      return {
        speed: speed,
        accel: 0,
        jerk: 0,
        energy: energy
      };
    }

    reset() {
      this.history = [];
    }
  }

  window.SS_MotionAnalytics = new MotionAnalytics();
})();
