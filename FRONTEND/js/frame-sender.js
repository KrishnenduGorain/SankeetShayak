/**
 * Sankeet Shayak - Frame Sender
 * Buffers and sends landmark coordinates from user webcam or video feed
 */
(function() {
  class FrameSender {
    constructor() {
      this.enabled = true;
      this.active = false;
    }

    send(landmarks) {
      if (!this.enabled || !window.SS_BackendAdapter) return;
      window.SS_BackendAdapter.sendLandmarks(landmarks);
    }

    setEnabled(val) {
      this.enabled = !!val;
    }
  }

  window.SS_FrameSender = new FrameSender();
})();
