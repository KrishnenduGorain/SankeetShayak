/**
 * Sankeet Shayak - Data Contract & Validation
 */
(function() {
  window.SS_DataContract = {
    normalizePrediction: function(raw) {
      if (!raw) return null;
      const label = raw.prediction || raw.label || null;
      const confidence = typeof raw.confidence === 'number' ? raw.confidence : 0;
      const probabilities = raw.probabilities || {};
      const telemetry = raw.telemetry || {};
      const isReliable = raw.is_reliable !== false;
      const warnings = Array.isArray(raw.warnings) ? raw.warnings : [];

      return {
        label: label,
        confidence: confidence,
        probabilities: probabilities,
        telemetry: telemetry,
        isReliable: isReliable,
        warnings: warnings,
        timestamp: Date.now()
      };
    },
    validateLandmarks: function(lm) {
      return Array.isArray(lm) && lm.length === 21;
    }
  };
})();
