// Shape-only classifier: rotation/scale-invariant finger-extension rules on MediaPipe world landmarks.
const d3 = (a, b) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
const FINGERS = [[5, 8], [9, 12], [13, 16], [17, 20]];   // [MCP, TIP] for index..pinky

export function fingerStates(w) {
  const four = FINGERS.map(([mcp, tip]) => d3(w[tip], w[0]) / Math.max(d3(w[mcp], w[0]), 1e-6) > 1.5);
  const thumb = d3(w[4], w[5]) / Math.max(d3(w[0], w[9]), 1e-6) > 0.6;
  return { thumb, four };
}

export function classifyShape(w) {
  if (!w || w.length < 21) return { label: 'Unknown', pattern: '' };
  const { thumb, four } = fingerStates(w);
  const k = four.map(b => (b ? 1 : 0)).join('');
  const pattern = (thumb ? 1 : 0) + k;
  const T = thumb;
  let label;
  switch (k) {
    case '0000': label = T ? 'Thumbs Up' : 'Fist'; break;
    case '1111': label = T ? 'Open Palm' : 'Four'; break;
    case '1000': label = T ? 'L-Shape' : 'Point'; break;
    case '1100': label = 'Peace'; break;
    case '1110': label = 'Three'; break;
    case '0001': label = T ? 'Call (Y)' : 'Pinky'; break;
    case '1001': label = 'Rock'; break;
    default: label = `Pattern ${pattern}`;
  }
  return { label, pattern };
}
