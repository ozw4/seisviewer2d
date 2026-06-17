export function getCanvas2dContext(canvas) {
  if (!canvas || typeof canvas.getContext !== 'function') return null;
  return canvas.getContext('2d');
}
