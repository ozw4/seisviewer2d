let staticCorrectionDom = null;
let staticCorrectionRender = () => {};

export function setStaticCorrectionDom(value) {
  staticCorrectionDom = value;
}

export function getStaticCorrectionDom() {
  return staticCorrectionDom;
}

export function setStaticCorrectionRender(callback) {
  staticCorrectionRender = typeof callback === 'function' ? callback : () => {};
}

export function requestStaticCorrectionRender() {
  staticCorrectionRender();
}
