import { initStaticCorrectionPage } from './refraction-static-run/main.js';

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initStaticCorrectionPage, { once: true });
} else {
  initStaticCorrectionPage();
}

export { initStaticCorrectionPage };
