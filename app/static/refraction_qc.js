import { initRefractionQcPage } from './refraction-qc/main.js';

export { initRefractionQcPage, loadJob } from './refraction-qc/main.js';

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initRefractionQcPage, { once: true });
} else {
  initRefractionQcPage();
}
