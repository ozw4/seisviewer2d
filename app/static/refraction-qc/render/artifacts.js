export function renderArtifactsView({
  root,
  bundle,
  viewState,
  appendText,
  createKv,
  createTable,
  filterArtifactRows,
}) {
  const state = viewState;

  function renderArtifacts(content, bundle) {
    if (!bundle) {
      appendText(content, 'No QC bundle loaded.');
      return;
    }
    const { rows, filteredRows } = filterArtifactRows(bundle, state);
    content.appendChild(createKv([
      ['Artifact count', rows.length],
      ['Visible artifacts', filteredRows.length],
      ['Available views', Array.isArray(bundle.available_views) ? bundle.available_views.join(', ') : ''],
      ['Unavailable views', Array.isArray(bundle.unavailable_views) ? bundle.unavailable_views.join(', ') : ''],
    ]));
    if (!filteredRows.length) {
      const missing = document.createElement('p');
      missing.className = 'refraction-qc-placeholder';
      missing.textContent = rows.length
        ? 'No artifact rows match the current artifact filters.'
        : 'No artifact manifest entries are present in this QC bundle.';
      content.appendChild(missing);
      return;
    }
    content.appendChild(createTable({
      columns: ['type', 'name', 'path'],
      records: filteredRows,
    }));
  }

  renderArtifacts(root, bundle);
}
