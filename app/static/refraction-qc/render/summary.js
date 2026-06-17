export function renderSummaryView({
  root,
  bundle,
  createKv,
  summaryValue,
  warningsCount,
}) {
  function renderSummary(content, bundle) {
    const summary = bundle && typeof bundle.summary === 'object' && bundle.summary ? bundle.summary : {};
    content.appendChild(createKv([
      ['Job ID', bundle.job_id],
      ['File ID', summaryValue(summary, ['file_id', 'input_file_id', 'corrected_file_id']) || bundle.file_id],
      ['State', summary.job_state || summary.status],
      ['Workflow', summary.workflow],
      ['Method', summary.method],
      ['Conversion', summary.conversion_mode],
      ['Layer count', summary.layer_count],
      ['Picks total', summaryValue(summary, ['total_picks', 'total_pick_count', 'pick_count', 'observation_count'])],
      ['Picks used', summaryValue(summary, ['used_picks', 'used_pick_count', 'used_observation_count'])],
      ['Picks rejected', summaryValue(summary, ['rejected_picks', 'rejected_pick_count', 'rejected_observation_count'])],
      ['RMS', summaryValue(summary, ['rms_ms', 'residual_rms_ms', 'used_rms_ms', 'all_rms_ms'])],
      ['MAD', summaryValue(summary, ['mad_ms', 'residual_mad_ms', 'used_mad_ms'])],
      ['Corrected TraceStore', summaryValue(summary, ['corrected_tracestore_status', 'corrected_trace_store_status', 'corrected_file_status', 'apply_status'])],
      ['Warnings', warningsCount(bundle, summary)],
      ['Coordinate mode', bundle.coordinate_mode],
      ['Available views', Array.isArray(bundle.available_views) ? bundle.available_views.join(', ') : ''],
      ['Unavailable views', Array.isArray(bundle.unavailable_views) ? bundle.unavailable_views.join(', ') : ''],
    ]));
  }

  renderSummary(root, bundle);
}
