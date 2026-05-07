# Static correction

`seisviewer2d` exposes backend-only static-correction workflows through the
`/statics/...` API family. The viewer can open corrected TraceStores that these
jobs register, but the static-correction APIs are primarily developer-facing.

## Workflows

- Datum statics: `POST /statics/datum/apply`
  - Computes datum trace shifts and can register a datum-corrected TraceStore.
- First-break QC: `POST /statics/first-break/qc`
  - Validates first-break picks against a datum solution and writes QC artifacts.
- Residual statics: `POST /statics/residual/apply`
  - Estimates residual applied event-time shifts and can register a residual-corrected TraceStore.
- Geometry linkage: `POST /statics/linkage/build`
  - Builds `geometry_linkage.npz`, which maps each sorted trace's source and receiver endpoints to near-surface node IDs.
- Time-term statics: `POST /statics/time-term/apply`
  - Estimates source/receiver node time terms, converts estimated delays to applied weathering shifts, writes artifacts, and can register a time-term-corrected TraceStore.

The detailed time-term inversion API, sign convention, apply modes, and artifact
contract are documented in [time_term_static_correction.md](time_term_static_correction.md).

## Job Lifecycle

Static-correction jobs use common job endpoints:

```bash
curl http://localhost:8000/statics/job/<job_id>/status
curl http://localhost:8000/statics/job/<job_id>/files
curl -L "http://localhost:8000/statics/job/<job_id>/download?name=<artifact_name>" -o <artifact_name>
```

`/statics/job/<job_id>/status` returns `state`, `progress`, and `message`.
`/statics/job/<job_id>/files` returns artifact names and sizes from the job
artifact directory. `/statics/job/<job_id>/download` accepts only a plain file
basename in the `name` query parameter.

## Trace Order

Static-correction artifacts that carry per-trace arrays use TraceStore sorted
trace order. The order is controlled by the uploaded/opened TraceStore key bytes:

- `key1_byte`: groups traces into sections, usually `189`
- `key2_byte`: sorts traces within a section, usually `193`

Do not reorder artifact arrays before applying them to a TraceStore. Per-trace
shift fields named `*_sorted` are aligned with this sorted order.
