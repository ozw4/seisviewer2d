# Static Correction UI Workflow

This page describes the browser workflow for running refraction statics from the
current viewer file and a directly selected first-break pick NPZ:

```text
SGY load -> direct first-break pick NPZ selection -> Static Correction tab -> refraction apply job -> Refraction QC
```

Synthetic data creation is not part of the Static Correction UI. If synthetic
SEG-Y files or synthetic first-break picks are used, prepare them outside the UI
and load them the same way as normal SEG-Y and pick NPZ files. For the repo's
synthetic smoke-test fixture, see
[refraction_static_ui_fixture.md](refraction_static_ui_fixture.md).

## Prerequisites

- A SEG-Y or TraceStore file is already open in the viewer.
- The viewer was loaded with the intended section and trace sort keys.
- A first-break pick NPZ is available locally and matches the loaded viewer
  file's sorted trace order.
- The TraceStore header bytes describe the source, receiver, coordinate,
  elevation, scalar, and offset fields needed by the selected model.

The viewer first-break probability cache is not a statics pick NPZ.

## Static Correction vs Refraction QC

Use the `Static Correction` tab to create a refraction static job. It collects
the current viewer target, selected pick NPZ, geometry headers, optional
endpoint linkage, model preset, output options, and export options, then submits
the request and NPZ to `/statics/refraction/apply-with-picks`.

Use the `Refraction QC` tab after a refraction job completes. It loads the
completed job's QC bundle and displays first-break residuals, reduced-time/LMO,
profiles, cell maps, static components, and gather preview where the required
artifacts are available. It does not create the static solution.

## Input and First-Break Picks

Open `/` in the browser, load an SGY/TraceStore in the viewer, then select
`Static Correction`.

In `Target`, confirm that the tab shows:

- `Current viewer file`: the active viewer file;
- `Sort keys`: the active viewer `key1_byte` and `key2_byte`;
- `Status`: the loaded target state.

The Static Correction tab does not ask users to type `file_id`, `key1_byte`, or
`key2_byte`. Those values come from the active viewer target.

In `First-break picks`, choose a `.npz` file with `First-break pick NPZ`. The
selected file is sent directly as multipart field `pick_npz`; users do not need
a pick `job_id`, `artifact_name`, or manual batch job registration.

## Geometry Header Preset

The `Geometry` preset controls which SEG-Y trace-header byte locations are sent
as geometry inputs:

- `SEG-Y default` fills the current default byte map for source and receiver
  IDs, coordinates, elevations, coordinate/elevation scalars, source depth, and
  offset.
- `custom` lets you edit those fields for files that use different header
  locations.

Coordinate and elevation units declare the units after header scalar handling.
The refraction model uses geometry distances in meters.

## Linkage

Endpoint linkage is off by default. With the checkbox unchecked, the apply
request uses:

```json
{
  "linkage": {
    "mode": "none"
  }
}
```

Check `Link matching source/receiver endpoints` only when source and receiver
endpoints that represent the same near-surface station should share endpoint
node IDs. The UI first submits `/statics/linkage/build`, waits for the linkage
job to finish, and then submits the refraction apply job with the generated
`geometry_linkage.npz` reference.

The linkage checkbox builds source/receiver endpoint linkage from station and
coordinate criteria. It is not a global merge of all sources and receivers into
one node, and it does not change the input SEG-Y or TraceStore geometry.

## Model Presets

`Model preset` selects the request shape:

- `One-layer global V2/T1`: one-layer T1LSST with
  `model.method="gli_variable_thickness"` and global or fixed V2.
- `Two-layer global V3/T2`: public two-layer T1LSST with V2/T1 and V3/T2
  layers. V3/T2 is global, not cell solved.
- `Three-layer global Vsub/T3`: public three-layer T1LSST with V2/T1, V3/T2,
  and Vsub/T3 layers. V3/T2 and Vsub/T3 are global, not cell solved.
- `Cell V2/T1 - 2D projected line`: one-layer T1LSST with cell-solved V2/T1
  on a projected 2D line. Provide line origin, azimuth, X origin, cell count,
  and cell size.
- `Cell V2/T1 - 3D grid`: one-layer T1LSST with cell-solved V2/T1 on an X/Y
  grid. Provide X/Y origin, X/Y cell count, and X/Y cell size.

For cell presets, observations are assigned by midpoint cell. The public apply
workflow supports cell solving for V2/T1 only.

## Output and Exports

`apply.register_corrected_file` controls whether the job registers a corrected
TraceStore:

- unchecked: artifact-only job; no corrected file is registered.
- checked: the final refraction trace shifts are applied with the repo sign
  convention `corrected(t) = raw(t - shift_s)` and a corrected TraceStore is
  registered when validation passes.

`Exports` controls optional exported artifact families. The UI can request the
canonical static table, LSST, LSST+, time-term spreadsheet, and first-break time
exports. The canonical static table is the repo-owned static-table interchange
format; LSST and LSST+ are export formats derived from the refraction solution.

All refraction jobs still write the standard solution, QC, static CSV, component
CSV, and artifact manifest files documented in
[../static_correction.md](../static_correction.md) and
[../refraction_static.md](../refraction_static.md).

## Run, Status, Cancel, and Artifacts

Select `Run` after the request preview validates. The tab shows the static
correction job ID, state, message, progress, and generated artifacts.

If linkage is enabled, the UI shows linkage progress first. A failed linkage job
prevents the refraction apply job from being submitted.

Use `Cancel` while the static correction job is active to request cancellation.
After completion, the artifact table lists generated files and download links.

## Automatic QC Loading

When a static correction job reaches a ready state, the UI loads its artifact
list and then automatically opens the `Refraction QC` tab with the completed
refraction job ID. You can also open `Refraction QC` manually, enter the job ID,
and select `Load QC Bundle`.

Use Refraction QC to review first-break fit residuals, reduced-time/LMO,
profiles, cell maps, static components, and raw/corrected gather preview when a
corrected TraceStore was registered and preview artifacts are available.

## Troubleshooting

No active viewer file:
Open an SGY/TraceStore in the viewer before running Static Correction. The
Target section must show the current viewer file and sort keys.

No pick NPZ selected:
Select a `.npz` file in `First-break pick NPZ`. The viewer first-break
probability cache is not a valid statics pick file.

Sorted order mismatch:
Reload the viewer file with the same `key1_byte` and `key2_byte` ordering used
when the pick NPZ was created. Pick arrays are interpreted in the active
TraceStore sorted trace order.

Geometry byte mismatch:
Switch the Geometry preset to `custom` and enter the source, receiver,
coordinate, scalar, elevation, and offset header bytes for the loaded SEG-Y.
Wrong geometry bytes can produce invalid endpoint tables or poor residuals.

## Known Limitations and Non-Goals

- The UI does not create synthetic data.
- The workflow does not write SEG-Y files or SEG-Y static headers.
- GRM, plus-minus, and refraction tomography are not available UI features.
- Public T1LSST apply supports cell V2/T1, but not cell V3/T2 or cell Vsub/T3.
- The UI does not provide browser controls for editing refraction cell models.
- Do not treat a successful synthetic run as real-data validation.
