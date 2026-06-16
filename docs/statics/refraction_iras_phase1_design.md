# IRAS-compatible 1-layer refraction statics design

## Purpose

This document defines the Phase 1 implementation target for making the repo's refraction statics workflow closer to an IRAS-style workflow, without checking the original IRAS materials into the repository.

The goal is to make the repo support a practical 1-layer refraction statics workflow:

```text
first-break picks
  -> resolve V1 weathering velocity
  -> solve variable-thickness GLI model for T1 and V2
  -> convert T1 to SH1 and WCOR using T1LSST-compatible 1-layer formulas
  -> write source/receiver static tables
  -> optionally apply trace shifts using the repo static-shift convention
```

The implementation must be self-contained. Developers and Codex should use this document as the implementation reference instead of relying on external IRAS manuals.

For the completed M3 multi-layer public workflow, also read
`docs/statics/refraction_multilayer_time_term.md`. Multi-layer items listed as
Phase 1 non-goals are milestone-scoped and are not current product limitations.

---

## Scope

### In scope for Phase 1

- Backward-compatible `model.weathering_velocity_m_s` support.
- New `model.first_layer` request block.
- Constant V1 mode.
- Global V1 estimation from near-offset direct arrivals.
- Existing 1-layer GLI variable-thickness workflow integration.
- T1LSST-compatible 1-layer component calculation.
- Source static table artifact.
- Receiver static table artifact.
- CSV/NPZ artifacts with explicit units and sign convention.
- Synthetic 1-layer tests for V1, T1, SH1, WCOR, and source/receiver static tables.
- Docs for the 1-layer workflow.

### Explicit non-goals for Phase 1

- Original IRAS documents stored in repo.
- Full IRAS compatibility.
- GRM / plus-minus method.
- V2 refractor cell grid.
- Spatially varying V2.
- Spatially varying V1 map.
- Multi-layer `V3/T2` or `Vsub/T3`.
- 2-layer / 3-layer T1LSST conversion.
- Refraction tomography.
- L1-norm refraction statics.
- SEG-Y header write-back.
- UI display work beyond artifact generation.

---

## Conceptual model

Phase 1 implements the simplified single-layer, variable-thickness GLI model for refraction statics.

The observed first-break travel time is modeled as:

```text
pick_time_s = T_source_s + T_receiver_s + offset_m / V2_m_s + error_s
```

Equivalently:

```text
pick_time_s = T_source_s + T_receiver_s + offset_m * s_b + error_s
```

where:

```text
s_b = 1 / V2
```

`T_source_s` and `T_receiver_s` are station/node time terms. In this Phase 1 design, these are treated as T1-style half-intercept time terms.

After solving for T1 and V2, compute weathering thickness SH1 and weathering correction WCOR using a single-layer replacement model.

---

## IRAS-style terms and repo terms

| IRAS-style term | Repo / implementation meaning | Unit |
|---|---|---:|
| V1 | Weathering velocity / first-layer velocity | m/s |
| V2 | Bedrock / refractor / sub-weathering velocity | m/s |
| T1 | Source/receiver half-intercept time term | s or ms in CSV |
| SH1 | Weathering-layer thickness derived from T1, V1, V2 | m |
| WCOR | Weathering replacement correction | s or ms in CSV |
| source static | Static component associated with source endpoint/station | s or ms in CSV |
| receiver static | Static component associated with receiver endpoint/station | s or ms in CSV |
| total applied shift | Repo sign-convention shift applied to traces | s or ms in CSV |

Keep existing internal names where already implemented, but add aliases in artifacts where useful:

```text
node_t1_time_s
source_t1_time_s
receiver_t1_time_s
v1_weathering_velocity_m_s
v2_refractor_velocity_m_s
sh1_weathering_thickness_m
weathering_correction_s
elevation_correction_s
total_static_s
total_applied_shift_s
```

---

## Sign convention

The repo static-shift convention is:

```text
corrected(t) = raw(t - shift_s)
```

Therefore:

```text
shift_s > 0  -> event appears later in corrected data
shift_s < 0  -> event appears earlier in corrected data
```

This convention must be written into:

- `refraction_static_qc.json`
- `refraction_t1lsst_1layer_components.csv` metadata, if supported
- docs
- test names or assertions

For V2 > V1, the 1-layer weathering correction is normally negative:

```text
WCOR = SH1 * (1 / V2 - 1 / V1)
```

because replacing a low-velocity weathering layer with a faster bedrock velocity removes delay.

Unless a future design intentionally separates display sign from apply sign, Phase 1 should keep:

```text
total_static_s == total_applied_shift_s
```

and document that this follows the repo convention.

---

## API design

### Legacy request compatibility

Existing requests must keep working:

```json
{
  "model": {
    "method": "gli_variable_thickness",
    "weathering_velocity_m_s": 800.0,
    "bedrock_velocity_mode": "solve_global"
  }
}
```

The legacy `weathering_velocity_m_s` value resolves to V1.

### New first-layer request

Add a first-layer block:

```json
{
  "model": {
    "method": "gli_variable_thickness",
    "first_layer": {
      "mode": "constant",
      "weathering_velocity_m_s": 800.0
    },
    "bedrock_velocity_mode": "solve_global"
  }
}
```

### V1 direct-arrival estimation request

```json
{
  "model": {
    "method": "gli_variable_thickness",
    "first_layer": {
      "mode": "estimate_direct_arrival",
      "min_weathering_velocity_m_s": 250.0,
      "max_weathering_velocity_m_s": 1800.0,
      "min_direct_offset_m": 0.0,
      "max_direct_offset_m": 300.0,
      "min_picks_per_fit": 5,
      "min_groups": 3,
      "robust_enabled": true,
      "robust_threshold": 3.5
    },
    "bedrock_velocity_mode": "solve_global"
  }
}
```

Suggested schema:

```python
class RefractionStaticFirstLayerRequest(BaseModel):
    mode: Literal["constant", "estimate_direct_arrival"] = "constant"
    weathering_velocity_m_s: float | None = None

    min_weathering_velocity_m_s: float = 250.0
    max_weathering_velocity_m_s: float = 1800.0

    min_direct_offset_m: float | None = None
    max_direct_offset_m: float | None = None

    min_picks_per_fit: int = 5
    min_groups: int = 3

    robust_enabled: bool = True
    robust_threshold: float = 3.5
```

### Conversion request

Add a conversion block:

```json
{
  "conversion": {
    "mode": "t1lsst_1layer"
  }
}
```

Suggested schema:

```python
class RefractionStaticConversionRequest(BaseModel):
    mode: Literal["existing", "t1lsst_1layer"] = "existing"
```

Phase 1 should make `t1lsst_1layer` numerically equivalent to the existing 1-layer weathering replacement calculation, while adding IRAS-style components and source/receiver static tables.

---

## V1 resolution

All downstream code must use one resolved value:

```python
@dataclass(frozen=True)
class ResolvedRefractionFirstLayer:
    mode: Literal["constant", "estimate_direct_arrival"]
    weathering_velocity_m_s: float
    status: str
    qc: dict[str, Any]
```

Refraction static job flow:

```text
run_refraction_static_apply_job
  -> load picks and geometry
  -> resolve_refraction_first_layer(...)
  -> solve GLI / bedrock model
  -> compute T1
  -> compute SH1
  -> compute WCOR
  -> compose datum/static components
  -> write artifacts
  -> optionally apply trace shifts
```

Do not read `req.model.weathering_velocity_m_s` directly after the first-layer resolution step.

---

## V1 direct-arrival estimation

MVP estimates a single global V1.

### Candidate picks

Use picks satisfying:

```text
valid_observation_mask_sorted == True
pick_time_s is finite
offset_m is finite
min_direct_offset_m <= offset_m <= max_direct_offset_m
```

### Grouping

Fit by source gather or source endpoint, depending on the existing input structures. Keep the group key in the output artifact.

For each group:

```text
t = a + s * x
V1_group = 1 / s
```

Do not force the line through the origin. The intercept `a` absorbs timing bias and small station delay.

### Robust fitting

MVP may use either:

- The existing robust/MAD pattern used elsewhere in statics code.
- A simple two-pass fit: fit, reject residuals above threshold, refit.

Reject invalid group estimates:

```text
n_used < min_picks_per_fit
slope <= 0
V1 outside [min_weathering_velocity_m_s, max_weathering_velocity_m_s]
non-finite residual statistics
```

Then resolve global V1:

```text
resolved V1 = median(valid group V1 estimates)
```

Require:

```text
n_valid_groups >= min_groups
```

### V1 output dataclass

```python
@dataclass(frozen=True)
class RefractionV1EstimateResult:
    mode: Literal["estimate_direct_arrival"]
    resolved_weathering_velocity_m_s: float
    group_key: np.ndarray
    group_v1_m_s: np.ndarray
    group_slope_s_per_m: np.ndarray
    group_intercept_s: np.ndarray
    group_n_candidates: np.ndarray
    group_n_used: np.ndarray
    group_residual_rms_s: np.ndarray
    group_residual_mad_s: np.ndarray
    group_status: np.ndarray
    qc: dict[str, Any]
```

---

## GLI variable-thickness model

The Phase 1 GLI model is:

```text
pick_time_s = T_source_s + T_receiver_s + offset_m / V2_m_s + error_s
```

For solved global V2:

```text
unknowns = source/receiver node T1 values + global bedrock slowness s_b
```

For fixed V2:

```text
unknowns = source/receiver node T1 values
known moveout = offset_m / V2
```

Existing sparse least-squares, damping, gauge fixing, and robust rejection should remain in place.

Validation:

```text
V2 > V1
V1 finite and positive
V2 finite and positive
T1 finite for active solved nodes
```

---

## T1LSST-compatible 1-layer formulas

Use the following formulas for each source/receiver endpoint/node.

### SH1 weathering thickness

Given:

```text
T1 = half-intercept time, s
V1 = weathering velocity, m/s
V2 = bedrock/refractor velocity, m/s
```

Compute:

```text
SH1 = T1 * V1 * V2 / sqrt(V2^2 - V1^2)
```

Equivalent form:

```text
SCOS1 = sqrt(1 - (V1 / V2)^2)
SH1 = T1 * V1 / SCOS1
```

### WCOR weathering correction

```text
WCOR = SH1 * (1 / V2 - 1 / V1)
```

This is normally negative when V2 > V1.

### Refractor elevation

If surface elevation is available:

```text
refractor_elevation_m = surface_elevation_m - SH1
```

If not available, output NaN with status.

### Datum/elevation correction

Use existing datum composition logic. Phase 1 should not redefine datum statics. It should expose existing components in IRAS-style tables.

---

## Output artifacts

### V1 artifacts

When `first_layer.mode="estimate_direct_arrival"`, write:

```text
refraction_v1_qc.json
refraction_v1_estimates.csv
```

`refraction_v1_estimates.csv` columns:

```text
group_kind
group_key
n_candidates
n_used
offset_min_m
offset_max_m
slope_s_per_m
v1_m_s
intercept_s
residual_rms_ms
residual_mad_ms
status
```

`refraction_v1_qc.json` example:

```json
{
  "v1_mode": "estimate_direct_arrival",
  "resolved_weathering_velocity_m_s": 812.3,
  "n_candidate_picks": 1200,
  "n_used_groups": 42,
  "v1_min_m_s": 650.0,
  "v1_median_m_s": 812.3,
  "v1_max_m_s": 980.0,
  "v1_status": "estimated",
  "warnings": []
}
```

For constant mode, include V1 details in `refraction_static_qc.json`; a separate V1 estimates CSV is not required.

### T1LSST 1-layer components artifact

Write:

```text
refraction_t1lsst_1layer_components.csv
```

Columns:

```text
endpoint_kind
endpoint_key
node_id
x_m
y_m
surface_elevation_m
floating_datum_elevation_m
flat_datum_elevation_m
t1_ms
v1_m_s
v2_m_s
sh1_weathering_thickness_m
refractor_elevation_m
weathering_correction_ms
floating_datum_correction_ms
flat_datum_correction_ms
elevation_correction_ms
total_static_ms
total_applied_shift_ms
solution_status
weathering_status
datum_status
static_status
```

### Source static table

Write:

```text
source_static_table.csv
```

Columns:

```text
endpoint_kind
source_endpoint_key
source_id
source_node_id
x_m
y_m
surface_elevation_m
floating_datum_elevation_m
flat_datum_elevation_m
t1_ms
v1_m_s
v2_m_s
sh1_weathering_thickness_m
refractor_elevation_m
weathering_correction_ms
floating_datum_correction_ms
flat_datum_correction_ms
elevation_correction_ms
total_static_ms
total_applied_shift_ms
solution_status
weathering_status
datum_status
static_status
pick_count
used_pick_count
residual_rms_ms
residual_mad_ms
```

### Receiver static table

Write:

```text
receiver_static_table.csv
```

Columns:

```text
endpoint_kind
receiver_endpoint_key
receiver_id
receiver_node_id
x_m
y_m
surface_elevation_m
floating_datum_elevation_m
flat_datum_elevation_m
t1_ms
v1_m_s
v2_m_s
sh1_weathering_thickness_m
refractor_elevation_m
weathering_correction_ms
floating_datum_correction_ms
flat_datum_correction_ms
elevation_correction_ms
total_static_ms
total_applied_shift_ms
solution_status
weathering_status
datum_status
static_status
pick_count
used_pick_count
residual_rms_ms
residual_mad_ms
```

### Combined NPZ

Write:

```text
source_receiver_static_table.npz
```

Arrays:

```text
source_endpoint_key
source_id
source_node_id
source_x_m
source_y_m
source_surface_elevation_m
source_t1_s
source_v1_m_s
source_v2_m_s
source_sh1_m
source_weathering_correction_s
source_elevation_correction_s
source_total_static_s
source_total_applied_shift_s
source_static_status

receiver_endpoint_key
receiver_id
receiver_node_id
receiver_x_m
receiver_y_m
receiver_surface_elevation_m
receiver_t1_s
receiver_v1_m_s
receiver_v2_m_s
receiver_sh1_m
receiver_weathering_correction_s
receiver_elevation_correction_s
receiver_total_static_s
receiver_total_applied_shift_s
receiver_static_status
```

Do not use pickled object arrays in NPZ artifacts.

### Existing solution NPZ aliases

Add the following alias arrays to the existing solution artifact where applicable:

```text
v1_weathering_velocity_m_s
resolved_weathering_velocity_m_s
v1_mode
v2_refractor_velocity_m_s
node_t1_time_s
node_sh1_weathering_thickness_m
node_weathering_correction_s
```

---

## QC JSON additions

Add a velocity section:

```json
{
  "velocity": {
    "v1_mode": "estimate_direct_arrival",
    "weathering_velocity_m_s": 812.3,
    "bedrock_velocity_m_s": 2460.0,
    "v2_mode": "solve_global"
  },
  "sign_convention": {
    "trace_shift_s": "corrected(t) = raw(t - shift_s)",
    "positive_shift": "event appears later in corrected data",
    "negative_shift": "event appears earlier in corrected data"
  }
}
```

Add artifact references:

```json
{
  "artifacts": {
    "refraction_v1_qc": "refraction_v1_qc.json",
    "refraction_v1_estimates": "refraction_v1_estimates.csv",
    "refraction_t1lsst_1layer_components": "refraction_t1lsst_1layer_components.csv",
    "source_static_table": "source_static_table.csv",
    "receiver_static_table": "receiver_static_table.csv",
    "source_receiver_static_table": "source_receiver_static_table.npz"
  }
}
```

---

## Validation and errors

Validation must produce explicit, actionable errors.

### V1 validation

- Constant V1 must be finite and positive.
- Estimated V1 must have enough groups.
- Estimated V1 must be within configured bounds.
- Request must not specify conflicting legacy and first-layer V1 values unless they are identical within tolerance.

### V2 validation

- V2 must be finite and positive.
- V2 must satisfy `V2 > V1`.
- If solving V2, lower bound must exceed resolved V1.

### T1 / SH1 validation

- T1 must be finite for active solved nodes.
- Negative T1 should not crash the job; it should be status-coded unless existing behavior rejects it.
- SH1 must be finite where status is valid.
- Max thickness guards should remain in effect.

### Static table validation

- One row per source endpoint.
- One row per receiver endpoint.
- Inactive or unsolved endpoints should appear with status, not disappear silently.
- Linked source/receiver endpoints sharing a node should have matching node-level T1 and SH1 values.

---

## Tests

### Unit tests

```text
test_refraction_static_legacy_weathering_velocity_request_still_valid
test_refraction_static_first_layer_constant_request_valid
test_refraction_static_rejects_v1_greater_than_bedrock_min
test_refraction_static_rejects_conflicting_legacy_and_first_layer_v1

test_v1_estimate_global_from_direct_arrivals
test_v1_estimate_robust_to_outlier_picks
test_v1_estimate_respects_direct_offset_gate
test_v1_estimate_fails_with_insufficient_picks
test_v1_estimate_rejects_velocity_outside_bounds

test_t1lsst_1layer_scalar_formula
test_t1lsst_1layer_vector_formula
test_t1lsst_1layer_matches_existing_weathering_thickness
test_t1lsst_1layer_matches_existing_weathering_replacement_shift
test_t1lsst_1layer_rejects_v2_less_than_or_equal_v1
test_t1lsst_1layer_artifact_contains_sign_convention

test_source_static_table_has_one_row_per_source_endpoint
test_receiver_static_table_has_one_row_per_receiver_endpoint
test_source_receiver_static_tables_match_npz
test_linked_source_receiver_share_same_node_t1_and_sh1
test_static_tables_include_inactive_endpoint_status
test_static_tables_are_pickle_free_npz
```

### Synthetic E2E test model

Use a small synthetic 1-layer model:

```text
V1 = 800 m/s
V2 = 2400 m/s
known source/receiver SH1 values
```

Compute:

```text
T1 = SH1 * sqrt(V2^2 - V1^2) / (V1 * V2)
pick_time = T1_source + T1_receiver + offset / V2
WCOR = SH1 * (1/V2 - 1/V1)
```

For direct arrivals:

```text
direct_pick_time = intercept + offset / V1 + noise
```

Assertions:

```text
estimated V1 ≈ 800 m/s
solved V2 ≈ 2400 m/s
estimated T1 ≈ known T1
estimated SH1 ≈ known SH1
WCOR matches expected
source table has one row per source endpoint
receiver table has one row per receiver endpoint
trace total shift equals source + receiver + datum components under repo convention
```

---

## Implementation file map

Current implementation files:

```text
app/statics/refraction/api/apply.py
app/statics/refraction/contracts/apply.py
app/statics/refraction/contracts/inputs.py
app/statics/refraction/contracts/model.py
app/statics/refraction/contracts/options.py
app/statics/refraction/contracts/field_corrections.py
app/statics/refraction/contracts/export.py
app/statics/refraction/contracts/table_apply.py
app/statics/refraction/application/workflow.py
app/statics/refraction/application/input_model.py
app/statics/refraction/application/job_status.py
app/statics/refraction/application/pick_source_loader.py
app/statics/refraction/application/weathering.py
app/statics/refraction/application/weathering_replacement.py
app/statics/refraction/application/bedrock.py
app/statics/refraction/application/design_matrix.py
app/statics/refraction/application/datum.py
app/statics/refraction/domain/v1.py
app/statics/refraction/domain/t1lsst.py
app/statics/refraction/domain/solver.py
app/statics/refraction/artifacts/
app/statics/refraction/ports/
app/statics/refraction/adapters/seisviewer2d/
app/services/job_artifact_refs.py

app/tests/test_refraction_static_schema.py
app/tests/test_refraction_static_v1.py
app/tests/test_refraction_static_t1lsst.py
app/tests/test_refraction_static_artifacts.py
app/tests/test_refraction_static_source_receiver_tables.py
app/tests/test_refraction_static_apply_job_api.py
app/tests/test_job_artifact_refs.py
```

---

## Implementation order

Recommended PR split:

```text
PR 1:
  schema / first_layer constant path / resolved V1 plumbing

PR 2:
  global V1 direct-arrival estimator and V1 artifacts

PR 3:
  T1LSST 1-layer formulas and component artifact

PR 4:
  source/receiver static tables and artifact manifest registration

PR 5:
  synthetic E2E tests and docs polish
```

---

## Notes for Codex

- Do not add original IRAS manuals or extracted copyrighted source material to the repo.
- Treat this document as the canonical implementation specification for Phase 1.
- Preserve backward compatibility with existing refraction statics requests.
- Prefer small, focused PRs and tests.
- Keep units explicit in CSV column names.
- Never silently change the repo static-shift convention.
