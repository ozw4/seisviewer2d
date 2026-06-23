from __future__ import annotations

import ast
from dataclasses import asdict, fields, is_dataclass
import importlib
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import BaseModel, ValidationError
from seis_statics.refraction import (
    RefractionStaticConversionOptions,
    RefractionStaticDatumOptions,
    RefractionStaticFirstLayerOptions,
    RefractionStaticLayerOptions,
    RefractionStaticModelOptions,
    RefractionStaticMoveoutOptions,
    RefractionStaticReducedTimeQcOptions,
    RefractionStaticRefractorCellOptions,
    RefractionStaticRobustOptions,
    RefractionStaticSolverOptions,
)

from app.statics.refraction.application.core_options import (
    conversion_options_from_request,
    datum_options_from_request,
    first_layer_options_from_request,
    layer_options_from_request,
    model_options_from_request,
    moveout_options_from_request,
    reduced_time_qc_options_from_request,
    refractor_cell_options_from_request,
    robust_options_from_request,
    solver_options_from_request,
)
from app.statics.refraction.contracts.model import (
    RefractionStaticFirstLayerRequest,
    RefractionStaticLayerRequest,
    RefractionStaticModelRequest,
    RefractionStaticRefractorCellRequest,
)
from app.statics.refraction.contracts.options import (
    RefractionStaticConversionRequest,
    RefractionStaticDatumRequest,
    RefractionStaticMoveoutRequest,
    RefractionStaticReducedTimeQcRequest,
    RefractionStaticRobustRequest,
    RefractionStaticSolverRequest,
)


def test_refraction_core_options_convert_field_by_field() -> None:
    first_layer = RefractionStaticFirstLayerRequest.model_validate(
        {
            'mode': 'estimate_direct_arrival',
            'min_weathering_velocity_m_s': 350.0,
            'max_weathering_velocity_m_s': 1700.0,
            'min_direct_offset_m': 12.5,
            'max_direct_offset_m': 145.0,
            'min_picks_per_fit': 6,
            'min_groups': 4,
            'robust_enabled': False,
            'robust_threshold': 4.25,
        }
    )
    refractor_cell = RefractionStaticRefractorCellRequest.model_validate(
        {
            'number_of_cell_x': 12,
            'size_of_cell_x_m': 250.0,
            'x_coordinate_origin_m': 1000.0,
            'coordinate_mode': 'line_2d_projected',
            'line_origin_x_m': 10.0,
            'line_origin_y_m': 20.0,
            'line_azimuth_deg': 35.0,
            'min_observations_per_cell': 7,
            'velocity_smoothing_weight': 0.2,
            'smoothing_reference_distance_m': 400.0,
        }
    )
    layer = RefractionStaticLayerRequest.model_validate(
        {
            'kind': 'v3_t2',
            'enabled': False,
            'min_offset_m': 1500.0,
            'max_offset_m': 2800.0,
            'velocity_mode': 'solve_global',
            'initial_velocity_m_s': 3600.0,
            'min_velocity_m_s': 2500.0,
            'max_velocity_m_s': 6500.0,
            'min_observations_per_cell': 8,
            'smoothing_weight': 0.05,
        }
    )
    moveout = RefractionStaticMoveoutRequest.model_validate(
        {
            'distance_source': 'offset_header',
            'offset_byte': 37,
            'min_offset_m': 300.0,
            'max_offset_m': 2500.0,
            'allow_missing_offset': True,
            'max_geometry_offset_mismatch_m': 25.0,
        }
    )
    robust = RefractionStaticRobustRequest(
        enabled=False,
        method='sigma',
        threshold=2.75,
        scale_floor_ms=0.1,
        max_iterations=3,
        min_used_fraction=0.8,
        min_used_observations=9,
    )
    solver = RefractionStaticSolverRequest(
        damping=0.125,
        min_picks_per_node=2,
        max_abs_half_intercept_time_ms=450.0,
        robust=robust,
    )
    datum = RefractionStaticDatumRequest.model_validate(
        {
            'mode': 'floating_and_flat',
            'floating_datum_mode': 'constant',
            'flat_datum_elevation_m': 100.0,
            'floating_datum_elevation_m': 150.0,
            'smoothing_radius_m': 500.0,
            'smoothing_window_nodes': 9,
            'smoothing_method': 'median',
            'allow_flat_datum_above_topography': False,
            'allow_flat_datum_below_refractor': True,
        }
    )
    conversion = RefractionStaticConversionRequest(
        mode='t1lsst_multilayer',
        layer_count=3,
    )
    reduced_time = RefractionStaticReducedTimeQcRequest(
        reduction_velocity_mode='fixed',
        fixed_velocity_m_s=2500.0,
    )

    assert first_layer_options_from_request(first_layer) == (
        RefractionStaticFirstLayerOptions(**first_layer.model_dump())
    )
    assert refractor_cell_options_from_request(refractor_cell) == (
        RefractionStaticRefractorCellOptions(**refractor_cell.model_dump())
    )
    assert layer_options_from_request(layer) == (
        RefractionStaticLayerOptions(**layer.model_dump())
    )
    assert moveout_options_from_request(moveout) == (
        RefractionStaticMoveoutOptions(**moveout.model_dump())
    )
    assert robust_options_from_request(robust) == (
        RefractionStaticRobustOptions(**robust.model_dump())
    )
    assert solver_options_from_request(solver) == RefractionStaticSolverOptions(
        half_intercept_damping_lambda=solver.damping,
        min_picks_per_node=solver.min_picks_per_node,
        max_abs_half_intercept_time_ms=solver.max_abs_half_intercept_time_ms,
        robust=RefractionStaticRobustOptions(**robust.model_dump()),
    )
    assert datum_options_from_request(datum) == RefractionStaticDatumOptions(
        mode=datum.mode,
        floating_datum_mode=datum.floating_datum_mode,
        flat_datum_elevation_m=datum.flat_datum_elevation_m,
        floating_datum_elevation_m=datum.floating_datum_elevation_m,
        smoothing_radius_m=datum.smoothing_radius_m,
        smoothing_window_nodes=datum.smoothing_window_nodes,
        smoothing_method=datum.smoothing_method,
        allow_flat_datum_above_topography=datum.allow_flat_datum_above_topography,
        allow_flat_datum_below_refractor=datum.allow_flat_datum_below_refractor,
    )
    assert conversion_options_from_request(conversion) == (
        RefractionStaticConversionOptions(**conversion.model_dump())
    )
    assert reduced_time_qc_options_from_request(reduced_time) == (
        RefractionStaticReducedTimeQcOptions(**reduced_time.model_dump())
    )


def test_refraction_core_options_preserve_defaults() -> None:
    assert asdict(
        first_layer_options_from_request(RefractionStaticFirstLayerRequest())
    ) == asdict(RefractionStaticFirstLayerOptions())
    assert asdict(
        moveout_options_from_request(RefractionStaticMoveoutRequest())
    ) == asdict(RefractionStaticMoveoutOptions())
    assert asdict(robust_options_from_request(RefractionStaticRobustRequest())) == (
        asdict(RefractionStaticRobustOptions())
    )
    assert asdict(solver_options_from_request(RefractionStaticSolverRequest())) == (
        asdict(RefractionStaticSolverOptions())
    )
    assert asdict(datum_options_from_request(RefractionStaticDatumRequest())) == (
        asdict(RefractionStaticDatumOptions())
    )
    assert asdict(
        conversion_options_from_request(RefractionStaticConversionRequest())
    ) == asdict(RefractionStaticConversionOptions())
    assert asdict(
        reduced_time_qc_options_from_request(RefractionStaticReducedTimeQcRequest())
    ) == asdict(RefractionStaticReducedTimeQcOptions())


def test_refraction_model_options_convert_nested_options() -> None:
    request = RefractionStaticModelRequest.model_validate(
        {
            'method': 'multilayer_time_term',
            'first_layer': {
                'mode': 'constant',
                'weathering_velocity_m_s': 850.0,
            },
            'initial_bedrock_velocity_m_s': 2450.0,
            'min_bedrock_velocity_m_s': 1300.0,
            'max_bedrock_velocity_m_s': 6200.0,
            'refractor_cell': {
                'number_of_cell_x': 4,
                'size_of_cell_x_m': 500.0,
                'x_coordinate_origin_m': 0.0,
            },
            'layers': [
                {
                    'kind': 'v2_t1',
                    'min_offset_m': 300.0,
                    'max_offset_m': 1600.0,
                    'velocity_mode': 'solve_cell',
                },
                {
                    'kind': 'v3_t2',
                    'min_offset_m': 1600.0,
                    'max_offset_m': None,
                    'velocity_mode': 'solve_global',
                    'initial_velocity_m_s': 3600.0,
                    'min_velocity_m_s': 2500.0,
                    'max_velocity_m_s': 7000.0,
                },
            ],
            'allow_overlapping_layer_gates': False,
        }
    )

    options = model_options_from_request(request)

    assert options == RefractionStaticModelOptions(
        method='multilayer_time_term',
        weathering_velocity_m_s=None,
        first_layer=RefractionStaticFirstLayerOptions(
            mode='constant',
            weathering_velocity_m_s=850.0,
        ),
        bedrock_velocity_mode='solve_global',
        bedrock_velocity_m_s=None,
        initial_bedrock_velocity_m_s=2450.0,
        min_bedrock_velocity_m_s=1300.0,
        max_bedrock_velocity_m_s=6200.0,
        max_weathering_thickness_m=None,
        refractor_cell=RefractionStaticRefractorCellOptions(
            number_of_cell_x=4,
            size_of_cell_x_m=500.0,
            x_coordinate_origin_m=0.0,
        ),
        layers=(
            RefractionStaticLayerOptions(
                kind='v2_t1',
                min_offset_m=300.0,
                max_offset_m=1600.0,
                velocity_mode='solve_cell',
            ),
            RefractionStaticLayerOptions(
                kind='v3_t2',
                min_offset_m=1600.0,
                max_offset_m=None,
                velocity_mode='solve_global',
                initial_velocity_m_s=3600.0,
                min_velocity_m_s=2500.0,
                max_velocity_m_s=7000.0,
            ),
        ),
        layer_assignment_policy='reject_overlap',
    )
    assert is_dataclass(options.first_layer)
    assert is_dataclass(options.refractor_cell)
    assert all(is_dataclass(layer) for layer in options.layers or ())
    assert not isinstance(options.first_layer, BaseModel)
    assert not isinstance(options.refractor_cell, BaseModel)
    assert not any(isinstance(layer, BaseModel) for layer in options.layers or ())


def test_refraction_model_options_convert_overlapping_layer_policy() -> None:
    request = RefractionStaticModelRequest.model_validate(
        {
            'method': 'multilayer_time_term',
            'first_layer': {
                'mode': 'constant',
                'weathering_velocity_m_s': 850.0,
            },
            'initial_bedrock_velocity_m_s': 2450.0,
            'layers': [
                {
                    'kind': 'v2_t1',
                    'min_offset_m': 300.0,
                    'max_offset_m': 1800.0,
                    'velocity_mode': 'solve_global',
                },
                {
                    'kind': 'v3_t2',
                    'min_offset_m': 1600.0,
                    'max_offset_m': None,
                    'velocity_mode': 'solve_global',
                    'initial_velocity_m_s': 3600.0,
                    'min_velocity_m_s': 2500.0,
                    'max_velocity_m_s': 7000.0,
                },
            ],
            'allow_overlapping_layer_gates': True,
        }
    )

    options = model_options_from_request(request)

    assert options.layer_assignment_policy == 'independent'


def test_refraction_datum_artifact_locator_stays_app_owned() -> None:
    request = RefractionStaticDatumRequest.model_validate(
        {
            'mode': 'floating_only',
            'floating_datum_mode': 'from_artifact',
            'floating_datum_job_id': 'datum-job',
            'floating_datum_artifact_name': 'floating_datum.csv',
        }
    )

    options = datum_options_from_request(request)

    assert options == RefractionStaticDatumOptions(
        mode='floating_only',
        floating_datum_mode='provided',
    )
    assert not hasattr(options, 'floating_datum_job_id')
    assert not hasattr(options, 'floating_datum_artifact_name')


@pytest.mark.parametrize(
    ('request_factory', 'payload', 'message'),
    [
        (
            RefractionStaticFirstLayerRequest.model_validate,
            {
                'mode': 'estimate_direct_arrival',
                'weathering_velocity_m_s': 800.0,
                'min_direct_offset_m': 0.0,
                'max_direct_offset_m': 100.0,
            },
            'weathering_velocity_m_s must be omitted',
        ),
        (
            RefractionStaticRefractorCellRequest.model_validate,
            {
                'number_of_cell_x': 2,
                'size_of_cell_x_m': 100.0,
                'x_coordinate_origin_m': 0.0,
                'coordinate_mode': 'line_2d_projected',
            },
            'line_origin_x_m',
        ),
        (
            RefractionStaticSolverRequest.model_validate,
            {'robust': {'min_used_fraction': 1.5}},
            'min_used_fraction must be <= 1',
        ),
        (
            RefractionStaticConversionRequest.model_validate,
            {'mode': 't1lsst_multilayer'},
            'layer_count is required',
        ),
    ],
)
def test_invalid_refraction_requests_are_rejected_before_conversion(
    request_factory: object,
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        request_factory(payload)


def test_refraction_core_options_import_is_lightweight() -> None:
    code = (
        'import sys; '
        'import app.statics.refraction.application.core_options; '
        'forbidden = ['
        '"app.statics.refraction.api.apply", '
        '"app.statics.refraction.artifacts.writer", '
        '"app.trace_store.reader", '
        '"app.services.job_manager"'
        ']; '
        'present = [name for name in forbidden if name in sys.modules]; '
        'assert present == [], present'
    )
    subprocess.run([sys.executable, '-c', code], check=True)


def test_seis_statics_refraction_does_not_import_app_modules() -> None:
    package = importlib.import_module('seis_statics.refraction')
    package_dir = Path(package.__file__).resolve().parent
    offenders: list[str] = []

    for source_path in sorted(package_dir.rglob('*.py')):
        tree = ast.parse(source_path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_modules = (node.module or '',)
            else:
                continue
            if any(
                module == 'app' or module.startswith('app.')
                for module in imported_modules
            ):
                offenders.append(str(source_path.relative_to(package_dir)))

    assert offenders == []


def test_external_core_types_are_canonical_numerical_options() -> None:
    option_types = (
        RefractionStaticConversionOptions,
        RefractionStaticDatumOptions,
        RefractionStaticFirstLayerOptions,
        RefractionStaticLayerOptions,
        RefractionStaticModelOptions,
        RefractionStaticMoveoutOptions,
        RefractionStaticReducedTimeQcOptions,
        RefractionStaticRefractorCellOptions,
        RefractionStaticRobustOptions,
        RefractionStaticSolverOptions,
    )

    for option_type in option_types:
        assert is_dataclass(option_type)
        assert option_type.__module__.startswith('seis_statics.refraction')
        assert all(field.init for field in fields(option_type))
