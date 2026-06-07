"""Hierarchical beta coral-cover model (Sully et al.)."""

from src.models.hbb._config import (
    CV_PREDICTORS,
    FEATURE_VARS,
    HAS_PYMC,
    OUTPUT_DIR,
    ModelSpec,
    SULLY_DATA_DIR,
    VARS_TO_STANDARDIZE,
)
from src.models.hbb.analysis import (
    calculate_coral_cover_by_ocean,
    create_output_dir_path,
    identify_bright_dark_spots,
    run_full_analysis,
)
from src.models.hbb.cv import predict_from_posterior_cv, prepare_cv_fold_arrays
from src.models.hbb.data import (
    clean_data,
    load_data,
    load_model_data_for_cv,
    load_model_data_from_pipeline,
    standardize_train_test,
    standardize_variables,
)
from src.models.hbb.design import (
    build_design_matrix,
    compute_correlation_matrix,
    inverse_transform_beta,
    transform_to_beta,
)
from src.models.hbb.indices import (
    prepare_hierarchical_indices,
    prepare_hierarchical_indices_legacy,
    prepare_hierarchical_indices_reparam,
)
from src.models.hbb.model import HierarchicalBetaModel
from src.models.hbb.projections import (
    build_current_design_matrix,
    load_model_and_project,
    project_future_coral_cover,
)
from src.plots.plot_config import COVARIATE_LABELS_DICT

__all__ = [
    "COVARIATE_LABELS_DICT",
    "CV_PREDICTORS",
    "FEATURE_VARS",
    "HAS_PYMC",
    "HierarchicalBetaModel",
    "ModelSpec",
    "OUTPUT_DIR",
    "SULLY_DATA_DIR",
    "VARS_TO_STANDARDIZE",
    "build_current_design_matrix",
    "build_design_matrix",
    "calculate_coral_cover_by_ocean",
    "clean_data",
    "compute_correlation_matrix",
    "create_output_dir_path",
    "identify_bright_dark_spots",
    "inverse_transform_beta",
    "load_data",
    "load_model_and_project",
    "load_model_data_for_cv",
    "load_model_data_from_pipeline",
    "predict_from_posterior_cv",
    "prepare_cv_fold_arrays",
    "prepare_hierarchical_indices",
    "prepare_hierarchical_indices_legacy",
    "prepare_hierarchical_indices_reparam",
    "project_future_coral_cover",
    "run_full_analysis",
    "standardize_train_test",
    "standardize_variables",
    "transform_to_beta",
]
