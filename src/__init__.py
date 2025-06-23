# src/__init__.py
"""
Модулі для проекту HAR-SKPR
"""

from .data_loader import (
    download_har,
    load_raw_signals_and_labels,
    load_561_features,
    get_activity_names,
    get_dataset_info
)

from .preprocessing import (
    StandardScaler3D,
    WindowNormalizer,
    SignalDenoiser
)

from .feature_extractors import (
    BaseKunchenkoExtractor,
    AggregatedFeatureExtractor,
    GroupedFeatureExtractor,
    FullFeatureExtractor,
    OptimalFeatureExtractor,
    EnsembleExpert
)

from .utils import (
    plot_confusion_matrix,
    save_classification_report,
    plot_feature_importance,
    plot_learning_curves,
    save_experiment_summary,
    compare_models_barplot
)

__all__ = [
    # data_loader
    'download_har',
    'load_raw_signals_and_labels',
    'load_561_features',
    'get_activity_names',
    'get_dataset_info',
    # preprocessing
    'StandardScaler3D',
    'WindowNormalizer', 
    'SignalDenoiser',
    # feature_extractors
    'BaseKunchenkoExtractor',
    'AggregatedFeatureExtractor',
    'GroupedFeatureExtractor',
    'FullFeatureExtractor',
    'OptimalFeatureExtractor',
    'EnsembleExpert',
    # utils
    'plot_confusion_matrix',
    'save_classification_report',
    'plot_feature_importance',
    'plot_learning_curves',
    'save_experiment_summary',
    'compare_models_barplot'
]