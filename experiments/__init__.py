# experiments/__init__.py
"""
Експериментальні модулі для проекту HAR-SKPR
"""

from . import exp1_basic_skpr
from . import exp2_ensemble_analysis
from . import exp3_optimal_basis
from . import exp4_hybrid_model
from . import exp5_full_validation

__all__ = [
    'exp1_basic_skpr',
    'exp2_ensemble_analysis',
    'exp3_optimal_basis',
    'exp4_hybrid_model',
    'exp5_full_validation'
]