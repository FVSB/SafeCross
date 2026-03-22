"""
SafeCross — Sistema de asistencia para personas con discapacidad visual.

Módulo principal que expone la función de decisión de cruce.
"""

from safecross.decide import can_cross, decide_verbose

__all__ = ["can_cross", "decide_verbose"]
__version__ = "0.1.0"
