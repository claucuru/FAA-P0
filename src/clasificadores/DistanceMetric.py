"""
Autores: Claudia Cuevas Ruano, Pablo Tejero Lascorz
Pareja: 03
"""
from enum import Enum


class DistanceMetric(Enum):
    CHEVYCHEV         = 0
    EUCLIDES          = 1
    EUCLIDES_WEIGHTED = 2
    MANHATTAN         = 3
