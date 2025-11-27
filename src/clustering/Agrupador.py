"""
Autores: Claudia Cuevas Ruano, Pablo Tejero Lascorz
Pareja: 03
"""
from abc import ABCMeta, abstractmethod
from Datos import Datos


class Agrupador:
    __metaclass__ = ABCMeta

    @abstractmethod
    def agrupar(self, datos: Datos):
        pass