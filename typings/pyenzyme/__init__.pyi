"""Type stubs for pyenzyme (upstream ships no .pyi).

Only symbols imported by chromhandler (and tests) are declared.
"""

from typing import Any

class DataTypes:
    CONCENTRATION: Any
    PEAK_AREA: Any

class MeasurementData:
    species_id: str
    initial: Any
    prepared: Any
    data_unit: Any
    data_type: Any
    time_unit: Any
    data: list[float]
    time: list[float]
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class Measurement:
    id: str
    name: str
    temperature: float
    temperature_unit: Any
    ph: float
    species_data: list[Any]
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class Protein:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class SmallMolecule:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class EnzymeMLDocument:
    name: str
    small_molecules: list[Any]
    proteins: list[Any]
    measurements: list[Any]
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
