"""
This file contains Pydantic model definitions for data validation.

Pydantic is a data validation library that uses Python type annotations.
It allows you to define data models with type hints that are validated
at runtime while providing static type checking.

Usage example:
```python
from my_model import MyModel

# Validates data at runtime
my_model = MyModel(name="John", age=30)

# Type-safe - my_model has correct type hints
print(my_model.name)

# Will raise error if validation fails
try:
    MyModel(name="", age=30)
except ValidationError as e:
    print(e)
```

For more information see:
https://docs.pydantic.dev/

WARNING: This is an auto-generated file.
Do not edit directly - any changes will be overwritten.
"""


## This is a generated file. Do not modify it manually!

from __future__ import annotations

from enum import Enum
from typing import Generic, Optional, TypeVar
from uuid import uuid4

from mdmodels.units.annotation import UnitDefinitionAnnot
from pydantic import BaseModel, ConfigDict, Field

# Filter Wrapper definition used to filter a list of objects
# based on their attributes
Cls = TypeVar("Cls")


class FilterWrapper(Generic[Cls]):
    """Wrapper class to filter a list of objects based on their attributes"""

    def __init__(self, collection: list[Cls], **kwargs):
        self.collection = collection
        self.kwargs = kwargs

    def filter(self) -> list[Cls]:
        for key, value in self.kwargs.items():
            self.collection = [
                item for item in self.collection if self._fetch_attr(key, item) == value
            ]
        return self.collection

    def _fetch_attr(self, name: str, item: Cls):
        try:
            return getattr(item, name)
        except AttributeError:
            raise AttributeError(f"{item} does not have attribute {name}")


# JSON-LD Helper Functions
def add_namespace(obj, prefix: str | None, iri: str | None):
    """Adds a namespace to the JSON-LD context

    Args:
        prefix (str): The prefix to add
        iri (str): The IRI to add
    """
    if prefix is None and iri is None:
        return
    elif prefix and iri is None:
        raise ValueError("If prefix is provided, iri must also be provided")
    elif iri and prefix is None:
        raise ValueError("If iri is provided, prefix must also be provided")

    obj.ld_context[prefix] = iri  # type: ignore


def validate_prefix(term: str | dict, prefix: str):
    """Validates that a term is prefixed with a given prefix

    Args:
        term (str): The term to validate
        prefix (str): The prefix to validate against

    Returns:
        bool: True if the term is prefixed with the prefix, False otherwise
    """

    if isinstance(term, dict) and not term["@id"].startswith(prefix + ":"):
        raise ValueError(f"Term {term} is not prefixed with {prefix}")
    elif isinstance(term, str) and not term.startswith(prefix + ":"):
        raise ValueError(f"Term {term} is not prefixed with {prefix}")


# Model Definitions


class Measurement(BaseModel):
    model_config: ConfigDict = ConfigDict(  # type: ignore
        validate_assignment=True,
    )

    id: str
    data: Data
    temperature: float
    temperature_unit: UnitDefinitionAnnot
    ph: float
    chromatograms: list[Chromatogram] = Field(default_factory=list)
    sample_name: Optional[Optional[str]] = Field(default=None)
    timestamp: Optional[Optional[str]] = Field(default=None)
    injection_volume: Optional[Optional[float]] = Field(default=None)
    dilution_factor: Optional[float] = 1
    injection_volume_unit: Optional[Optional[UnitDefinitionAnnot]] = Field(default=None)

    # JSON-LD fields
    ld_id: str = Field(
        serialization_alias="@id",
        default_factory=lambda: "chromatopy:Measurement/" + str(uuid4()),
    )
    ld_type: list[str] = Field(
        serialization_alias="@type",
        default_factory=lambda: [
            "chromatopy:Measurement",
        ],
    )
    ld_context: dict[str, str | dict] = Field(
        serialization_alias="@context",
        default_factory=lambda: {
            "chromatopy": "https://github.com/FAIRChemistry/chromatopy",
        },
    )

    def filter_chromatograms(self, **kwargs) -> list[Chromatogram]:
        """Filters the chromatograms attribute based on the given kwargs

        Args:
            **kwargs: The attributes to filter by.

        Returns:
            list[Chromatogram]: The filtered list of Chromatogram objects
        """

        return FilterWrapper[Chromatogram](self.chromatograms, **kwargs).filter()

    def set_attr_term(
        self,
        attr: str,
        term: str | dict,
        prefix: str | None = None,
        iri: str | None = None,
    ):
        """Sets the term for a given attribute in the JSON-LD object

        Example:
            # Using an IRI term
            >> obj.set_attr_term("name", "http://schema.org/givenName")

            # Using a prefix and term
            >> obj.set_attr_term("name", "schema:givenName", "schema", "http://schema.org")

            # Usinng a dictionary term
            >> obj.set_attr_term("name", {"@id": "http://schema.org/givenName", "@type": "@id"})

        Args:
            attr (str): The attribute to set the term for
            term (str | dict): The term to set for the attribute

        Raises:
            AssertionError: If the attribute is not found in the model
        """

        assert (
            attr in self.model_fields
        ), f"Attribute {attr} not found in {self.__class__.__name__}"

        if prefix:
            validate_prefix(term, prefix)

        add_namespace(self, prefix, iri)
        self.ld_context[attr] = term

    def add_type_term(
        self, term: str, prefix: str | None = None, iri: str | None = None
    ):
        """Adds a term to the @type field of the JSON-LD object

        Example:
            # Using a term
            >> obj.add_type_term("https://schema.org/Person")

            # Using a prefixed term
            >> obj.add_type_term("schema:Person", "schema", "https://schema.org/Person")

        Args:
            term (str): The term to add to the @type field
            prefix (str, optional): The prefix to use for the term. Defaults to None.
            iri (str, optional): The IRI to use for the term prefix. Defaults to None.

        Raises:
            ValueError: If prefix is provided but iri is not
            ValueError: If iri is provided but prefix is not
        """

        if prefix:
            validate_prefix(term, prefix)

        add_namespace(self, prefix, iri)
        self.ld_type.append(term)

    def add_to_chromatograms(
        self,
        type: Optional[SignalType] = None,
        peaks: list[Peak] = [],
        signals: list[float] = [],
        times: list[float] = [],
        processed_signal: list[float] = [],
        wavelength: Optional[float] = None,
        **kwargs,
    ):
        params = {
            "type": type,
            "peaks": peaks,
            "signals": signals,
            "times": times,
            "processed_signal": processed_signal,
            "wavelength": wavelength,
        }

        if "id" in kwargs:
            params["id"] = kwargs["id"]

        self.chromatograms.append(Chromatogram(**params))

        return self.chromatograms[-1]


class Data(BaseModel):
    model_config: ConfigDict = ConfigDict(  # type: ignore
        validate_assignment=True,
    )  # type: ignore

    value: float
    unit: UnitDefinitionAnnot
    data_type: DataType

    # JSON-LD fields
    ld_id: str = Field(
        serialization_alias="@id",
        default_factory=lambda: "chromatopy:Data/" + str(uuid4()),
    )
    ld_type: list[str] = Field(
        serialization_alias="@type",
        default_factory=lambda: [
            "chromatopy:Data",
        ],
    )
    ld_context: dict[str, str | dict] = Field(
        serialization_alias="@context",
        default_factory=lambda: {
            "chromatopy": "https://github.com/FAIRChemistry/chromatopy",
        },
    )

    def set_attr_term(
        self,
        attr: str,
        term: str | dict,
        prefix: str | None = None,
        iri: str | None = None,
    ):
        """Sets the term for a given attribute in the JSON-LD object

        Example:
            # Using an IRI term
            >> obj.set_attr_term("name", "http://schema.org/givenName")

            # Using a prefix and term
            >> obj.set_attr_term("name", "schema:givenName", "schema", "http://schema.org")

            # Usinng a dictionary term
            >> obj.set_attr_term("name", {"@id": "http://schema.org/givenName", "@type": "@id"})

        Args:
            attr (str): The attribute to set the term for
            term (str | dict): The term to set for the attribute

        Raises:
            AssertionError: If the attribute is not found in the model
        """

        assert (
            attr in self.model_fields
        ), f"Attribute {attr} not found in {self.__class__.__name__}"

        if prefix:
            validate_prefix(term, prefix)

        add_namespace(self, prefix, iri)
        self.ld_context[attr] = term

    def add_type_term(
        self, term: str, prefix: str | None = None, iri: str | None = None
    ):
        """Adds a term to the @type field of the JSON-LD object

        Example:
            # Using a term
            >> obj.add_type_term("https://schema.org/Person")

            # Using a prefixed term
            >> obj.add_type_term("schema:Person", "schema", "https://schema.org/Person")

        Args:
            term (str): The term to add to the @type field
            prefix (str, optional): The prefix to use for the term. Defaults to None.
            iri (str, optional): The IRI to use for the term prefix. Defaults to None.

        Raises:
            ValueError: If prefix is provided but iri is not
            ValueError: If iri is provided but prefix is not
        """

        if prefix:
            validate_prefix(term, prefix)

        add_namespace(self, prefix, iri)
        self.ld_type.append(term)


class Chromatogram(BaseModel):
    model_config: ConfigDict = ConfigDict(  # type: ignore
        validate_assignment=True,
    )  # type: ignore

    type: Optional[Optional[SignalType]] = Field(default=None)
    peaks: list[Peak] = Field(default_factory=list)
    signals: list[float] = Field(default_factory=list)
    times: list[float] = Field(default_factory=list)
    processed_signal: list[float] = Field(default_factory=list)
    wavelength: Optional[Optional[float]] = Field(default=None)

    # JSON-LD fields
    ld_id: str = Field(
        serialization_alias="@id",
        default_factory=lambda: "chromatopy:Chromatogram/" + str(uuid4()),
    )
    ld_type: list[str] = Field(
        serialization_alias="@type",
        default_factory=lambda: [
            "chromatopy:Chromatogram",
        ],
    )
    ld_context: dict[str, str | dict] = Field(
        serialization_alias="@context",
        default_factory=lambda: {
            "chromatopy": "https://github.com/FAIRChemistry/chromatopy",
        },
    )

    def filter_peaks(self, **kwargs) -> list[Peak]:
        """Filters the peaks attribute based on the given kwargs

        Args:
            **kwargs: The attributes to filter by.

        Returns:
            list[Peak]: The filtered list of Peak objects
        """

        return FilterWrapper[Peak](self.peaks, **kwargs).filter()

    def set_attr_term(
        self,
        attr: str,
        term: str | dict,
        prefix: str | None = None,
        iri: str | None = None,
    ):
        """Sets the term for a given attribute in the JSON-LD object

        Example:
            # Using an IRI term
            >> obj.set_attr_term("name", "http://schema.org/givenName")

            # Using a prefix and term
            >> obj.set_attr_term("name", "schema:givenName", "schema", "http://schema.org")

            # Usinng a dictionary term
            >> obj.set_attr_term("name", {"@id": "http://schema.org/givenName", "@type": "@id"})

        Args:
            attr (str): The attribute to set the term for
            term (str | dict): The term to set for the attribute

        Raises:
            AssertionError: If the attribute is not found in the model
        """

        assert (
            attr in self.model_fields
        ), f"Attribute {attr} not found in {self.__class__.__name__}"

        if prefix:
            validate_prefix(term, prefix)

        add_namespace(self, prefix, iri)
        self.ld_context[attr] = term

    def add_type_term(
        self, term: str, prefix: str | None = None, iri: str | None = None
    ):
        """Adds a term to the @type field of the JSON-LD object

        Example:
            # Using a term
            >> obj.add_type_term("https://schema.org/Person")

            # Using a prefixed term
            >> obj.add_type_term("schema:Person", "schema", "https://schema.org/Person")

        Args:
            term (str): The term to add to the @type field
            prefix (str, optional): The prefix to use for the term. Defaults to None.
            iri (str, optional): The IRI to use for the term prefix. Defaults to None.

        Raises:
            ValueError: If prefix is provided but iri is not
            ValueError: If iri is provided but prefix is not
        """

        if prefix:
            validate_prefix(term, prefix)

        add_namespace(self, prefix, iri)
        self.ld_type.append(term)

    def add_to_peaks(
        self,
        retention_time: float,
        area: float,
        molecule_id: Optional[str] = None,
        type: Optional[str] = None,
        width: Optional[float] = None,
        amplitude: Optional[float] = None,
        max_signal: Optional[float] = None,
        skew: Optional[float] = None,
        percent_area: Optional[float] = None,
        tailing_factor: Optional[float] = None,
        separation_factor: Optional[float] = None,
        peak_start: Optional[float] = None,
        peak_end: Optional[float] = None,
        **kwargs,
    ):
        params = {
            "retention_time": retention_time,
            "area": area,
            "molecule_id": molecule_id,
            "type": type,
            "width": width,
            "amplitude": amplitude,
            "max_signal": max_signal,
            "skew": skew,
            "percent_area": percent_area,
            "tailing_factor": tailing_factor,
            "separation_factor": separation_factor,
            "peak_start": peak_start,
            "peak_end": peak_end,
        }

        if "id" in kwargs:
            params["id"] = kwargs["id"]

        self.peaks.append(Peak(**params))

        return self.peaks[-1]


class Peak(BaseModel):
    model_config: ConfigDict = ConfigDict(  # type: ignore
        validate_assignment=True,
    )  # type: ignore

    retention_time: float
    area: float
    molecule_id: Optional[Optional[str]] = Field(default=None)
    type: Optional[Optional[str]] = Field(default=None)
    width: Optional[Optional[float]] = Field(default=None)
    amplitude: Optional[Optional[float]] = Field(default=None)
    max_signal: Optional[Optional[float]] = Field(default=None)
    skew: Optional[Optional[float]] = Field(default=None)
    percent_area: Optional[Optional[float]] = Field(default=None)
    tailing_factor: Optional[Optional[float]] = Field(default=None)
    separation_factor: Optional[Optional[float]] = Field(default=None)
    peak_start: Optional[Optional[float]] = Field(default=None)
    peak_end: Optional[Optional[float]] = Field(default=None)

    # JSON-LD fields
    ld_id: str = Field(
        serialization_alias="@id",
        default_factory=lambda: "chromatopy:Peak/" + str(uuid4()),
    )
    ld_type: list[str] = Field(
        serialization_alias="@type",
        default_factory=lambda: [
            "chromatopy:Peak",
        ],
    )
    ld_context: dict[str, str | dict] = Field(
        serialization_alias="@context",
        default_factory=lambda: {
            "chromatopy": "https://github.com/FAIRChemistry/chromatopy",
        },
    )

    def set_attr_term(
        self,
        attr: str,
        term: str | dict,
        prefix: str | None = None,
        iri: str | None = None,
    ):
        """Sets the term for a given attribute in the JSON-LD object

        Example:
            # Using an IRI term
            >> obj.set_attr_term("name", "http://schema.org/givenName")

            # Using a prefix and term
            >> obj.set_attr_term("name", "schema:givenName", "schema", "http://schema.org")

            # Usinng a dictionary term
            >> obj.set_attr_term("name", {"@id": "http://schema.org/givenName", "@type": "@id"})

        Args:
            attr (str): The attribute to set the term for
            term (str | dict): The term to set for the attribute

        Raises:
            AssertionError: If the attribute is not found in the model
        """

        assert (
            attr in self.model_fields
        ), f"Attribute {attr} not found in {self.__class__.__name__}"

        if prefix:
            validate_prefix(term, prefix)

        add_namespace(self, prefix, iri)
        self.ld_context[attr] = term

    def add_type_term(
        self, term: str, prefix: str | None = None, iri: str | None = None
    ):
        """Adds a term to the @type field of the JSON-LD object

        Example:
            # Using a term
            >> obj.add_type_term("https://schema.org/Person")

            # Using a prefixed term
            >> obj.add_type_term("schema:Person", "schema", "https://schema.org/Person")

        Args:
            term (str): The term to add to the @type field
            prefix (str, optional): The prefix to use for the term. Defaults to None.
            iri (str, optional): The IRI to use for the term prefix. Defaults to None.

        Raises:
            ValueError: If prefix is provided but iri is not
            ValueError: If iri is provided but prefix is not
        """

        if prefix:
            validate_prefix(term, prefix)

        add_namespace(self, prefix, iri)
        self.ld_type.append(term)


class SignalType(Enum):
    DAD = "diode array detector"
    ELSD = "evaporative light scattering detector"
    FID = "flame ionization detector"
    FLD = "fluorescence detector"
    MS = "mass spectrometry"
    RID = "refractive index detector"
    TCD = "thermal conductivity detector"
    UV = "uv/visible absorbance detector"


class DataType(Enum):
    CALIBRATION = "calibration"
    TIMECOURSE = "timecourse"


# Rebuild all the classes within this file
for cls in [
    Measurement,
    Data,
    Chromatogram,
    Peak,
]:
    cls.model_rebuild()
