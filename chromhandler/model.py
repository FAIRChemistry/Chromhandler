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
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Generic, TypeVar, Union
from enum import Enum
from uuid import uuid4
from datetime import date, datetime
from mdmodels.units.annotation import UnitDefinitionAnnot

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

    obj.ld_context[prefix] = iri # type: ignore

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

class Sample(BaseModel):

    model_config: ConfigDict = ConfigDict( # type: ignore
        validate_assignment = True,
    ) # type: ignore

    id: str = Field(
        default=...,
        description="""Unique identifier of the sample.""",
    )
    chromatograms: list[Chromatogram] = Field(
        default_factory=list,
        description="""Measured chromatogram and peaks.""",
    )
    initial_conditions: list[InitialCondition] = Field(
        default_factory=list,
        description="""Initial conditions of the sample.""",
    )
    timestamp: Optional[str] = Field(
        default=None,
        description="""Timestamp of sample injection into the column.""",
    )
    injection_volume: Optional[float] = Field(
        default=None,
        description="""Injection volume.""",
    )
    dilution_factor: Optional[float] = Field(
        default= 1,
        description="""Dilution factor.""",
    )
    injection_volume_unit: Optional[UnitDefinitionAnnot] = Field(
        default=None,
        description="""Unit of injection volume.""",
    )

    # JSON-LD fields
    ld_id: str = Field(
        serialization_alias="@id",
        default_factory=lambda: "chromhander:Sample/" + str(uuid4())
    )
    ld_type: list[str] = Field(
        serialization_alias="@type",
        default_factory = lambda: [
            "chromhander:Sample",
        ],
    )
    ld_context: dict[str, str | dict] = Field(
        serialization_alias="@context",
        default_factory = lambda: {
            "chromhander": "https://github.com/FAIRChemistry/chromhandler",
            "om": "http://www.ontology-of-units-of-measure.org/resource/om-2/",
            "qudt": "http://qudt.org/schema/qudt#/",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#/",
            "schema": "http://schema.org/",
            "unit": "http://qudt.org/vocab/unit#/",
            "xsd": "http://www.w3.org/2001/XMLSchema#/",
            "Chromatogram": "https://github.com/FAIRChemistry/chromhandler#Chromatogram/",
            "InitialCondition": "https://github.com/FAIRChemistry/chromhandler#InitialCondition/",
        }
    )

    def filter_chromatograms(self, **kwargs) -> list[Chromatogram]:
        """Filters the chromatograms attribute based on the given kwargs

        Args:
            **kwargs: The attributes to filter by.

        Returns:
            list[Chromatogram]: The filtered list of Chromatogram objects
        """

        return FilterWrapper[Chromatogram](self.chromatograms, **kwargs).filter()

    def filter_initial_conditions(self, **kwargs) -> list[InitialCondition]:
        """Filters the initial_conditions attribute based on the given kwargs

        Args:
            **kwargs: The attributes to filter by.

        Returns:
            list[InitialCondition]: The filtered list of InitialCondition objects
        """

        return FilterWrapper[InitialCondition](self.initial_conditions, **kwargs).filter()


    def set_attr_term(
        self,
        attr: str,
        term: str | dict,
        prefix: str | None = None,
        iri: str | None = None
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

        assert attr in self.model_fields, f"Attribute {attr} not found in {self.__class__.__name__}"

        if prefix:
            validate_prefix(term, prefix)

        add_namespace(self, prefix, iri)
        self.ld_context[attr] = term

    def add_type_term(
        self,
        term: str,
        prefix: str | None = None,
        iri: str | None = None
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
        id: str,
        sample_id: str,
        signal: list[float]= [],
        time: list[float]= [],
        peaks: list[Peak]= [],
        wavelength: Optional[float]= None,
        reaction_time: Optional[float]= None,
        reaction_time_unit: Optional[UnitDefinitionAnnot]= None,
        **kwargs,
    ):
        params = {
            "id": id,
            "sample_id": sample_id,
            "signal": signal,
            "time": time,
            "peaks": peaks,
            "wavelength": wavelength,
            "reaction_time": reaction_time,
            "reaction_time_unit": reaction_time_unit
        }

        if "id" in kwargs:
            params["id"] = kwargs["id"]

        self.chromatograms.append(
            Chromatogram(**params)
        )

        return self.chromatograms[-1]


    def add_to_initial_conditions(
        self,
        molecule_id: str,
        init_conc: float,
        conc_unit: UnitDefinitionAnnot,
        **kwargs,
    ):
        params = {
            "molecule_id": molecule_id,
            "init_conc": init_conc,
            "conc_unit": conc_unit
        }

        if "id" in kwargs:
            params["id"] = kwargs["id"]

        self.initial_conditions.append(
            InitialCondition(**params)
        )

        return self.initial_conditions[-1]


class Chromatogram(BaseModel):

    model_config: ConfigDict = ConfigDict( # type: ignore
        validate_assignment = True,
    ) # type: ignore

    id: str = Field(
        default=...,
        description="""Unique identifier of the sample.""",
    )
    sample_id: str = Field(
        default=...,
        description="""Identifier of the sample this chromatogram is
        part of.""",
    )
    signal: list[float] = Field(
        default_factory=list,
        description="""Signal values.""",
    )
    time: list[float] = Field(
        default_factory=list,
        description="""Time values of the signal in minutes.""",
    )
    peaks: list[Peak] = Field(
        default_factory=list,
        description="""Peaks in the signal.""",
    )
    wavelength: Optional[float] = Field(
        default=None,
        description="""Wavelength of the signal in nm.""",
    )
    reaction_time: Optional[float] = Field(
        default=None,
        description="""Time relative to reaction start""",
    )
    reaction_time_unit: Optional[UnitDefinitionAnnot] = Field(
        default=None,
        description="""Unit of reaction time""",
    )

    # JSON-LD fields
    ld_id: str = Field(
        serialization_alias="@id",
        default_factory=lambda: "chromhander:Chromatogram/" + str(uuid4())
    )
    ld_type: list[str] = Field(
        serialization_alias="@type",
        default_factory = lambda: [
            "chromhander:Chromatogram",
        ],
    )
    ld_context: dict[str, str | dict] = Field(
        serialization_alias="@context",
        default_factory = lambda: {
            "chromhander": "https://github.com/FAIRChemistry/chromhandler",
            "om": "http://www.ontology-of-units-of-measure.org/resource/om-2/",
            "qudt": "http://qudt.org/schema/qudt#/",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#/",
            "schema": "http://schema.org/",
            "unit": "http://qudt.org/vocab/unit#/",
            "xsd": "http://www.w3.org/2001/XMLSchema#/",
            "Peak": "https://github.com/FAIRChemistry/chromhandler#Peak/",
        }
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
        iri: str | None = None
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

        assert attr in self.model_fields, f"Attribute {attr} not found in {self.__class__.__name__}"

        if prefix:
            validate_prefix(term, prefix)

        add_namespace(self, prefix, iri)
        self.ld_context[attr] = term

    def add_type_term(
        self,
        term: str,
        prefix: str | None = None,
        iri: str | None = None
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
        chromatogram_id: str,
        location: Estimate,
        area: Estimate,
        skew: Optional[Estimate]= None,
        width: Optional[Estimate]= None,
        molecule_id: Optional[str]= None,
        type: Optional[str]= None,
        amplitude: Optional[float]= None,
        max_signal: Optional[float]= None,
        percent_area: Optional[float]= None,
        tailing_factor: Optional[float]= None,
        separation_factor: Optional[float]= None,
        peak_start: Optional[float]= None,
        peak_end: Optional[float]= None,
        **kwargs,
    ):
        params = {
            "chromatogram_id": chromatogram_id,
            "location": location,
            "area": area,
            "skew": skew,
            "width": width,
            "molecule_id": molecule_id,
            "type": type,
            "amplitude": amplitude,
            "max_signal": max_signal,
            "percent_area": percent_area,
            "tailing_factor": tailing_factor,
            "separation_factor": separation_factor,
            "peak_start": peak_start,
            "peak_end": peak_end
        }

        if "id" in kwargs:
            params["id"] = kwargs["id"]

        self.peaks.append(
            Peak(**params)
        )

        return self.peaks[-1]


class Estimate(BaseModel):

    model_config: ConfigDict = ConfigDict( # type: ignore
        validate_assignment = True,
    ) # type: ignore

    mean: float = Field(
        default=...,
        description="""Mean value of the estimate.""",
    )
    median: Optional[float] = Field(
        default=None,
        description="""Median of the estimate.""",
    )
    std: Optional[float] = Field(
        default=None,
        description="""One sigma standard deviation of the estimate.""",
    )
    q05: Optional[float] = Field(
        default=None,
        description="""5th percentile of the estimate.""",
    )
    q95: Optional[float] = Field(
        default=None,
        description="""95th percentile of the estimate.""",
    )
    samples: list[float] = Field(
        default_factory=list,
        description="""Samples from the posterior distribution.""",
    )

    # JSON-LD fields
    ld_id: str = Field(
        serialization_alias="@id",
        default_factory=lambda: "chromhander:Estimate/" + str(uuid4())
    )
    ld_type: list[str] = Field(
        serialization_alias="@type",
        default_factory = lambda: [
            "chromhander:Estimate",
        ],
    )
    ld_context: dict[str, str | dict] = Field(
        serialization_alias="@context",
        default_factory = lambda: {
            "chromhander": "https://github.com/FAIRChemistry/chromhandler",
            "om": "http://www.ontology-of-units-of-measure.org/resource/om-2/",
            "qudt": "http://qudt.org/schema/qudt#/",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#/",
            "schema": "http://schema.org/",
            "unit": "http://qudt.org/vocab/unit#/",
            "xsd": "http://www.w3.org/2001/XMLSchema#/",
        }
    )


    def set_attr_term(
        self,
        attr: str,
        term: str | dict,
        prefix: str | None = None,
        iri: str | None = None
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

        assert attr in self.model_fields, f"Attribute {attr} not found in {self.__class__.__name__}"

        if prefix:
            validate_prefix(term, prefix)

        add_namespace(self, prefix, iri)
        self.ld_context[attr] = term

    def add_type_term(
        self,
        term: str,
        prefix: str | None = None,
        iri: str | None = None
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


class Peak(BaseModel):

    model_config: ConfigDict = ConfigDict( # type: ignore
        validate_assignment = True,
    ) # type: ignore

    chromatogram_id: str = Field(
        default=...,
        description="""Identifier of the chromatogram this peak is part
        of.""",
    )
    location: Estimate = Field(
        default=...,
        description="""Retention time of the peak in minutes.""",
    )
    area: Estimate = Field(
        default=...,
        description="""Area of the peak.""",
    )
    skew: Optional[Estimate] = Field(
        default=None,
        description="""Skew of the peak.""",
    )
    width: Optional[Estimate] = Field(
        default=None,
        description="""Width of the peak.""",
    )
    molecule_id: Optional[str] = Field(
        default=None,
        description="""Identifier of the molecule.""",
    )
    type: Optional[str] = Field(
        default=None,
        description="""Type of peak (baseline-baseline / baseline-
        valley / ...)""",
    )
    amplitude: Optional[float] = Field(
        default=None,
        description="""Amplitude of the peak.""",
    )
    max_signal: Optional[float] = Field(
        default=None,
        description="""Maximum signal of the peak.""",
    )
    percent_area: Optional[float] = Field(
        default=None,
        description="""Percent area of the peak.""",
    )
    tailing_factor: Optional[float] = Field(
        default=None,
        description="""Tailing factor of the peak.""",
    )
    separation_factor: Optional[float] = Field(
        default=None,
        description="""Separation factor of the peak.""",
    )
    peak_start: Optional[float] = Field(
        default=None,
        description="""Start time of the peak.""",
    )
    peak_end: Optional[float] = Field(
        default=None,
        description="""End time of the peak.""",
    )

    # JSON-LD fields
    ld_id: str = Field(
        serialization_alias="@id",
        default_factory=lambda: "chromhander:Peak/" + str(uuid4())
    )
    ld_type: list[str] = Field(
        serialization_alias="@type",
        default_factory = lambda: [
            "chromhander:Peak",
        ],
    )
    ld_context: dict[str, str | dict] = Field(
        serialization_alias="@context",
        default_factory = lambda: {
            "chromhander": "https://github.com/FAIRChemistry/chromhandler",
            "om": "http://www.ontology-of-units-of-measure.org/resource/om-2/",
            "qudt": "http://qudt.org/schema/qudt#/",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#/",
            "schema": "http://schema.org/",
            "unit": "http://qudt.org/vocab/unit#/",
            "xsd": "http://www.w3.org/2001/XMLSchema#/",
            "Estimate": "https://github.com/FAIRChemistry/chromhandler#Estimate/",
            "Estimate": "https://github.com/FAIRChemistry/chromhandler#Estimate/",
            "Estimate": "https://github.com/FAIRChemistry/chromhandler#Estimate/",
            "Estimate": "https://github.com/FAIRChemistry/chromhandler#Estimate/",
        }
    )


    def set_attr_term(
        self,
        attr: str,
        term: str | dict,
        prefix: str | None = None,
        iri: str | None = None
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

        assert attr in self.model_fields, f"Attribute {attr} not found in {self.__class__.__name__}"

        if prefix:
            validate_prefix(term, prefix)

        add_namespace(self, prefix, iri)
        self.ld_context[attr] = term

    def add_type_term(
        self,
        term: str,
        prefix: str | None = None,
        iri: str | None = None
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


class InitialCondition(BaseModel):

    model_config: ConfigDict = ConfigDict( # type: ignore
        validate_assignment = True,
    ) # type: ignore

    molecule_id: str = Field(
        default=...,
        description="""Identifier of the molecule.""",
    )
    init_conc: float = Field(
        default=...,
        description="""Initial concentration of the molecule.""",
    )
    conc_unit: UnitDefinitionAnnot = Field(
        default=...,
        description="""Unit of the concentration.""",
    )

    # JSON-LD fields
    ld_id: str = Field(
        serialization_alias="@id",
        default_factory=lambda: "chromhander:InitialCondition/" + str(uuid4())
    )
    ld_type: list[str] = Field(
        serialization_alias="@type",
        default_factory = lambda: [
            "chromhander:InitialCondition",
        ],
    )
    ld_context: dict[str, str | dict] = Field(
        serialization_alias="@context",
        default_factory = lambda: {
            "chromhander": "https://github.com/FAIRChemistry/chromhandler",
            "om": "http://www.ontology-of-units-of-measure.org/resource/om-2/",
            "qudt": "http://qudt.org/schema/qudt#/",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#/",
            "schema": "http://schema.org/",
            "unit": "http://qudt.org/vocab/unit#/",
            "xsd": "http://www.w3.org/2001/XMLSchema#/",
        }
    )


    def set_attr_term(
        self,
        attr: str,
        term: str | dict,
        prefix: str | None = None,
        iri: str | None = None
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

        assert attr in self.model_fields, f"Attribute {attr} not found in {self.__class__.__name__}"

        if prefix:
            validate_prefix(term, prefix)

        add_namespace(self, prefix, iri)
        self.ld_context[attr] = term

    def add_type_term(
        self,
        term: str,
        prefix: str | None = None,
        iri: str | None = None
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