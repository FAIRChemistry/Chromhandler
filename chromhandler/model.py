from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar
from uuid import uuid4

from mdmodels.units.annotation import UnitDefinitionAnnot  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    pass

# Filter Wrapper definition used to filter a list of objects
# based on their attributes
Cls = TypeVar("Cls")


class FilterWrapper(Generic[Cls]):
    """Wrapper class to filter a list of objects based on their attributes"""

    def __init__(self, collection: list[Cls], **kwargs: Any) -> None:
        self.collection = collection
        self.kwargs = kwargs

    def filter(self) -> list[Cls]:
        for key, value in self.kwargs.items():
            self.collection = [item for item in self.collection if self._fetch_attr(key, item) == value]
        return self.collection

    def _fetch_attr(self, name: str, item: Cls) -> Any:
        try:
            return getattr(item, name)
        except AttributeError as err:
            raise AttributeError(f"{item} does not have attribute {name}") from err


# JSON-LD Helper Functions
def add_namespace(obj: Any, prefix: str | None, iri: str | None) -> None:
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

    obj.ld_context[prefix] = iri  # type: ignore[index]


def validate_prefix(term: str | dict[str, str], prefix: str) -> None:
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
    model_config: ConfigDict = ConfigDict(  # type: ignore
        validate_assignment=True,
    )  # type: ignore

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
    timestamp: str | None = Field(
        default=None,
        description="""Timestamp of sample injection into the column.""",
    )
    injection_volume: float | None = Field(
        default=None,
        description="""Injection volume.""",
    )
    dilution_factor: float | None = Field(
        default=1,
        description="""Dilution factor.""",
    )
    injection_volume_unit: str | UnitDefinitionAnnot | None = Field(
        default=None,
        description="""Unit of injection volume.""",
    )

    # JSON-LD fields
    ld_id: str = Field(
        serialization_alias="@id", default_factory=lambda: "chromhander:Sample/" + str(uuid4())
    )
    ld_type: list[str] = Field(
        serialization_alias="@type",
        default_factory=lambda: [
            "chromhander:Sample",
        ],
    )
    ld_context: dict[str, str | dict[str, str]] = Field(
        serialization_alias="@context",
        default_factory=lambda: {
            "chromhander": "https://github.com/FAIRChemistry/chromhandler",
            "om": "http://www.ontology-of-units-of-measure.org/resource/om-2/",
            "qudt": "http://qudt.org/schema/qudt#/",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#/",
            "schema": "http://schema.org/",
            "unit": "http://qudt.org/vocab/unit#/",
            "xsd": "http://www.w3.org/2001/XMLSchema#/",
            "Chromatogram": "https://github.com/FAIRChemistry/chromhandler#Chromatogram/",
            "InitialCondition": "https://github.com/FAIRChemistry/chromhandler#InitialCondition/",
        },
    )

    def filter_chromatograms(self, **kwargs: Any) -> list[Chromatogram]:
        """Filters the chromatograms attribute based on the given kwargs

        Args:
            **kwargs: The attributes to filter by.

        Returns:
            list[Chromatogram]: The filtered list of Chromatogram objects
        """

        return FilterWrapper[Chromatogram](self.chromatograms, **kwargs).filter()

    def filter_initial_conditions(self, **kwargs: Any) -> list[InitialCondition]:
        """Filters the initial_conditions attribute based on the given kwargs

        Args:
            **kwargs: The attributes to filter by.

        Returns:
            list[InitialCondition]: The filtered list of InitialCondition objects
        """

        return FilterWrapper[InitialCondition](self.initial_conditions, **kwargs).filter()

    def set_attr_term(
        self, attr: str, term: str | dict[str, str], prefix: str | None = None, iri: str | None = None
    ) -> None:
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

        assert attr in self.__class__.model_fields, f"Attribute {attr} not found in {self.__class__.__name__}"

        if prefix:
            validate_prefix(term, prefix)

        add_namespace(self, prefix, iri)
        self.ld_context[attr] = term

    def add_type_term(self, term: str, prefix: str | None = None, iri: str | None = None) -> None:
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
        signal: list[float] | None = None,
        time: list[float] | None = None,
        peaks: list[Peak] | None = None,
        wavelength: float | None = None,
        reaction_time: float | None = None,
        reaction_time_unit: str | UnitDefinitionAnnot | None = None,
        **kwargs: Any,
    ) -> Any:
        if signal is None:
            signal = []
        if time is None:
            time = []
        if peaks is None:
            peaks = []
        params: dict[str, Any] = {
            "id": id,
            "sample_id": sample_id,
            "signal": signal,
            "time": time,
            "peaks": peaks,
            "wavelength": wavelength,
            "reaction_time": reaction_time,
            "reaction_time_unit": reaction_time_unit,
        }

        if "id" in kwargs:
            params["id"] = kwargs["id"]

        self.chromatograms.append(Chromatogram(**params))

        return self.chromatograms[-1]

    def add_to_initial_conditions(
        self,
        molecule_id: str,
        init_conc: float,
        conc_unit: str | UnitDefinitionAnnot,
        **kwargs: Any,
    ) -> Any:
        params: dict[str, Any] = {"molecule_id": molecule_id, "init_conc": init_conc, "conc_unit": conc_unit}

        if "id" in kwargs:
            params["id"] = kwargs["id"]

        self.initial_conditions.append(InitialCondition(**params))

        return self.initial_conditions[-1]


class Chromatogram(BaseModel):
    model_config: ConfigDict = ConfigDict(  # type: ignore
        validate_assignment=True,
    )  # type: ignore

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
    wavelength: float | None = Field(
        default=None,
        description="""Wavelength of the signal in nm.""",
    )
    reaction_time: float | None = Field(
        default=None,
        description="""Time relative to reaction start""",
    )
    reaction_time_unit: str | UnitDefinitionAnnot | None = Field(
        default=None,
        description="""Unit of reaction time""",
    )

    # JSON-LD fields
    ld_id: str = Field(
        serialization_alias="@id", default_factory=lambda: "chromhander:Chromatogram/" + str(uuid4())
    )
    ld_type: list[str] = Field(
        serialization_alias="@type",
        default_factory=lambda: [
            "chromhander:Chromatogram",
        ],
    )
    ld_context: dict[str, str | dict[str, str]] = Field(
        serialization_alias="@context",
        default_factory=lambda: {
            "chromhander": "https://github.com/FAIRChemistry/chromhandler",
            "om": "http://www.ontology-of-units-of-measure.org/resource/om-2/",
            "qudt": "http://qudt.org/schema/qudt#/",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#/",
            "schema": "http://schema.org/",
            "unit": "http://qudt.org/vocab/unit#/",
            "xsd": "http://www.w3.org/2001/XMLSchema#/",
            "Peak": "https://github.com/FAIRChemistry/chromhandler#Peak/",
        },
    )

    def filter_peaks(self, **kwargs: Any) -> list[Peak]:
        """Filters the peaks attribute based on the given kwargs

        Args:
            **kwargs: The attributes to filter by.

        Returns:
            list[Peak]: The filtered list of Peak objects
        """

        return FilterWrapper[Peak](self.peaks, **kwargs).filter()

    def set_attr_term(
        self, attr: str, term: str | dict[str, str], prefix: str | None = None, iri: str | None = None
    ) -> None:
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

        assert attr in self.__class__.model_fields, f"Attribute {attr} not found in {self.__class__.__name__}"

        if prefix:
            validate_prefix(term, prefix)

        add_namespace(self, prefix, iri)
        self.ld_context[attr] = term

    def add_type_term(self, term: str, prefix: str | None = None, iri: str | None = None) -> None:
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
        skew: Estimate | None = None,
        width: Estimate | None = None,
        molecule_id: str | None = None,
        type: str | None = None,
        amplitude: float | None = None,
        max_signal: float | None = None,
        percent_area: float | None = None,
        tailing_factor: float | None = None,
        separation_factor: float | None = None,
        peak_start: float | None = None,
        peak_end: float | None = None,
        **kwargs: Any,
    ) -> Any:
        params: dict[str, Any] = {
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
            "peak_end": peak_end,
        }

        if "id" in kwargs:
            params["id"] = kwargs["id"]

        self.peaks.append(Peak(**params))

        return self.peaks[-1]


class Estimate(BaseModel):
    model_config: ConfigDict = ConfigDict(  # type: ignore
        validate_assignment=True,
    )  # type: ignore

    mean: float = Field(
        default=...,
        description="""Mean value of the estimate.""",
    )
    median: float | None = Field(
        default=None,
        description="""Median of the estimate.""",
    )
    std: float | None = Field(
        default=None,
        description="""One sigma standard deviation of the estimate.""",
    )
    q05: float | None = Field(
        default=None,
        description="""5th percentile of the estimate.""",
    )
    q95: float | None = Field(
        default=None,
        description="""95th percentile of the estimate.""",
    )
    samples: list[float] = Field(
        default_factory=list,
        description="""Samples from the posterior distribution.""",
    )

    # JSON-LD fields
    ld_id: str = Field(
        serialization_alias="@id", default_factory=lambda: "chromhander:Estimate/" + str(uuid4())
    )
    ld_type: list[str] = Field(
        serialization_alias="@type",
        default_factory=lambda: [
            "chromhander:Estimate",
        ],
    )
    ld_context: dict[str, str | dict[str, str]] = Field(
        serialization_alias="@context",
        default_factory=lambda: {
            "chromhander": "https://github.com/FAIRChemistry/chromhandler",
            "om": "http://www.ontology-of-units-of-measure.org/resource/om-2/",
            "qudt": "http://qudt.org/schema/qudt#/",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#/",
            "schema": "http://schema.org/",
            "unit": "http://qudt.org/vocab/unit#/",
            "xsd": "http://www.w3.org/2001/XMLSchema#/",
        },
    )

    def set_attr_term(
        self, attr: str, term: str | dict[str, str], prefix: str | None = None, iri: str | None = None
    ) -> None:
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

        assert attr in self.__class__.model_fields, f"Attribute {attr} not found in {self.__class__.__name__}"

        if prefix:
            validate_prefix(term, prefix)

        add_namespace(self, prefix, iri)
        self.ld_context[attr] = term

    def add_type_term(self, term: str, prefix: str | None = None, iri: str | None = None) -> None:
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
    model_config: ConfigDict = ConfigDict(  # type: ignore
        validate_assignment=True,
    )  # type: ignore

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
    skew: Estimate | None = Field(
        default=None,
        description="""Skew of the peak.""",
    )
    width: Estimate | None = Field(
        default=None,
        description="""Width of the peak.""",
    )
    molecule_id: str | None = Field(
        default=None,
        description="""Identifier of the molecule.""",
    )
    type: str | None = Field(
        default=None,
        description="""Type of peak (baseline-baseline / baseline-
        valley / ...)""",
    )
    amplitude: float | None = Field(
        default=None,
        description="""Amplitude of the peak.""",
    )
    max_signal: float | None = Field(
        default=None,
        description="""Maximum signal of the peak.""",
    )
    percent_area: float | None = Field(
        default=None,
        description="""Percent area of the peak.""",
    )
    tailing_factor: float | None = Field(
        default=None,
        description="""Tailing factor of the peak.""",
    )
    separation_factor: float | None = Field(
        default=None,
        description="""Separation factor of the peak.""",
    )
    peak_start: float | None = Field(
        default=None,
        description="""Start time of the peak.""",
    )
    peak_end: float | None = Field(
        default=None,
        description="""End time of the peak.""",
    )

    # JSON-LD fields
    ld_id: str = Field(serialization_alias="@id", default_factory=lambda: "chromhander:Peak/" + str(uuid4()))
    ld_type: list[str] = Field(
        serialization_alias="@type",
        default_factory=lambda: [
            "chromhander:Peak",
        ],
    )
    ld_context: dict[str, str | dict[str, str]] = Field(
        serialization_alias="@context",
        default_factory=lambda: {
            "chromhander": "https://github.com/FAIRChemistry/chromhandler",
            "om": "http://www.ontology-of-units-of-measure.org/resource/om-2/",
            "qudt": "http://qudt.org/schema/qudt#/",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#/",
            "schema": "http://schema.org/",
            "unit": "http://qudt.org/vocab/unit#/",
            "xsd": "http://www.w3.org/2001/XMLSchema#/",
            "Estimate": "https://github.com/FAIRChemistry/chromhandler#Estimate/",
        },
    )

    def set_attr_term(
        self, attr: str, term: str | dict[str, str], prefix: str | None = None, iri: str | None = None
    ) -> None:
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

        assert attr in self.__class__.model_fields, f"Attribute {attr} not found in {self.__class__.__name__}"

        if prefix:
            validate_prefix(term, prefix)

        add_namespace(self, prefix, iri)
        self.ld_context[attr] = term

    def add_type_term(self, term: str, prefix: str | None = None, iri: str | None = None) -> None:
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
    model_config: ConfigDict = ConfigDict(  # type: ignore
        validate_assignment=True,
    )  # type: ignore

    molecule_id: str = Field(
        default=...,
        description="""Identifier of the molecule.""",
    )
    init_conc: float = Field(
        default=...,
        description="""Initial concentration of the molecule.""",
    )
    conc_unit: str | UnitDefinitionAnnot = Field(
        default=...,
        description="""Unit of the concentration.""",
    )

    # JSON-LD fields
    ld_id: str = Field(
        serialization_alias="@id", default_factory=lambda: "chromhander:InitialCondition/" + str(uuid4())
    )
    ld_type: list[str] = Field(
        serialization_alias="@type",
        default_factory=lambda: [
            "chromhander:InitialCondition",
        ],
    )
    ld_context: dict[str, str | dict[str, str]] = Field(
        serialization_alias="@context",
        default_factory=lambda: {
            "chromhander": "https://github.com/FAIRChemistry/chromhandler",
            "om": "http://www.ontology-of-units-of-measure.org/resource/om-2/",
            "qudt": "http://qudt.org/schema/qudt#/",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#/",
            "schema": "http://schema.org/",
            "unit": "http://qudt.org/vocab/unit#/",
            "xsd": "http://www.w3.org/2001/XMLSchema#/",
        },
    )

    def set_attr_term(
        self, attr: str, term: str | dict[str, str], prefix: str | None = None, iri: str | None = None
    ) -> None:
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

        assert attr in self.__class__.model_fields, f"Attribute {attr} not found in {self.__class__.__name__}"

        if prefix:
            validate_prefix(term, prefix)

        add_namespace(self, prefix, iri)
        self.ld_context[attr] = term

    def add_type_term(self, term: str, prefix: str | None = None, iri: str | None = None) -> None:
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
