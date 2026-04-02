from __future__ import annotations

import json
import keyword

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .calibration import LinearCalibration  # noqa: TC001 — Pydantic needs runtime access


def validate_python_identifier(value: str, *, field_name: str) -> str:
    """Require identifiers that can be used safely as DottedDict attribute names."""
    if not value.isidentifier() or keyword.iskeyword(value):
        raise ValueError(
            f"{field_name} must be a valid Python identifier, got {value!r}. "
            "Use letters, digits, and underscores only, without a leading digit "
            "and without Python keywords."
        )
    return value


class Molecule(BaseModel):
    model_config: ConfigDict = ConfigDict(  # type: ignore
        validate_assignment=True,
        use_enum_values=True,
    )

    id: str = Field(
        description="ID of the molecule",
    )
    pubchem_cid: int = Field(
        description="PubChem CID of the molecule",
    )
    name: str = Field(
        description="Name of the molecule",
    )
    standard: LinearCalibration | None = Field(
        description="Standard associated with the molecule",
        default=None,
    )
    constant: bool = Field(
        description=(
            "Boolean indicating whether the molecule concentration is constant throughout the experiment"
        ),
        default=False,
    )
    internal_standard: bool = Field(
        description="Boolean indicating whether the molecule is an internal standard",
        default=False,
    )
    calibration: LinearCalibration | None = Field(
        default=None,
        description="Linear calibration model fitted from t=0 calibration standards.",
    )

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return validate_python_identifier(value, field_name="Molecule.id")

    @classmethod
    def read_json(cls, path: str) -> Molecule:
        """Creates a Molecule instance from a JSON file.

        Args:
            path (str): The path to the JSON file.

        Returns:
            Molecule: The created Molecule instance.
        """

        with open(path) as f:
            data = json.load(f)

        return cls(**data)

    def save_json(self, path: str) -> None:
        """Saves the Molecule instance to a JSON file.

        Args:
            path (str): The path to the JSON file.

        Returns:
            None
        """

        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=4))
