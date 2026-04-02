import json
import re

import requests
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .molecule import validate_python_identifier


class Protein(BaseModel):
    model_config: ConfigDict = ConfigDict(  # type: ignore
        validate_assignment=True,
        use_enum_values=True,
    )

    id: str = Field(
        description="ID of the Protein",
    )
    name: str = Field(
        description="Name of the protein",
    )
    sequence: str | None = Field(
        description="Amino acid sequence of the protein",
        default=None,
    )
    organism: str | None = Field(
        description="Organism from which the protein originates",
        default=None,
    )
    organism_tax_id: str | None = Field(
        description="Taxonomic ID of the organism",
        default=None,
    )
    constant: bool = Field(
        description="Boolean indicating whether the protein concentration is constant",
        default=True,
    )

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return validate_python_identifier(value, field_name="Protein.id")

    @classmethod
    def read_json(cls, path: str) -> "Protein":
        """Creates a Protein instance from a JSON file.

        Args:
            path (str): The path to the JSON file.

        Returns:
            Protein: The created Protein instance.
        """
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def save_json(self, path: str) -> None:
        """Saves the Protein instance to a JSON file.

        Args:
            path (str): The path to the JSON file.

        Returns:
            None
        """
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=4))

    @classmethod
    def from_uniprot(cls, uniprot_id: str, name: str | None = None) -> "Protein":
        """
        Creates a Protein instance from a UniProt ID using the UniProt API.

        Args:
            uniprot_id (str): The UniProt accession or entry name.

        Returns:
            Protein: The created Protein instance.

        Raises:
            ValueError: If the UniProt entry cannot be found or parsed.
        """
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(
                f"Could not fetch UniProt entry for ID '{uniprot_id}'. Status code: {response.status_code}"
            )

        data = response.json()

        # Extract fields from UniProt JSON
        protein_name = None
        try:
            protein_name = data["proteinDescription"]["recommendedName"]["fullName"]["value"]
        except Exception:
            # fallback to submittedName or alternativeName if available
            try:
                protein_name = data["proteinDescription"]["submittedName"][0]["fullName"]["value"]
            except Exception:
                protein_name = None

        sequence = data.get("sequence", {}).get("value", None)

        organism = None
        organism_tax_id = None
        if "organism" in data:
            organism = data["organism"].get("scientificName", None)
            taxon_ids = data["organism"].get("taxonId", None)
            if taxon_ids is not None:
                organism_tax_id = str(taxon_ids)

        return cls(
            id=uniprot_id,
            name=name or protein_name or uniprot_id,
            sequence=sequence,
            organism=organism,
            organism_tax_id=organism_tax_id,
        )

    @property
    def ld_id_url(self) -> str | None:
        """Returns the URL of the UniProt page of the protein based on the protein ID

        Returns:
            str | None: URL of the UniProt page of the protein if the protein ID is defined, None otherwise.
        """

        uniprot_pattern = r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}"

        if re.fullmatch(uniprot_pattern, self.id) is None:
            return None
        else:
            return f"https://www.uniprot.org/uniprotkb/{self.id}/entry"
