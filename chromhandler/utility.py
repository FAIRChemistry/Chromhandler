import sys

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pyenzyme import DataTypes, EnzymeMLDocument, SmallMolecule

from chromhandler.model import Chromatogram

logger.remove()
logger.add(sys.stderr, level="INFO")


def _resolve_chromatogram(
    chromatograms: list[Chromatogram], wavelength: float | None
) -> Chromatogram:
    if len(chromatograms) == 1:
        return chromatograms[0]

    if len(chromatograms) > 1:
        assert (
            wavelength is not None
        ), "Multiple chromatograms found, but no wavelength is specified."

        # check that any of the chromatograms has the specified wavelength
        assert any(
            [chrom.wavelength == wavelength for chrom in chromatograms]
        ), f"No chromatogram found with wavelength {wavelength} nm."

        return next(chrom for chrom in chromatograms if chrom.wavelength == wavelength)

    raise ValueError("No chromatogram found.")


def generate_gaussian_data(
    amplitude: float,
    center: float,
    half_height_diameter: float,
    start: float,
    end: float,
    num_points: int = 100,
) -> tuple[list[float], list[float]]:
    """
    Generate x and y data for a Gaussian curve.

    Parameters:
    - amplitude: The peak height of the Gaussian.
    - center: The position of the center of the peak.
    - half_height_diameter: The full width at half maximum (FWHM) of the peak.
    - start: The starting x-value.
    - end: The ending x-value.
    - num_points: Number of points to generate (default is 100).

    Returns:
    - x_values: List of x-values.
    - y_values: List of y-values corresponding to the Gaussian curve.
    """
    # Calculate sigma from the half-height diameter (FWHM)
    sigma = half_height_diameter / (2 * np.sqrt(2 * np.log(2)))

    # Generate x values
    x_values = np.linspace(start, end, num_points)

    # Generate y values using the Gaussian function
    y_values = amplitude * np.exp(-((x_values - center) ** 2) / (2 * sigma**2))

    return x_values.tolist(), y_values.tolist()


def visualize_enzymeml(
    enzymeml_doc: EnzymeMLDocument, return_fig: bool = False
) -> Figure | None:
    """visualize the data in the EnzymeML document

    Args:
        enzymeml_doc (EnzymeMLDocument): The EnzymeML document to visualize
        return_fig (bool, optional): Whether to return the figure. Defaults to False.
    """
    for species in enzymeml_doc.measurements[0].species_data:
        if species.data:
            plt.scatter(
                species.time,
                species.data,
                label=get_species_by_id(enzymeml_doc, species.species_id).name,
            )
    plt.legend()

    # handel y label
    if species.data_type == DataTypes.PEAK_AREA:
        plt.ylabel("Peak Area [-]")
    elif species.data_type == DataTypes.CONCENTRATION:
        plt.ylabel(f"concentration [{species.data_unit.name}]")
    plt.xlabel(f"reaction time [{species.time_unit.name}]")

    plt.show()
    if return_fig:
        return plt.gcf()
    return None


def get_species_by_id(enzymeml_doc: EnzymeMLDocument, species_id: str) -> SmallMolecule:
    for species in enzymeml_doc.small_molecules:
        if species.id == species_id:
            return species
    raise ValueError(f"Species with ID {species_id} not found.")
