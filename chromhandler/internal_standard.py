# from loguru import logger
# from mdmodels.units.annotation import UnitDefinition
# from pydantic import BaseModel, Field

# from chromhandler.model import Sample
# from chromhandler.utility import _resolve_chromatogram

# logger.level("INFO")


# class InternalStandard(BaseModel):
#     molecule_id: str = Field(
#         description="The ID of the molecule.",
#     )
#     standard_molecule_id: str = Field(
#         description="The ID of the standard molecule.",
#     )
#     molecule_init_conc: float = Field(description="The initial concentration of the molecule.")
#     molecule_conc_unit: UnitDefinition = Field(description="The unit of concentration for the molecule.")
#     standard_init_conc: float = Field(description="The initial concentration of the standard.")
#     molecule_t0_signal: float | None = Field(description="The t0 signal of the molecule.", default=None)
#     standard_t0_signal: float | None = Field(description="The t0 signal of the standard.", default=None)

#     def _set_t0_signals(
#         self,
#         sample: "Sample",
#     ) -> None:
#         """Sets the t0 signals of the molecule and the standard from the t=0 chromatogram.

#         The first chromatogram in *sample* (sorted by reaction_time) is used as t=0.
#         """
#         chroms_at_t0 = sorted(
#             (c for c in sample.chromatograms if c.reaction_time is not None),
#             key=lambda c: c.reaction_time,  # type: ignore[arg-type]
#         )
#         assert chroms_at_t0, "No chromatogram with a reaction_time found in the sample."
#         assert chroms_at_t0[0].reaction_time == 0, (
#             "No measurement data is available at t=0. Concentration calculation is not "
#             "possible without a chromatogram at reaction_time=0."
#         )

#         for peak in _resolve_chromatogram(chromatograms=[chroms_at_t0[0]], wavelength=None).peaks:
#             if peak.molecule_id == self.molecule_id:
#                 self.molecule_t0_signal = peak.area
#             elif peak.molecule_id == self.standard_molecule_id:
#                 self.standard_t0_signal = peak.area

#         assert self.molecule_t0_signal is not None, f"""
#         No peak area for {self.molecule_id} was found in measurement at t=0.
#         """

#         assert self.standard_t0_signal is not None, f"""
#         No peak area for {self.standard_molecule_id} was found in measurement at t=0.
#         """

#         logger.debug(f"molecule t0: {self.molecule_t0_signal}, standard t0: {self.standard_t0_signal}")

#     def calculate_conc(self, molecue_signal: float, standard_molecule_signal: float) -> float:
#         """Calculates the concentration of the internal standard based on the peak area.

#         Args:
#             molecue_signal (float): The signal of the molecule.
#             standard_molecule_signal (float): The signal of the standard molecule.

#         Returns:
#             float: The concentration of the internal standard.

#         Raises:
#             ValueError: If any signal is below the minimum threshold.
#         """
#         MIN_SIGNAL_THRESHOLD = 100  # Adjust this based on your typical noise level

#         assert self.molecule_t0_signal is not None, """
#         The t0 signal of the molecule is not defined.
#         Use the `_set_t0_signals` method to set the t0 signals.
#         """

#         assert self.standard_t0_signal is not None, """
#         The t0 signal of the standard is not defined.
#         Use the `_set_t0_signals` method to set the t0 signals.
#         """

#         # Validate signal strengths
#         if standard_molecule_signal < MIN_SIGNAL_THRESHOLD:
#             raise ValueError(
#                 f"Internal standard signal {standard_molecule_signal} is below minimum threshold {MIN_SIGNAL_THRESHOLD}"
#             )

#         if molecue_signal < MIN_SIGNAL_THRESHOLD:
#             raise ValueError(
#                 f"Molecule signal {molecue_signal} is below minimum threshold {MIN_SIGNAL_THRESHOLD}"
#             )

#         if self.standard_t0_signal < MIN_SIGNAL_THRESHOLD:
#             raise ValueError(
#                 f"Initial internal standard signal {self.standard_t0_signal} is below minimum threshold {MIN_SIGNAL_THRESHOLD}"
#             )

#         if self.molecule_t0_signal < MIN_SIGNAL_THRESHOLD:
#             raise ValueError(
#                 f"Initial molecule signal {self.molecule_t0_signal} is below minimum threshold {MIN_SIGNAL_THRESHOLD}"
#             )

#         t0_ratio = self.molecule_t0_signal / self.standard_t0_signal
#         ratio = molecue_signal / standard_molecule_signal

#         conc = self.molecule_init_conc * (ratio / t0_ratio)

#         logger.debug(f"initial conc: {self.molecule_init_conc}")
#         logger.debug(f"t0 ratio: {t0_ratio}, ratio: {ratio}, conc: {conc}")

#         return conc
