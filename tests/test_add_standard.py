"""`add_standard` must build each molecule's standard from that molecule's peaks only (issue #14)."""

from chromhandler.handler import Handler
from chromhandler.model import (
    Chromatogram,
    Data,
    DataType,
    Measurement,
    Peak,
    SignalType,
)

CONCS = [1.0, 2.0, 3.0]
AREAS = {
    "glc": [101.0, 198.0, 303.0],
    "lac": [510.0, 990.0, 1520.0],
}  # per molecule, per conc


def _handler() -> Handler:
    measurements = []
    for i, conc in enumerate(CONCS):
        peaks = [
            Peak(retention_time=5.0, area=AREAS["glc"][i]),
            Peak(retention_time=8.0, area=AREAS["lac"][i]),
        ]
        measurements.append(
            Measurement(
                id=f"std_{conc}",
                data=Data(value=conc, unit="mmol/L", data_type=DataType.CALIBRATION),
                temperature=25.0,
                temperature_unit="Celsius",
                ph=7.0,
                chromatograms=[Chromatogram(type=SignalType.UV, peaks=peaks)],
            )
        )
    return Handler(id="cal", name="cal", mode="calibration", measurements=measurements)


def test_add_standard_for_two_molecules_uses_each_molecules_own_areas() -> None:
    handler = _handler()
    mol_a = handler.define_molecule(
        id="glc", name="glucose", pubchem_cid=1, retention_time=5.0, auto_assign=True
    )
    mol_b = handler.define_molecule(
        id="lac", name="lactate", pubchem_cid=2, retention_time=8.0, auto_assign=True
    )

    for mol in (mol_a, mol_b):
        handler.add_standard(mol, visualize=False)
        assert mol.standard is not None
        samples = sorted(mol.standard.samples, key=lambda s: s.concentration)
        assert [s.concentration for s in samples] == CONCS
        assert [s.signal for s in samples] == AREAS[mol.id]
