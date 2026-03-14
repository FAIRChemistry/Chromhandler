---
repo: "https://github.com/FAIRChemistry/chromhandler"
prefix: "chromhander"
---

# Chromatography Data Model

## Objects

### Sample


- __id__
  - Type: string
  - Description: Unique identifier of the sample.
- __chromatograms__
  - Type: Chromatogram[]
  - Description: Measured chromatogram and peaks.
- initial_conditions
  - Type: InitialCondition[]
  - Description: Initial conditions of the sample.
- timestamp
  - Type: string
  - Description: Timestamp of sample injection into the column.
- injection_volume
  - Type: float
  - Description: Injection volume.
- dilution_factor
  - Type: float
  - Description: Dilution factor.
  - Default: 1.0
- injection_volume_unit
  - Type: UnitDefinition
  - Description: Unit of injection volume.


### Chromatogram

- __id__
  - Type: string
  - Description: Unique identifier of the sample.
- __sample_id__
  - Type: string
  - Description: Identifier of the sample this chromatogram is part of.
- signal
  - Type: float[]
  - Description: Signal values.
- time
  - Type: float[]
  - Description: Time values of the signal in minutes.
- peaks
  - Type: Peak[]
  - Description: Peaks in the signal.
- wavelength
  - Type: float
  - Description: Wavelength of the signal in nm.
- reaction_time
  - Type: float
  - Description: Time relative to reaction start
- reaction_time_unit
  - Type: UnitDefinition
  - Description: Unit of reaction time

### Estimate

- __mean__
  - Type: float
  - Description: Mean value of the estimate.
- median
  - Type: float
  - Description: Median of the estimate.
- std
  - Type: float
  - Description: One sigma standard deviation of the estimate.
- q05
  - Type: float
  - Description: 5th percentile of the estimate.
- q95
  - Type: float
  - Description: 95th percentile of the estimate.
- samples
  - Type: float[]
  - Description: Samples from the posterior distribution.


### Peak

- __chromatogram_id__
  - Type: string
  - Description: Identifier of the chromatogram this peak is part of.
- __location__
  - Type: Estimate
  - Description: Retention time of the peak in minutes.
- __area__
  - Type: Estimate
  - Description: Area of the peak.
- skew
  - Type: Estimate
  - Description: Skew of the peak.
- width
  - Type: Estimate
  - Description: Width of the peak.
- molecule_id
  - Type: string
  - Description: Identifier of the molecule.
- type
  - Type: string
  - Description: Type of peak (baseline-baseline / baseline-valley / ...)
- amplitude
  - Type: float
  - Description: Amplitude of the peak.
- max_signal
  - Type: float
  - Description: Maximum signal of the peak.
- percent_area
  - Type: float
  - Description: Percent area of the peak.
- tailing_factor
  - Type: float
  - Description: Tailing factor of the peak.
- separation_factor
  - Type: float
  - Description: Separation factor of the peak.
- peak_start
  - Type: float
  - Description: Start time of the peak.
- peak_end
  - Type: float
  - Description: End time of the peak.

### InitialCondition

- __molecule_id__
  - Type: string
  - Description: Identifier of the molecule.
- __init_conc__
  - Type: float
  - Description: Initial concentration of the molecule.
- __conc_unit__
  - Type: UnitDefinition
  - Description: Unit of the concentration.
