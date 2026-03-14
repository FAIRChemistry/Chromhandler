A `Calibrator` can be created by reading data from multiple measurements. After defining the calibration object, which is specific to a measured species, the calibrator can be used to calculate the concentration of peaks of the same species.

First, the data is read in, and the peaks from which the calibration object should be created are assigned to a species.
```python
from chromatopy.tools import ChromAnalyzer, Calibrator

calibration_path = "path/to/calibration/data"

# read in data from path
calibration_analyzer = ChromAnalyzer.read_data(calibration_path)

# Get peaks of the species to create calibration curve
oxalac = calib_analyzer.add_species(
    name="Oxaloacetate", retention_time=21.35, chebi=chebi_oxaloacetate
)
```

After defining the species, the calibration object can be created by specifying the concentrations of the measured species.
```python
# Define concentrations and corresponding unit
concentrations = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
conc_unit = "mmol/L"

# Create a calibrator for the substrate
substrate_calibrator = Calibrator.from_species(
    species=substrate_species, concentrations=concentrations, conc_unit=conc_unit
)

# Plot the calibration curve
substrate_calibrator.plot()
```

### Defining a calibrator without measured species

Alternatively, a calibrator can be created without defining a species. In this case, the species is not observed in the chromatogram, but the slope is provided.
```python
product_calibrator = Calibrator(
    name="Citrate", slope=199030, conc_unit="mmol/L", chebi=12345
)
```

## Calculating concentrations

After defining the calibrator, the calibrator can be added to a `ChromAnalyzer` by adding it with the `add_calibrator` method. 

```python
analyzer.add_calibrator(substrate_calibrator)
```

The added calibrators are then used to calculate concentrations of the peaks, which share the same `chebi_id` or the same `name` attribute as the calibrator.

In this way, a `DataFrame` with the calculated concentrations can be obtained by using the `calculate_concentrations` method.
```python
df = analyzer._create_df()
```

## Next step
In the next development steps, instead of yielding a `DataFrame`, the results are mapped to `EnzymeML`, allowing to store the results in a standardized format and apply further analysis steps.