Chromatopy can currently read data from Shimadzu chromatographic system. In the future, it will be able to read data from Agilent and Thermo Fischer chromatographic systems as well. The tool is designed to efficiently handle data from time-course experiments, where multiple samples are analyzed in a sequence.

The `ChromAnalyzer` is the central object to read and process chromatographic data. The scope of one `ChromAnalyzer` instance is one sample which is measured over time

## Reading data

Data can be read by using the `read_data` method. Thereby, the path to a directory, or to a file can be passed. The method will automatically detect the file format and read the data accordingly.

```python
from chromatopy.tools import ChromAnalyzer

path = "path/to/data"
analyzer = ChromAnalyzer.read_data(calibration_path)
```

## Visualization of raw data

The raw data can be visualized by using the `plot_measurements` method. The method will plot the raw chromatograms of the samples.

```python
analyzer.plot_measurements()
```

## Assigning measurement conditions

The `add_species` method allows to assign information on the measured species to the `ChromAnalyzer` object. In this way peaks of a given retention time can be assigned, or the concentration of protein sequence of a catalyst can be added to the `ChromAnalyzer`.

### Adding measurement conditions to peaks

By specifying setting the `retention_time` attribute, the `add_species` method can be used to assign a peak to a given species. Thereby, all peaks loaded to the analyzer will be extracted from the chromatogram and assigned to the species. Additionally, the `chebi_id` attribute can be set to assign a ChEBI identifier to the species. Furthermore, the `init_conc` and `conc_unit` can be specified. The corresponding reaction times of the species can be added by setting the `reaction_times` and `time_unit` attributes.

```python
analyzer.add_species(
    retention_time=10.5, 
    chebi_id="12345", 
    init_conc=1, 
    conc_unit="mmol/l"
    reaction_times=[0, 10, 20],
    time_unit="min"
)
```

### Adding measurement conditions to unobserved species

If a species is not observed in the chromatogram, the `add_species` method can be used to assign the species to the `ChromAnalyzer` object. In this way e.g. a protein catalyst can be defined by specifying either the `uniprot_id`` or the `sequence` attribute. Additionally, the `init_conc` and `conc_unit` can be specified.

```python
analyzer.add_species(
    name="Protein catalyst", 
    uniprot_id="P12345", 
    init_conc=1, 
    conc_unit="nmol/l"
)
```
