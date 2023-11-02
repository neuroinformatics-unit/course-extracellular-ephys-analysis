
import spikeinterface.extractors as si_extractors
from spikeinterface import extract_waveforms

from pathlib import Path

from my_first_project.pipeline_functions import (
    display_the_probe,
    display_recording,
    run_standard_preprocessing,
    load_or_run_sorting,
    show_waveform_and_template,
    save_quality_metrics,
)

# Setup Options and Paths
show_probe = False
show_preprocessing = False
show_waveform = False

base_path = Path(r"C:\data\ephys\extracellular-ephys-analysis-2023\example_data")
data_path = base_path / r"rawdata" / "sub-001" / "ses-001" / "ephys"
output_path = base_path / "derivatives" / "sub-001" / "ses-001" / "ephys"

# load Raw Data
raw_recording = si_extractors.read_spikeglx(data_path)

if show_probe:
    display_the_probe(raw_recording)

# Preprocess Data
preprocessed_recording = run_standard_preprocessing(raw_recording)

if show_preprocessing:
    display_recording(preprocessed_recording, time_range=(1, 2))

# Sorting
sorting_output_path = output_path / "sorting"

sorting = load_or_run_sorting(sorting_output_path, preprocessed_recording)

# Waveform Extraction
waveforms = extract_waveforms(
    preprocessed_recording,
    sorting,
    folder=(output_path / "postprocessing").as_posix(),
    overwrite=True,
    ms_before=2,
    ms_after=2,
    max_spikes_per_unit=500,
    return_scaled=True,
    # Sparsity Options
    radius_um=75,
    method="radius",
    sparse=True,
)

if show_waveform:
    show_waveform_and_template(waveforms, unit_id=2)

# Save Quality Metrics
quality_metrics_path = output_path / "quality_metrics.csv"

save_quality_metrics(waveforms, quality_metrics_path)

