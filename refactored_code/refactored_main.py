
import spikeinterface.extractors as si_extractors
from spikeinterface import extract_waveforms


from pathlib import Path

from myproject_pipeline_functions import (
    display_with_index,
    preprocess_for_mountainsort5,
    show_recording_heatmap,
    get_mountainsort5_sorting_object,
    save_waveform_quality_metrics,
)

show_probe = True
show_preprocessing = False
show_waveform = False

base_path = Path(r"C:\fMRIData\git-repo\extracellular-ephys-analysis-course-2023\example_data")
data_path = base_path / r"rawdata" / "sub-001" / "ses-001" / "ephys"
output_path = base_path / "derivatives" / "sub-001" / "ses-001" / "ephys"

# Loading Raw Data ---------------------------------------------------------------------

assert data_path.is_dir(), ("`data_path` does not exist! This should be a "
                            "folder containing data.")

raw_recording = si_extractors.read_spikeglx(data_path)

assert raw_recording.get_sampling_frequency() == 30000, (f"Sampling frequency of the loaded file is not 30,000 Hz!" 
                                                         f"It is {raw_recording.get_sampling_frequency()}")
if show_probe:
    display_with_index(raw_recording)

# Preprocessing ------------------------------------------------------------------------

preprocessed_recording = preprocess_for_mountainsort5(raw_recording)

if show_preprocessing:
    show_recording_heatmap(recording, time_range=(1, 2))

# Sorting ------------------------------------------------------------------------------

sorting = get_mountainsort5_sorting_object(output_path, preprocessed_recording)

# Waveforms and Quality Metrics --------------------------------------------------------

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

save_waveform_quality_metrics(output_path, waveforms)