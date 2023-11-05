import probeinterface.plotting as pi_plot

import spikeinterface.extractors as si_extractors
import spikeinterface.preprocessing as si_prepro
import spikeinterface.sorters as si_sorters
import spikeinterface.widgets as si_widgets
import spikeinterface.curation as si_curation
import spikeinterface.postprocessing as si_postprocess
from spikeinterface import extract_waveforms, qualitymetrics

from pathlib import Path
import matplotlib.pyplot as plt

show_probe = True
show_preprocessing = False
show_waveform = False

base_path = Path(r"C:\fMRIData\git-repo\extracellular-ephys-analysis-course-2023\example_data")
data_path = base_path / r"rawdata" / "sub-001" / "ses-001" / "ephys"
output_path = base_path / "derivatives" / "sub-001" / "ses-001" / "ephys"

# Loading Raw Data ---------------------------------------------------------------------

raw_recording = si_extractors.read_spikeglx(data_path)

if show_probe:
    probe = raw_recording.get_probe()
    pi_plot.plot_probe(probe, with_contact_id=True)
    plt.show()


print(raw_recording)

print(dir(raw_recording))

print(raw_recording.get_sampling_frequency())

example_data = raw_recording.get_traces(start_frame=0, end_frame=1000, return_scaled=True)

print(example_data)

# Preprocessing ------------------------------------------------------------------------

shifted_recording = si_prepro.phase_shift(raw_recording)

filtered_recording = si_prepro.bandpass_filter(
    shifted_recording, freq_min=300, freq_max=6000
)

common_referenced_recording = si_prepro.common_reference(
    filtered_recording, reference="global", operator="median"
)

preprocessed_recording = si_prepro.whiten(common_referenced_recording, dtype='float32')

if show_preprocessing:
    si_widgets.plot_timeseries(
        preprocessed_recording ,
        order_channel_by_depth=True,
        time_range=(1, 2),
        return_scaled=True,
        show_channel_ids=True,
        mode="map",  # "map", "line"
    )
    plt.show()


# TODO: manage existing projects
# TODO: get this working on macos
sorting_output_path = output_path / "sorting"

if (sorting_output_path / "sorter_output").is_dir():
    sorting = si_extractors.NpzSortingExtractor(
        sorting_output_path / "sorter_output" / "firings.npz"
    )
else:
    sorting = si_sorters.run_sorter(
       "mountainsort5",
       preprocessed_recording,
       output_folder=output_path / "sorting",
    )

sorting = sorting.remove_empty_units()

sorting = si_curation.remove_excess_spikes(
    sorting, preprocessed_recording
)

waveforms = extract_waveforms(
    preprocessed_recording,
    sorting,
    folder=output_path / "postprocessing",
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
    valid_unit_ids = waveforms.unit_ids
    unit_to_show = valid_unit_ids[0]
    unit_waveform_data = waveforms.get_waveforms(unit_id=unit_to_show)

    print(f"The shape of the waveform data is "
          f"num_waveforms x num_samples x num_channels: {unit_waveform_data.shape}")

    single_waveform_data = unit_waveform_data[0, :, :]
    plt.plot(single_waveform_data)
    plt.title("Data from a single waveform")
    plt.show()

    unit_template_data = waveforms.get_template(unit_id=unit_to_show)
    print(f"The template is averaged over all waveforms. The shape"
          f"of the template data is num_samples x num_channels: {unit_template_data.shape}")

    plt.plot(unit_template_data)
    plt.title(f"Template for unit: {unit_to_show}")
    plt.show()

# Save Quality Metrics
quality_metrics_path = output_path / "quality_metrics.csv"

si_postprocess.compute_principal_components(
    waveforms, n_components=5, mode='by_channel_local'
)
quality_metrics = qualitymetrics.compute_quality_metrics(waveforms)
quality_metrics.to_csv(quality_metrics_path)
