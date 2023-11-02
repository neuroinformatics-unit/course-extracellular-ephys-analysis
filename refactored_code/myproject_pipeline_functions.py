import probeinterface.plotting as pi_plot

import spikeinterface.extractors as si_extractors
import spikeinterface.preprocessing as si_prepro
import spikeinterface.sorters as si_sorters
import spikeinterface.widgets as si_widgets
import spikeinterface.curation as si_curation
import spikeinterface.postprocessing as si_postprocess
from spikeinterface import qualitymetrics

import matplotlib.pyplot as plt


def display_the_probe(recording):
    """
    """
    probe = recording.get_probe()
    pi_plot.plot_probe(probe, with_device_index=True)
    plt.show()


def display_recording(recording, time_range):
    """
    """
    si_widgets.plot_timeseries(
        recording,
        order_channel_by_depth=True,
        time_range=time_range,
        return_scaled=True,
        show_channel_ids=True,
        mode="map",  # "map", "line"
    )
    plt.show()


def run_standard_preprocessing(raw_recording):
    """
    """
    shifted_recording = si_prepro.phase_shift(raw_recording)

    filtered_recording = si_prepro.bandpass_filter(
        shifted_recording, freq_min=300, freq_max=6000
    )
    preprocessed_recording = si_prepro.common_reference(
        filtered_recording, reference="global", operator="median"
    )

    return preprocessed_recording


def load_or_run_sorting(sorting_output_path, preprocessed_recording):
    """
    """
    if (sorting_output_path / "sorter_output").is_dir():
        sorting = si_extractors.NpzSortingExtractor(
            (sorting_output_path / "sorter_output" / "firings.npz").as_posix())
    else:
        sorting = si_sorters.run_sorter(
            "mountainsort5",
            preprocessed_recording,
            output_folder=sorting_output_path.as_posix(),
        )

    sorting = sorting.remove_empty_units()

    sorting = si_curation.remove_excess_spikes(
        sorting, preprocessed_recording
    )

    return sorting


def show_waveform_and_template(waveforms, unit_id=2):
    """
    """
    unit_waveform_data = waveforms.get_waveforms(unit_id=unit_id)

    single_waveform_data = unit_waveform_data[0, :, :]
    plt.plot(single_waveform_data)
    plt.title("Data from a single waveform")
    plt.show()

    unit_template_data = waveforms.get_template(unit_id=unit_id)

    plt.plot(unit_template_data)
    plt.title(f"Template for unit: {unit_id}")
    plt.show()


def save_quality_metrics(waveforms, quality_metrics_path):
    """
    """
    si_postprocess.compute_principal_components(waveforms, n_components=5, mode='by_channel_local')  # in place
    quality_metrics = qualitymetrics.compute_quality_metrics(waveforms)
    quality_metrics.to_csv(quality_metrics_path)
