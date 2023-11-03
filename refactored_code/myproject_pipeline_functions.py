import probeinterface.plotting as pi_plot

import spikeinterface.extractors as si_extractors
import spikeinterface.preprocessing as si_prepro
import spikeinterface.sorters as si_sorters
import spikeinterface.widgets as si_widgets
import spikeinterface.curation as si_curation
import spikeinterface.postprocessing as si_postprocess
from spikeinterface import qualitymetrics


import matplotlib.pyplot as plt


def display_with_index(recording):
    """
    Plot the probe associated with a recording.

    Parameters
    ----------

    recording :
        SpikeInterface recording object.
    """
    probe = recording.get_probe()
    pi_plot.plot_probe(probe, with_device_index=True)
    plt.show()


def preprocess_for_mountainsort5(raw_recording):
    """
    <Requires documentation>
    """
    shifted_recording = si_prepro.phase_shift(raw_recording)

    filtered_recording = si_prepro.bandpass_filter(
        shifted_recording, freq_min=300, freq_max=6000
    )

    common_referenced_recording = si_prepro.common_reference(
        filtered_recording, reference="global", operator="median"
    )

    preprocessed_recording = si_prepro.whiten(common_referenced_recording,
                                              dtype='float32')

    return preprocessed_recording


def show_recording_heatmap(recording, time_range):
    """
    <Requires Documentation>
    """
    si_widgets.plot_timeseries(
        recording,
        order_channel_by_depth=True,
        time_range=time_range,
        return_scaled=True,
        show_channel_ids=True,
        mode="map",
    )
    plt.show()


def get_mountainsort5_sorting_object(output_path, preprocessed_recording):
    """
    <Requires Documentation>
    """
    sorting_output_path = output_path / "sorting"

    if (sorting_output_path / "sorter_output").is_dir():
        sorting = si_extractors.NpzSortingExtractor(
            (sorting_output_path / "sorter_output" / "firings.npz").as_posix()
        )
    else:
        sorting = si_sorters.run_sorter(
           "mountainsort5",
           preprocessed_recording,
           output_folder=(output_path / "sorting").as_posix(),
        )

    sorting = sorting.remove_empty_units()

    sorting = si_curation.remove_excess_spikes(
        sorting, preprocessed_recording
    )

    return sorting


def save_waveform_quality_metrics(output_path, waveforms):
    """
    <Requires Documentation>
    """
    quality_metrics_path = output_path / "quality_metrics.csv"

    si_postprocess.compute_principal_components(
        waveforms, n_components=5, mode='by_channel_local'
    )
    quality_metrics = qualitymetrics.compute_quality_metrics(waveforms)
    quality_metrics.to_csv(quality_metrics_path)