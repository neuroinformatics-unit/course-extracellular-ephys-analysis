import probeinterface.plotting as pi_plot

import spikeinterface.extractors as si_extractors
import spikeinterface.preprocessing as si_prepro
import spikeinterface.sorters as si_sorters
import spikeinterface.widgets as si_widgets
import spikeinterface.curation as si_curation
import spikeinterface.postprocessing as si_postprocess
from spikeinterface import extract_waveforms, qualitymetrics, load_waveforms, load_extractor

from pathlib import Path
import matplotlib.pyplot as plt

show_probe = True
show_preprocessing = True
show_waveform = True

base_path = Path(r"C:\Users\Joe\PycharmProjects\ephys-workshop-2023\extracellular-ephys-analysis-course-2023\example_data")
data_path = base_path / "rawdata" / "sub-001" / "ses-001" / "ephys"
output_path = base_path / "derivatives" / "sub-001" / "ses-001" / "ephys"

# Loading Raw Data ---------------------------------------------------------------------

raw_recording = si_extractors.read_spikeglx(data_path)

if show_probe:
    probe = raw_recording.get_probe()
    pi_plot.plot_probe(probe, with_contact_id=True)
    plt.show()

# Extra things to try
print(raw_recording)
# SpikeGLXRecordingExtractor: 384 channels - 30.0kHz - 1 segments - 90,000 samples - 3.00s
#                             int16 dtype - 65.92 MiB
# It is a SpikeGLXRecordingExtractor class

print(dir(raw_recording))
# dir() shows all class attributes and methods for the class

print(raw_recording.get_sampling_frequency())
# 30 kHz. This class method can be seen on the results from dir()

example_data = raw_recording.get_traces(start_frame=0, end_frame=1000, return_scaled=True)
# Returns the data, as a num_samples x num_channels array. We index from first (index 0) sample
# 1000th (index 999, as the end_frame is upper-bound exclusive).
# The `raw_recording` is lazy object that only loads data into memory when requested.

# Create and plot the Fourier Transform of a single channel
import numpy as np
sampling_frequency = 30000
single_channgel_data = example_data[:, 0]
num_samples = single_channgel_data.size
freqs = np.fft.fftfreq(num_samples, d=1/sampling_frequency)
signal_fft = np.fft.fft(single_channgel_data)
magnitude_fft = np.abs(signal_fft)
scale_magnitude_fft = magnitude_fft * 2 / single_channgel_data.size

plt.plot(freqs, scale_magnitude_fft)
plt.ylabel("Scaled Magnitude")
plt.xlabel("Frequency (Hz)")
plt.title("Demeaned Signal Frequency Spectrum")
plt.show()



# Preprocessing ------------------------------------------------------------------------

preprocessed_output_path = output_path / "preprocessed"

if preprocessed_output_path.is_dir():
    preprocessed_recording = load_extractor(preprocessed_output_path)
else:
    shifted_recording = si_prepro.phase_shift(raw_recording)

    filtered_recording = si_prepro.bandpass_filter(
        shifted_recording, freq_min=300, freq_max=6000
    )
    common_referenced_recording = si_prepro.common_reference(
        filtered_recording, reference="global", operator="median"
    )
    whitened_recording = si_prepro.whiten(
        common_referenced_recording, dtype='float32'
    )
    preprocessed_recording = si_prepro.correct_motion(
        whitened_recording, preset="kilosort_like"
    )  # see also 'nonrigid_accurate'

    preprocessed_recording.save(folder=preprocessed_output_path)


if show_preprocessing:
    si_widgets.plot_traces(
        preprocessed_recording,
        order_channel_by_depth=True,
        time_range=(2, 3),
        return_scaled=True,
        show_channel_ids=True,
        mode="map",  # "map", "line"
        clim=(-10, 10),  # after whitening, use (-10, 10) otherwise use (-200, 200)
    )
    plt.show()

# Preprocessing - Extra things to try
# Whitening completely changes the scaling of the data.

# The data looks very similar when unscaled, as int16. This is because the range
# and precision is not changed by scaling, only the scaling of the values, placing them
# in more interpretable units. Just because we are using float rather than int16, because
# the data is acquired as int16 we do not increse the resolution of the recording simply
# by scaling.

# We can save the data with
# preprocessed_data_path = output_path / "preprocessed_data"
# preprocessed_recording.save(folder=preprocessed_data_path)

# Setting the bandpass filter minimum cutoff to zero would include freuqencies
# in the range 0 - 6000 Hz.
# Setting the bandpass filter maximum (having returned the minimum to 300) would include
# 300 - 15000 Hz. 15000 Hz is chosen as it is the Nyquist frequency (half the sampling
# rate of 30 kHz) that represents the largest detectable frequency in the recorded signal.

# using preprocessed_recording.get_traces(start_frame=0, end_frame=1000, return_scaled=True)
# (or False, 1000 samples are taken arbitarily) shows the int16 data as acquired (when False)
# or the same data scaled to microvolts. You will see that if two datapoints are the same
# value when int16, they are the same value as microvolts.

example_prepro_data = preprocessed_recording.get_traces(0, 60000)
single_channgel_data = example_prepro_data[:, 300]
standard_dev = np.std(single_channgel_data)  # ideally we would measure this with spikes removed
mean_ = np.mean(single_channgel_data)

standard_dev_cutoff = mean_ - 3 * standard_dev
spike_indicies = np.where(single_channgel_data < standard_dev_cutoff)
# Using the std methods with adjustment is not good  but takes multiple points on single AP


# Instead use scipy inbuilt function for the distance
import scipy
distance_between_peaks_in_samples = int(0.001 * sampling_frequency)  # spikes at least 1ms apart

# However, scipy function is extremely annoying and does not have a simple threshold
# cutoff, which we want for this example. By visualising we see that the threshold is
# under-estimating the peaks because of the way prominenec works:
# We can emulate this here by re-thresholding to 3x the std
# Note, there are peak finding algorithms in other packages you can use for threshold
# cutoffs, see https://stackoverflow.com/questions/1713335/peak-finding-algorithm-for-python-scipy
# Also, in general there are better ways to find peaks, i.e. template matching
spike_indicies = scipy.signal.find_peaks(single_channgel_data * -1,  # need to invert to find negative peaks
                                         distance=distance_between_peaks_in_samples,
                                         prominence=standard_dev_cutoff * -1)[0]  # we inverted so looking for 'positive

spike_indicies = spike_indicies[single_channgel_data[spike_indicies] < standard_dev_cutoff]

plt.plot(single_channgel_data)
plt.scatter(spike_indicies, single_channgel_data[spike_indicies])
plt.show()

sorting_path = output_path / "sorting"

if (expected_filepath := sorting_path / "sorter_output" / "firings.npz").is_dir():
    sorting = si_extractors.NpzSortingExtractor(
        expected_filepath
    )
else:
    sorting = si_sorters.run_sorter(
       "mountainsort5",
       preprocessed_recording,
       output_folder=sorting_path,
       remove_existing_folder=True,
       filter=False,
       whiten=False,
    )

sorting = sorting.remove_empty_units()

sorting = si_curation.remove_excess_spikes(
    sorting, preprocessed_recording
)

# Sorting - Extra things to try
# Use this function to get the times of all APs for a unit.
spike_times = sorting.get_unit_spike_train(unit_id=2, return_times=True)

# use the si_widgets.post_rasters functino as below to view the unit spikes
# as a raster plot.
si_widgets.plot_rasters(sorting, unit_ids=[2])
plt.show()

# The conditional statement is as above.

waveforms_path = output_path / "postprocessing"

if waveforms_path.is_dir():
    waveforms = load_waveforms(waveforms_path)
else:
    waveforms = extract_waveforms(
        preprocessed_recording,
        sorting,
        folder=waveforms_path,
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

# Extra things to try
# Index out the most-negtive channel and plot
unit_waveform_data = waveforms.get_waveforms(unit_id=2)

import numpy as np
# Lets index out the data from a single action potential.
# `first_ap_data` is a num_samples x num_channels array
first_ap_data = unit_waveform_data[0, :, :]

# Let's just find the most-negative (i.e. the largest negative peak_ value
# across all timepoints, all channels). This is over a 2D array, but np.argmin
# flattens the array, and so the index of the value is this flattened array (it is
# something like 700). We need to convert it back to a 2D index (index of the
# timepoint, index of the channel). To do this, we use `unravel_index` function.
largest_neg_idx = np.argmin(first_ap_data)
largest_neg_idx = np.unravel_index(largest_neg_idx, first_ap_data.shape)

# Now,  we index out the num_samples x num_channel array at the channel in which
# the most negative value was found. `largest_neg_index` has two values,
# the first is the timepoint where the minimum value was found, the second is the
# channel where the minimum value was found.
largest_neg_channel = first_ap_data[:, largest_neg_idx[1]]

plt.plot(largest_neg_channel)
plt.show()

# Now, we want to plot a dot over the plot of the AP. The axis of the AP
# plot are the voltage on the y-axis, and the sample index on the x-axis
# The sample index is the timepoint of where the negative value was found,
# as above. We can get the voltage by finding the minimum value of the AP.
# Then, we can plot as a plt.scatter(<index of min value>, <min value>).
peak_value = np.min(largest_neg_channel)
plt.plot(largest_neg_channel)
plt.scatter(largest_neg_idx[0], peak_value)
plt.show()

# Save Quality Metrics
quality_metrics_path = output_path / "quality_metrics.csv"

si_postprocess.compute_principal_components(
    waveforms, n_components=5, mode='by_channel_local'
)
quality_metrics = qualitymetrics.compute_quality_metrics(waveforms)
quality_metrics.to_csv(quality_metrics_path)
