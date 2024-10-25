
import spikeinterface.extractors as si_extractors
import spikeinterface.preprocessing as si_prepro
import spikeinterface.sorters as si_sorters

from pathlib import Path

base_path = Path(r"/Users/joeziminski/PycharmProjects/ephys-course-2024-2/course-extracellular-ephys-analysis/example_data")
data_path = base_path / r"rawdata" / "sub-001" / "ses-001" / "ephys"
output_path = base_path / "derivatives" / "sub-001" / "ses-001" / "ephys"

# Loading Raw Data ---------------------------------------------------------------------

print("Loading data)")

raw_recording = si_extractors.read_spikeglx(data_path)

# Preprocessing ------------------------------------------------------------------------
# We skip the whitening step, because it is performed in Kilosort

print("Running preprocessing")

shifted_recording = si_prepro.phase_shift(raw_recording)

if False:
    filtered_recording = si_prepro.bandpass_filter(
        shifted_recording, freq_min=300, freq_max=6000
    )

    preprocessed_recording = si_prepro.common_reference(
        filtered_recording, reference="global", operator="median"
    )

# Sorting ------------------------------------------------------------------------------
# We do common median referencing in SpikeInterface, and skip Common Average Reference
# in Kilosort. We also set the frequency cutoff for Kilosort#s lowpass filter
# very low (so it has no effect) because we bandpass filter in SpikeInterface.

print("Starting sorting")

sorting_output_path = output_path / "sorting"

sorting = si_sorters.run_sorter(
   "kilosort4",
   shifted_recording,
    remove_existing_folder=True
#   do_CAR=False,
#  freq_min=300,
)
