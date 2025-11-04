"""
This is an example script for running kilosort2.5 on the HPC.
It requires a GPU node, and expects singularity is installed.
"""
import spikeinterface.full as si
from pathlib import Path

base_path = Path(__file__).parent / "example_data"
data_path = base_path / r"rawdata" / "sub-001" / "ses-001" / "ephys"
output_path = base_path / "derivatives" / "sub-001" / "ses-001" / "ephys"

# Loading Raw Data ---------------------------------------------------------------------

print("Loading data)")

raw_recording = si.read_spikeglx(data_path, stream_id="imec0.ap")

# Preprocessing ------------------------------------------------------------------------
# We skip the whitening step, because it is performed in Kilosort

print("Running preprocessing")

shifted_recording = si.phase_shift(raw_recording)

filtered_recording = si.bandpass_filter(
    shifted_recording, freq_min=300, freq_max=6000
)

preprocessed_recording = si.common_reference(
    filtered_recording, reference="global", operator="median"
)

# Sorting ------------------------------------------------------------------------------
# We do common median referencing in SpikeInterface, and skip Common Average Reference
# in Kilosort. We also set the frequency cutoff for Kilosort#s lowpass filter
# very low (so it has no effect) because we bandpass filter in SpikeInterface.

print("Starting sorting")

sorting_output_path = output_path / "sorting"

sorting = si.run_sorter(
   "kilosort2_5",
   preprocessed_recording,
   folder=(output_path / "sorting").as_posix(),
   singularity_image=True,
   car=False,
   freq_min=150,
)