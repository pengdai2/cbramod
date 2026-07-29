import argparse
from datetime import timedelta
import os
import sys
import pyedflib


def slice_edf_time_window(input_path, output_path, start_time_sec, end_time_sec):
    """Slices a time window from an EDF file and saves it to a new file.

    Parameters:
    - input_path: str, path to source .edf file
    - output_path: str, path for sliced output .edf file
    - start_time_sec: float, start of window in seconds
    - end_time_sec: float, end of window in seconds
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    duration_sec = end_time_sec - start_time_sec
    if duration_sec <= 0:
        print("Error: End time must be strictly greater than start time.", file=sys.stderr)
        sys.exit(1)

    # Open the source EDF file
    with pyedflib.EdfReader(input_path) as reader:
        n_channels = reader.signals_in_file
        sample_frequencies = [reader.getSampleFrequency(i) for i in range(n_channels)]
        signal_headers = reader.getSignalHeaders()
        header = reader.getHeader()
        orig_start_time = reader.getStartdatetime()

        # Read data and slice into an array for each channel
        sliced_signals = []
        for i in range(n_channels):
            sf = sample_frequencies[i]
            start_sample = int(round(start_time_sec * sf))
            n_samples_to_read = int(round(duration_sec * sf))

            try:
                # Read physical signal values
                signal_data = reader.readSignal(i, start=start_sample, n=n_samples_to_read)
                sliced_signals.append(signal_data)
            except Exception as e:
                print(f"Error reading channel {i}: {e}", file=sys.stderr)
                sys.exit(1)

        # Adjust start time of the new file
        new_start_time = orig_start_time + timedelta(seconds=start_time_sec)

        # Update global header metadata for the new duration
        new_header = header.copy()
        new_header["starttime"] = new_start_time

    # Write the sliced data to a new file
    writer = pyedflib.EdfWriter(output_path, n_channels, file_type=pyedflib.FILETYPE_EDFPLUS)
    writer.setHeader(new_header)
    writer.setSignalHeaders(signal_headers)
    writer.writeSamples(sliced_signals)
    writer.close()
    print(f"Successfully saved sliced window to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Slice a specific time window from an EDF file using pyedflib."
    )
    
    # Required arguments
    parser.add_argument("-i", "--input", required=True, help="Path to the input EDF file")
    parser.add_argument("-o", "--output", required=True, help="Path to save the sliced output EDF file")
    parser.add_argument("-s", "--start", type=float, required=True, help="Start time of the window in seconds")
    parser.add_argument("-e", "--end", type=float, required=True, help="End time of the window in seconds")

    args = parser.parse_args()

    slice_edf_time_window(
        input_path=args.input,
        output_path=args.output,
        start_time_sec=args.start,
        end_time_sec=args.end
    )


if __name__ == "__main__":
    main()
