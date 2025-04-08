import numpy as np
import argparse

def compare_npz_files(file1, file2):
    """Compare two NPZ files by computing the sum of absolute differences between arrays."""
    npz1 = np.load(file1)
    npz2 = np.load(file2)

    keys1 = set(npz1.files)
    keys2 = set(npz2.files)

    # Check if both files contain the same keys
    if keys1 != keys2:
        print("Mismatch in internal file names:")
        print("Only in", file1, ":", keys1 - keys2)
        print("Only in", file2, ":", keys2 - keys1)
        return

    differences_found = False

    # Compare each corresponding array
    for key in keys1:
        arr1 = npz1[key]
        arr2 = npz2[key]

        # Check if the arrays have the same shape
        print(key)
        if arr1.shape != arr2.shape:
            print(f"Shape mismatch in '{key}': {arr1.shape} vs {arr2.shape}")
            differences_found = True
            continue  # Skip difference calculation for mismatched shapes

        # Compute sum of absolute differences
        diff_sum = np.sum(np.abs(arr1 - arr2))

        # Report difference if any
        if diff_sum != 0:
            print(f"Difference in '{key}': sum of absolute differences = {diff_sum}, array size = {arr1.size}")
            differences_found = True

    if not differences_found:
        print("All internal files are identical (zero difference).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two NPZ files by summing absolute differences between arrays.")
    parser.add_argument("file1", type=str, help="Path to the first NPZ file")
    parser.add_argument("file2", type=str, help="Path to the second NPZ file")
    
    args = parser.parse_args()
    compare_npz_files(args.file1, args.file2)
