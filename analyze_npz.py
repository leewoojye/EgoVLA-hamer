import numpy as np
import sys
import os

def analyze_npz(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        # Load the NPZ file
        data = np.load(file_path, allow_pickle=True)
        
        print(f"\nAnalysis of: {os.path.basename(file_path)}")
        print(f"Location:   {os.path.abspath(file_path)}")
        print("=" * 80)
        print(f"{'Key':<25} | {'Type':<25} | {'Shape / Length':<20}")
        print("-" * 80)

        # Get all keys and sort them for finding easier
        keys = sorted(data.files)
        
        first_dim_counts = []

        for key in keys:
            val = data[key]
            
            # Determine type string
            type_str = type(val).__name__
            if isinstance(val, np.ndarray):
                 type_str += f" ({val.dtype})"

            # Determine shape or length string
            shape_str = "N/A"
            if hasattr(val, 'shape'):
                shape_str = str(val.shape)
                if len(val.shape) > 0:
                    first_dim_counts.append(val.shape[0])
            elif hasattr(val, '__len__'):
                shape_str = str(len(val))
                first_dim_counts.append(len(val))
            
            print(f"{key:<25} | {type_str:<25} | {shape_str:<20}")

        print("=" * 80)

        # Heuristic to guess sample count (N)
        if first_dim_counts:
            from collections import Counter
            counts = Counter(first_dim_counts)
            most_common_n, freq = counts.most_common(1)[0]
            if freq > 1:
                print(f"\nInferred Number of Samples (Frames/Instances): {most_common_n}")
                print(f"(Based on {freq} fields having this as their first dimension)")

        print("-" * 80)
        # Check specific known 4D/3D fields for Hamer checks
        if 'hand_pose_pca15' in data:
            print(f"\n[Check] hand_pose_pca15 is present.")
            print(f"        Shape: {data['hand_pose_pca15'].shape} (Expected: (N, 15))")
        
        if 'theta' in data:
             print(f"\n[Check] theta is present.")
             print(f"        Shape: {data['theta'].shape} (Expected: (N, 48))")

    except Exception as e:
        print(f"\nError analyzing file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_npz.py <path_to_npz_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    analyze_npz(file_path)
