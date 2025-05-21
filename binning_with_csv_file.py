import pandas as pd

# Load the radar data from CSV
df = pd.read_csv("csv_data/side_radars_corrected_transformation.csv")

# Get the starting timestamp to use as a reference point
t0 = df["timestamp"].min()

# Create frame_id by grouping timestamps into 1ms bins
# This allows grouping radar points that arrived at similar times
df["frame_id"] = ((df["timestamp"] - t0) // 1).astype(int)  # 1ms bins

# Save the new DataFrame with frame_id assignments
df.to_csv("csv_data/side_radars_binned_051ms.csv", index=False)

print("Done! Frame IDs assigned with 1ms binning.")
