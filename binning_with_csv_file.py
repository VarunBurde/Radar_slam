import pandas as pd

# Load your radar data
df = pd.read_csv("side_radars_corrected_transformation.csv")  # Replace with your actual filename

# Use the same logic as the 20ms binning, but with 5ms bins
t0 = df["timestamp"].min()  # starting timestamp
df["frame_id"] = ((df["timestamp"] - t0) // 1).astype(int)  # 5ms = 0.005 seconds

# Save the new DataFrame
df.to_csv("side_radars_binned_051ms.csv", index=False)

print("Done! Frame IDs assigned every 5ms like the 20ms version.")
