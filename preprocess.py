import os
import numpy as np
import pandas as pd


def load_raw_dataset(data_folder):
    """
    Loads the UCI Daily and Sports Activities dataset by reading all txt files,
    flattening time-series matrices, extracting labels and participant IDs.

    Parameters
    ----------
    data_folder : str
        Path to the extracted dataset root folder.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing flattened features + label + participant.
    """

    rows = []
    labels = []
    participants = []

    # Walk through all files in the directory
    for root, dirs, files in os.walk(data_folder):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)

                # Extract activity label (e.g., a01)
                activity_label = os.path.basename(root)

                # Extract participant (e.g., p1, p2 ...)
                participant = os.path.basename(os.path.dirname(root))

                try:
                    # Load multivariate time-series (matrix)
                    data = np.loadtxt(file_path)

                    # Flatten the matrix into a single long feature vector
                    flat_data = data.flatten()

                    rows.append(flat_data)
                    labels.append(activity_label)
                    participants.append(participant)

                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

    # Convert list → DataFrame
    X = np.vstack(rows)
    df = pd.DataFrame(X)

    # Add label + participant metadata
    df["label"] = labels
    df["participant"] = participants

    return df


def save_dataset(df, output_path="daily_sports_raw.csv"):
    """
    Saves the processed dataset to CSV format.
    """
    df.to_csv(output_path, index=False)
    print(f"Saved processed dataset to: {output_path}")


if __name__ == "__main__":
    # === Change this to your dataset folder path in Colab or local ===
    DATASET_PATH = "Daily and Sports Activities Dataset"

    print("Loading raw dataset...")

    df = load_raw_dataset(DATASET_PATH)

    print("Dataset loaded.")
    print("Shape:", df.shape)

    # Save output CSV
    save_dataset(df, "daily_sports_raw.csv")

    print("Processing complete.")

