import csv
import os

class DatasetManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self._init_csv()

    def _init_csv(self):
        """Initializes the CSV with features first and the label at the very end."""
        if not os.path.exists(self.file_path):
            header = []
            
            # 1. Left Hand Features (21 joints * 3 coordinates = 63 columns)
            for i in range(21):
                header.extend([f"L{i}_x", f"L{i}_y", f"L{i}_z"])
                
            # 2. Right Hand Features (21 joints * 3 coordinates = 63 columns)
            for i in range(21):
                header.extend([f"R{i}_x", f"R{i}_y", f"R{i}_z"])
            
            # 3. Label placed at the absolute END
            header.append("label")

            with open(self.file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)

    def save(self, label, features):
        """Appends a new row with features first, followed by the label."""
        # Row structure: [126 numbers] + ["LABEL"]
        row = list(features) + [label]
        with open(self.file_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def count_label(self, label):
        """Counts how many samples exist for a specific label."""
        if not os.path.exists(self.file_path):
            return 0
        
        count = 0
        with open(self.file_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header row
            for row in reader:
                if row and row[-1] == label:  # Check the last column
                    count += 1
        return count