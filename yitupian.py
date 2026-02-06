import os
import shutil

# Set main directory
train_dir = r"zhiwubindu\images\test"  # Your train directory path

# Iterate through all subdirectories in the train directory
for subdir in os.listdir(train_dir):
    subdir_path = os.path.join(train_dir, subdir)

    # Ensure it is a directory
    if os.path.isdir(subdir_path):
        for filename in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, filename)

            # Check if it is an image file (adjust as needed)
            if os.path.isfile(file_path) and filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                new_path = os.path.join(train_dir, filename)  # Target path

                # If filename duplicates, rename to avoid overwriting
                counter = 1
                while os.path.exists(new_path):
                    name, ext = os.path.splitext(filename)
                    new_path = os.path.join(train_dir, f"{name}_{counter}{ext}")
                    counter += 1

                shutil.move(file_path, new_path)  # Move file

        # Remove empty subdirectory
        os.rmdir(subdir_path)

print("All images have been moved to the train directory, and empty subdirectories have been deleted.")
