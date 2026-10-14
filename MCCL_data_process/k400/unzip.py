import os
import re
import shutil
import tarfile

def merge_and_extract(source_dir, destination_dir, temp_dir):
    """
    Merge and extract .tar.gz.partX-Y files from the source directory to the destination directory.

    Args:
        source_dir (str): The directory containing .tar.gz.part files.
        destination_dir (str): The directory to extract files into.
        temp_dir (str): Temporary directory for merging files.
    """
    # Ensure destination and temporary directories exist
    os.makedirs(destination_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Regular expression to match .tar.gz.partX-Y files
    part_file_pattern = re.compile(r'(.*\.tar\.gz)\.part\d+-\d+$')
    
    # Find all part files grouped by their base filename
    part_files = {}
    for root, _, files in os.walk(source_dir):
        for file in files:
            match = part_file_pattern.match(file)
            if match:
                base_name = match.group(1)
                if base_name not in part_files:
                    part_files[base_name] = []
                part_files[base_name].append(os.path.join(root, file))
    
    # Process each set of part files
    for base_name, parts in part_files.items():
        parts.sort()  # Sort parts to ensure correct order
        merged_file_path = os.path.join(temp_dir, os.path.basename(base_name))
        print(f"Merging parts for: {base_name}")
        
        # Merge parts
        with open(merged_file_path, 'wb') as merged_file:
            for part in parts:
                with open(part, 'rb') as part_file:
                    shutil.copyfileobj(part_file, merged_file)
        
        # Extract merged file
        print(f"Extracting: {merged_file_path}")
        try:
            with tarfile.open(merged_file_path, 'r:gz') as tar:
                tar.extractall(path=destination_dir)
                print(f"Extracted: {merged_file_path} to {destination_dir}")
        except tarfile.ReadError as e:
            print(f"Failed to extract {merged_file_path}: {e}")
        finally:
            # Optionally, remove the merged file
            os.remove(merged_file_path)

# Example usage
source_directory = "/mnt/newdisk/mnt/newdisk/dataset/action/k400/kinetics-400/"  # Replace with the source directory
destination_directory = "/mnt/newdisk/mnt/newdisk/dataset/action/k400/video/"  # Replace with the destination directory
temporary_directory = "/mnt/newdisk/mnt/newdisk/dataset/action/k400/temp/"  # Replace with a temporary directory for merging

merge_and_extract(source_directory, destination_directory, temporary_directory)

