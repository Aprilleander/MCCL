import os
import random
import shutil

def select_and_copy_subfolders(source_dir, target_dir, selection_probability=0.05):
    """
    Randomly select subfolders from source_dir with a given probability
    and copy them to target_dir.

    Args:
        source_dir (str): Path to the source directory.
        target_dir (str): Path to the target directory.
        selection_probability (float): Probability of selecting a subfolder (default is 0.3).
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # List all subfolders in the source directory
    subfolders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
    
    for subfolder in subfolders:
        if random.random() < selection_probability:
            source_path = os.path.join(source_dir, subfolder)
            target_path = os.path.join(target_dir, subfolder)
            
            # Copy the subfolder to the target directory
            shutil.copytree(source_path, target_path)
            print(f"Copied: {subfolder}")

if __name__ == "__main__":
    source_directory = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/ssv2_frame_all/'
    target_directory = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/ssv2_frame_v5/'
    
    select_and_copy_subfolders(source_directory, target_directory)