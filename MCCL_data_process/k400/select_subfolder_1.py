import os
import random
import shutil

def select_and_copy_subfolders(source_dir, target_dir, selection_probability=0.3):
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
        sub_dir = os.path.join(source_dir,subfolder)
        target_sub_dir = os.path.join(target_dir,subfolder)
        source_sub_dir = os.path.join(source_dir,subfolder)
        if not os.path.exists(target_sub_dir):
            os.makedirs(target_sub_dir)
        sub = [f for f in os.listdir(sub_dir) if os.path.isdir(sub_dir)]
        for video_name in sub:
            if random.random() < selection_probability and video_name.endswith('.mp4'):
                sub = os.path.join(subfolder,video_name)
                source_path = os.path.join(source_dir, sub)
                target_path = os.path.join(target_dir, sub)
                
                # Copy the subfolder to the target directory

                shutil.copyfile(source_path, target_path)
                print(f"Copied: {subfolder}/{video_name}")

if __name__ == "__main__":
    source_directory = '/mnt/newdisk/mnt/newdisk/dataset/action/k400/video/compress/val_256/'
    target_directory = '/mnt/newdisk/mnt/newdisk/dataset/action/k400/video/v30/val_256/'
    
    select_and_copy_subfolders(source_directory, target_directory)