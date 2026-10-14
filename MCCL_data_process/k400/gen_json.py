import os
import json
import random
from pathlib import Path

def load_classes(txt_file):
    """Load class names from the txt file."""
    with open(txt_file, 'r') as file:
        return {line.strip(): idx for idx, line in enumerate(file)}

def sanitize(class_name):
    return class_name.replace(' ','_')

def classify_folders(base_path, class_dict):
    """Classify folders into label_train, label_val, and unlabel."""
    label_train, label_val, unlabel = [], [], []

    for class_name, class_id in class_dict.items():
        class_name = sanitize(class_name)
        class_path = Path(base_path) / class_name
        if not class_path.exists() or not class_path.is_dir():
            print(class_name)
            continue
            

        subfolders = [subfolder.name for subfolder in class_path.iterdir() if subfolder.is_dir()]
        if class_id % 2 == 0:  # Even classes
            random.shuffle(subfolders)
            split_index = (len(subfolders) // 2) +1
            labeled_subfolders = subfolders[:split_index]
            unlabeled_subfolders = subfolders[split_index:]

            # Further split labeled_subfolders into train and val
            random.shuffle(labeled_subfolders)
            train_split_idx = int(len(labeled_subfolders) * 0.9)
            label_train += [{"id": folder, "label": str(class_id)} for folder in labeled_subfolders[:train_split_idx]]
            label_val += [{"id": folder, "label": str(class_id)} for folder in labeled_subfolders[train_split_idx:]]

            # Add remaining to unlabel
            unlabel += [{"id": folder, "label": str(class_id)} for folder in unlabeled_subfolders]
        else:  # Odd classes
            unlabel += [{"id": folder, "label": str(class_id)} for folder in subfolders]

    return label_train, label_val, unlabel

def save_json(data, output_file):
    """Save data to a JSON file."""
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)

def main(base_path, txt_file, output_dir):
    """Main function to generate the JSON files."""
    # Load classes
    class_dict = load_classes(txt_file)

    # Classify folders
    label_train, label_val, unlabel = classify_folders(base_path, class_dict)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON files
    save_json(label_train, os.path.join(output_dir, 'label_train.json'))
    save_json(label_val, os.path.join(output_dir, 'label_val.json'))
    save_json(unlabel, os.path.join(output_dir, 'unlabel.json'))

# Example usage
if __name__ == "__main__":
    base_path = "/mnt/newdisk/mnt/newdisk/dataset/action/k400/video/v30/frame/train_256/"  # Replace with the path to your folders
    txt_file = "/mnt/newdisk/mnt/newdisk/dataset/action/k400/label/initial/kinetics_label_map.txt"       # Replace with your class txt file
    output_dir = "/mnt/newdisk/mnt/newdisk/dataset/action/k400/label/timesformer/"     # Replace with your desired output directory

    main(base_path, txt_file, output_dir)