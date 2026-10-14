import os
import json

def filter_ids_with_existing_folders(input_json_path, target_directory, output_json_path):
    """
    Filters IDs from a JSON file and checks if corresponding folders exist in the target directory.
    If a folder exists, the ID's information is written to a new JSON file.

    Args:
        input_json_path (str): Path to the input JSON file.
        target_directory (str): Path to the directory containing folders.
        output_json_path (str): Path to save the filtered JSON file.
    """
    # Load the input JSON file
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Filter IDs where corresponding folders exist in the target directory
    filtered_data = [
        item for item in data
        if os.path.isdir(os.path.join(target_directory, item['id']+'_frame'))
    ]

    # Write the filtered data to the output JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)

    print(f"Filtered data written to {output_json_path}")

if __name__ == "__main__":
    input_json = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/22w/something-something-v2-train.json'
    target_dir = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/ssv2_frame_v30/'
    output_json = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v30_for/train_v30.json'

    filter_ids_with_existing_folders(input_json, target_dir, output_json)