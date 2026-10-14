import os
import csv
import json


def load_json(json_path):
    """Load the JSON file and return a list of IDs (in order)."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return [str(item['id']) for item in data]  # Keep IDs as strings for matching folder names


def generate_csv(json_path, folder_path, output_csv):
    ordered_ids = load_json(json_path)

    # Build a mapping from video ID to its folder name
    folder_map = {}
    for subfolder_name in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, subfolder_name)):
            original_video_id = subfolder_name.split('_')[0]
            folder_map[original_video_id] = subfolder_name

    # Initialize video_id counter
    video_id_counter = 0

    # Create the CSV file with the required headers
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["original_vido_id video_id frame_id path labels"])

        # Process videos in the same order as in JSON
        for original_video_id in ordered_ids:
            if original_video_id in folder_map:
                subfolder_name = folder_map[original_video_id]
                subfolder_path = os.path.join(folder_path, subfolder_name)

                images = sorted(os.listdir(subfolder_path))
                for frame_id, image_name in enumerate(images):
                    if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        image_path = os.path.join(subfolder_name, image_name)
                        combined_info = f'{original_video_id} {video_id_counter} {frame_id} {image_path} ""'
                        if frame_id % 2 == 0:
                            writer.writerow([combined_info])

                video_id_counter += 1

    print("Finished writing CSV in JSON order.")


# === Usage ===
path = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/ssv2_frame_v30/'
out_csv = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v30/even_sort/label_validation_v30.csv'
json_path = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v30/even_sort/something-something-v2-label_validation_v30.json'

generate_csv(json_path, path, out_csv)