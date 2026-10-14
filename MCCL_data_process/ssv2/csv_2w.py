import os
import csv
import json


def load_json(json_path):
    """Load the JSON file and return a set of valid IDs."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return {item['id'] for item in data}
    
def generate_csv(json_path,folder_path, output_csv):

    valid_ids = load_json(json_path)

    # Initialize video_id counter
    video_id_counter = 0
    
    # Create the CSV file with the required headers
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        #writer.writerow(["original_video_id", "video_id", "frame_id", "path", "labels"])
        writer.writerow(["original_vido_id video_id frame_id path labels"])

        # Loop through each subfolder in the given folder
        for subfolder_name in sorted(os.listdir(folder_path)):
            subfolder_path = os.path.join(folder_path, subfolder_name)
            
            if os.path.isdir(subfolder_path):
                # Extract the original_video_id from the subfolder name
                original_video_id = subfolder_name.split('_')[0]

                if original_video_id in valid_ids:
                    # Get all images in the subfolder
                    images = sorted(os.listdir(subfolder_path))
                    
                    for frame_id, image_name in enumerate(images):
                        if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                            image_path = os.path.join(subfolder_name, image_name)
                            
                            # Write a row to the CSV file
                            combined_info = f'{original_video_id} {video_id_counter} {frame_id} {image_path} \"\"'
                            
                            # Write a single-column row to the CSV file
                            if frame_id %3 ==0:
                                writer.writerow([combined_info])
                            #writer.writerow([original_video_id, video_id_counter, frame_id, image_path, " "])
                    
                    # Increment video_id for the next subfolder
                    video_id_counter += 1


path = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/ssv2_frame_all/'
out_csv = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v100_cutv15/label_validation_v70.csv'
json_path = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v100_cutv15/something-something-v2-validation_v70.json'

generate_csv(json_path,path,out_csv)
print("finished")