import os
import json
import csv
from pathlib import Path

def load_class_mapping(txt_file):
    """Load class mapping from the txt file."""
    with open(txt_file, 'r') as file:
        return {str(idx): line.strip() for idx, line in enumerate(file)}

def load_json(json_file):
    """Load JSON data from a file."""
    with open(json_file, 'r') as file:
        return json.load(file)

def sanitize(class_name):
    return class_name.replace(' ','_')

def generate_csv(json_data, class_mapping, base_path, output_csv):
    """Generate CSV file from JSON data and class mapping."""
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(["original_vido_id video_id frame_id path labels"])

        video_id =0

        for item in json_data:
            original_video_id = item["id"]
            label = item["label"]

            class_name = class_mapping.get(label, "Unknown")
            class_name = sanitize(class_name)
            video_path = Path(base_path) / class_name / original_video_id
            # vid_path = class_name / original_video_id

            if not video_path.exists():
                print(f"Warning: Path {video_path} does not exist.")
                continue
            
            # Simulating frame paths (replace with actual frame paths if available)
            frame_files = sorted(video_path.glob("*.jpg"))

            for frame_id,frame_file in enumerate(frame_files):  # Example: two frames per video
                rela_path = f"{class_name}/{original_video_id}/{frame_file.name}"
                row_data = f"{original_video_id} {video_id} {frame_id} {rela_path} \"\""
                # csv_writer.writerow([original_video_id, label, frame_id, str(frame_file), ""])
                if frame_id %2 ==0:
                    csv_writer.writerow([row_data])
            video_id+=1

def main(json_file, txt_file, base_path, output_csv):
    """Main function to generate the CSV file."""
    # Load class mapping
    class_mapping = load_class_mapping(txt_file)

    # Load JSON data
    json_data = load_json(json_file)

    # Generate CSV
    generate_csv(json_data, class_mapping, base_path, output_csv)

# Example usage
if __name__ == "__main__":
    json_file = "/mnt/newdisk/mnt/newdisk/dataset/action/k400/label/timesformer_v15_for/label_validation.json"       # Replace with the path to your JSON file
    txt_file = "/mnt/newdisk/mnt/newdisk/dataset/action/k400/label/initial/kinetics_label_map.txt"       # Replace with the path to your class mapping file
    base_path = "/mnt/newdisk/mnt/newdisk/dataset/action/k400/video/v30/frame/train_256/"  # Replace with the base path to your video folders
    output_csv = "/mnt/newdisk/mnt/newdisk/dataset/action/k400/label/timesformer_v15_for/label_validation.csv"      # Replace with the desired output CSV file

    main(json_file, txt_file, base_path, output_csv)