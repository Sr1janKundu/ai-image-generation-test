import os, json

import config

def create_json_map(root_dir):
    all_files = []

    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        for file in os.listdir(folder_path):
            all_files.append(os.path.join(folder_path, file))

    data = {}
    for filename in all_files:
        try:
            with open(filename, 'r') as file:
                content = file.read()
                sentences = [line.strip() for line in content.strip().splitlines() if line.strip()]
                data[os.path.basename(filename).split('.')[0]] = sentences
        except FileNotFoundError:
            print(f"File not found: {filename}")
        except Exception as e:
            print(f"Error reading file {filename}: {e}")

    output_file = config.birds_caps_file
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f"JSON file created: {output_file}")

if __name__ == '__main__':
    create_json_map(config.birds_caps_dir)