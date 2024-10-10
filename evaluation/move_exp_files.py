import os
import shutil

def copy_directory(src, dest):
    # Ensure the destination directory exists
    if not os.path.exists(dest):
        os.makedirs(dest)

    # Copy the contents of the source directory to the destination directory
    for item in os.listdir(src):
        src_item = os.path.join(src, item)
        dest_item = os.path.join(dest, item)

        if os.path.isdir(src_item):
            shutil.copytree(src_item, dest_item, dirs_exist_ok=True)
        else:
            shutil.copy2(src_item, dest_item)

    print(f"Copied contents from {src} to {dest}")

def clean_file_content(file_path):
    """Read the file content, remove 'experiment_files/', and write it back."""
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        cleaned_content = content.replace("experiment_files/", "")
        with open(file_path, 'w') as file:
            file.write(cleaned_content)
        print(f"Processed {file_path}")
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except IOError as e:
        print(f"Error reading/writing file {file_path}: {e}")

def copy_sampled_ts(source_folder, dest_file):
    """Find and copy 'sampled_ts.pkl' or 'age_samples.pkl' from source_folder to dest_file."""
    possible_files = ["sampled_ts.pkl", "age_samples.pkl"]
    for file_name in possible_files:
        src_file = os.path.join(source_folder, file_name)
        if os.path.exists(src_file):
            shutil.copy(src_file, dest_file)
            print(f"Copied {src_file} to {dest_file}")
            return
    print(f"Error: None of the files {possible_files} found in {source_folder}.")

def process_files():
    # List of files to clean
    files_to_clean = [
        "generated_graphs/last_test.txt",
        "generated_graphs/last_train.txt",
        "generated_graphs/last_age_test.txt",
        "generated_graphs/last_age_train.txt"
    ]

    # Clean the content of specified files
    for file_name in files_to_clean:
        file_path = os.path.join("/workdir", file_name)
        clean_file_content(file_path)
    
    # Folder names from last_test.txt and last_age_test.txt
    last_test_path = os.path.join("/workdir", "generated_graphs/last_test.txt")
    last_age_test_path = os.path.join("/workdir", "generated_graphs/last_age_test.txt")

    with open(last_test_path, 'r') as file:
        test_folder = file.read().strip()
    with open(last_age_test_path, 'r') as file:
        age_test_folder = file.read().strip()

    # Define paths for the copied files
    dest_age_pkl = os.path.join("/workdir/generated_graphs", "age-encovid.pkl")
    dest_damnet_pkl = os.path.join("/workdir/generated_graphs", "damnets-encovid.pkl")

    # Base directory
    base_dir = "/workdir/generated_graphs"

    # Copy the sampled_ts.pkl or age_samples.pkl files from the folders mentioned
    copy_sampled_ts(os.path.join(base_dir, test_folder), dest_damnet_pkl)
    copy_sampled_ts(os.path.join(base_dir, age_test_folder), dest_age_pkl)

if __name__ == "__main__":
    src = "/workdir/sota_paper_implementation/damnets-tagGen-dymonds/experiment_files"
    dest = "/workdir/generated_graphs"
    

    copy_directory(src, dest)
    process_files()
