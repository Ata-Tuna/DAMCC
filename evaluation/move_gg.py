import os
import shutil
import argparse

def read_file_content(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return None
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def copy_and_rename(src, dest):
    if not src:
        # base_dir = "sota_paper_implementation/damnets-tagGen-dymonds/"
        base_dir = ""
        last_test_path = "sota_paper_implementation/damnets-tagGen-dymonds/experiment_files/last_test.txt"
        last_train_path = "sota_paper_implementation/damnets-tagGen-dymonds/experiment_files/last_train.txt"

        new_name = read_file_content(last_test_path).replace("experiment_files/", "").strip()
        new_name2 = read_file_content(last_train_path).replace("experiment_files/", "").strip()
        exp_name = read_file_content(last_test_path)
        train_name = read_file_content(last_train_path)
        src = os.path.join("sota_paper_implementation/damnets-tagGen-dymonds/", exp_name)
        print(f"src is {src}")
        src_train = os.path.join(base_dir, last_train_path)


    dest_path = os.path.join(dest, new_name)
    dest_path2 = os.path.join(dest_path,new_name2)
    
    print(f"source is {src}")
    if os.path.isdir(src):
        shutil.copytree(src, dest_path, dirs_exist_ok=True)
        print("hey", dest_path)
    else:
        print(f"dest {dest}")
        if not os.path.exists(dest):
            os.makedirs(dest)
        shutil.copy2(src, dest_path)

    # print(dest_path2)
    print("hoy", src_train)
    if os.path.isdir(src_train):
        shutil.copytree(src_train, dest_path2, dirs_exist_ok=True)
    else:
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        shutil.copy2(src_train, dest_path)
    
    # print(f"Copied {src} to {dest_path}")
    # print(f"Copied {src_train} to {dest_path2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy and rename a file or folder.")
    
    parser.add_argument("-s", "--src", type=str, default=None, help="Source file or directory (default from last_test.txt).")
    parser.add_argument("-d", "--dest", type=str, default="generated_graphs", help="Destination directory (default: generated_graphs).")
    # parser.add_argument("-n", "--new_name", type=str, default="damnets-encovid", help="New name for the copied file or folder (default: 'damnets-encovid').")
    
    args = parser.parse_args()
    
    copy_and_rename(args.src, args.dest)
