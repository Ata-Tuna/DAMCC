import os
import shutil
import argparse


def read_file_content(file_path):
    """
    Reads the content of a file and returns it.
    
    Args:
    - file_path (str): The path to the file to read.
    
    Returns:
    - str: The content of the file.
    """
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


def copy_and_rename_file(src_dir, dest_dir, src_filename, new_filename):
    # Ensure the source file exists
    src_file_path = os.path.join(src_dir, src_filename)
    if src_file_path == "":
        dir1 = "sota_paper_implementation/damnets-tagGen-dymonds/" 
        dir2 = "data/"
        if src_filename == "":
            dir3 = "encovid_test_graphs.pkl"
        else:
            dir3 = src_filename
        src_file_path = dir1+dir2+dir3
        print(f"src_file_path is {src_file_path}.")
    
    if not os.path.isfile(src_file_path):
        raise FileNotFoundError(f"The file {src_file_path} does not exist.")

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Create the destination file path
    dest_file_path = os.path.join(dest_dir, new_filename)

    # Copy and rename the file
    shutil.copy2(src_file_path, dest_file_path)

    print(f"Copied and renamed {src_filename} to {new_filename} in {dest_dir}")

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Copy and rename a file.")

    # Adding optional arguments with short names and default values
    parser.add_argument(
        "-s", "--src_dir", 
        type=str, 
        default="", 
        help="Source directory of the file (default in under damnets/data/)."
    )
    parser.add_argument(
        "-d", "--dest_dir", 
        type=str, 
        default="generated_graphs_test", 
        help="Destination directory for the file (default: generated_graphs_test)."
    )
    parser.add_argument(
        "-f", "--src_filename", 
        type=str, 
        default="", 
        help="Name of the file to be copied (default encovid_test_graphs.pkl)."
    )
    parser.add_argument(
        "-n", "--new_filename", 
        type=str, 
        default="encovid-test-graphs.pkl", 
        help="New name for the copied file (default: 'encovid-test-graphs.pkl')."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    copy_and_rename_file(args.src_dir, args.dest_dir, args.src_filename, args.new_filename)
