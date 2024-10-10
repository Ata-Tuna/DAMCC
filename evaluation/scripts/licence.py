
import os

def add_header_to_python_files(directory, header_text):
    # Iterate over all files in the specified directory
    for filename in os.listdir(directory):
        # Check if the file has a .py extension
        if filename.endswith('.py'):
            filepath = os.path.join(directory, filename)
            # Read the existing content of the file
            with open(filepath, 'r') as file:
                existing_content = file.read()

            # Write the header and existing content back to the file
            with open(filepath, 'w') as file:
                file.write(header_text + '\n\n' + existing_content)

if __name__ == '__main__':
    # Specify the directory containing your .py files
    directory_path = "/home/houssem.souid/brainiac-1-temporal/experiments"
    # Specify the header text you want to add
    header_text = """
### **
###  * Copyright (C) Euranova - All Rights Reserved 2024
###  * 
###  * This source code is protected under international copyright law.  All rights
###  * reserved and protected by the copyright holders.
###  * This file is confidential and only available to authorized individuals with the
###  * permission of the copyright holders.  If you encounter this file and do not have
###  * permission, please contact the copyright holders and delete this file at 
###  * research@euranova.eu
###  * It may only be used for academic research purposes as authorized by a written
###  * agreement issued by Euranova. 
###  * Any commercial use, redistribution, or modification of this software without 
###  * explicit written permission from Euranova is strictly prohibited. 
###  * By using this software, you agree to abide by the terms of this license. 
###  **"""

    add_header_to_python_files(directory_path, header_text)