import os
import glob
import subprocess
from pathlib import Path


# Define the source and destination directories
def convert_pdf_to_png(source_dir):
    print(f"Converting {source_dir} pdf to png...")
    destination_dir_post = source_dir / "summary_figures_posterior"
    destination_dir_param = source_dir / "summary_figures_parameter"

    # Create the destination directories if they don't exist
    if not os.path.exists(destination_dir_post):
        os.makedirs(destination_dir_post)
    if not os.path.exists(destination_dir_param):
        os.makedirs(destination_dir_param)

    # Find all directories matching the pattern 'learn_posterior_*'
    pattern = os.path.join(source_dir, "learn_posterior_*")
    directories = glob.glob(pattern)

    # Process each directory
    for dir_path in directories:
        if os.path.isdir(dir_path):
            folder_name = os.path.basename(dir_path)
            pdf_path = os.path.join(dir_path, "posterior_predictive_check.pdf")

            # Check if the PDF file exists
            if os.path.exists(pdf_path):
                # Define the new PNG file name
                new_png_name = f"{folder_name}_ppc.png"
                new_png_path = os.path.join(destination_dir_post, new_png_name)

                # Skip conversion if PNG already exists
                if not os.path.exists(new_png_path):
                    # Convert PDF to PNG
                    subprocess.run(["convert", "-density", "300", pdf_path, "-quality", "99", new_png_path])

    # Process each directory
    for dir_path in directories:
        if os.path.isdir(dir_path):
            folder_name = os.path.basename(dir_path)
            pdf_path = os.path.join(dir_path, "recruitment_curves.pdf")

            # Check if the PDF file exists
            if os.path.exists(pdf_path):
                # Define the new PNG file name
                new_png_name = f"{folder_name}_rc.png"
                new_png_path = os.path.join(destination_dir_param, new_png_name)

                # Skip conversion if PNG already exists
                if not os.path.exists(new_png_path):
                    # Convert PDF to PNG
                    subprocess.run(["convert", "-density", "300", pdf_path, "-quality", "99", new_png_path])

    print("Conversion completed.")


if __name__ == "__main__":
    source_dir = Path("/home/mcintosh/Local/temp/test_hbmep/hbmep_sim/build")

    # Automatically populate d with directories matching the pattern 'test*'
    pattern = os.path.join(source_dir, "test*")
    d = [Path(dir_path).name for dir_path in glob.glob(pattern) if os.path.isdir(dir_path)]

    for dir_name in d:
        convert_pdf_to_png(source_dir / dir_name)
