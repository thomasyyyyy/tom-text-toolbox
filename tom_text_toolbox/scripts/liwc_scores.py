import pandas as pd
import subprocess
import os
import psutil

def is_license_server_running(process_name="LIWC-22-license-server.exe"):
    for proc in psutil.process_iter(attrs=['pid', 'name']):
        if proc.info['name'] == process_name:
            return True
    return False

def start_liwc_license_server():
    license_server_path = r"C:\Program Files\LIWC-22\LIWC-22-license-server\LIWC-22-license-server.exe" # Assumes default windows installation path

    if is_license_server_running():
        print("LIWC license server is already running.")
        return

    try:
        subprocess.Popen(license_server_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("LIWC license server launched successfully.")
    except Exception as e:
        print(f"Failed to launch license server: {e}. Double-check the path.")

def classify_liwc(file: str, column: str, dependent: bool = False, merge_back: bool = False, concise: bool = False):
    """ Run LIWC analysis on the specified file and column.
    Args:
        file (str): Path to the input file.
        column (str): Column name to analyze.
        dependent (bool): If True, output will be in a specific format.
        merge_back (bool): If True, merge LIWC output back into original DataFrame.
        concise (bool): If True, only keep a subset of LIWC columns.
    Returns:
        A DataFrame with LIWC analysis results or None if an error occurs.
    """
    print("Running LIWC analysis...")

    start_liwc_license_server()

    # Get the directory where this script resides
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Build absolute paths
    input_path = os.path.join(base_dir, file)
    output_file = "liwc_captions.csv" if dependent else "total_linguistic_analysis.csv"
    output_path = os.path.join(base_dir, output_file)

    if column is not None and isinstance(column, str):
        # Convert column name to index (1-based for LIWC)
        df = pd.read_csv(input_path)
        if column in df.columns:
            col_index = df.columns.get_loc(column)
            if isinstance(col_index, int):
                column = str(col_index + 1)
            else:
                raise ValueError(f"Column '{column}' is not unique or cannot be resolved to a single index.")

    try:
        cmd_to_execute = [
            "LIWC-22-cli.exe",
            "--mode", "wc",
            "--input", input_path,
            "--column-indices", column,
            "--output", output_path
        ]
        print("Running command:", cmd_to_execute)
        subprocess.run(cmd_to_execute, check=True)
        print("LIWC analysis completed successfully.")

        if merge_back:
            print("Merging LIWC output back into original file...")
            original_df = pd.read_csv(input_path)
            liwc_df = pd.read_csv(output_path)

            if concise:
                columns_to_keep = ["achieve", "Affect", "affiliation", "Analytic",
                                   "article", "Authentic", "cogproc", "emo_anx", "emo_sad",
                                   "filler", "function", "home", "motion", "Drives",
                                   'reward', 'risk', 'curiosity', "negate", "number", "Perception",
                                   "we", "i", "you", "relig", "emo_pos", "emo_neg", "tone_pos",
                                   "tone_neg", "Social", "space", "swear", "time", "work"]
                liwc_df = liwc_df[columns_to_keep]
                
            # Assume LIWC output has same row order; concat columns
            merged_df = pd.concat([original_df.reset_index(drop=True), liwc_df.reset_index(drop=True)], axis=1)

            # Save to output file
            merged_df.to_csv(output_path, index=False)
            print(f"Merged LIWC output written back to: {output_path}")

    except subprocess.CalledProcessError as e:
        print(f"Error running LIWC analysis: {e}")

if __name__ == "__main__":
    # Example usage
    file = "text_data_TEST.csv"
    column = "caption"
    classify_liwc(file, column, dependent=True, merge_back=True, concise = True)
