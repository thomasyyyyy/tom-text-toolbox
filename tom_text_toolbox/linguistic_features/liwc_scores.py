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
    license_server_path = r"C:\Program Files\LIWC-22\LIWC-22-license-server\LIWC-22-license-server.exe"
    if is_license_server_running():
        print("LIWC license server is already running.")
        return
    try:
        subprocess.Popen(license_server_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("LIWC license server launched successfully.")
    except Exception as e:
        print(f"Failed to launch license server: {e}. Double-check the path.")

def classify_liwc(file: str, column: str, dependent: bool = False, merge_back: bool = False,
                  concise: bool = False, custom_dictionary: str = None):
    """
    Run LIWC twice — default + optional custom dictionary — and merge results into the original DataFrame.
    Outputs are created next to the input file unless a custom path is provided.
    """
    print("Running LIWC analysis...")
    start_liwc_license_server()

    # Keep file path as given, compute input directory
    input_path = file
    input_dir = os.path.dirname(input_path) or "."
    
    # Output file names
    output_file_default = "liwc_captions.csv" if dependent else "total_linguistic_analysis.csv"
    output_file_custom = "liwc_captions_custom.csv" if dependent else "total_linguistic_analysis_custom.csv"
    merged_output_file = "liwc_combined.csv"

    output_path_default = os.path.join(input_dir, output_file_default)
    output_path_custom = os.path.join(input_dir, output_file_custom)
    merged_output_path = os.path.join(input_dir, merged_output_file)

    # Convert column name to index (1-based for LIWC CLI)
    df = pd.read_csv(input_path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in file.")
    col_index = df.columns.get_loc(column)
    column_index = str(col_index + 1)

    original_df = df.copy()

    # --- First Run: Default Dictionary ---
    try:
        cmd_default = [
            "LIWC-22-cli.exe",
            "--mode", "wc",
            "--input", input_path,
            "--column-indices", column_index,
            "--output", output_path_default
        ]
        print("\nRunning LIWC with default dictionary...")
        subprocess.run(cmd_default, check=True, cwd=input_dir)
        print("Default LIWC analysis completed successfully.")

        liwc_df_default = pd.read_csv(output_path_default)
        if concise:
            columns_to_keep = ["achieve", "Affect", "affiliation", "Analytic",
                               "article", "Authentic", "cogproc", "emo_anx", "emo_sad", "exclusive",
                               "filler", "function", "home", "motion", "Drives",
                               'reward', 'risk', 'curiosity', "negate", "number", "Perception",
                               "we", "i", "you", "relig", "emo_pos", "emo_neg", "tone_pos",
                               "tone_neg", "Social", "space", "swear", "time", "work"]
            liwc_df_default = liwc_df_default[[col for col in columns_to_keep if col in liwc_df_default.columns]]

    except subprocess.CalledProcessError as e:
        print(f"Error running LIWC default analysis: {e}")
        return

    # --- Second Run: Custom Dictionary ---
    liwc_df_custom = None
    if custom_dictionary:
        if not os.path.exists(custom_dictionary):
            raise FileNotFoundError(f"Custom dictionary file not found: {custom_dictionary}")

        try:
            cmd_custom = [
                "LIWC-22-cli.exe",
                "--mode", "wc",
                "--input", input_path,
                "--column-indices", column_index,
                "--dictionary", custom_dictionary,
                "--output", output_path_custom
            ]
            print("\nRunning LIWC with custom dictionary...")
            subprocess.run(cmd_custom, check=True, cwd=input_dir)
            print("Custom LIWC analysis completed successfully.")

            liwc_df_custom = pd.read_csv(output_path_custom)
            liwc_df_custom = liwc_df_custom.add_suffix("_custom")

        except subprocess.CalledProcessError as e:
            print(f"Error running LIWC custom analysis: {e}")

    # --- Merge Back Results ---
    if merge_back:
        print("\nMerging LIWC outputs back into original file...")
        merged_df = pd.concat([original_df.reset_index(drop=True),
                               liwc_df_default.reset_index(drop=True)], axis=1)
        if liwc_df_custom is not None:
            merged_df = pd.concat([merged_df.reset_index(drop=True),
                                   liwc_df_custom.reset_index(drop=True)], axis=1)

        merged_df.to_csv(merged_output_path, index=False)
        print(f"Combined LIWC output written to: {merged_output_path}")

        print(list(merged_df.columns))

# Example usage
if __name__ == "__main__":
    file = "text_data_TEST.csv"  # relative path or full path
    column = "caption"
    custom_dict_path = r"tom_text_toolbox\dictionaries\regulatory-mode-dictionary.dicx"
    classify_liwc(file, column, dependent=True, merge_back=True, concise=True, custom_dictionary=custom_dict_path)
