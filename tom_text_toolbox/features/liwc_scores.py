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

def classify_liwc(file: str, column: str, dependent: bool = False, merge_back: bool = False, concise: bool = False, custom_dictionary: str = None):
    """Run LIWC twice — default + optional custom dictionary — and merge both results into the original DataFrame."""
    print("Running LIWC analysis...")
    start_liwc_license_server()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, file)

    # Output file names
    output_file_default = "liwc_captions.csv" if dependent else "total_linguistic_analysis.csv"
    output_path_default = os.path.join(base_dir, output_file_default)

    output_file_custom = "liwc_captions_custom.csv" if dependent else "total_linguistic_analysis_custom.csv"
    output_path_custom = os.path.join(base_dir, output_file_custom)

    # Convert column name to index (1-based for LIWC CLI)
    if column and isinstance(column, str):
        df = pd.read_csv(input_path)
        if column in df.columns:
            col_index = df.columns.get_loc(column)
            column = str(int(col_index) + 1)
        else:
            raise ValueError(f"Column '{column}' not found in file.")

    # Load original data once for merging later
    original_df = pd.read_csv(input_path)

    # --- First Run: Default Dictionary ---
    try:
        cmd_default = [
            "LIWC-22-cli.exe",
            "--mode", "wc",
            "--input", input_path,
            "--column-indices", column,
            "--output", output_path_default
        ]
        print("\nRunning LIWC with default dictionary...")
        subprocess.run(cmd_default, check=True)
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
        try:
            if not os.path.exists(custom_dictionary):
                raise FileNotFoundError(f"Custom dictionary file not found: {custom_dictionary}")

            cmd_custom = [
                "LIWC-22-cli.exe",
                "--mode", "wc",
                "--input", input_path,
                "--column-indices", column,
                "--dictionary", custom_dictionary,
                "--output", output_path_custom
            ]
            print("\nRunning LIWC with custom dictionary...")
            subprocess.run(cmd_custom, check=True)
            print("Custom LIWC analysis completed successfully.")

            liwc_df_custom = pd.read_csv(output_path_custom)

            # Rename columns from custom run to avoid collisions
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

        merged_output_path = os.path.join(base_dir, "liwc_combined.csv")
        merged_df.to_csv(merged_output_path, index=False)
        print(f"Combined LIWC output written to: {merged_output_path}")

if __name__ == "__main__":
    file = "text_data_TEST.csv"
    column = "caption"
    custom_dict_path = r"tom_text_toolbox\dictionaries\regulatory-mode-dictionary.dicx"
    classify_liwc(file, column, dependent=True, merge_back=True, concise=True, custom_dictionary=custom_dict_path)
