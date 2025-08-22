import pandas as pd
import os
import subprocess
import argparse
import shutil


def patch_file(file_path, replacements):
    """Patch a file in place by replacing given text patterns."""
    with open(file_path, "r") as f:
        content = f.read()
    for search, replace in replacements:
        content = content.replace(search, replace)
    with open(file_path, "w") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description="Run Speciteller on a CSV file and merge predictions.")
    parser.add_argument("--csv", required=True, help="Path to input CSV file")
    parser.add_argument("--column", required=True, help="Name of text column in CSV")
    parser.add_argument("--repo", required=True, help="Path to Domain-Agnostic-Sentence-Specificity-Prediction repo")
    parser.add_argument("--env", default="speciteller", help="Conda environment name (default: speciteller)")
    parser.add_argument("--output", default="output_with_specificity.csv", help="Output CSV file")
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    repo_path = os.path.abspath(args.repo)
    tweets_path = os.path.join(repo_path, "my_tweets.txt")

    # Export text column to txt
    df = pd.read_csv(csv_path)
    if args.column not in df.columns:
        raise ValueError(f"Column '{args.column}' not found in CSV.")
    df[args.column].dropna().to_csv(tweets_path, index=False, header=False)
    print(f"[INFO] Exported {len(df)} rows to {tweets_path}")

    # File paths in repo
    data2_path = os.path.join(repo_path, "data2.py")
    test_path = os.path.join(repo_path, "test.py")

    # Backup originals
    data2_backup = data2_path + ".bak"
    test_backup = test_path + ".bak"
    shutil.copyfile(data2_path, data2_backup)
    shutil.copyfile(test_path, test_backup)
    print("[INFO] Backed up original files")

    try:
        # Patch files
        patch_file(data2_path, [
            ("s1['test']['path'] = os.path.join(data_path, 'twitters.txt')",
             f"s1['test']['path'] = '{tweets_path}'"),
            ("s1['unlab']['path'] ='dataset/data/twitteru.txt'",
             f"s1['unlab']['path'] = '{tweets_path}'")
        ])

        patch_file(test_path, [
            ("_, xst = getFeatures(os.path.join(params.nlipath,'twitters.txt'))",
             f"_, xst = getFeatures('{tweets_path}')"),
            ("_, xsu = getFeatures('dataset/data/twitteru.txt')",
             f"_, xsu = getFeatures('{tweets_path}')")
        ])
        print("[INFO] Patched data2.py and test.py")

        # Run model
        subprocess.run(
            ["conda", "run", "-n", args.env, "python", "test.py", "--gpu_id", "0", "--test_data", "twitter"],
            cwd=repo_path,
            check=True
        )

        # Merge predictions
        pred_file = os.path.join(repo_path, "pred.txt")
        if not os.path.exists(pred_file):
            raise FileNotFoundError("Prediction file 'pred.txt' not found. Check model run.")

        preds = pd.read_csv(pred_file, header=None, names=["specificity"])
        df["specificity"] = preds["specificity"]
        df.to_csv(args.output, index=False)
        print(f"[INFO] Predictions saved to {args.output}")

    finally:
        # Restore originals
        shutil.move(data2_backup, data2_path)
        shutil.move(test_backup, test_path)
        print("[INFO] Restored original data2.py and test.py")


if __name__ == "__main__":
    main()
