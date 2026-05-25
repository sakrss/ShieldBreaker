import os
import gzip
import shutil


def unzip_all_gz(src_dir=".", out_dir="unzip"):
    """
    Unzip all .gz files in a directory.

    Args:
        src_dir (str): Source directory containing .gz files.
        out_dir (str): Output directory to store uncompressed files.
    """
    # Create output directory if it does not exist
    os.makedirs(out_dir, exist_ok=True)

    for filename in os.listdir(src_dir):
        # Only process .gz files
        if not filename.endswith(".gz"):
            continue

        gz_path = os.path.join(src_dir, filename)
        out_name = filename[:-3]          # Remove ".gz" suffix
        out_path = os.path.join(out_dir, out_name)

        print(f"[+] Unzipping: {filename} -> {out_path}")

        try:
            # Open gzip file in binary read mode
            with gzip.open(gz_path, "rb") as f_in:
                # Write the decompressed content to output file
                with open(out_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        except Exception as e:
            print(f"[!] Failed to unzip {filename}: {e}")


if __name__ == "__main__":
    # Default behavior:
    #   - Read all .gz files in the current directory
    #   - Output decompressed files into ./unzip/
    unzip_all_gz()
