import os
import subprocess
import sys

LINKS_FILE = os.path.join(os.path.dirname(__file__), "candor_links")
LOCAL_DIR = "/home/zonghuan/tudelft/projects/datasets"
REMOTE_HOST = "zli33@login.delftblue.tudelft.nl"
REMOTE_DIR = "/scratch/zli33/data/candor"
PASSWORD = ""  # <-- set your cluster password here


def main():
    if not PASSWORD:
        print("ERROR: Please set the PASSWORD variable in the script.")
        sys.exit(1)

    os.makedirs(LOCAL_DIR, exist_ok=True)

    with open(LINKS_FILE, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    total = len(urls)
    print(f"Found {total} links to download.\n")


    for idx, url in enumerate(urls, start=1):
        # Derive filename from the URL path (e.g. processed_media_part_001.zip)
        filename = url.split("?")[0].rsplit("/", 1)[-1]
        local_path = os.path.join(LOCAL_DIR, filename)

        # --- Download ---
        print(f"[{idx}/{total}] Downloading {filename} ...")
        dl_cmd = ["wget", "-q", "--show-progress", "-O", local_path, url]
        ret = subprocess.run(dl_cmd)
        if ret.returncode != 0:
            print(f"  ERROR: Download failed for {filename}, skipping.")
            continue
        print(f"  Download complete.")

        # --- Upload to remote cluster via sshpass + scp ---
        remote_path = f"{REMOTE_HOST}:{REMOTE_DIR}/{filename}"
        print(f"  Uploading to {remote_path} ...")
        scp_cmd = [
            "sshpass", "-p", PASSWORD,
            "scp", "-o", "StrictHostKeyChecking=no",
            local_path, remote_path,
        ]
        ret = subprocess.run(scp_cmd)
        if ret.returncode != 0:
            print(f"  ERROR: Upload failed for {filename}, keeping local copy.")
            continue
        print(f"  Upload complete.")

        # --- Delete local file (keep only the first one) ---
        if idx == 1:
            print(f"  Keeping local copy: {local_path}")
        else:
            os.remove(local_path)
            print(f"  Deleted local copy.")

        print()

    print("All done.")


if __name__ == "__main__":
    main()
