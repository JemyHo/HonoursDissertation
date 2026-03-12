import os
import sys
import subprocess

DCMVC_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    cmd = [sys.executable, "train.py"] + sys.argv[1:]
    subprocess.check_call(cmd, cwd=DCMVC_DIR)

if __name__ == "__main__":
    main()