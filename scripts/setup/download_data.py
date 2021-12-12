"""
Downloads the data required for the project to a local directory.
Code adapted from: https://github.com/github/CodeSearchNet/blob/master/script/download_dataset.py 
under the MIT license
"""

import argparse
import os
from subprocess import call
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Download Data", description="Downloads data for the project")
    parser.add_argument("-d", "--destination-dir", type=Path, help="Where to download the data to")
    parser.add_argument("-l", "--languages", choices=('python', 'javascript', 'java', 'ruby', 'php', 'go'), help="What languages to download data for", nargs="+")
    args = parser.parse_args()

    destination_dir = args.destination_dir
    destination_dir.mkdir(exist_ok=True)
    os.chdir(destination_dir)

    for language in args.languages:
        call(['wget', 'https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{}.zip'.format(language), '-P', destination_dir, '-O', '{}.zip'.format(language)])
        call(['unzip', '{}.zip'.format(language)])
        call(['rm', '{}.zip'.format(language)])
