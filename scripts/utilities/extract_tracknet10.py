"""
Extract the TrackNet10 spectrum from a FLUKA/USRTRACK listing file.

Usage:
    python scripts/utilities/extract_tracknet10.py [--lis-file PATH] [--output PATH]
"""

import re
import argparse
import numpy as np

from shielding_ml.pipelines.paths import REPO_ROOT, SPECTRA_DIR

parser = argparse.ArgumentParser(description='Extract TrackNet10 spectrum from .lis file')
parser.add_argument(
    '--lis-file',
    default=str(REPO_ROOT / 'analysis' / 'lcls2CP_usrtrack_25_tab.lis'),
    help='Path to the USRTRACK .lis file',
)
parser.add_argument(
    '--output',
    default=str(SPECTRA_DIR / 'TrackNet10_spectrum.dat'),
    help='Output path for the extracted spectrum',
)
args = parser.parse_args()

with open(args.lis_file) as f:
    lines = f.readlines()

detector_pattern = re.compile(r'#\s+Detector\s+n:\s+(\d+)\s+(Track\S+)')
weights = None
for i, line in enumerate(lines):
    m = detector_pattern.search(line)
    if m and m.group(2).strip() == 'TrackNet10':
        data_start = i + 2
        w = []
        for dl in lines[data_start: data_start + 251]:
            nums = dl.split()
            if len(nums) >= 3:
                w.append(float(nums[2]))
        weights = np.array(w[:250])
        break

if weights is None:
    raise SystemExit('TrackNet10 detector not found in the .lis file.')

E = np.arange(250)
np.savetxt(args.output, np.column_stack([E, weights]), fmt='%.6e')
print(f'Saved {args.output}, shape: {weights.shape}')
