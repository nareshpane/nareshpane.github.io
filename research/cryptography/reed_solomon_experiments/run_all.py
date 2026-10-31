from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "01_polynomial_walkthrough.py",
    "02_lagrange_interpolation.py",
    "03_codebook_and_distance.py",
    "04_exhaustive_list_sizes.py",
    "05_interpolation_crowding.py",
    "06_constructed_affine_line.py",
    "07_exact_small_line_census.py",
    "08_random_line_search.py",
    "09_scaling_estimates.py",
    "10_gf8_extension_field.py",
]

for script in SCRIPTS:
    print(f"\n=== Running {script} ===", flush=True)
    subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)

print(f"\nAll experiments completed. Outputs are in: {ROOT / 'outputs'}")
