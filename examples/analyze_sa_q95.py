#!/usr/bin/env python3
"""Detailed analysis of when SA Q95 table is helpful."""

import csv
from pathlib import Path

def main():
    csv_path = Path("comparison_outputs/sa_tables/sa_comparison.csv")

    # Parse CSV data
    data = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'codec': row['codec'],
                'q': int(float(row['quality'])),
                'bpp': float(row['bpp']),
                'ssim2': float(row['ssimulacra2']),
                'ba': float(row['butteraugli']),
                'dssim': float(row['dssim'])
            })

    # Find SA Q95 data
    sa_q95 = [d for d in data if d['codec'] == 'mozjpeg-sa-nearest' and d['q'] == 95][0]

    print("=== SA Q95 Table Analysis ===\n")
    print(f"SA Q95 produces: {sa_q95['bpp']:.3f} bpp, SSIM2 {sa_q95['ssim2']:.2f}")
    print()

    # Find standard JPEG at similar quality levels
    std_data = [d for d in data if d['codec'] == 'mozjpeg-std']

    print("Comparison with Standard JPEG:\n")

    # Find closest standard by BPP
    closest_bpp = min(std_data, key=lambda d: abs(d['bpp'] - sa_q95['bpp']))
    print(f"Closest BPP match: Standard Q{closest_bpp['q']}")
    print(f"  Standard: {closest_bpp['bpp']:.3f} bpp, SSIM2 {closest_bpp['ssim2']:.2f}")
    print(f"  SA Q95:   {sa_q95['bpp']:.3f} bpp, SSIM2 {sa_q95['ssim2']:.2f}")
    print(f"  → SA is {sa_q95['bpp'] - closest_bpp['bpp']:+.3f} bpp, {sa_q95['ssim2'] - closest_bpp['ssim2']:+.2f} SSIM2")
    print()

    # Find closest standard by SSIM2
    closest_ssim2 = min(std_data, key=lambda d: abs(d['ssim2'] - sa_q95['ssim2']))
    print(f"Closest SSIM2 match: Standard Q{closest_ssim2['q']}")
    print(f"  Standard: {closest_ssim2['bpp']:.3f} bpp, SSIM2 {closest_ssim2['ssim2']:.2f}")
    print(f"  SA Q95:   {sa_q95['bpp']:.3f} bpp, SSIM2 {sa_q95['ssim2']:.2f}")
    print(f"  → SA is {(sa_q95['bpp'] - closest_ssim2['bpp']) / closest_ssim2['bpp'] * 100:+.1f}% file size for {sa_q95['ssim2'] - closest_ssim2['ssim2']:+.2f} SSIM2")
    print()

    # Check ALL metrics at matched BPP
    print("All metrics at ~0.645 bpp (interpolated Q45-Q50):\n")

    q45 = [d for d in std_data if d['q'] == 45][0]
    q50 = [d for d in std_data if d['q'] == 50][0]

    # Interpolate
    weight = (sa_q95['bpp'] - q45['bpp']) / (q50['bpp'] - q45['bpp'])
    std_interp = {
        'ssim2': q45['ssim2'] + weight * (q50['ssim2'] - q45['ssim2']),
        'ba': q45['ba'] + weight * (q50['ba'] - q45['ba']),
        'dssim': q45['dssim'] + weight * (q50['dssim'] - q45['dssim'])
    }

    print(f"               SA Q95    Std (interp)   Difference   Winner")
    print(f"  SSIM2:       {sa_q95['ssim2']:6.2f}    {std_interp['ssim2']:6.2f}       {sa_q95['ssim2'] - std_interp['ssim2']:+.2f}      {'SA' if sa_q95['ssim2'] > std_interp['ssim2'] else 'Std'}")
    print(f"  Butteraugli: {sa_q95['ba']:6.3f}    {std_interp['ba']:6.3f}       {sa_q95['ba'] - std_interp['ba']:+.3f}      {'SA' if sa_q95['ba'] < std_interp['ba'] else 'Std'}")
    print(f"  DSSIM:       {sa_q95['dssim']:.6f}  {std_interp['dssim']:.6f}     {sa_q95['dssim'] - std_interp['dssim']:+.6f}  {'SA' if sa_q95['dssim'] < std_interp['dssim'] else 'Std'}")
    print()

    # Verdict
    print("=== VERDICT ===\n")
    print("SA Q95 is helpful ONLY when:")
    print("  1. Target bitrate is ~0.64-0.65 bpp")
    print("  2. Optimizing for SSIMULACRA2 specifically")
    print("  3. You DON'T care about Butteraugli or DSSIM")
    print()
    print("Gain: +1.64 SSIM2 at matched BPP")
    print("Cost: Slightly worse Butteraugli (+0.54) and DSSIM (+0.00005)")
    print()
    print("Equivalent standard quality range: Q45-Q50 (closer to Q47-48)")
    print()
    print("Recommendation:")
    print("  → Use SA Q95 when user requests Q47-48 AND optimizing for SSIMULACRA2")
    print("  → Skip it otherwise (minimal benefit, adds complexity)")

if __name__ == '__main__':
    main()
