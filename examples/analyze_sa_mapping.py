#!/usr/bin/env python3
"""Analyze SA tables to find equivalent standard JPEG quality levels."""

import csv
from pathlib import Path

def main():
    csv_path = Path("comparison_outputs/sa_tables/sa_comparison.csv")

    # Parse CSV data
    std_data = {}  # quality -> {bpp, ssim2, ba, dssim}
    sa_data = {}   # quality -> {bpp, ssim2, ba, dssim}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            codec = row['codec']
            q = int(float(row['quality']))
            bpp = float(row['bpp'])
            ssim2 = float(row['ssimulacra2'])
            ba = float(row['butteraugli'])
            dssim = float(row['dssim'])

            if codec == 'mozjpeg-std':
                std_data[q] = {'bpp': bpp, 'ssim2': ssim2, 'ba': ba, 'dssim': dssim}
            elif codec == 'mozjpeg-sa-nearest':
                sa_data[q] = {'bpp': bpp, 'ssim2': ssim2, 'ba': ba, 'dssim': dssim}

    # Find unique SA table BPP values (since multiple qualities use same table)
    sa_tables = {}  # bpp -> {qualities, ssim2, ba, dssim}
    for q, data in sa_data.items():
        bpp = round(data['bpp'], 3)  # Round to avoid float comparison issues
        if bpp not in sa_tables:
            sa_tables[bpp] = {'qualities': [], 'ssim2': data['ssim2'], 'ba': data['ba'], 'dssim': data['dssim']}
        sa_tables[bpp]['qualities'].append(q)

    print("=== SA Table Analysis ===\n")
    print(f"Found {len(sa_tables)} unique SA tables:\n")

    for bpp in sorted(sa_tables.keys()):
        qualities = sa_tables[bpp]['qualities']
        print(f"SA Table (Q{min(qualities)}-Q{max(qualities)}): {bpp:.3f} bpp, SSIM2 {sa_tables[bpp]['ssim2']:.2f}")

    print("\n=== BPP-Matched Quality Mapping (with interpolation) ===\n")

    # For each unique SA table, interpolate standard quality at same BPP
    mappings = []
    for sa_bpp in sorted(sa_tables.keys()):
        sa_info = sa_tables[sa_bpp]
        sa_qs = sa_info['qualities']

        # Find two standard qualities that bracket this BPP
        std_sorted = sorted(std_data.items())

        lower_q, upper_q = None, None
        for i in range(len(std_sorted) - 1):
            q1, data1 = std_sorted[i]
            q2, data2 = std_sorted[i + 1]

            if data1['bpp'] <= sa_bpp <= data2['bpp']:
                lower_q, upper_q = (q1, data1), (q2, data2)
                break

        if lower_q is None:
            # SA BPP is outside standard range, use closest
            if sa_bpp < std_sorted[0][1]['bpp']:
                lower_q = upper_q = std_sorted[0]
            else:
                lower_q = upper_q = std_sorted[-1]

        # Interpolate standard quality metrics at SA's BPP
        if lower_q == upper_q:
            std_ssim2 = lower_q[1]['ssim2']
            std_ba = lower_q[1]['ba']
            std_dssim = lower_q[1]['dssim']
            interp_note = f"(at Q{lower_q[0]})"
        else:
            # Linear interpolation
            bpp1, bpp2 = lower_q[1]['bpp'], upper_q[1]['bpp']
            weight = (sa_bpp - bpp1) / (bpp2 - bpp1)

            std_ssim2 = lower_q[1]['ssim2'] + weight * (upper_q[1]['ssim2'] - lower_q[1]['ssim2'])
            std_ba = lower_q[1]['ba'] + weight * (upper_q[1]['ba'] - lower_q[1]['ba'])
            std_dssim = lower_q[1]['dssim'] + weight * (upper_q[1]['dssim'] - lower_q[1]['dssim'])
            interp_note = f"(interpolated Q{lower_q[0]}-Q{upper_q[0]})"

        # Calculate quality difference at matched BPP
        ssim2_diff = sa_info['ssim2'] - std_ssim2
        ba_diff = std_ba - sa_info['ba']  # Lower BA is better
        dssim_diff = std_dssim - sa_info['dssim']  # Lower DSSIM is better

        print(f"SA Q{min(sa_qs)} table ({sa_bpp:.3f} bpp):")
        print(f"  SSIM2:  SA {sa_info['ssim2']:.2f} vs Std {std_ssim2:.2f} {interp_note} ({ssim2_diff:+.2f})")
        print(f"  BA:     SA {sa_info['ba']:.3f} vs Std {std_ba:.3f} ({ba_diff:+.3f} = {'SA better' if ba_diff > 0 else 'Std better'})")
        print(f"  DSSIM:  SA {sa_info['dssim']:.6f} vs Std {std_dssim:.6f} ({dssim_diff:+.6f} = {'SA better' if dssim_diff > 0 else 'Std better'})")

        # Overall winner based on SSIM2 (primary metric)
        winner = "✓ SA BETTER" if ssim2_diff > 1.0 else "✗ STD BETTER/EQUAL"
        print(f"  → {winner}\n")

        mappings.append({
            'sa_qualities': sa_qs,
            'sa_bpp': sa_bpp,
            'std_quality_range': (lower_q[0], upper_q[0]) if lower_q != upper_q else (lower_q[0],),
            'std_bpp': sa_bpp,  # Same BPP for matched comparison
            'ssim2_diff': ssim2_diff,
            'ba_diff': ba_diff,
            'dssim_diff': dssim_diff
        })

    print("\n=== Recommended Mapping for zenjpeg ===\n")
    print("Summary: SA tables are ONLY worth using if they provide >1 SSIM2 improvement at matched BPP.\n")

    for m in sorted(mappings, key=lambda x: x['sa_bpp']):
        std_range = m['std_quality_range']
        if len(std_range) == 1:
            std_q_str = f"Q{std_range[0]}"
        else:
            std_q_str = f"Q{std_range[0]}-Q{std_range[1]}"

        if m['ssim2_diff'] > 1.0:
            marker = "✓"
            rec = f"USE SA Q{min(m['sa_qualities'])} table ({m['ssim2_diff']:+.2f} SSIM2 gain)"
        else:
            marker = "✗"
            rec = f"Use standard quantization ({m['ssim2_diff']:+.2f} SSIM2)"

        print(f"{marker} {std_q_str} ({m['sa_bpp']:.3f} bpp): {rec}")

    # Check if ANY SA table is worth using
    any_good = any(m['ssim2_diff'] > 1.0 for m in mappings)
    if not any_good:
        print("\n⚠️  CONCLUSION: No SA tables provide meaningful improvement over standard JPEG.")
        print("    Recommend NOT integrating SA tables into zenjpeg.")

if __name__ == '__main__':
    main()
