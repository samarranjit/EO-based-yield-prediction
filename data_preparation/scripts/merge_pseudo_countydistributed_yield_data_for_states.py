"""
Mosaic per-county pseudo-yield rasters into a single state-year GeoTIFF.

`distribute_county_yield_to_pixels.py` writes one pseudo-yield raster per
(state, year, county) into data/pixel_distributed_yield/Intermediate/, named

    {crop}_{STATE}_{YEAR}_{GEOID}_{County}_pseudo_yield.tif

This script collects every county raster for one (state, year), validates that
they share a CRS / band layout / dtype, merges them into one raster, prints a
summary with per-band statistics, and writes two files under the output root
(data/pixel_distributed_yield/ by default):

    tifs/{crop}_{STATE}_{YEAR}_pseudo_yield.tif        the merged raster
    pdfs/{crop}_{STATE}_{YEAR}_pseudo_yield_bands.pdf  per-band QC figure

Counties are non-overlapping by construction, so the default merge method
("first") is effectively a simple mosaic; overlapping pixels take the value of
the first raster in sorted order.

    python scripts/merge_pseudo_countydistributed_yield_data_for_states.py --state MD --year 2022

Use --output-root to relocate both outputs together, or --output / --plot-path
to place either file individually. The Maryland rasters consumed by model/ live
in data/Maryland/pseudo_yield_files/, so pass --output explicitly when
regenerating those.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to disk; never open a window

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.plot import plotting_extent

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CROP_NAME, DATA_DIR, END_YEAR, NODATA, START_YEAR, STATE_FIPS

# Per-county rasters written by distribute_county_yield_to_pixels.py.
DEFAULT_INPUT_DIR = DATA_DIR / "pixel_distributed_yield" / "Intermediate"

# Output root; merged rasters and QC figures go in separate subdirectories.
DEFAULT_OUTPUT_ROOT = DATA_DIR / "pixel_distributed_yield"
RASTER_SUBDIR = "tifs"
PLOT_SUBDIR = "pdfs"

# rasterio.merge pixel-selection strategies.
MERGE_METHODS = ("first", "last", "min", "max")

# Cap on how many bands the QC figure shows.
MAX_BANDS_TO_PLOT = 6


# ============================================================
# 1) ARGUMENTS
# ============================================================


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge per-county pseudo-yield GeoTIFFs into one state-year raster."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--state",
        required=True,
        help="Two-letter state code, e.g. MD.",
    )
    parser.add_argument(
        "--year",
        required=True,
        type=int,
        help="Crop year, e.g. 2022.",
    )
    parser.add_argument(
        "--crop",
        default=CROP_NAME.lower(),
        help="Crop name used as the filename prefix.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory holding the per-county pseudo-yield rasters.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            f"Output root. The merged raster goes in <root>/{RASTER_SUBDIR}/ "
            f"and the QC figure in <root>/{PLOT_SUBDIR}/."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Full path for the merged raster. Overrides --output-root.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help=(
            "Full path for the QC figure. Overrides --output-root. "
            "The file extension selects the format (.pdf, .png, ...)."
        ),
    )
    parser.add_argument(
        "--merge-method",
        default="first",
        choices=MERGE_METHODS,
        help="How to resolve pixels covered by more than one input raster.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output raster if it already exists.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip writing the per-band QC figure.",
    )

    args = parser.parse_args(argv)

    args.state = args.state.upper()
    if args.state not in STATE_FIPS:
        parser.error(
            f"Unknown state code {args.state!r}. "
            f"Expected one of: {', '.join(sorted(STATE_FIPS))}"
        )

    if not START_YEAR <= args.year <= END_YEAR:
        print(
            f"Warning: year {args.year} is outside the study period "
            f"{START_YEAR}-{END_YEAR}."
        )

    # Both outputs share one stem so the raster and its QC figure stay paired.
    stem = f"{args.crop}_{args.state}_{args.year}_pseudo_yield"

    if args.output is None:
        args.output = args.output_root / RASTER_SUBDIR / f"{stem}.tif"

    if args.plot_path is None:
        args.plot_path = args.output_root / PLOT_SUBDIR / f"{stem}_bands.pdf"

    return args


# ============================================================
# 2) INPUT DISCOVERY AND VALIDATION
# ============================================================


def find_county_rasters(
    input_dir: Path,
    crop: str,
    state: str,
    year: int,
) -> list[Path]:
    """Return the sorted per-county pseudo-yield rasters for one state-year."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    pattern = f"{crop}_{state}_{year}_*_pseudo_yield.tif"
    tif_files = sorted(input_dir.glob(pattern))

    if not tif_files:
        raise FileNotFoundError(
            f"No rasters matched {pattern!r} in {input_dir}"
        )

    print(f"Found {len(tif_files)} county raster(s) matching {pattern!r}:")
    for index, path in enumerate(tif_files, start=1):
        print(f"  {index:>3}. {path.name}")

    return tif_files


def read_raster_metadata(tif_files: list[Path]) -> list[dict]:
    """Collect the metadata needed to check that the inputs are mergeable."""
    metadata = []
    for tif_path in tif_files:
        with rasterio.open(tif_path) as src:
            metadata.append(
                {
                    "file": tif_path.name,
                    "width": src.width,
                    "height": src.height,
                    "band_count": src.count,
                    "crs": src.crs,
                    "dtype": src.dtypes,
                    "nodata": src.nodata,
                    "resolution": src.res,
                    "bounds": src.bounds,
                    "band_descriptions": src.descriptions,
                }
            )
    return metadata


def validate_inputs(metadata: list[dict]) -> dict:
    """Fail on incompatible inputs, warn on merely suspicious ones.

    Returns the reference (first) raster's metadata.
    """
    reference = metadata[0]

    print("\nReference raster:")
    print(f"  File:              {reference['file']}")
    print(f"  CRS:               {reference['crs']}")
    print(f"  Number of bands:   {reference['band_count']}")
    print(f"  Data type:         {reference['dtype']}")
    print(f"  Resolution:        {reference['resolution']}")
    print(f"  NoData value:      {reference['nodata']}")
    print(f"  Band descriptions: {reference['band_descriptions']}")

    for info in metadata[1:]:
        if info["crs"] != reference["crs"]:
            raise ValueError(
                "CRS mismatch:\n"
                f"  {reference['file']}: {reference['crs']}\n"
                f"  {info['file']}: {info['crs']}\n"
                "Reproject the rasters to a common CRS before merging."
            )

        if info["band_count"] != reference["band_count"]:
            raise ValueError(
                "Band-count mismatch:\n"
                f"  {reference['file']}: {reference['band_count']} bands\n"
                f"  {info['file']}: {info['band_count']} bands"
            )

        if info["dtype"] != reference["dtype"]:
            raise ValueError(
                "Data-type mismatch:\n"
                f"  {reference['file']}: {reference['dtype']}\n"
                f"  {info['file']}: {info['dtype']}"
            )

        if not np.allclose(info["resolution"], reference["resolution"]):
            print(
                f"Warning: {info['file']} has resolution {info['resolution']}, "
                f"while the reference resolution is {reference['resolution']}."
            )

        if info["nodata"] != reference["nodata"]:
            print(
                f"Warning: {info['file']} has NoData {info['nodata']}, "
                f"while the reference NoData is {reference['nodata']}."
            )

        if info["band_descriptions"] != reference["band_descriptions"]:
            print(
                f"Warning: band descriptions differ in {info['file']}.\n"
                f"  Reference: {reference['band_descriptions']}\n"
                f"  Current:   {info['band_descriptions']}"
            )

    print("\nValidation completed.")
    return reference


# ============================================================
# 3) MERGE
# ============================================================


def merge_rasters(tif_files: list[Path], merge_method: str):
    """Mosaic the inputs in memory.

    Returns (array, transform, profile, band_descriptions).
    """
    source_datasets = [rasterio.open(path) for path in tif_files]
    try:
        nodata_value = source_datasets[0].nodata
        if nodata_value is None:
            nodata_value = NODATA
            print(
                f"Warning: inputs declare no NoData value; using {NODATA} "
                "from config for the merge."
            )

        combined_array, combined_transform = merge(
            source_datasets,
            method=merge_method,
            nodata=nodata_value,
        )

        # The first raster defines the output profile and band names.
        combined_profile = source_datasets[0].profile.copy()
        band_descriptions = source_datasets[0].descriptions
    finally:
        for dataset in source_datasets:
            dataset.close()

    combined_profile.update(
        {
            "driver": "GTiff",
            "height": combined_array.shape[1],
            "width": combined_array.shape[2],
            "count": combined_array.shape[0],
            "transform": combined_transform,
            "nodata": nodata_value,
            "compress": "deflate",
            "tiled": True,
            "BIGTIFF": "IF_SAFER",
        }
    )

    print(f"\nMerged {len(tif_files)} raster(s) using method {merge_method!r}.")
    print(
        f"Combined array shape: {combined_array.shape} "
        "(bands, height, width)"
    )

    return combined_array, combined_transform, combined_profile, band_descriptions


# ============================================================
# 4) REPORTING
# ============================================================


def valid_pixel_mask(band: np.ndarray, nodata_value) -> np.ndarray:
    """Finite pixels that are not NoData."""
    mask = np.isfinite(band)
    if nodata_value is not None and not np.isnan(nodata_value):
        mask &= band != nodata_value
    return mask


def band_label(band_descriptions, band_index: int) -> str:
    description = band_descriptions[band_index] if band_descriptions else None
    return description or f"Band {band_index + 1}"


def report_combined_raster(
    combined_array: np.ndarray,
    combined_transform,
    combined_profile: dict,
    band_descriptions,
) -> None:
    nodata_value = combined_profile.get("nodata")

    print("\nCOMBINED RASTER INFORMATION")
    print("=" * 65)
    print(f"Driver:             {combined_profile['driver']}")
    print(f"CRS:                {combined_profile['crs']}")
    print(f"Band count:         {combined_profile['count']}")
    print(f"Width:              {combined_profile['width']:,} pixels")
    print(f"Height:             {combined_profile['height']:,} pixels")
    print(f"Data type:          {combined_profile['dtype']}")
    print(f"NoData value:       {nodata_value}")
    print(
        "Pixel resolution:   "
        f"{combined_transform.a}, {abs(combined_transform.e)}"
    )
    print(f"Affine transform:   {combined_transform}")
    print(f"Band descriptions:  {band_descriptions}")

    left = combined_transform.c
    top = combined_transform.f
    right = left + combined_profile["width"] * combined_transform.a
    bottom = top + combined_profile["height"] * combined_transform.e

    print("Bounds:")
    print(f"  Left:   {left}")
    print(f"  Bottom: {bottom}")
    print(f"  Right:  {right}")
    print(f"  Top:    {top}")

    print("\nBAND STATISTICS")
    print("=" * 65)

    for band_index in range(combined_array.shape[0]):
        band = combined_array[band_index]
        valid_values = band[valid_pixel_mask(band, nodata_value)]

        print(f"\nBand {band_index + 1}: {band_label(band_descriptions, band_index)}")
        print(f"  Total pixels:       {band.size:,}")
        print(f"  Valid pixels:       {valid_values.size:,}")
        print(f"  Missing pixels:     {band.size - valid_values.size:,}")

        if valid_values.size == 0:
            print("  No valid pixel values were found.")
            continue

        print(f"  Minimum:            {np.min(valid_values):.6g}")
        print(f"  Maximum:            {np.max(valid_values):.6g}")
        print(f"  Mean:               {np.mean(valid_values):.6g}")
        print(f"  Median:             {np.median(valid_values):.6g}")
        print(f"  Standard deviation: {np.std(valid_values):.6g}")
        print(f"  2nd percentile:     {np.percentile(valid_values, 2):.6g}")
        print(f"  98th percentile:    {np.percentile(valid_values, 98):.6g}")


def plot_bands(
    combined_array: np.ndarray,
    combined_transform,
    nodata_value,
    band_descriptions,
    plot_path: Path,
) -> None:
    """Write a QC figure with one panel per band (2-98 percentile stretch)."""
    band_count = combined_array.shape[0]
    bands_to_plot = min(band_count, MAX_BANDS_TO_PLOT)

    fig, axes = plt.subplots(
        nrows=bands_to_plot,
        ncols=1,
        figsize=(12, 5 * bands_to_plot),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    extent = plotting_extent(combined_array[0], combined_transform)

    for band_index, axis in enumerate(axes):
        band = combined_array[band_index].astype("float64")
        band[~valid_pixel_mask(band, nodata_value)] = np.nan
        valid_values = band[np.isfinite(band)]

        name = band_label(band_descriptions, band_index)

        if valid_values.size == 0:
            axis.text(
                0.5,
                0.5,
                "No valid data",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_title(name)
            axis.set_axis_off()
            continue

        lower_limit, upper_limit = np.percentile(valid_values, [2, 98])
        if lower_limit == upper_limit:
            # Constant or near-constant band: fall back to the full range.
            lower_limit = np.min(valid_values)
            upper_limit = np.max(valid_values)

        image = axis.imshow(
            band,
            extent=extent,
            vmin=lower_limit,
            vmax=upper_limit,
        )
        axis.set_title(
            f"{name}\nDisplay range: {lower_limit:.4g} to {upper_limit:.4g}"
        )
        axis.set_xlabel("X coordinate")
        axis.set_ylabel("Y coordinate")

        colorbar = fig.colorbar(image, ax=axis, shrink=0.8)
        colorbar.set_label(name)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"\nQC figure saved: {plot_path}")
    if band_count > bands_to_plot:
        print(
            f"The raster contains {band_count} bands; "
            f"only the first {bands_to_plot} were plotted."
        )


# ============================================================
# 5) OUTPUT
# ============================================================


def write_output(
    output_path: Path,
    combined_array: np.ndarray,
    combined_profile: dict,
    band_descriptions,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **combined_profile) as dst:
        dst.write(combined_array)

        # Carry the original band names over to the merged raster.
        for band_number, description in enumerate(band_descriptions, start=1):
            if description:
                dst.set_band_description(band_number, description)

    print(f"\nCombined GeoTIFF saved: {output_path}")


# ============================================================
# 6) ENTRY POINT
# ============================================================


def main(argv=None) -> int:
    args = parse_args(argv)

    print("=" * 65)
    print(f"Merging {args.crop} pseudo-yield rasters for {args.state} {args.year}")
    print("=" * 65)

    if args.output.exists() and not args.overwrite:
        print(
            f"Output already exists: {args.output}\n"
            "Pass --overwrite to replace it."
        )
        return 1

    try:
        tif_files = find_county_rasters(
            args.input_dir,
            args.crop,
            args.state,
            args.year,
        )
        metadata = read_raster_metadata(tif_files)
        validate_inputs(metadata)
    except (FileNotFoundError, ValueError) as error:
        print(f"\nError: {error}")
        return 1

    combined_array, combined_transform, combined_profile, band_descriptions = (
        merge_rasters(tif_files, args.merge_method)
    )

    report_combined_raster(
        combined_array,
        combined_transform,
        combined_profile,
        band_descriptions,
    )

    if not args.no_plot:
        plot_bands(
            combined_array,
            combined_transform,
            combined_profile.get("nodata"),
            band_descriptions,
            args.plot_path,
        )

    write_output(args.output, combined_array, combined_profile, band_descriptions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
