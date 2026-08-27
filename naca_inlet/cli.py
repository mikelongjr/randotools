"""Command-line interface for the NACA inlet STEP tool."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cadquery import Vector

from .cad import (
    build_cavity,
    cut_inlet,
    export_step,
    export_stl,
    import_step,
    make_demo_plate,
    place_cavity,
    placement_for_location,
    thicken_if_surface,
)
from .geometry import resolve_spec, spec_summary
from .surface import (
    build_surface_inlet,
    demo_surface_location,
    make_demo_surface,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="naca-inlet",
        description=(
            "Import a STEP surface, drape a parametric NACA submerged inlet "
            "onto it (conforming to local curvature), and export the resulting faces."
        ),
    )
    p.add_argument(
        "input",
        nargs="?",
        help="Input STEP file. Omit with --demo to generate a flat plate.",
    )
    p.add_argument(
        "output",
        nargs="?",
        help="Output STEP path (default: <input>_naca.step)",
    )
    p.add_argument("--gui", action="store_true", help="Open the parameter GUI")
    p.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Build a sample host and add the inlet (no input STEP). "
            "A single positional path is treated as the output file. "
            "Default --mode surfaces uses a curved panel; --mode solid uses a plate."
        ),
    )
    p.add_argument(
        "--area",
        type=float,
        default=None,
        help="Throat area in mm² (primary size parameter)",
    )
    p.add_argument("--width", type=float, default=None, help="Throat width in mm")
    p.add_argument("--height", type=float, default=None, help="Throat height in mm")
    p.add_argument(
        "--aspect",
        type=float,
        default=4.0,
        help="Width/height ratio when sizing from --area (NACA optimum 3–5, default 4)",
    )
    p.add_argument(
        "--ramp-angle",
        type=float,
        default=7.0,
        help="Ramp angle in degrees (NACA optimum 5–7, default 7)",
    )
    p.add_argument(
        "--location",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Point on/near the surface (mm). Snapped to the nearest face.",
    )
    p.add_argument(
        "--location-ref",
        choices=("throat", "tip"),
        default="throat",
        help="What --location refers to (default: throat centre)",
    )
    p.add_argument(
        "--heading",
        type=float,
        default=0.0,
        help="Yaw about the surface normal in degrees. 0 = world +X projected onto the surface.",
    )
    p.add_argument(
        "--mode",
        choices=("surfaces", "solid"),
        default="surfaces",
        help=(
            "surfaces (default): punch a hole in the skin and add conforming "
            "ramp/wall/throat faces. solid: boolean-cut a cavity from a solid body."
        ),
    )
    p.add_argument(
        "--flip-normal",
        action="store_true",
        help="Reverse the inward cut direction",
    )
    p.add_argument(
        "--skin-thickness",
        type=float,
        default=0.0,
        help="If the STEP is an open surface, thicken it by this many mm before cutting",
    )
    p.add_argument(
        "--duct-extension",
        type=float,
        default=None,
        help="Internal duct length aft of the lip (mm). Default: max(height, 0.25×width)",
    )
    p.add_argument(
        "--overshoot",
        type=float,
        default=None,
        help="How far the cutter protrudes above the surface (mm)",
    )
    p.add_argument(
        "--stations",
        type=int,
        default=48,
        help="Loft stations (higher = smoother, slower)",
    )
    p.add_argument(
        "--units",
        default="MM",
        choices=("MM", "CM", "M", "INCH", "FT"),
        help="Unit to interpret the STEP file in (default MM)",
    )
    p.add_argument(
        "--cavity",
        metavar="PATH",
        help="Also export the inlet cavity solid to this STEP path",
    )
    p.add_argument(
        "--stl",
        metavar="PATH",
        help="Also export the result as STL",
    )
    p.add_argument(
        "--print-dims",
        action="store_true",
        help="Print derived dimensions and exit (no CAD)",
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="Run a built-in plate cut and report volumes",
    )
    return p


def _spec_from_args(args: argparse.Namespace):
    return resolve_spec(
        area=args.area,
        width=args.width,
        height=args.height,
        aspect_ratio=args.aspect,
        ramp_angle_deg=args.ramp_angle,
        duct_extension=args.duct_extension,
        overshoot=args.overshoot,
        n_stations=args.stations,
    )


def run_self_test() -> int:
    spec = resolve_spec(area=4000.0, aspect_ratio=4.0, ramp_angle_deg=7.0)
    print(spec_summary(spec))

    print("--- solid cut ---")
    host = make_demo_plate(spec)
    cavity = build_cavity(spec)
    v0 = host.val().Volume()
    result = cut_inlet(host, cavity)
    v1 = result.val().Volume()
    removed = v0 - v1
    print(f"Host volume     {v0:.1f} mm³")
    print(f"Result volume   {v1:.1f} mm³")
    print(f"Removed         {removed:.1f} mm³")
    if removed <= 0:
        print("SELF-TEST FAILED: cut removed no volume", file=sys.stderr)
        return 1
    if removed < spec.area * spec.height * 0.2:
        print("SELF-TEST FAILED: removed volume looks too small", file=sys.stderr)
        return 1

    print("--- surface drape ---")
    spec_s = resolve_spec(area=800.0, aspect_ratio=4.0, ramp_angle_deg=7.0)
    skin = make_demo_surface(spec_s)
    loc = demo_surface_location(spec_s, skin)
    place = placement_for_location(
        skin, spec_s, loc, 0.0, location_ref="throat"
    )
    draped = build_surface_inlet(skin, spec_s, place)
    n_faces = len(draped.faces().vals())
    n_solids = len(draped.solids().vals())
    print(f"Draped faces    {n_faces}")
    print(f"Draped solids   {n_solids}")
    if n_faces < 5:
        print(
            "SELF-TEST FAILED: expected OML + floor + 2 walls + throat",
            file=sys.stderr,
        )
        return 1
    if n_solids != 0:
        print("SELF-TEST FAILED: surface result should not be a solid", file=sys.stderr)
        return 1
    print("SELF-TEST OK")
    return 0


def run(args: argparse.Namespace) -> int:
    if args.gui:
        from .gui import launch_gui

        launch_gui(args)
        return 0

    if args.self_test:
        return run_self_test()

    if args.area is None and (args.width is None or args.height is None) and not args.demo:
        if args.print_dims:
            pass
        else:
            print(
                "Specify --area, or both --width and --height. "
                "Use --gui for an interactive form.",
                file=sys.stderr,
            )
            return 2

    if args.demo and args.area is None and args.width is None:
        args.area = 4000.0

    spec = _spec_from_args(args)
    print(spec_summary(spec))
    if args.print_dims:
        return 0

    if args.demo:
        # With --demo a single positional is the output path, not an input file.
        output = Path(args.output or args.input or "naca_inlet_demo.step")
        if args.mode == "solid":
            host = make_demo_plate(spec)
            location = Vector(spec.cutout_length, 0, 0)
        else:
            host = make_demo_surface(spec)
            location = demo_surface_location(spec, host)
            print(
                f"Demo surface throat at "
                f"({location.x:.3f}, {location.y:.3f}, {location.z:.3f})"
            )
    else:
        if not args.input:
            print("Input STEP file is required (or pass --demo / --gui).", file=sys.stderr)
            return 2
        in_path = Path(args.input)
        if not in_path.is_file():
            print(f"Input file not found: {in_path}", file=sys.stderr)
            return 2
        host = import_step(str(in_path), unit=args.units)
        if args.mode == "solid":
            host = thicken_if_surface(host, args.skin_thickness)
        output = Path(args.output or in_path.with_name(in_path.stem + "_naca.step"))
        if args.location is None:
            bb = host.val().BoundingBox()
            location = Vector(
                0.5 * (bb.xmin + bb.xmax),
                0.5 * (bb.ymin + bb.ymax),
                bb.zmax,
            )
            print(
                f"No --location given; using bbox top centre "
                f"({location.x:.3f}, {location.y:.3f}, {location.z:.3f})"
            )
        else:
            location = Vector(*args.location)

    placement = placement_for_location(
        host,
        spec,
        location,
        args.heading,
        location_ref=args.location_ref,
        flip_normal=args.flip_normal,
    )
    print(
        f"Snapped to     ({placement.point_on_surface.x:.3f}, "
        f"{placement.point_on_surface.y:.3f}, {placement.point_on_surface.z:.3f})"
    )
    print(
        f"Tip origin     ({placement.origin.x:.3f}, "
        f"{placement.origin.y:.3f}, {placement.origin.z:.3f})"
    )
    print(
        f"Downstream X   ({placement.x_dir.x:.4f}, {placement.x_dir.y:.4f}, {placement.x_dir.z:.4f})"
    )
    print(
        f"Outward Z      ({placement.z_dir.x:.4f}, {placement.z_dir.y:.4f}, {placement.z_dir.z:.4f})"
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "surfaces":
        result = build_surface_inlet(host, spec, placement)
        print(f"Result faces   {len(result.faces().vals())}")
    else:
        cavity_local = build_cavity(spec)
        cavity = place_cavity(cavity_local, placement)
        if args.cavity:
            export_step(cavity, args.cavity)
            print(f"Wrote cavity   {args.cavity}")
        result = cut_inlet(host, cavity)

    export_step(result, str(output))
    print(f"Wrote result   {output}")
    if args.stl:
        export_stl(result, args.stl)
        print(f"Wrote STL      {args.stl}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
