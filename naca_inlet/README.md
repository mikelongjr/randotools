# NACA Inlet STEP tool

Import a **non-solid STEP surface**, add a parametric **NACA submerged inlet** that
follows local curvature, and export a set of faces you can bring back into CAD.

The inlet is sized primarily by **throat area**. Width/height follow a NACA aspect
ratio (default 4). Ramp length follows the Whitehouse construction of NACA RM-A7I30
(5–7° ramp). Side walls follow the published NACA curved-divergence
ordinates (Pignier Table 3 / Whitehouse unit-width table), interpolated
through every station.

## Install

```bash
pip install cadquery
python3 -m naca_inlet --help
```

## Usage

```bash
# GUI
python3 -m naca_inlet --gui

# Drape an inlet onto a skin surface (default --mode surfaces)
python3 -m naca_inlet fuselage_skin.step fuselage_naca.step \
    --area 4000 \
    --location 1200 0 80 \
    --heading 0

# Demo: curved panel + conforming duct faces
python3 -m naca_inlet --demo demo_naca.step --area 2500

# Old solid-body boolean cut
python3 -m naca_inlet body.step body_naca.step \
    --mode solid --area 4000 --location 1200 0 80
```

`--location` is snapped to the nearest face. By default it is the **throat centre**
(the visible opening’s aft end). Use `--location-ref tip` for the forward point.

`--heading` is yaw about the surface normal. `0` means downstream is world +X
projected onto the tangent plane. If the duct goes the wrong way, `--flip-normal`.

## What `--mode surfaces` exports

A STEP compound of faces, not a solid:

1. The original skin with a NACA-shaped hole (outline projected onto the surface).
2. Ramp **floor** at the design angle from the nose (single loft, one face).
3. Left and right **sidewalls** — ruled from the skin edges to the matching
   floor rail edges (shared topology for stitching).
4. **Throat** face at the aft opening (shared edge with the floor loft).

Re-import that STEP and use the faces as trimming/loft references.

## Parameters

| Parameter | Default | Notes |
|---|---|---|
| `--area` | (required unless width+height) | Throat area, mm² |
| `--aspect` | 4 | Width/height; NACA optimum 3–5 |
| `--ramp-angle` | 7° | NACA optimum 5–7° |
| `--mode` | `surfaces` | `surfaces` = conforming faces; `solid` = boolean cut |
| `--width` / `--height` | derived | Override area sizing |

Units are millimetres unless `--units` is set to match the STEP file.
