"""Drape a NACA inlet onto an open surface, producing CAD faces.

The planform is walked along the host face so the cutout, ramp floor, and
sidewalls follow local curvature. The original face is given a hole; the
duct is added as separate faces (floor, walls, throat) for re-import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cadquery as cq
from cadquery import Vector
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.BRepLProp import BRepLProp_SLProps
from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCP.Geom import Geom_CylindricalSurface
from OCP.GeomAPI import GeomAPI_PointsToBSplineSurface, GeomAPI_ProjectPointOnSurf
from OCP.GeomAbs import GeomAbs_C0
from OCP.TColgp import TColgp_Array2OfPnt
from OCP.TopAbs import TopAbs_REVERSED
from OCP.gp import gp_Ax3, gp_Dir, gp_Pnt

from .cad import Placement, _as_workplane, _faces_of
from .geometry import InletSpec, nose_half_width, planform_half_width, ramp_floor_z

# First inclined floor station downstream of the nose line (mm).
NOSE_RAMP_STATION = 1.0


@dataclass
class SurfaceSample:
    oml: Vector
    normal: Vector  # outward
    x_dir: Vector
    y_dir: Vector


def project_point_on_face(
    face: cq.Face,
    point: Vector,
    outward_hint: Vector,
) -> Tuple[Vector, Vector]:
    """Closest point on ``face`` and an outward unit normal."""
    pnt = gp_Pnt(point.x, point.y, point.z)
    surf = BRep_Tool.Surface_s(face.wrapped)
    proj = GeomAPI_ProjectPointOnSurf(pnt, surf)
    if proj.NbPoints() <= 0:
        raise RuntimeError(f"Failed to project point {point} onto the surface")
    u, v = proj.LowerDistanceParameters()
    gp = proj.NearestPoint()
    adaptor = BRepAdaptor_Surface(face.wrapped)
    sl = BRepLProp_SLProps(adaptor, u, v, 1, 1e-6)
    if sl.IsNormalDefined():
        gn = sl.Normal()
        normal = Vector(gn.X(), gn.Y(), gn.Z())
        if face.wrapped.Orientation() == TopAbs_REVERSED:
            normal = normal.multiply(-1.0)
    else:
        normal = Vector(face.normalAt())
    if normal.Length < 1e-12:
        raise RuntimeError("Surface normal is undefined")
    normal = normal.normalized()
    if normal.dot(outward_hint) < 0:
        normal = normal.multiply(-1.0)
    return Vector(gp.X(), gp.Y(), gp.Z()), normal


def _tangent_frame(normal: Vector, x_hint: Vector) -> Tuple[Vector, Vector]:
    x = x_hint - normal.multiply(x_hint.dot(normal))
    if x.Length < 1e-9:
        # x_hint was parallel to the normal; pick an arbitrary tangent.
        fallback = Vector(1, 0, 0)
        if abs(normal.dot(fallback)) > 0.9:
            fallback = Vector(0, 1, 0)
        x = fallback - normal.multiply(fallback.dot(normal))
    x = x.normalized()
    y = normal.cross(x).normalized()
    return x, y


def walk_on_surface(
    face: cq.Face,
    start: Vector,
    direction: Vector,
    distance: float,
    outward_hint: Vector,
    n_steps: int = 60,
) -> Tuple[Vector, Vector]:
    """Step along the surface by approximately ``distance``.

    Returns the end point and the unit tangent at the end (same sense as
    ``direction``).
    """
    if distance <= 1e-12:
        p, n = project_point_on_face(face, start, outward_hint)
        x, _ = _tangent_frame(n, direction)
        return p, x
    p, n = project_point_on_face(face, start, outward_hint)
    d = direction.normalized()
    d, _ = _tangent_frame(n, d)
    step = distance / max(1, n_steps)
    for _ in range(max(1, n_steps)):
        trial = p + d.multiply(step)
        p, n = project_point_on_face(face, trial, outward_hint)
        d, _ = _tangent_frame(n, d)
    return p, d


def snap_placement_to_surface(
    face: cq.Face,
    spec: InletSpec,
    placement: Placement,
) -> Placement:
    """Put the tip on the face; walk from the throat if needed."""
    hint = placement.z_dir
    start, n0 = project_point_on_face(face, placement.point_on_surface, hint)
    x0, y0 = _tangent_frame(n0, placement.x_dir)
    placement.point_on_surface = start
    placement.z_dir = n0
    placement.x_dir = x0
    placement.y_dir = y0
    if (placement.origin - start).Length < 1e-6:
        # Already at the tip (location_ref=tip).
        return placement
    # Origin was computed in the tangent plane; walk from the snap point
    # back to the tip along -X.
    tip, x_back = walk_on_surface(
        face, start, x0.multiply(-1.0), spec.cutout_length, n0
    )
    placement.origin = tip
    xt, yt = _tangent_frame(
        project_point_on_face(face, tip, hint)[1],
        x_back.multiply(-1.0),
    )
    _p, nt = project_point_on_face(face, tip, hint)
    placement.origin = _p
    placement.z_dir = nt
    placement.x_dir = xt
    placement.y_dir = yt
    return placement


def _centerline_frames(
    face: cq.Face,
    spec: InletSpec,
    placement: Placement,
) -> List[Tuple[float, SurfaceSample]]:
    n = max(24, spec.n_stations)
    xs = [spec.cutout_length * i / (n - 1) for i in range(n)]
    xs.append(min(NOSE_RAMP_STATION, 0.25 * spec.cutout_length))
    xs = sorted({round(x, 8) for x in xs if 0.0 <= x <= spec.cutout_length})
    frames: List[Tuple[float, SurfaceSample]] = []
    p = placement.origin
    xdir = placement.x_dir
    outward = placement.z_dir
    prev = 0.0
    for x in xs:
        if x - prev > 1e-9:
            p, xdir = walk_on_surface(face, p, xdir, x - prev, outward, n_steps=8)
        p, nrm = project_point_on_face(face, p, outward)
        xdir, ydir = _tangent_frame(nrm, xdir)
        frames.append(
            (x, SurfaceSample(oml=p, normal=nrm, x_dir=xdir, y_dir=ydir))
        )
        prev = x
        outward = nrm
    return frames


def _sample_xy(
    face: cq.Face,
    frame: SurfaceSample,
    y: float,
    outward: Vector,
) -> Tuple[Vector, Vector]:
    trial = frame.oml + frame.y_dir.multiply(y)
    return project_point_on_face(face, trial, outward)


def _offset(oml: Vector, normal: Vector, depth: float) -> Vector:
    return oml - normal.multiply(depth)


def _dedupe(points: Sequence[Vector], tol: float = 1e-6) -> List[Vector]:
    out: List[Vector] = []
    for p in points:
        if not out or (p - out[-1]).Length > tol:
            out.append(p)
    return out


def _spline(points: Sequence[Vector]) -> cq.Edge:
    pts = _dedupe(points)
    if len(pts) < 2:
        raise RuntimeError("Need at least two distinct points for a spline")
    if len(pts) == 2:
        return cq.Edge.makeLine(pts[0], pts[1])
    return cq.Edge.makeSpline(pts)


def _span(points: Sequence[Vector]) -> float:
    if not points:
        return 0.0
    return max((p - points[0]).Length for p in points)


def _section_wire(pts: Sequence[Vector]) -> cq.Wire:
    """Closed-ish polyline wire across a station (for hole outlines)."""
    pts = _dedupe(pts)
    if len(pts) < 2:
        raise RuntimeError("Need at least two distinct points for a section wire")
    edges = [cq.Edge.makeLine(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    return cq.Wire.assembleEdges(edges)


def _loft_wire(pts: Sequence[Vector]) -> cq.Wire:
    """Wire for a loft section. Splines avoid ThruSections splitting the result
    into many ruled patches when SetMaxDegree(1) is used with polylines."""
    pts = _dedupe(pts)
    if len(pts) < 2:
        raise RuntimeError("Need at least two distinct points for a loft wire")
    return cq.Wire.assembleEdges([_spline(pts)])


def _closed_planform_wire(
    oml_left: Sequence[Vector],
    oml_throat: Sequence[Vector],
    oml_right: Sequence[Vector],
    *,
    nose_edge: cq.Edge | None = None,
    nose_pts: Sequence[Vector] | None = None,
) -> cq.Wire:
    # left: tip → throat, throat: left → right, right: throat → tip
    e_l = _spline(oml_left)
    e_t = _loft_wire(oml_throat)
    e_r = _spline(list(reversed(oml_right)))
    edges: List[cq.Edge] = [e_l] + list(e_t.Edges()) + [e_r]
    gap = (oml_right[0] - oml_left[0]).Length
    if gap > 1e-4:
        if nose_edge is not None:
            edges.append(cq.Edge(nose_edge.wrapped.Reversed()))
        elif nose_pts is not None and len(nose_pts) >= 2:
            edges.extend(_section_wire(list(reversed(nose_pts))).Edges())
        else:
            edges.append(cq.Edge.makeLine(oml_right[0], oml_left[0]))
    return cq.Wire.assembleEdges(edges)


def _punch_hole(face: cq.Face, wire: cq.Wire) -> cq.Face:
    try:
        holed = face.addHole(wire)
        if holed.Area() < face.Area() * 0.999:
            return holed
    except Exception:
        pass
    try:
        rev = cq.Wire(wire.wrapped.Reversed())
        holed = face.addHole(rev)
        if holed.Area() < face.Area() * 0.999:
            return holed
    except Exception:
        pass
    raise RuntimeError(
        "Could not punch the NACA outline as a hole in the surface. "
        "The inlet may fall off the face — move --location or reduce --area."
    )


def _floor_face(grid: List[List[Vector]]) -> cq.Face:
    nu = len(grid)
    nv = len(grid[0])
    arr = TColgp_Array2OfPnt(1, nu, 1, nv)
    for i, row in enumerate(grid, start=1):
        for j, p in enumerate(row, start=1):
            arr.SetValue(i, j, gp_Pnt(p.x, p.y, p.z))
    builder = GeomAPI_PointsToBSplineSurface()
    builder.Init(arr, 1, 1, GeomAbs_C0, 1e-6)
    surf = builder.Surface()
    return cq.Face(BRepBuilderAPI_MakeFace(surf, 1e-6).Face())


def _loft_sections(sections: List[Sequence[Vector]]) -> List[cq.Face]:
    """Loft cross-section wires into face(s) (degree 1, no longitudinal overshoot)."""
    thru = BRepOffsetAPI_ThruSections(False, False)
    thru.CheckCompatibility(False)
    thru.SetMaxDegree(1)
    added = 0
    for pts in sections:
        if _span(pts) < 1e-4:
            continue
        thru.AddWire(_loft_wire(pts).wrapped)
        added += 1
    if added < 2:
        raise RuntimeError("Loft needs at least two sections")
    thru.Build()
    if not thru.IsDone():
        raise RuntimeError("Loft failed")
    faces = cq.Shape.cast(thru.Shape()).Faces()
    if not faces:
        raise RuntimeError("Loft produced no face")
    return list(faces)


def _floor_thru(sections: List[Sequence[Vector]]) -> List[cq.Face]:
    return _loft_sections(sections)


def _orient_edge_like(reference: cq.Wire, edge: cq.Edge) -> cq.Edge:
    """Flip ``edge`` so it runs the same way as ``reference`` (tip→throat, etc.).

    Ruled surfaces twist when the two rails run in opposite directions.
    """
    rs, re = reference.startPoint(), reference.endPoint()
    es, ee = edge.startPoint(), edge.endPoint()
    aligned = (es - rs).Length + (ee - re).Length
    flipped = (ee - rs).Length + (es - re).Length
    if flipped + 1e-9 < aligned:
        return cq.Edge(edge.wrapped.Reversed())
    return edge


def _wire_from_edge(edge: cq.Edge) -> cq.Wire:
    return cq.Wire.assembleEdges([edge])


def _floor_boundary_edges(
    floor: cq.Face,
    *,
    nose_pt: Vector,
    throat_left_pt: Vector,
) -> Tuple[cq.Edge, cq.Edge, cq.Edge, cq.Edge]:
    """Return ``(nose, right_rail, throat, left_rail)`` edges of a lofted floor."""
    edges = list(floor.Edges())
    if len(edges) != 4:
        raise RuntimeError(f"Expected 4 floor boundary edges, got {len(edges)}")
    by_len = sorted(edges, key=lambda e: e.Length())
    nose = by_len[0]
    throat = by_len[1]
    rails = by_len[2:]

    def _near(edge: cq.Edge, pt: Vector) -> float:
        return min(
            (edge.startPoint() - pt).Length,
            (edge.endPoint() - pt).Length,
        )

    if _near(rails[0], throat_left_pt) <= _near(rails[1], throat_left_pt):
        left_rail, right_rail = rails[0], rails[1]
    else:
        left_rail, right_rail = rails[1], rails[0]

    if _near(nose, nose_pt) > 1.0:
        raise RuntimeError("Could not identify the nose edge on the lofted floor")
    return nose, right_rail, throat, left_rail


def build_surface_inlet(
    host: cq.Workplane | cq.Shape,
    spec: InletSpec,
    placement: Placement,
) -> cq.Workplane:
    """Return the host faces with a NACA hole plus conforming duct faces."""
    face = placement.face
    placement = snap_placement_to_surface(face, spec, placement)
    frames = _centerline_frames(face, spec, placement)
    outward = placement.z_dir

    n_across = 11
    nose_hw = nose_half_width(spec)
    oml_left: List[Vector] = []
    oml_right: List[Vector] = []
    floor_left: List[Vector] = []
    floor_right: List[Vector] = []
    section_pts: List[List[Vector]] = []
    oml_rows: List[List[Vector]] = []

    for x, fr in frames:
        hw = planform_half_width(x, spec, min_width=nose_hw)
        depth = -ramp_floor_z(x, spec)
        row: List[Vector] = []
        oml_row: List[Vector] = []
        for k in range(n_across):
            y = -hw + (2.0 * hw) * k / (n_across - 1)
            oml, nrm = _sample_xy(face, fr, y, outward)
            floor_pt = _offset(oml, nrm, depth)
            oml_row.append(oml)
            row.append(floor_pt)
            if k == 0:
                oml_left.append(oml)
                floor_left.append(floor_pt)
            if k == n_across - 1:
                oml_right.append(oml)
                floor_right.append(floor_pt)
        section_pts.append(row)
        oml_rows.append(oml_row)

    oml_throat = oml_rows[-1]
    floor_throat = section_pts[-1]

    # Build the floor first so its boundary edges can be reused verbatim on the
    # side walls, throat, and skin-hole nose cap (clean quilt stitching).
    try:
        floor_faces = _loft_sections(section_pts)
        floor = floor_faces[0]
        if floor.Area() < 1.0:
            raise RuntimeError("degenerate floor")
    except Exception:
        floor = _floor_face(section_pts)

    nose_edge, right_rail, throat_edge, left_rail = _floor_boundary_edges(
        floor,
        nose_pt=section_pts[0][0],
        throat_left_pt=floor_left[-1],
    )

    oml_left_wire = _loft_wire(oml_left)
    oml_right_wire = _loft_wire(oml_right)
    oml_throat_wire = _loft_wire(oml_throat)

    left_rail = _orient_edge_like(oml_left_wire, left_rail)
    right_rail = _orient_edge_like(oml_right_wire, right_rail)
    throat_edge = _orient_edge_like(oml_throat_wire, throat_edge)

    outline = _closed_planform_wire(
        oml_left, oml_throat, oml_right, nose_edge=nose_edge
    )
    holed = _punch_hole(face, outline)
    try:
        if not holed.isValid():
            holed = holed.fix()
    except Exception:
        pass

    wall_l = cq.Face.makeRuledSurface(oml_left_wire, _wire_from_edge(left_rail))
    wall_r = cq.Face.makeRuledSurface(oml_right_wire, _wire_from_edge(right_rail))
    throat = cq.Face.makeRuledSurface(oml_throat_wire, _wire_from_edge(throat_edge))

    duct: List[cq.Face] = [floor, wall_l, wall_r, throat]
    for f in duct:
        if not f.isValid():
            f.fix()

    kept: List[cq.Shape] = []
    for other in _faces_of(host):
        if other.isSame(face):
            continue
        kept.append(other)
    kept.append(holed)
    kept.extend(duct)
    return cq.Workplane().newObject(kept)


def make_demo_surface(spec: InletSpec, *, radius: float | None = None) -> cq.Workplane:
    """Open cylindrical panel (single bounded face) large enough for ``spec``.

    Cylinder axis is world +X so default heading 0 flows along the panel.
    """
    if radius is None:
        radius = max(280.0, spec.width * 5.0)
    length = spec.cutout_length + spec.width + 160.0
    half_arc = spec.width / 2.0 + 80.0
    du = half_arc / radius
    ax = gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0), gp_Dir(0, 1, 0))
    cyl = Geom_CylindricalSurface(ax, radius)
    face = cq.Face(
        BRepBuilderAPI_MakeFace(
            cyl, -du, du, -length / 2.0, length / 2.0, 1e-6
        ).Face()
    )
    return cq.Workplane().newObject([face])


def demo_surface_location(spec: InletSpec, host: cq.Workplane | cq.Shape) -> Vector:
    """Throat location on the +Y crest of the demo cylinder."""
    bb = _as_workplane(host).val().BoundingBox()
    # Crest at max Y, mid X, z=0.
    return Vector(0.5 * (bb.xmin + bb.xmax), bb.ymax, 0.5 * (bb.zmin + bb.zmax))
