"""CadQuery / OpenCASCADE construction: build, place, and cut a NACA inlet."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cadquery as cq
from cadquery import Plane, Vector
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepLProp import BRepLProp_SLProps
from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCP.TopAbs import TopAbs_IN, TopAbs_REVERSED
from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec

from .geometry import (
    InletSpec,
    lip_lower_depth,
    planform_polyline,
    ramp_floor_z,
)

CQShape = cq.Workplane | cq.Shape | cq.Solid


@dataclass
class Placement:
    origin: Vector  # world location of the inlet tip (on the surface)
    x_dir: Vector  # downstream
    z_dir: Vector  # outward normal
    point_on_surface: Vector
    face: cq.Face
    y_dir: Vector = None  # type: ignore[assignment]  # lateral; set by apply_heading


def _as_workplane(obj: CQShape) -> cq.Workplane:
    if isinstance(obj, cq.Workplane):
        return obj
    return cq.Workplane().newObject([obj])


def _rotate_vector(v: Vector, axis: Vector, deg: float) -> Vector:
    if abs(deg) < 1e-12:
        return Vector(v)
    tr = gp_Trsf()
    tr.SetRotation(
        gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(axis.x, axis.y, axis.z)),
        math.radians(deg),
    )
    gv = gp_Vec(v.x, v.y, v.z).Transformed(tr)
    return Vector(gv.X(), gv.Y(), gv.Z())


def _project_onto_tangent(vec: Vector, normal: Vector) -> Vector:
    n = normal.normalized()
    tang = vec - n.multiply(vec.dot(n))
    if tang.Length < 1e-9:
        raise ValueError("Vector is parallel to the surface normal")
    return tang.normalized()


def _station_wire(x: float, width: float, z_bot: float, z_top: float) -> cq.Wire:
    height = z_top - z_bot
    if height < 0.25:
        z_top = z_bot + 0.25
        height = 0.25
    zc = 0.5 * (z_top + z_bot)
    return (
        cq.Workplane("YZ")
        .workplane(offset=x)
        .center(0, zc)
        .rect(width, height)
        .wires()
        .val()
    )


def _opening_cavity(spec: InletSpec) -> cq.Workplane:
    """Surface cutout: planform prism with a straight ramp floor."""
    poly = planform_polyline(spec, n=max(48, spec.n_stations))
    depth = spec.total_depth + max(4.0, spec.overshoot)
    prism = cq.Workplane("XY").polyline(poly).close().extrude(-depth)
    cap = cq.Workplane("XY").polyline(poly).close().extrude(spec.overshoot)
    body = prism.union(cap)
    tan_a = math.tan(spec.ramp_angle_rad)
    x_min, x_max = -0.5 * spec.cutout_length, spec.x_end + spec.cutout_length
    z_bot = -spec.total_depth * 12.0
    wedge = [
        (x_min, z_bot),
        (x_max, z_bot),
        (x_max, -x_max * tan_a - 0.05),
        (x_min, -x_min * tan_a - 0.05),
    ]
    cutter = (
        cq.Workplane("XZ")
        .polyline(wedge)
        .close()
        .extrude(spec.width * 4.0, both=True)
    )
    return body.cut(cutter)


def _throat_cavity(spec: InletSpec) -> cq.Workplane:
    """Internal duct from the lip LE aft, roof follows the blunt lip."""
    x0 = spec.cutout_length - 0.5
    n = max(16, spec.n_stations // 2)
    xs = [x0 + (spec.x_end - x0) * i / (n - 1) for i in range(n)]
    xs += [
        spec.cutout_length,
        spec.cutout_length + 0.094 * spec.lip_chord,
        spec.x_rotation,
        spec.cutout_length + spec.lip_chord,
        spec.x_end,
    ]
    xs = sorted({round(x, 5) for x in xs if x0 - 1e-9 <= x <= spec.x_end + 1e-9})
    wires: List[cq.Wire] = []
    for x in xs:
        z_bot = ramp_floor_z(max(x, 0.0), spec)
        s = (x - spec.cutout_length) / spec.lip_chord
        z_top = -lip_lower_depth(max(s, 0.0), spec)
        # Stay just below the OML so we do not recut the outer skin aft of the lip.
        z_top = min(z_top, -0.02)
        wires.append(_station_wire(x, spec.width, z_bot, z_top))
    return cq.Workplane().newObject([cq.Solid.makeLoft(wires)])


def build_cavity(spec: InletSpec) -> cq.Workplane:
    """Build the inlet cavity solid in local coordinates (tip at origin)."""
    opening = _opening_cavity(spec)
    throat = _throat_cavity(spec)
    try:
        fused = opening.union(throat)
        if fused.val().isValid() and fused.val().Volume() > 0.5 * opening.val().Volume():
            return fused
    except Exception:
        pass
    return cq.Workplane().newObject([opening.val(), throat.val()])


def place_cavity(cavity: cq.Workplane, placement: Placement) -> cq.Workplane:
    plane = Plane(
        origin=placement.origin,
        xDir=placement.x_dir,
        normal=placement.z_dir,
    )
    solids = _as_workplane(cavity).solids().vals()
    if not solids:
        raise RuntimeError("Inlet cavity is empty")
    shape: cq.Shape = (
        solids[0] if len(solids) == 1 else cq.Compound.makeCompound(solids)
    )
    located = shape.located(plane.location)
    return cq.Workplane().newObject([located])


def _faces_of(shape: CQShape) -> List[cq.Face]:
    return list(_as_workplane(shape).faces().vals())


def _solids_of(shape: CQShape) -> List[cq.Solid]:
    return list(_as_workplane(shape).solids().vals())


def closest_surface_point(
    host: CQShape,
    approx: Vector,
    *,
    flip_normal: bool = False,
) -> Placement:
    """Snap ``approx`` onto the nearest host face and build a local frame.

    Downstream (X) is world +X projected onto the tangent plane; the caller
    rotates that with ``heading_deg``.
    """
    vert = BRepBuilderAPI_MakeVertex(gp_Pnt(approx.x, approx.y, approx.z)).Vertex()
    best: Optional[Tuple[float, gp_Pnt, Vector, cq.Face]] = None

    for face in _faces_of(host):
        ext = BRepExtrema_DistShapeShape(vert, face.wrapped)
        ext.Perform()
        if ext.NbSolution() <= 0:
            continue
        dist = ext.Value()
        p2 = ext.PointOnShape2(1)
        surf = BRep_Tool.Surface_s(face.wrapped)
        proj = GeomAPI_ProjectPointOnSurf(p2, surf)
        if proj.NbPoints() <= 0:
            continue
        u, v = proj.LowerDistanceParameters()
        adaptor = BRepAdaptor_Surface(face.wrapped)
        sl = BRepLProp_SLProps(adaptor, u, v, 1, 1e-6)
        if not sl.IsNormalDefined():
            n_vec = Vector(face.normalAt())
        else:
            n = sl.Normal()
            n_vec = Vector(n.X(), n.Y(), n.Z())
            if face.wrapped.Orientation() == TopAbs_REVERSED:
                n_vec = n_vec.multiply(-1.0)
        if best is None or dist < best[0]:
            best = (dist, p2, n_vec, face)

    if best is None:
        raise RuntimeError("Could not project the location onto any face of the STEP model")

    _, p2, normal, face = best
    point = Vector(p2.X(), p2.Y(), p2.Z())
    if normal.Length < 1e-12:
        raise RuntimeError("Surface normal is undefined at the projected point")
    normal = normal.normalized()

    # For a solid, ensure Z is outward (pointing out of the material).
    solids = _solids_of(host)
    if solids:
        test = gp_Pnt(
            point.x + 0.5 * normal.x,
            point.y + 0.5 * normal.y,
            point.z + 0.5 * normal.z,
        )
        classifier = BRepClass3d_SolidClassifier(solids[0].wrapped)
        classifier.Perform(test, 1e-4)
        if classifier.State() == TopAbs_IN:
            normal = normal.multiply(-1.0)

    if flip_normal:
        normal = normal.multiply(-1.0)

    return Placement(
        origin=point,
        x_dir=Vector(1, 0, 0),  # filled in by apply_heading
        z_dir=normal,
        point_on_surface=point,
        face=face,
        y_dir=Vector(0, 1, 0),
    )


def apply_heading(placement: Placement, heading_deg: float) -> Placement:
    """Set downstream X by projecting world +X onto the tangent plane, then yaw."""
    n = placement.z_dir.normalized()
    try:
        x = _project_onto_tangent(Vector(1, 0, 0), n)
    except ValueError:
        x = _project_onto_tangent(Vector(0, 1, 0), n)
    x = _rotate_vector(x, n, heading_deg).normalized()
    placement.x_dir = x
    placement.y_dir = n.cross(x).normalized()
    return placement


def placement_for_location(
    host: CQShape,
    spec: InletSpec,
    location: Vector,
    heading_deg: float,
    *,
    location_ref: str = "throat",
    flip_normal: bool = False,
) -> Placement:
    """Build a placement whose origin is the inlet tip on the surface."""
    snap = closest_surface_point(host, location, flip_normal=flip_normal)
    snap = apply_heading(snap, heading_deg)
    if location_ref not in {"throat", "tip"}:
        raise ValueError("location_ref must be 'throat' or 'tip'")
    if location_ref == "throat":
        # The given point is the throat centre (aft end of the cutout).
        snap.origin = snap.point_on_surface - snap.x_dir.multiply(spec.cutout_length)
    else:
        snap.origin = snap.point_on_surface
    return snap


def thicken_if_surface(host: CQShape, thickness: float) -> cq.Workplane:
    """If the STEP has faces but no solid, thicken the faces into a solid skin."""
    wp = _as_workplane(host)
    solids = wp.solids().vals()
    if solids:
        return wp
    faces = wp.faces().vals()
    if not faces:
        raise RuntimeError("STEP file contains neither solids nor faces")
    if thickness <= 0:
        raise RuntimeError(
            "Imported STEP is a surface (no solid). Pass --skin-thickness "
            "to thicken it before cutting, or export a solid from CAD."
        )
    thickened: List[cq.Shape] = []
    for face in faces:
        thickened.append(face.thicken(thickness))
    if len(thickened) == 1:
        return cq.Workplane().newObject(thickened)
    fused = thickened[0]
    for extra in thickened[1:]:
        fused = fused.fuse(extra)
    return cq.Workplane().newObject([fused])


def _keep_major_solids(shape: cq.Shape, min_fraction: float = 1e-4) -> cq.Shape:
    solids = list(shape.Solids())
    if not solids:
        return shape
    solids.sort(key=lambda s: s.Volume(), reverse=True)
    threshold = max(solids[0].Volume() * min_fraction, 1.0)
    keep = [s for s in solids if s.Volume() >= threshold]
    if not keep:
        keep = [solids[0]]
    if len(keep) == 1:
        return keep[0]
    return cq.Compound.makeCompound(keep)


def cut_inlet(host: CQShape, cavity: cq.Workplane) -> cq.Workplane:
    host_wp = _as_workplane(host)
    solids = host_wp.solids().vals()
    if not solids:
        raise RuntimeError("Host has no solid to cut")
    cutters = _as_workplane(cavity).solids().vals()
    if not cutters:
        raise RuntimeError("Inlet cavity has no solid to cut with")
    result_solids = []
    for solid in solids:
        result: cq.Shape = solid
        try:
            for cutter in cutters:
                result = result.cut(cutter)
        except Exception as exc:
            raise RuntimeError(f"Boolean cut failed: {exc}") from exc
        result = _keep_major_solids(result)
        try:
            if not result.isValid():
                result = result.fix()
        except Exception:
            pass
        result_solids.append(result)
    return cq.Workplane().newObject(result_solids)


def make_demo_plate(
    spec: InletSpec,
    *,
    margin: float = 40.0,
    thickness: float | None = None,
) -> cq.Workplane:
    """A flat plate large enough to receive ``spec``, sitting on z=0 (top face)."""
    if thickness is None:
        thickness = max(spec.total_depth + spec.overshoot + 4.0, 8.0)
    length = spec.x_end + 2.0 * margin
    width = spec.width + 2.0 * margin
    # Top face at z=0, plate extends to −Z so the inlet cuts into it.
    return (
        cq.Workplane("XY")
        .center(spec.cutout_length * 0.5, 0)
        .box(length, width, thickness)
        .translate((0, 0, -thickness / 2.0))
    )


def import_step(path: str, unit: str = "MM") -> cq.Workplane:
    return cq.importers.importStep(path, unit=unit)  # type: ignore[arg-type]


def export_step(shape: CQShape, path: str) -> None:
    cq.exporters.export(_as_workplane(shape), path)


def export_stl(shape: CQShape, path: str, tolerance: float = 0.1) -> None:
    cq.exporters.export(_as_workplane(shape), path, tolerance=tolerance)
