"""Parametric NACA submerged-inlet (flush duct) geometry.

Sizing follows the construction method distilled from:

* Mossman & Randall, NACA RM-A7I30 (curved-divergence walls, 5–7° ramp,
  width-to-depth 3–5).
* Whitehouse, "NACA Submerged Duct Construction" (lip rotation point at
  0.75 chord, rotation-point depth 0.481 × entrance height, lip LE radius
  0.094 × entrance height).

Sidewall planform follows the NACA RM-A7I30 curved-divergence ordinates
(Pignier et al. 2016, Table 3; Whitehouse unit-width construction: scale
length by cutout length and width by throat width, curve through the nodes).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

# Whitehouse / NACA lip proportions, as fractions of entrance height h.
LIP_LE_RADIUS_FRAC = 0.094
LIP_ROTATION_STATION = 0.75
LIP_ROTATION_DEPTH_FRAC = 0.481
LIP_AFT_DEPTH_FRAC = 0.50  # continue until thickness is 1/2 chord, then flat

Point2 = Tuple[float, float]


def smoothstep(s: float) -> float:
    """Cubic Hermite S-curve: 0 at s=0, 1 at s=1, zero slope at both ends."""
    s = max(0.0, min(1.0, s))
    return s * s * (3.0 - 2.0 * s)


# Pignier (2016) Table 3, restated from the tip. Published as x/L, y/L with
# x/L = 0 at the throat; y = 0.500 there is the half-width / throat-width
# (Whitehouse "width column" × duct width). x/L = 1 is the finite nose.
NACA_CURVED_DIVERGENCE: Tuple[Tuple[float, float], ...] = (
    (0.00, 0.042),
    (0.10, 0.080),
    (0.20, 0.118),
    (0.30, 0.157),
    (0.40, 0.195),
    (0.50, 0.233),
    (0.60, 0.307),
    (0.70, 0.382),
    (0.80, 0.457),
    (0.90, 0.497),
    (1.00, 0.500),
)
NACA_TIP_HALF_WIDTH_FRAC = NACA_CURVED_DIVERGENCE[0][1]
NACA_THROAT_HALF_WIDTH_FRAC = NACA_CURVED_DIVERGENCE[-1][1]


def _pchip_slopes(xs: Sequence[float], ys: Sequence[float]) -> List[float]:
    """Fritsch–Carlson monotone cubic slopes; interpolant hits every knot."""
    n = len(xs)
    h = [xs[i + 1] - xs[i] for i in range(n - 1)]
    delta = [(ys[i + 1] - ys[i]) / h[i] for i in range(n - 1)]
    m = [0.0] * n
    m[0] = delta[0]
    m[-1] = delta[-1]
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0.0:
            m[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
    return m


_NACA_S = [p[0] for p in NACA_CURVED_DIVERGENCE]
_NACA_Y = [p[1] for p in NACA_CURVED_DIVERGENCE]
_NACA_M = _pchip_slopes(_NACA_S, _NACA_Y)


def naca_wall_half_width_frac(s_from_tip: float) -> float:
    """Half-width / throat-width at station s (0 = tip, 1 = throat)."""
    s = max(0.0, min(1.0, s_from_tip))
    xs, ys, ms = _NACA_S, _NACA_Y, _NACA_M
    if s <= xs[0]:
        return ys[0]
    if s >= xs[-1]:
        return ys[-1]
    i = 0
    while i < len(xs) - 2 and s > xs[i + 1]:
        i += 1
    x1, x2 = xs[i], xs[i + 1]
    h = x2 - x1
    t = (s - x1) / h
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return h00 * ys[i] + h10 * h * ms[i] + h01 * ys[i + 1] + h11 * h * ms[i + 1]


def hermite_zero_slope(t: float, a: float, b: float) -> float:
    """Interpolate a→b on t∈[0,1] with zero endpoint derivatives."""
    t = max(0.0, min(1.0, t))
    t2 = t * t
    t3 = t2 * t
    return (2.0 * t3 - 3.0 * t2 + 1.0) * a + (-2.0 * t3 + 3.0 * t2) * b


@dataclass(frozen=True)
class InletSpec:
    """Fully resolved NACA inlet dimensions, in millimetres unless noted.

    Local coordinates used by the builder:
      X  downstream (flow direction), 0 at the forward tip
      Y  lateral, 0 on the centreline
      Z  outward surface normal, 0 on the host surface; cavity is −Z
    """

    area: float
    width: float
    height: float
    aspect_ratio: float
    ramp_angle_deg: float
    cutout_length: float
    rotation_length: float
    total_depth: float
    lip_chord: float
    duct_extension: float
    overshoot: float
    n_stations: int
    min_half_width: float

    @property
    def ramp_angle_rad(self) -> float:
        return math.radians(self.ramp_angle_deg)

    @property
    def x_lip(self) -> float:
        return self.cutout_length

    @property
    def x_rotation(self) -> float:
        return self.rotation_length

    @property
    def x_end(self) -> float:
        return self.cutout_length + self.lip_chord + self.duct_extension

    @property
    def throat_area(self) -> float:
        return self.width * self.height


def resolve_spec(
    *,
    area: float | None = None,
    width: float | None = None,
    height: float | None = None,
    aspect_ratio: float = 4.0,
    ramp_angle_deg: float = 7.0,
    duct_extension: float | None = None,
    overshoot: float | None = None,
    n_stations: int = 48,
    min_half_width: float = 0.4,
) -> InletSpec:
    """Derive a complete inlet from area and/or width/height.

    Priority:
      1. width and height both given → area = width × height
      2. area and width given → height = area / width
      3. area and height given → width = area / height
      4. area (and aspect ratio) → height = sqrt(area / AR), width = AR × height
    """
    if aspect_ratio <= 0:
        raise ValueError("aspect_ratio must be positive")
    if not (2.0 <= ramp_angle_deg <= 20.0):
        raise ValueError("ramp_angle_deg should be in the range 2–20 (NACA optimum is 5–7)")

    if width is not None and height is not None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        area = width * height
        aspect_ratio = width / height
    elif area is not None and width is not None:
        if area <= 0 or width <= 0:
            raise ValueError("area and width must be positive")
        height = area / width
        aspect_ratio = width / height
    elif area is not None and height is not None:
        if area <= 0 or height <= 0:
            raise ValueError("area and height must be positive")
        width = area / height
        aspect_ratio = width / height
    elif area is not None:
        if area <= 0:
            raise ValueError("area must be positive")
        height = math.sqrt(area / aspect_ratio)
        width = aspect_ratio * height
    else:
        raise ValueError("Provide --area, or both --width and --height")

    # Whitehouse length from rotation point to tip.
    total_depth = (1.0 + LIP_ROTATION_DEPTH_FRAC) * height
    ramp_rad = math.radians(ramp_angle_deg)
    tan_a = math.tan(ramp_rad)
    if tan_a <= 1e-9:
        raise ValueError("ramp angle is too small")
    rotation_length = total_depth / tan_a
    lip_chord = height  # Whitehouse: airfoil chord equals entrance height
    cutout_length = rotation_length - LIP_ROTATION_STATION * lip_chord
    if cutout_length <= 0:
        raise ValueError(
            "Cutout length is non-positive; decrease ramp angle or entrance height"
        )

    if duct_extension is None:
        duct_extension = max(height, 0.25 * width)
    if overshoot is None:
        overshoot = max(2.0, 0.15 * height)

    return InletSpec(
        area=width * height,
        width=width,
        height=height,
        aspect_ratio=width / height,
        ramp_angle_deg=ramp_angle_deg,
        cutout_length=cutout_length,
        rotation_length=rotation_length,
        total_depth=total_depth,
        lip_chord=lip_chord,
        duct_extension=duct_extension,
        overshoot=overshoot,
        n_stations=max(16, int(n_stations)),
        min_half_width=min_half_width,
    )


def planform_half_width(
    x: float, spec: InletSpec, *, min_width: float | None = None
) -> float:
    """Half-width of the surface cutout at station x (from the tip).

    Interpolates the NACA curved-divergence table (through every published
    station). ``min_width`` is a CAD floor for the blunt leading edge;
    defaults to ``spec.min_half_width``.
    """
    floor_w = spec.min_half_width if min_width is None else min_width
    if spec.cutout_length <= 1e-12 or x >= spec.cutout_length:
        return spec.width / 2.0
    hw = naca_wall_half_width_frac(x / spec.cutout_length) * spec.width
    return max(floor_w, hw)


def nose_half_width(spec: InletSpec) -> float:
    """Leading-edge half-width from the NACA table (0.042 × throat width)."""
    return max(spec.min_half_width, NACA_TIP_HALF_WIDTH_FRAC * spec.width)


# Fraction of cutout length used to fair the ramp onto the skin (G1 at the nose).
FLUSH_BLEND_FRAC = 0.15


def ramp_floor_z(x: float, spec: InletSpec, *, flush: bool = False) -> float:
    """Z of the ramp floor (negative, into the body).

    Straight ramp by default. With ``flush=True`` the depth and slope are
    zero at the leading edge so the floor lies on the skin and leaves it
    tangentially, then matches the design ramp angle after a short blend.
    """
    if x <= 0.0:
        return 0.0
    z_lin = -x * math.tan(spec.ramp_angle_rad)
    z_lin = max(z_lin, -spec.total_depth)
    if not flush:
        return z_lin
    blend = FLUSH_BLEND_FRAC * spec.cutout_length
    if x >= blend:
        return z_lin
    return z_lin * smoothstep(x / blend)


def lip_lower_depth(s: float, spec: InletSpec) -> float:
    """Inward depth of the aft-lip lower surface.

    ``s`` is distance downstream of the lip leading edge, divided by lip
    chord (entrance height). Depth is 0 at the surface.
    """
    h = spec.height
    r = LIP_LE_RADIUS_FRAC
    if s <= 0.0:
        return 0.0
    if s <= r:
        # Circle centred at (r, 0) with radius r: vertical tangent at the LE.
        return h * math.sqrt(max(0.0, r * r - (s - r) ** 2))
    if s >= 1.0:
        return LIP_AFT_DEPTH_FRAC * h
    if s <= LIP_ROTATION_STATION:
        t = (s - r) / (LIP_ROTATION_STATION - r)
        return hermite_zero_slope(t, r * h, LIP_ROTATION_DEPTH_FRAC * h)
    t = (s - LIP_ROTATION_STATION) / (1.0 - LIP_ROTATION_STATION)
    return hermite_zero_slope(t, LIP_ROTATION_DEPTH_FRAC * h, LIP_AFT_DEPTH_FRAC * h)


def planform_polyline(spec: InletSpec, n: int = 80) -> List[Point2]:
    """Closed planform polyline (tip → left wall → throat → right wall)."""
    nose = nose_half_width(spec)
    xs = [spec.cutout_length * i / (n - 1) for i in range(n)]
    left = [(x, -planform_half_width(x, spec, min_width=nose)) for x in xs]
    right = [(x, planform_half_width(x, spec, min_width=nose)) for x in reversed(xs)]
    return left + right


def side_profile(spec: InletSpec, n: int = 80) -> Tuple[List[Point2], List[Point2]]:
    """Centreline side view: (floor polyline, roof polyline) in XZ."""
    xs = [spec.x_end * i / (n - 1) for i in range(n)]
    floor = [(x, ramp_floor_z(x, spec)) for x in xs]
    roof: List[Point2] = []
    for x in xs:
        if x <= spec.cutout_length:
            roof.append((x, 0.0))
        else:
            s = (x - spec.cutout_length) / spec.lip_chord
            roof.append((x, -lip_lower_depth(s, spec)))
    return floor, roof


def spec_summary(spec: InletSpec) -> str:
    lines = [
        f"Throat area        {spec.area:.3f} mm²",
        f"Throat width       {spec.width:.3f} mm",
        f"Throat height      {spec.height:.3f} mm",
        f"Aspect ratio W/H   {spec.aspect_ratio:.3f}",
        f"Ramp angle         {spec.ramp_angle_deg:.2f} deg",
        f"Cutout length      {spec.cutout_length:.3f} mm  (tip → lip LE)",
        f"Rotation length    {spec.rotation_length:.3f} mm  (tip → 0.75c)",
        f"Ramp depth         {spec.total_depth:.3f} mm",
        f"Lip chord          {spec.lip_chord:.3f} mm",
        f"Duct extension     {spec.duct_extension:.3f} mm",
        f"Loft stations      {spec.n_stations}  (nose → throat sections)",
    ]
    return "\n".join(lines)
