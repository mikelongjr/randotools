"""Unit tests for NACA inlet sizing (no CAD kernel required)."""

from __future__ import annotations

import math
import unittest

from naca_inlet.geometry import (
    LIP_ROTATION_DEPTH_FRAC,
    LIP_ROTATION_STATION,
    NACA_CURVED_DIVERGENCE,
    NACA_TIP_HALF_WIDTH_FRAC,
    naca_wall_half_width_frac,
    nose_half_width,
    planform_half_width,
    ramp_floor_z,
    resolve_spec,
)


class ResolveSpecTests(unittest.TestCase):
    def test_whitehouse_example_inches_as_mm(self) -> None:
        # 1 in height, 4 in width, 7 deg ramp → cutout 11.312 in from lip LE to tip.
        inch = 25.4
        spec = resolve_spec(width=4.0 * inch, height=1.0 * inch, ramp_angle_deg=7.0)
        self.assertAlmostEqual(spec.area, 4.0 * inch * inch, places=6)
        self.assertAlmostEqual(spec.total_depth / spec.height, 1.0 + LIP_ROTATION_DEPTH_FRAC)
        self.assertAlmostEqual(spec.cutout_length / inch, 11.312, places=2)
        self.assertAlmostEqual(spec.rotation_length / inch, 12.062, places=2)

    def test_area_and_aspect(self) -> None:
        spec = resolve_spec(area=4000.0, aspect_ratio=4.0)
        self.assertAlmostEqual(spec.width / spec.height, 4.0)
        self.assertAlmostEqual(spec.width * spec.height, 4000.0, places=6)

    def test_area_and_width(self) -> None:
        spec = resolve_spec(area=4000.0, width=80.0)
        self.assertAlmostEqual(spec.height, 50.0)
        self.assertAlmostEqual(spec.aspect_ratio, 1.6)

    def test_planform_endpoints(self) -> None:
        spec = resolve_spec(area=2000.0, aspect_ratio=4.0)
        self.assertAlmostEqual(
            planform_half_width(0.0, spec), NACA_TIP_HALF_WIDTH_FRAC * spec.width
        )
        self.assertAlmostEqual(
            planform_half_width(spec.cutout_length, spec), spec.width / 2.0
        )

    def test_naca_wall_ordinates(self) -> None:
        spec = resolve_spec(width=40.0, height=10.0)
        nose = nose_half_width(spec)
        for s, y_over_w in NACA_CURVED_DIVERGENCE:
            x = s * spec.cutout_length
            self.assertAlmostEqual(
                naca_wall_half_width_frac(s), y_over_w, places=9
            )
            self.assertAlmostEqual(
                planform_half_width(x, spec, min_width=nose),
                y_over_w * spec.width,
                places=6,
            )
        # Forward stations are wider than a cubic S-curve (NACA blunt nose).
        s = 0.2
        naca = naca_wall_half_width_frac(s)
        smooth = s * s * (3.0 - 2.0 * s) * 0.5
        self.assertGreater(naca, smooth)

    def test_ramp_matches_angle(self) -> None:
        spec = resolve_spec(height=10.0, width=40.0, ramp_angle_deg=7.0)
        expected = spec.total_depth / math.tan(math.radians(7.0))
        self.assertAlmostEqual(spec.rotation_length, expected)
        self.assertAlmostEqual(
            spec.cutout_length,
            spec.rotation_length - LIP_ROTATION_STATION * spec.lip_chord,
        )

    def test_ramp_angle_from_nose(self) -> None:
        spec = resolve_spec(height=10.0, width=40.0, ramp_angle_deg=7.0)
        self.assertEqual(ramp_floor_z(0.0, spec), 0.0)
        x = 0.02 * spec.cutout_length
        z = ramp_floor_z(x, spec)
        expected = -x * math.tan(math.radians(7.0))
        self.assertAlmostEqual(z, expected)
        # Straight ramp is shallower than the old tangent blend near the nose.
        blend = 0.15 * spec.cutout_length
        z_blend = expected * (3.0 * (x / blend) ** 2 - 2.0 * (x / blend) ** 3)
        self.assertLess(z, z_blend)
        self.assertAlmostEqual(
            nose_half_width(spec), NACA_TIP_HALF_WIDTH_FRAC * spec.width
        )
        self.assertLess(2.0 * nose_half_width(spec), 0.2 * spec.width)


if __name__ == "__main__":
    unittest.main()
