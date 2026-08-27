"""Tkinter GUI for placing a parametric NACA inlet on a STEP surface."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from tkinter import (
    BooleanVar,
    Canvas,
    DoubleVar,
    Frame,
    IntVar,
    StringVar,
    Text,
    Tk,
    filedialog,
    messagebox,
    ttk,
)

from cadquery import Vector

from .cad import (
    build_cavity,
    cut_inlet,
    export_step,
    import_step,
    make_demo_plate,
    place_cavity,
    placement_for_location,
    thicken_if_surface,
)
from .geometry import InletSpec, planform_polyline, resolve_spec, side_profile, spec_summary
from .surface import build_surface_inlet, demo_surface_location, make_demo_surface


class NacaInletApp:
    def __init__(self, root: Tk, args: argparse.Namespace | None = None) -> None:
        self.root = root
        root.title("NACA Inlet — STEP")
        root.minsize(820, 640)

        args = args or argparse.Namespace()

        self.input_path = StringVar(value=getattr(args, "input", None) or "")
        self.output_path = StringVar(value=getattr(args, "output", None) or "")
        self.area = DoubleVar(value=getattr(args, "area", None) or 4000.0)
        self.width = DoubleVar(value=getattr(args, "width", None) or 0.0)
        self.height = DoubleVar(value=getattr(args, "height", None) or 0.0)
        self.aspect = DoubleVar(value=getattr(args, "aspect", None) or 4.0)
        self.ramp = DoubleVar(value=getattr(args, "ramp_angle", None) or 7.0)
        self.stations = IntVar(value=int(getattr(args, "stations", None) or 48))
        self.loc_x = DoubleVar(value=0.0)
        self.loc_y = DoubleVar(value=0.0)
        self.loc_z = DoubleVar(value=0.0)
        if getattr(args, "location", None):
            self.loc_x.set(args.location[0])
            self.loc_y.set(args.location[1])
            self.loc_z.set(args.location[2])
        self.heading = DoubleVar(value=getattr(args, "heading", None) or 0.0)
        self.location_ref = StringVar(value=getattr(args, "location_ref", None) or "throat")
        self.flip_normal = BooleanVar(value=bool(getattr(args, "flip_normal", False)))
        self.demo_plate = BooleanVar(value=bool(getattr(args, "demo", False)))
        self.skin = DoubleVar(value=getattr(args, "skin_thickness", None) or 0.0)
        self.use_wh = BooleanVar(value=False)
        self.mode = StringVar(value=getattr(args, "mode", None) or "surfaces")

        self._build_form()
        self._redraw_preview()

    def _build_form(self) -> None:
        pad = {"padx": 8, "pady": 4}
        files = ttk.LabelFrame(self.root, text="Files")
        files.pack(fill="x", **pad)
        self._path_row(files, "Input STEP", self.input_path, self._browse_in, 0)
        self._path_row(files, "Output STEP", self.output_path, self._browse_out, 1)
        ttk.Checkbutton(
            files,
            text="No input — use a sample host",
            variable=self.demo_plate,
            command=self._redraw_preview,
        ).grid(row=2, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(files, text="Result").grid(row=3, column=0, sticky="e", padx=4)
        ttk.Combobox(
            files,
            textvariable=self.mode,
            values=("surfaces", "solid"),
            width=12,
            state="readonly",
        ).grid(row=3, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(files, text="surfaces = hole + conforming duct faces").grid(
            row=3, column=2, sticky="w"
        )

        size = ttk.LabelFrame(self.root, text="Inlet size")
        size.pack(fill="x", **pad)
        self._spin(size, "Throat area (mm²)", self.area, 10, 1e7, 100, 0, 0)
        self._spin(size, "Aspect ratio W/H", self.aspect, 1.5, 10, 0.1, 0, 2)
        self._spin(size, "Ramp angle (deg)", self.ramp, 3, 15, 0.5, 1, 0)
        self._ispin(size, "Loft stations", self.stations, 16, 200, 4, 1, 2)
        ttk.Label(
            size,
            text=(
                "Cross-sections from nose to throat used to loft the ramp floor and "
                "trace the side walls on curved skins. Default 48; use more on tight "
                "curvature, fewer for faster builds."
            ),
            wraplength=520,
            justify="left",
        ).grid(row=2, column=0, columnspan=5, sticky="w", padx=8, pady=(0, 4))
        ttk.Checkbutton(
            size,
            text="Size from width × height instead of area",
            variable=self.use_wh,
            command=self._redraw_preview,
        ).grid(row=3, column=2, columnspan=2, sticky="w", padx=8)
        self._spin(size, "Width (mm)", self.width, 1, 1e5, 1, 4, 0)
        self._spin(size, "Height (mm)", self.height, 1, 1e5, 0.5, 4, 2)

        place = ttk.LabelFrame(self.root, text="Location on surface")
        place.pack(fill="x", **pad)
        self._spin(place, "X (mm)", self.loc_x, -1e6, 1e6, 1, 0, 0)
        self._spin(place, "Y (mm)", self.loc_y, -1e6, 1e6, 1, 0, 2)
        self._spin(place, "Z (mm)", self.loc_z, -1e6, 1e6, 1, 0, 4)
        self._spin(place, "Heading (deg)", self.heading, -180, 180, 1, 1, 0)
        ttk.Label(place, text="Location is").grid(row=1, column=2, sticky="e", padx=4)
        ttk.Combobox(
            place,
            textvariable=self.location_ref,
            values=("throat", "tip"),
            width=10,
            state="readonly",
        ).grid(row=1, column=3, sticky="w")
        ttk.Checkbutton(place, text="Flip normal", variable=self.flip_normal).grid(
            row=1, column=4, sticky="w", padx=8
        )
        self._spin(place, "Skin thicken (mm)", self.skin, 0, 1e4, 0.5, 1, 5)

        for var in (
            self.area,
            self.aspect,
            self.ramp,
            self.width,
            self.height,
            self.stations,
            self.use_wh,
        ):
            var.trace_add("write", lambda *_: self._redraw_preview())

        mid = Frame(self.root)
        mid.pack(fill="both", expand=True, **pad)
        preview = ttk.LabelFrame(mid, text="Planform and side view")
        preview.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self.canvas = Canvas(preview, background="#111418", height=280)
        self.canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.canvas.bind("<Configure>", lambda _e: self._redraw_preview())

        dims = ttk.LabelFrame(mid, text="Derived dimensions")
        dims.pack(side="right", fill="y")
        self.dim_text = Text(dims, width=36, height=16, wrap="word", state="disabled")
        self.dim_text.pack(fill="both", expand=True, padx=4, pady=4)

        btns = Frame(self.root)
        btns.pack(fill="x", **pad)
        ttk.Button(btns, text="Generate STEP", command=self._generate).pack(
            side="left", padx=4
        )
        ttk.Button(btns, text="Quit", command=self.root.destroy).pack(side="right", padx=4)

        self.log = Text(self.root, height=8, wrap="word")
        self.log.pack(fill="x", padx=8, pady=(0, 8))
        self._log("Ready. Set inlet area and a location, then Generate STEP.")

    def _path_row(self, parent, label, var, cmd, row) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="e", padx=4, pady=2)
        ttk.Entry(parent, textvariable=var, width=64).grid(
            row=row, column=1, sticky="ew", padx=4, pady=2
        )
        ttk.Button(parent, text="Browse", command=cmd).grid(row=row, column=2, padx=4)
        parent.columnconfigure(1, weight=1)

    def _spin(self, parent, label, var, lo, hi, inc, row, col) -> None:
        ttk.Label(parent, text=label).grid(
            row=row, column=col, sticky="e", padx=4, pady=2
        )
        ttk.Spinbox(
            parent,
            textvariable=var,
            from_=lo,
            to=hi,
            increment=inc,
            width=12,
        ).grid(row=row, column=col + 1, sticky="w", padx=4, pady=2)

    def _ispin(self, parent, label, var, lo, hi, inc, row, col) -> None:
        ttk.Label(parent, text=label).grid(
            row=row, column=col, sticky="e", padx=4, pady=2
        )
        ttk.Spinbox(
            parent,
            textvariable=var,
            from_=lo,
            to=hi,
            increment=inc,
            width=12,
        ).grid(row=row, column=col + 1, sticky="w", padx=4, pady=2)

    def _browse_in(self) -> None:
        path = filedialog.askopenfilename(
            title="Input STEP",
            filetypes=[("STEP", "*.step *.stp *.STEP *.STP"), ("All", "*")],
        )
        if path:
            self.input_path.set(path)
            if not self.output_path.get():
                p = Path(path)
                self.output_path.set(str(p.with_name(p.stem + "_naca.step")))

    def _browse_out(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Output STEP",
            defaultextension=".step",
            filetypes=[("STEP", "*.step *.stp"), ("All", "*")],
        )
        if path:
            self.output_path.set(path)

    def _current_spec(self) -> InletSpec:
        stations = max(16, int(self.stations.get()))
        if self.use_wh.get():
            return resolve_spec(
                width=float(self.width.get()),
                height=float(self.height.get()),
                ramp_angle_deg=float(self.ramp.get()),
                n_stations=stations,
            )
        return resolve_spec(
            area=float(self.area.get()),
            aspect_ratio=float(self.aspect.get()),
            ramp_angle_deg=float(self.ramp.get()),
            n_stations=stations,
        )

    def _set_dims(self, text: str) -> None:
        self.dim_text.configure(state="normal")
        self.dim_text.delete("1.0", "end")
        self.dim_text.insert("1.0", text)
        self.dim_text.configure(state="disabled")

    def _log(self, msg: str) -> None:
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.root.update_idletasks()

    def _redraw_preview(self) -> None:
        try:
            spec = self._current_spec()
        except Exception as exc:
            self._set_dims(str(exc))
            return
        self._set_dims(spec_summary(spec))
        c = self.canvas
        c.delete("all")
        w = max(c.winfo_width(), 200)
        h = max(c.winfo_height(), 160)
        c.create_line(w // 2, 8, w // 2, h - 8, fill="#2a3138")
        self._draw_plan(c, spec, 8, 8, w // 2 - 16, h - 16)
        self._draw_side(c, spec, w // 2 + 8, 8, w // 2 - 16, h - 16)

    def _fit(self, pts, x0, y0, box_w, box_h, y_sign=-1):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        span_x = max(maxx - minx, 1e-6)
        span_y = max(maxy - miny, 1e-6)
        scale = 0.86 * min(box_w / span_x, box_h / span_y)
        cx = 0.5 * (minx + maxx)
        cy = 0.5 * (miny + maxy)
        mx = x0 + box_w / 2
        my = y0 + box_h / 2

        def xf(p):
            return (
                mx + (p[0] - cx) * scale,
                my + y_sign * (p[1] - cy) * scale,
            )

        return xf, scale

    def _draw_plan(self, c: Canvas, spec: InletSpec, x0, y0, bw, bh) -> None:
        poly = planform_polyline(spec)
        xf, _ = self._fit(poly, x0, y0, bw, bh, y_sign=-1)
        flat = [coord for p in poly for coord in xf(p)]
        c.create_polygon(flat, outline="#7ec8e3", fill="#1b3a4b", width=2)
        c.create_text(
            x0 + 8, y0 + 8, anchor="nw", fill="#9aa7b2", text="Plan (flow →)"
        )

    def _draw_side(self, c: Canvas, spec: InletSpec, x0, y0, bw, bh) -> None:
        floor, roof = side_profile(spec)
        pts = floor + list(reversed(roof))
        xf, _ = self._fit(pts, x0, y0, bw, bh, y_sign=-1)
        # Surface line z=0
        z0a = xf((0.0, 0.0))
        z0b = xf((spec.x_end, 0.0))
        c.create_line(*z0a, *z0b, fill="#5c6770", dash=(4, 3))
        floor_xy = [coord for p in floor for coord in xf(p)]
        roof_xy = [coord for p in roof for coord in xf(p)]
        c.create_line(*floor_xy, fill="#e0a458", width=2)
        c.create_line(*roof_xy, fill="#7ec8e3", width=2)
        c.create_text(
            x0 + 8, y0 + 8, anchor="nw", fill="#9aa7b2", text="Side (inward down)"
        )

    def _generate(self) -> None:
        try:
            spec = self._current_spec()
        except Exception as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return
        out = self.output_path.get().strip()
        if not out:
            messagebox.showerror("Output", "Choose an output STEP path.")
            return
        self._log("Building NACA inlet…")
        try:
            if self.demo_plate.get() or not self.input_path.get().strip():
                if self.mode.get() == "solid":
                    self._log("Using demo plate (solid cut).")
                    host = make_demo_plate(spec)
                    location = Vector(spec.cutout_length, 0, 0)
                else:
                    self._log("Using demo curved panel (surfaces).")
                    host = make_demo_surface(spec)
                    location = demo_surface_location(spec, host)
                self.location_ref.set("throat")
                self.loc_x.set(location.x)
                self.loc_y.set(location.y)
                self.loc_z.set(location.z)
            else:
                self._log(f"Importing {self.input_path.get()}…")
                host = import_step(self.input_path.get())
                if self.mode.get() == "solid":
                    host = thicken_if_surface(host, float(self.skin.get()))
                location = Vector(
                    float(self.loc_x.get()),
                    float(self.loc_y.get()),
                    float(self.loc_z.get()),
                )
            self._log("Placing on surface…")
            placement = placement_for_location(
                host,
                spec,
                location,
                float(self.heading.get()),
                location_ref=self.location_ref.get(),
                flip_normal=self.flip_normal.get(),
            )
            if self.mode.get() == "surfaces":
                self._log("Draping inlet faces (conforming to curvature)…")
                result = build_surface_inlet(host, spec, placement)
                self._log(f"{len(result.faces().vals())} faces")
            else:
                self._log("Boolean cut…")
                cavity_local = build_cavity(spec)
                cavity = place_cavity(cavity_local, placement)
                result = cut_inlet(host, cavity)
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            export_step(result, out)
            self._log(f"Wrote {out}")
            self._log(spec_summary(spec))
            messagebox.showinfo("Done", f"Wrote\n{out}")
        except Exception as exc:
            self._log(traceback.format_exc())
            messagebox.showerror("Generate failed", str(exc))


def launch_gui(args: argparse.Namespace | None = None) -> None:
    root = Tk()
    # Dark-ish ttk on systems that support it; ignore failures.
    try:
        style = ttk.Style(root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    NacaInletApp(root, args)
    root.mainloop()
