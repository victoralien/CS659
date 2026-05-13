import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

# US coin diameters in mm (approx) and values in USD
COIN_SPECS = {
    "penny":  {"diam_mm": 19.05, "value": 0.01},
    "nickel": {"diam_mm": 21.21, "value": 0.05},
    "dime":   {"diam_mm": 17.91, "value": 0.10},
    "quarter":{"diam_mm": 24.26, "value": 0.25},
}
COIN_ORDER = ["dime", "penny", "nickel", "quarter"]

def preprocess(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (9, 9), 1.5)
    return gray

def detect_circles(gray: np.ndarray, dp: float, min_dist: int, p1: int, p2: int, min_r: int, max_r: int):
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=p1,
        param2=p2,
        minRadius=min_r,
        maxRadius=max_r
    )
    if circles is None:
        return []
    circles = np.round(circles[0, :]).astype(int)
    return [(int(x), int(y), int(r)) for x, y, r in circles]

def estimate_scale(diams_px: np.ndarray):
    """
    Infer mm/px by trying candidate scales derived from mapping each observed diameter
    to each coin type. Choose the scale that minimizes median absolute error to reduce
    sensitivity to a single bad circle.
    """
    coin_types = list(COIN_SPECS.keys())
    diam_mm = np.array([COIN_SPECS[c]["diam_mm"] for c in coin_types], dtype=float)

    best = None
    for dpx in diams_px:
        for dmm in diam_mm:
            mm_per_px = dmm / max(dpx, 1e-6)
            est_mm = diams_px * mm_per_px
            idx = np.argmin((est_mm[:, None] - diam_mm[None, :]) ** 2, axis=1)
            err = float(np.median(np.abs(est_mm - diam_mm[idx])))

            # mild penalty if resulting sizes are implausible for US coins
            if np.any(est_mm < 15) or np.any(est_mm > 35):
                err *= 1.2

            if best is None or err < best[0]:
                best = (err, mm_per_px, idx)

    if best is None:
        return None

    _, mm_per_px, _ = best
    return mm_per_px


def assign_coins(
    diams_px: np.ndarray,
    mm_per_px: float,
    size_tol_mm: float = 0.6,
    margin_mm: float = 0.35,
    margin_score: float = 0.05,
    area_weight: float = 0.45,
    rel_tol: float = 0.08,
):
    """
    Classify each diameter (in px) using the estimated scale.
    Reject if size is implausible, if the closest match is still too far,
    or if the gap to the second-best match is too small (ambiguous).
    Uses diameter as primary signal with area as a secondary cue.
    """
    coin_types = list(COIN_SPECS.keys())
    diam_mm = np.array([COIN_SPECS[c]["diam_mm"] for c in coin_types], dtype=float)
    area_mm2 = np.pi * (diam_mm / 2.0) ** 2

    assigned = []
    for dpx in diams_px:
        mm = dpx * mm_per_px
        area = np.pi * (mm / 2.0) ** 2

        if mm < 15 or mm > 35:
            assigned.append("unknown")
            continue

        diff_d = np.abs(mm - diam_mm)
        diff_a = np.abs(area - area_mm2)
        rel_d = diff_d / diam_mm
        rel_a = diff_a / area_mm2

        score = (1.0 - area_weight) * rel_d + area_weight * rel_a

        best_idx = int(np.argmin(score))
        best_score = float(score[best_idx])
        best_diff_d = float(diff_d[best_idx])
        best_rel_d = float(rel_d[best_idx])
        if len(score) > 1:
            second_score = float(np.partition(score, 1)[1])
        else:
            second_score = float("inf")
        gap_score = second_score - best_score

        if best_diff_d > size_tol_mm or best_rel_d > rel_tol or gap_score < margin_score:
            assigned.append("unknown")
        else:
            assigned.append(coin_types[best_idx])

    return assigned

def analyze_image(bgr: np.ndarray, settings: dict):
    gray = preprocess(bgr)
    circles = detect_circles(
        gray,
        dp=settings["dp"],
        min_dist=settings["min_dist"],
        p1=settings["param1"],
        p2=settings["param2"],
        min_r=settings["min_radius"],
        max_r=settings["max_radius"],
    )

    assigned = []
    mm_per_px = None
    if circles:
        diams_px = np.array([2.0 * r for (_, _, r) in circles], dtype=float)
        mm_per_px = estimate_scale(diams_px)
        if mm_per_px is None:
            assigned = ["unknown"] * len(circles)
        else:
            assigned = assign_coins(diams_px, mm_per_px)

    annotated = bgr.copy()
    for (x, y, r), label in zip(circles, assigned):
        is_unknown = (label == "unknown")
        color = (0, 0, 255) if is_unknown else (0, 200, 0)  # red for unknown, green for labeled
        cv2.circle(annotated, (x, y), r, color, 2)
        cv2.circle(annotated, (x, y), 2, color, 3)
        cv2.putText(
            annotated, label, (x - r, y - r - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
        )

    counts = {k: 0 for k in COIN_SPECS.keys()}
    for a in assigned:
        if a in counts:
            counts[a] += 1
    unknown_count = sum(1 for a in assigned if a == "unknown")
    classified_count = len(assigned) - unknown_count

    rows = []
    total = 0.0
    for c in COIN_ORDER:
        n = counts.get(c, 0)
        v = COIN_SPECS[c]["value"]
        subtotal = n * v
        total += subtotal
        rows.append((c, n, f"${v:.2f}", f"${subtotal:.2f}"))

    return {
        "circles": circles,
        "assigned": assigned,
        "annotated": annotated,
        "counts": counts,
        "rows": rows,
        "total": total,
        "mm_per_px": mm_per_px,
        "total_found": len(circles),
        "classified_count": classified_count,
        "unknown_count": unknown_count,
    }

class CoinCounterUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("US Coin Counter (Tkinter + OpenCV)")
        self.geometry("1200x720")

        self.original_bgr = None
        self.annotated_bgr = None

        # --- Controls (left panel) ---
        left = ttk.Frame(self, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text="Controls", font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 8))

        ttk.Button(left, text="Upload Image", command=self.upload_image).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Analyze", command=self.run_analysis).pack(fill=tk.X, pady=4)

        ttk.Separator(left).pack(fill=tk.X, pady=10)

        ttk.Label(left, text="Detection tuning", font=("Arial", 11, "bold")).pack(anchor="w", pady=(0, 6))

        self.dp = tk.DoubleVar(value=1.2)
        self.min_dist = tk.IntVar(value=40)
        self.param1 = tk.IntVar(value=120)
        self.param2 = tk.IntVar(value=35)
        self.min_radius = tk.IntVar(value=12)
        self.max_radius = tk.IntVar(value=180)

        self._slider(left, "dp", self.dp, 1.0, 2.5, is_float=True)
        self._slider(left, "minDist", self.min_dist, 10, 200)
        self._slider(left, "param1 (Canny high)", self.param1, 50, 300)
        self._slider(left, "param2 (Hough threshold)", self.param2, 10, 80)
        self._slider(left, "minRadius", self.min_radius, 5, 80)
        self._slider(left, "maxRadius", self.max_radius, 50, 400)

        ttk.Separator(left).pack(fill=tk.X, pady=10)

        self.scale_label = ttk.Label(left, text="Estimated scale: —", wraplength=240)
        self.scale_label.pack(anchor="w", pady=4)

        self.total_label = ttk.Label(left, text="Total: —", font=("Arial", 12, "bold"))
        self.total_label.pack(anchor="w", pady=6)
        self.count_label = ttk.Label(left, text="Coins: —", font=("Arial", 11))
        self.count_label.pack(anchor="w", pady=(0, 6))

        # --- Main area (right) ---
        right = ttk.Frame(self, padding=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        top = ttk.Frame(right)
        top.pack(fill=tk.BOTH, expand=True)

        self.orig_panel = ttk.Label(top, text="Original image will appear here", anchor="center")
        self.orig_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.ann_panel = ttk.Label(top, text="Annotated image will appear here", anchor="center")
        self.ann_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        bottom = ttk.Frame(right)
        bottom.pack(fill=tk.BOTH, expand=False, pady=(10, 0))

        ttk.Label(bottom, text="Breakdown", font=("Arial", 12, "bold")).pack(anchor="w")

        cols = ("coin", "count", "value_each", "subtotal")
        self.tree = ttk.Treeview(bottom, columns=cols, show="headings", height=6)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=140, anchor="center")
        self.tree.pack(fill=tk.X)

        # Keep references to PhotoImage to prevent GC
        self._orig_photo = None
        self._ann_photo = None

    def _slider(self, parent, label, var, lo, hi, is_float=False):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=label).pack(anchor="w")
        if is_float:
            s = ttk.Scale(row, from_=lo, to=hi, variable=var, orient=tk.HORIZONTAL)
        else:
            s = ttk.Scale(row, from_=lo, to=hi, variable=var, orient=tk.HORIZONTAL)
        s.pack(fill=tk.X)
        # show current value
        val = ttk.Label(row, textvariable=var)
        val.pack(anchor="e")

    def upload_image(self):
        path = filedialog.askopenfilename(
            title="Select coin image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")]
        )
        if not path:
            return

        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            messagebox.showerror("Error", "Could not load that image.")
            return

        self.original_bgr = bgr
        self.annotated_bgr = None

        self.show_image(self.orig_panel, bgr, which="orig")
        self.ann_panel.configure(text="Click Analyze to detect coins")
        self._ann_photo = None

        self.clear_table()
        self.scale_label.configure(text="Estimated scale: —")
        self.total_label.configure(text="Total: —")
        self.count_label.configure(text="Coins: —")

    def run_analysis(self):
        if self.original_bgr is None:
            messagebox.showinfo("No image", "Upload an image first.")
            return

        settings = {
            "dp": float(self.dp.get()),
            "min_dist": int(self.min_dist.get()),
            "param1": int(self.param1.get()),
            "param2": int(self.param2.get()),
            "min_radius": int(self.min_radius.get()),
            "max_radius": int(self.max_radius.get()),
        }

        result = analyze_image(self.original_bgr, settings)
        self.annotated_bgr = result["annotated"]

        self.show_image(self.ann_panel, self.annotated_bgr, which="ann")

        self.clear_table()
        for row in result["rows"]:
            self.tree.insert("", tk.END, values=row)

        total = result["total"]
        self.total_label.configure(text=f"Total: ${total:.2f}")
        found = result.get("total_found", 0)
        classified = result.get("classified_count", 0)
        unknown = result.get("unknown_count", 0)
        self.count_label.configure(text=f"Coins: {found} (classified {classified}, unknown {unknown})")

        mm_per_px = result.get("mm_per_px")
        if mm_per_px is None:
            self.scale_label.configure(text="Estimated scale: — (not enough detections)")
        else:
            self.scale_label.configure(text=f"Estimated scale: {mm_per_px:.6f} mm/px")

        if len(result["circles"]) == 0:
            messagebox.showwarning(
                "No coins detected",
                "No circles detected. Try adjusting param2, minRadius/maxRadius, or use a clearer photo."
            )

    def clear_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

    def show_image(self, panel: ttk.Label, bgr: np.ndarray, which: str):
        # Convert BGR -> RGB for PIL
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        # Fit into panel-ish size; use a reasonable default before layout settles
        target_w = max(400, panel.winfo_width())
        target_h = max(300, panel.winfo_height())
        pil.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(pil)
        panel.configure(image=photo, text="")

        if which == "orig":
            self._orig_photo = photo
        else:
            self._ann_photo = photo

if __name__ == "__main__":
    app = CoinCounterUI()
    app.mainloop()