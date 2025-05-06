import tempfile
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from tkinterdnd2 import DND_FILES, DND_TEXT, TkinterDnD
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont, ImageTk


class GelGUI(TkinterDnD.Tk):
    """GUI for SDS-PAGE gel quantification."""

    def __init__(self):
        super().__init__()
        self.title("SDS-PAGE Gel Quantifier")
        self.geometry("650x600")

        # Scrollable main canvas
        container = tk.Frame(self)
        container.pack(fill="both", expand=True)

        self.canvas_main = tk.Canvas(container)
        scrollbar = tk.Scrollbar(
            container,
            orient="vertical",
            command=self.canvas_main.yview,
        )
        self.canvas_main.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas_main.pack(side="left", fill="both", expand=True)

        # Mouse-wheel scrolling
        self.canvas_main.bind_all(
            '<MouseWheel>',
            lambda e: self.canvas_main.yview_scroll(int(-1 * (e.delta / 120)), 'units')
        )
        self.canvas_main.bind_all(
            '<Button-4>',
            lambda e: self.canvas_main.yview_scroll(-1, 'units')
        )
        self.canvas_main.bind_all(
            '<Button-5>',
            lambda e: self.canvas_main.yview_scroll(1, 'units')
        )

        # Inner frame
        self.main_frame = tk.Frame(self.canvas_main)
        self.main_window = self.canvas_main.create_window(
            (0, 0), window=self.main_frame, anchor="nw"
        )
        self.canvas_main.bind(
            '<Configure>',
            lambda e: self.canvas_main.itemconfig(
                self.main_window, width=e.width
            )
        )
        self.main_frame.bind(
            '<Configure>',
            lambda e: self.canvas_main.configure(
                scrollregion=self.canvas_main.bbox("all")
            )
        )

        # State variables
        self.save_path = tk.StringVar()
        self.current_image = None
        self.edited_base = None
        self.rotation_var = tk.DoubleVar(value=0)
        self.undo_stack = []
        self.lane_widgets = []
        self.boundaries = None

        self._build_load_edit_section()
        self._build_label_section()
        self._build_ladder_type_section()
        self._build_layout_section()
        self._build_save_run_section()

    def _build_load_edit_section(self):
        frame = tk.LabelFrame(
            self.main_frame,
            text="1) Load & Edit Gel Image",
            padx=5,
            pady=5,
        )
        frame.pack(fill="x", padx=10, pady=5)

        self.preview_label = tk.Label(
            frame,
            text="Drag & Drop image here",
            bg="lightgray",
        )
        self.preview_label.pack(
            fill="x", padx=5, pady=5, expand=True
        )
        self.preview_label.drop_target_register(
            DND_FILES, DND_TEXT
        )
        self.preview_label.dnd_bind(
            "<<Drop>>", self.on_drop
        )

        btn_frame = tk.Frame(frame)
        btn_frame.pack(fill="x", padx=5)
        tk.Button(
            btn_frame,
            text="Browse…",
            command=self.browse_image,
        ).pack(side="left")
        tk.Label(
            btn_frame, text="or URL:"
        ).pack(side="left", padx=(10, 0))
        self.url_entry = tk.Entry(btn_frame, width=40)
        self.url_entry.pack(side="left", padx=5)
        tk.Button(
            btn_frame,
            text="Load URL",
            command=self.load_url,
        ).pack(side="left")

        edit_frame = tk.Frame(frame)
        edit_frame.pack(fill="x", padx=5, pady=(5, 0))
        tk.Button(
            edit_frame,
            text="Undo",
            command=self.undo,
        ).pack(side="left", padx=2)
        tk.Button(
            edit_frame,
            text="Crop",
            command=self.crop_image,
        ).pack(side="left", padx=2)
        tk.Button(
            edit_frame,
            text="Flip H",
            command=lambda: self.apply_edit(
                lambda img: img.transpose(
                    Image.FLIP_LEFT_RIGHT
                )
            ),
        ).pack(side="left", padx=2)
        tk.Button(
            edit_frame,
            text="Flip V",
            command=lambda: self.apply_edit(
                lambda img: img.transpose(
                    Image.FLIP_TOP_BOTTOM
                )
            ),
        ).pack(side="left", padx=2)
        tk.Label(
            edit_frame, text="Rotate:"
        ).pack(side="left", padx=(10, 2))
        tk.Scale(
            edit_frame,
            from_=0,
            to=360,
            orient='horizontal',
            variable=self.rotation_var,
            length=180,
            command=self.on_rotate_slider,
        ).pack(side="left")

        self.lbl_imgpath = tk.Label(
            frame,
            text="No file selected",
            anchor="w",
        )
        self.lbl_imgpath.pack(
            fill="x", padx=5, pady=(5, 0)
        )

    def _build_label_section(self):
        frame = tk.LabelFrame(
            self.main_frame,
            text="2) Load Lane Labels",
            padx=5,
            pady=5,
        )
        frame.pack(fill="x", padx=10, pady=5)

        tk.Button(
            frame,
            text="Import Excel…",
            command=self.load_labels_excel,
        ).pack(side="left", padx=5)
        tk.Button(
            frame,
            text="Parse Paste…",
            command=self.load_labels_paste,
        ).pack(side="left", padx=5)
        self.paste_text = tk.Text(frame, height=4)
        self.paste_text.pack(fill="x", padx=5, pady=(5, 0))
        tk.Label(
            frame,
            text="Format: label[TAB]mw or 'Ladder'",
        ).pack(anchor="w", padx=5)

    def _build_layout_section(self):
        frame = tk.LabelFrame(
            self.main_frame,
            text="4) Gel Layout & Labels",
            padx=5,
            pady=5,
        )
        frame.pack(fill="x", padx=10, pady=5)
        tk.Label(
            frame, text="# Lanes:"
        ).pack(side="left")
        self.lanes_var = tk.IntVar(value=6)
        ttk.OptionMenu(
            frame,
            self.lanes_var,
            6,
            *range(1, 25),
            command=lambda _: self.on_lanes_change(),
        ).pack(side="left", padx=5)
        self.lane_sub = tk.Frame(frame)
        self.lane_sub.pack(fill="x", pady=(5, 0))
        self.on_lanes_change()

    def _build_ladder_type_section(self):
        """Add ladder selector: SDS-PAGE vs Agarose."""
        frame = tk.LabelFrame(
            self.main_frame,
            text="3) Ladder Type",
            padx=5,
            pady=5,
        )
        frame.pack(fill="x", padx=10, pady=5)
        self.ladder_type = tk.StringVar(value="SDS-PAGE")
        tk.Radiobutton(
            frame,
            text="SDS-PAGE",
            variable=self.ladder_type,
            value="SDS-PAGE",
        ).pack(side="left", padx=5)
        tk.Radiobutton(
            frame,
            text="Agarose",
            variable=self.ladder_type,
            value="Agarose",
        ).pack(side="left", padx=5)
        tk.Button(
            frame,
            text="Calibrate Ladder",
            command=self.calibrate_ladder).pack(side="right", padx=5)

    def _build_save_run_section(self):
        frame = tk.LabelFrame(
            self.main_frame,
            text="5) Save Settings & Run",
            padx=5,
            pady=5,
        )
        frame.pack(fill="x", padx=10, pady=5)
        tk.Entry(
            frame,
            textvariable=self.save_path,
            width=50,
        ).pack(side="left", padx=5)
        tk.Button(
            frame,
            text="Save As…",
            command=self.browse_save,
        ).pack(side="left")
        tk.Button(
            self.main_frame,
            text="Run Quantification",
            bg="#4caf50",
            fg="white",
            command=self.on_run,
        ).pack(pady=10)

    def push_undo(self):
        """Save current image to undo stack."""
        if self.current_image:
            self.undo_stack.append(self.current_image.copy())

    def undo(self):
        """Revert to previous image."""
        if self.undo_stack:
            self.current_image = self.undo_stack.pop()
            self.edited_base = self.current_image.copy()
            self.rotation_var.set(0)
            self.update_preview()

    def apply_edit(self, fn):
        """Apply an edit function to the image."""
        if self.current_image:
            self.push_undo()
            self.current_image = fn(self.current_image)
            self.edited_base = self.current_image.copy()
            self.rotation_var.set(0)
            self.update_preview()

    def on_drop(self, event):
        """Handle file drops or URLs."""
        data = event.data.strip('{}')
        if data.lower().startswith(('http://', 'https://')):
            self.url_entry.delete(0, tk.END)
            self.url_entry.insert(0, data)
            self.load_url()
        else:
            if data.startswith('file://'):
                data = data[7:]
            self.set_image(data)

    def browse_image(self):
        """Open file dialog to select an image."""
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png;*.jpg;*.tif;*.bmp")]
        )
        if path:
            self.set_image(path)

    def load_url(self):
        """Fetch an image from a URL."""
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showerror("Error", "No URL provided!")
            return
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except Exception as exc:
            messagebox.showerror(
                "Error", f"Failed to fetch URL:\n{exc}"
            )
            return

        temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".png"
        )
        temp_file.write(response.content)
        temp_file.close()
        self.set_image(temp_file.name)

    def set_image(self, path):
        """Load and preview the selected image."""
        self.push_undo()
        self.lbl_imgpath.config(text=path)
        image = Image.open(path)
        self.current_image = image.copy()
        self.edited_base = image.copy()
        self.rotation_var.set(0)
        self.update_preview()

    def update_preview(self):
        """Refresh the thumbnail preview."""
        if self.current_image:
            thumb = self.current_image.copy()
            thumb.thumbnail((550, 250))
            tk_img = ImageTk.PhotoImage(thumb)
            self.preview_label.config(image=tk_img, text='')
            self.preview_label.image = tk_img

    def on_rotate_slider(self, angle):
        """Rotate image preview based on slider."""
        if self.edited_base:
            self.current_image = self.edited_base.rotate(
                float(angle), expand=True
            )
            self.update_preview()

    def crop_image(self):
        """Crop the current image with mouse drag."""
        if not self.current_image:
            return

        top = tk.Toplevel(self)
        top.title("Crop Image")

        img = self.current_image
        w, h = img.size
        scale = min(800 / w, 600 / h, 1)
        display_img = img.resize((int(w * scale), int(h * scale)))
        tk_img = ImageTk.PhotoImage(display_img)

        canvas = tk.Canvas(
            top, width=int(w * scale), height=int(h * scale),
            cursor='cross'
        )
        canvas.pack()
        canvas.create_image(0, 0, anchor='nw', image=tk_img)
        canvas.image = tk_img

        coords = {}

        def start(event):
            coords['x0'], coords['y0'] = event.x, event.y
            coords['rect'] = canvas.create_rectangle(
                event.x, event.y, event.x, event.y,
                outline='red'
            )

        def drag(event):
            canvas.coords(
                coords['rect'],
                coords['x0'], coords['y0'],
                event.x, event.y
            )

        def end(event):
            x0, y0 = coords['x0'], coords['y0']
            x1, y1 = event.x, event.y
            sx0, sx1 = sorted((x0, x1))
            sy0, sy1 = sorted((y0, y1))
            ox0, oy0 = int(sx0 / scale), int(sy0 / scale)
            ox1, oy1 = int(sx1 / scale), int(sy1 / scale)

            self.push_undo()
            self.current_image = self.current_image.crop(
                (ox0, oy0, ox1, oy1)
            )
            self.edited_base = self.current_image.copy()
            self.rotation_var.set(0)
            top.destroy()
            self.update_preview()

        canvas.bind('<ButtonPress-1>', start)
        canvas.bind('<B1-Motion>', drag)
        canvas.bind('<ButtonRelease-1>', end)

    def load_labels_excel(self):
        """Import lane labels from an Excel file."""
        path = filedialog.askopenfilename(
            filetypes=[("Excel files", "*.xls;*.xlsx")]
        )
        if not path:
            return

        try:
            df = pd.read_excel(path, header=None)
        except Exception as exc:
            messagebox.showerror(
                "Error", f"Excel read failed:\n{exc}"
            )
            return

        labels = []
        for _, row in df.iterrows():
            lab = str(row.iloc[0]).strip()
            mw = None
            if len(row) > 1:
                try:
                    mw = float(row.iloc[1])
                except ValueError:
                    mw = None
            labels.append((lab, mw))

        if labels:
            self.fill_labels(labels)

    def load_labels_paste(self):
        """Parse pasted text for lane labels."""
        text = self.paste_text.get("1.0", "end").strip()
        labels = []
        for line in text.splitlines():
            entry = line.strip()
            if not entry:
                continue
            if "ladder" in entry.lower() and "\t" not in entry:
                labels.append((entry, None))
                continue

            parts = (
                entry.split("\t")
                if "\t" in entry
                else entry.rsplit(None, 1)
            )
            if len(parts) == 2:
                lab = parts[0].strip()
                try:
                    mw = float(parts[1])
                except ValueError:
                    mw = None
                labels.append((lab, mw))

        if labels:
            self.fill_labels(labels)

    def fill_labels(self, labels):
        """Fill lane entries from label list."""
        self.lanes_var.set(len(labels))
        self.on_lanes_change()
        for idx, (lab, mw) in enumerate(labels):
            ne, me, lv = self.lane_widgets[idx]
            ne.delete(0, tk.END)
            ne.insert(0, lab)
            me.delete(0, tk.END)
            if mw is not None:
                me.insert(0, str(mw))
            lv.set(mw is None or 'ladder' in lab.lower())

    def on_lanes_change(self):
        """Update lane entry widgets based on lane count."""
        for child in self.lane_sub.winfo_children():
            child.destroy()
        self.lane_widgets.clear()

        for i in range(1, self.lanes_var.get() + 1):
            row = tk.Frame(self.lane_sub)
            row.pack(fill="x", pady=2)

            tk.Label(row, text=f"Lane {i}:").pack(side="left")
            ne = tk.Entry(row, width=20)
            ne.pack(side="left", padx=5)

            tk.Label(row, text="MW:").pack(side="left")
            me = tk.Entry(row, width=8)
            me.pack(side="left", padx=5)

            lv = tk.BooleanVar()
            tk.Checkbutton(
                row,
                text="Ladder?",
                variable=lv
            ).pack(side="left", padx=5)

            self.lane_widgets.append((ne, me, lv))

    def browse_save(self):
        """Prompt for save path and update entry."""
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
        )
        if path:
            self.save_path.set(path)

    def detect_boundaries(self):
        """Compute interior lane boundary x-coordinates via intensity, ignoring 5% of either side."""
        if not self.current_image:
            return
        w, h = self.current_image.size
        gray = self.current_image.convert('L')
        arr = np.array(gray, dtype=float)
        # build horizontal intensity profile and ignore edges
        profile = arr.mean(axis=0)
        edge = int(len(profile) * 0.05)
        trimmed = profile[edge:-edge] if edge > 0 else profile.copy()
        # find candidate peaks (brightest columns)
        candidates_rel = np.argsort(-trimmed)
        candidates = [i + edge for i in candidates_rel]
        # select only interior dividers, one less than lane count
        n = len(self.lane_widgets)
        num_dividers = max(n - 1, 0)
        min_dist = max(2, int(w / (2 * n)))
        boundaries = []
        for idx in candidates:
            if all(abs(idx - b) > min_dist for b in boundaries):
                boundaries.append(idx)
                if len(boundaries) >= num_dividers:
                    break
        self.boundaries = sorted(boundaries)

    def calibrate_ladder(self):
        """Open a window to click green (25 kDa) and red (75 kDa) bands."""
        if not self.current_image:
            messagebox.showerror("Error", "Load image first!")
            return

        # find the ladder checkbox
        lanes = [lv.get() for _, _, lv in self.lane_widgets]
        try:
            lid = lanes.index(True)
        except ValueError:
            messagebox.showerror("Error", "Check a 'Ladder?' box first.")
            return

        # make sure we have the same boundaries we draw on the final gel
        self.detect_boundaries()
        w, h = self.current_image.size
        bnds = self.boundaries or []
        full_bounds = [0] + bnds + [w]

        # crop exactly that lane
        x0, x1 = full_bounds[lid], full_bounds[lid + 1]
        stripe = self.current_image.crop((x0, 0, x1, h))

        # now pop up the calibration canvas
        top = tk.Toplevel(self)
        top.title("Calibrate Ladder")
        canvas = tk.Canvas(top, width=x1 - x0, height=h, cursor="cross")
        canvas.pack()
        tk_img = ImageTk.PhotoImage(stripe)
        canvas.create_image(0, 0, anchor="nw", image=tk_img)
        canvas.image = tk_img
        state = {"step": 0}
        prompt = canvas.create_text(
            10,
            10,
            anchor="nw",
            text="Click green band",
            fill="green")

        def on_click(evt):
            y = evt.y
            cx = (x1 - x0) // 2
            size = max(int((x1 - x0) / 10), 10)
            color = "green" if state["step"] == 0 else "red"
            canvas.create_line(
                cx - size,
                y - size,
                cx + size,
                y + size,
                fill=color,
                width=3)
            canvas.create_line(
                cx - size,
                y + size,
                cx + size,
                y - size,
                fill=color,
                width=3)
            if state["step"] == 0:
                self.cal_y25 = y
                state["step"] = 1
                canvas.itemconfig(prompt, text="Click red band", fill="red")
            else:
                self.cal_y75 = y
                top.destroy()

        canvas.bind("<Button-1>", on_click)

    def annotate_image(self, img, lanes):
        w, h = img.size
        n = len(lanes)

        # 1) detect lane boundaries
        self.detect_boundaries()
        boundaries = self.boundaries or []
        draw = ImageDraw.Draw(img)
        div_w = max(int(w * 0.0015), 1)
        # draw interior dividers
        for x in boundaries:
            draw.line((x, 0, x, h), fill='black', width=div_w)
        # extend last boundary for visual closure
        if len(boundaries) >= 2:
            right = boundaries[-1]
            prev = boundaries[-2]
            ext = right + (right - prev)
            draw.line((ext, 0, ext, h), fill='black', width=div_w)

        # 2) label wrapping
        lane_w = w / n
        wrapped, max_chars = [], 1
        for lane in lanes:
            words = lane['name'].split()
            lines, cur = [], ''
            for word in words:
                cand = f"{cur} {word}".strip()
                if len(cand) <= 11:
                    cur = cand
                else:
                    lines.append(cur)
                    cur = word
            lines.append(cur)
            wrapped.append(lines)
            for ln in lines:
                max_chars = max(max_chars, len(ln))

        fsize = int(lane_w / max_chars)
        try:
            font = ImageFont.truetype('arial.ttf', fsize)
        except OSError:
            font = ImageFont.load_default()
        max_lines = max(len(lines) for lines in wrapped)
        margin = (fsize + 4) * max_lines + 20

        # 3) extend canvas
        out = Image.new('RGB', (w, h + margin), 'white')
        out.paste(img, (0, 0))
        draw = ImageDraw.Draw(out)

        # 4) calc lane midpoints including edges
        full_bounds = [0] + boundaries + [w]
        mids = [(full_bounds[i] + full_bounds[i + 1]) //
                2 for i in range(len(full_bounds) - 1)]

        # 5) draw labels
        label_y0 = h + 5
        for idx, lines in enumerate(wrapped):
            x = mids[idx]
            for j, ln in enumerate(lines):
                bb = draw.textbbox((0, 0), ln, font=font)
                tw = bb[2] - bb[0]
                y = label_y0 + j * (fsize + 4)
                draw.text((x - tw / 2, y), ln, fill='black', font=font)

        # 6) draw calibration X's
        if self.ladder_type.get() == 'SDS-PAGE':
            lid = next(
                (i for i, l in enumerate(lanes) if l['is_ladder']), None)
            if lid is not None and hasattr(
                    self, 'cal_y25') and hasattr(
                    self, 'cal_y75'):
                x0, x1 = full_bounds[lid], full_bounds[lid + 1]
                cx = (x0 + x1) // 2
                size = max(int((x1 - x0) / 10), 10)
                for y, color in [
                        (self.cal_y25, 'green'), (self.cal_y75, 'red')]:
                    draw.line(
                        (cx - size,
                         y - size,
                         cx + size,
                         y + size),
                        fill=color,
                        width=3)
                    draw.line(
                        (cx - size,
                         y + size,
                         cx + size,
                         y - size),
                        fill=color,
                        width=3)
            else:
                messagebox.showwarning(
                    "Calibration Missing",
                    "Please calibrate ladder before running quantification.")

        # 7) draw markers
        for idx, lane in enumerate(lanes):
            mw = lane['mw']
            if mw is None or self.ladder_type.get() != 'SDS-PAGE':
                continue
            ratio = (mw - 25) / 50
            if ratio < 0:
                ratio *= 4
            ymk = self.cal_y25 + int((self.cal_y75 - self.cal_y25) * ratio)
            xmk = mids[idx]
            r = 8
            draw.ellipse((xmk - r, ymk - r, xmk + r, ymk + r),
                         outline='white', width=4)
            draw.ellipse((xmk - r, ymk - r, xmk + r, ymk + r),
                         outline='black', width=1)

        return out

    def on_run(self):
        """Collect inputs and generate annotated image."""
        if not self.current_image:
            messagebox.showerror(
                "Error",
                "No image loaded!"
            )
            return

        lanes = []
        for ne, me, lv in self.lane_widgets:
            name = ne.get().strip()
            try:
                mw_value = me.get().strip()
                mw = float(mw_value) if mw_value else None
            except ValueError:
                messagebox.showerror(
                    "Error", f"Bad MW: {me.get()}"
                )
                return
            lanes.append(
                {
                    'name': name,
                    'mw': mw,
                    'is_ladder': lv.get()
                }
            )

        save_path = self.save_path.get().strip()
        if not save_path:
            messagebox.showerror(
                "Error",
                "No save path!"
            )
            return

        annotated_img = self.annotate_image(
            self.current_image.copy(),
            lanes
        )
        annotated_img.save(save_path)
        messagebox.showinfo(
            "Done",
            "Gel quantification complete!"
        )


if __name__ == "__main__":
    app = GelGUI()
    app.mainloop()
