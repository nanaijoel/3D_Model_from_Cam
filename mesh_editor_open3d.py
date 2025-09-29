#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import fnmatch
from typing import List, Optional
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import open3d as o3d

SUPPORTED_EXTS = [".ply", ".pcd", ".obj", ".off", ".stl", ".gltf", ".glb"]


def detect_mode_from_ext(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".obj", ".off", ".stl", ".gltf", ".glb"]:
        return "mesh"
    if ext == ".pcd":
        return "pcd"
    if ext == ".ply":
        tmp = o3d.io.read_triangle_mesh(path)
        if (not tmp.is_empty()) and len(tmp.triangles) > 0:
            return "mesh"
        return "pcd"
    return "auto"


def pick_points(geometry, window_title="Pick points/vertices"):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=window_title, width=1600, height=900)
    vis.add_geometry(geometry)
    # Shift+LMB = select, U = undo, Q = quit
    vis.run()
    vis.destroy_window()
    return vis.get_picked_points()  # list of indices


def edit_point_cloud(in_path: str, out_path: str, keep_unselected: bool) -> bool:
    print(f"[info] Punktwolke laden: {in_path}")
    pcd = o3d.io.read_point_cloud(in_path)
    if pcd.is_empty():
        messagebox.showwarning("Open3D", "Leere Punktwolke – nichts zu bearbeiten.")
        return False

    picked = pick_points(pcd, f"Open3D | DELETE points: {os.path.basename(in_path)}")
    if len(picked) == 0:
        if keep_unselected:
            return bool(o3d.io.write_point_cloud(out_path, pcd, write_ascii=False, compressed=True))
        return False

    edited = pcd.select_by_index(picked, invert=True)
    ok = o3d.io.write_point_cloud(out_path, edited, write_ascii=False, compressed=True)
    return bool(ok)


def edit_triangle_mesh(in_path: str, out_path: str, keep_unselected: bool) -> bool:
    print(f"[info] Mesh laden: {in_path}")
    mesh = o3d.io.read_triangle_mesh(in_path)
    if mesh.is_empty():
        messagebox.showwarning("Open3D", "Leeres Mesh – nichts zu bearbeiten.")
        return False

    mesh.compute_vertex_normals()
    picked = pick_points(mesh, f"Open3D | DELETE vertices: {os.path.basename(in_path)}")
    if len(picked) == 0:
        if keep_unselected:
            return bool(o3d.io.write_triangle_mesh(out_path, mesh, write_triangle_uvs=True))
        return False

    nV = np.asarray(mesh.vertices).shape[0]
    mask_remove = np.zeros(nV, dtype=bool)
    mask_remove[np.array(picked, dtype=int)] = True

    mesh.remove_vertices_by_mask(mask_remove)
    mesh.remove_unreferenced_vertices()
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    # Optional: leicht glätten
    # mesh = mesh.filter_smooth_taubin(number_of_iterations=5)

    ext = os.path.splitext(out_path)[1].lower()
    if ext not in [".ply", ".stl", ".obj", ".off", ".gltf", ".glb"]:
        out_path = os.path.splitext(out_path)[0] + ".ply"
    return bool(o3d.io.write_triangle_mesh(out_path, mesh, write_triangle_uvs=True))


def list_geometry_files(base_dir: str, pattern: Optional[str], recursive: bool) -> List[str]:
    files: List[str] = []
    if recursive:
        for root, _, fnames in os.walk(base_dir):
            for f in fnames:
                ext = os.path.splitext(f)[1].lower()
                if ext in SUPPORTED_EXTS:
                    if pattern and not fnmatch.fnmatch(f, pattern):
                        continue
                    files.append(os.path.join(root, f))
    else:
        for f in sorted(os.listdir(base_dir)):
            path = os.path.join(base_dir, f)
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTS:
                if pattern and not fnmatch.fnmatch(f, pattern):
                    continue
                files.append(path)
    files.sort()
    return files


class MeshSelectorApp(tk.Tk):
    def __init__(self, start_dir: Optional[str] = None):
        super().__init__()
        self.title("Projekt-Mesh-Auswahl")
        self.geometry("900x600")

        self.project_dir = tk.StringVar(value=start_dir or self._guess_projects_dir())
        self.pattern = tk.StringVar(value="*.ply")
        self.recursive = tk.BooleanVar(value=False)
        self.inplace = tk.BooleanVar(value=False)
        self.keep_unselected = tk.BooleanVar(value=True)
        self.suffix = tk.StringVar(value="_edited")
        self.output_dir = tk.StringVar(value="")

        self._build_ui()
        self._refresh_list()

    def _guess_projects_dir(self) -> str:
        here = os.path.abspath(os.path.dirname(__file__))
        guess = os.path.join(here, "projects")
        return guess if os.path.isdir(guess) else here

    def _build_ui(self):
        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Projektordner:").pack(side=tk.LEFT)
        self.entry_dir = ttk.Entry(top, textvariable=self.project_dir, width=60)
        self.entry_dir.pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Ordner wählen…", command=self._choose_dir).pack(side=tk.LEFT)
        ttk.Button(top, text="Neu scannen", command=self._refresh_list).pack(side=tk.LEFT, padx=6)

        opt = ttk.Frame(self, padding=8)
        opt.pack(fill=tk.X)
        ttk.Label(opt, text="Filter:").pack(side=tk.LEFT)
        ttk.Entry(opt, textvariable=self.pattern, width=18).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(opt, text="Rekursiv", variable=self.recursive, command=self._refresh_list).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(opt, text="Inplace überschreiben", variable=self.inplace).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(opt, text="Ohne Auswahl trotzdem speichern", variable=self.keep_unselected).pack(side=tk.LEFT, padx=8)

        opt2 = ttk.Frame(self, padding=8)
        opt2.pack(fill=tk.X)
        ttk.Label(opt2, text="Suffix:").pack(side=tk.LEFT)
        ttk.Entry(opt2, textvariable=self.suffix, width=12).pack(side=tk.LEFT, padx=4)
        ttk.Label(opt2, text="Ausgabeordner (optional):").pack(side=tk.LEFT, padx=8)
        ttk.Entry(opt2, textvariable=self.output_dir, width=40).pack(side=tk.LEFT, padx=4)
        ttk.Button(opt2, text="…", command=self._choose_out_dir).pack(side=tk.LEFT)

        mid = ttk.Frame(self, padding=8)
        mid.pack(fill=tk.BOTH, expand=True)

        self.listbox = tk.Listbox(mid, selectmode=tk.BROWSE)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox.bind("<Double-1>", lambda e: self._open_selected())

        sb = ttk.Scrollbar(mid, orient=tk.VERTICAL, command=self.listbox.yview)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        self.listbox.config(yscrollcommand=sb.set)

        right = ttk.Frame(mid, padding=8)
        right.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Button(right, text="Öffnen (Editor)", command=self._open_selected, width=22).pack(pady=4)
        ttk.Button(right, text="Alle nacheinander öffnen", command=self._open_all, width=22).pack(pady=4)
        ttk.Button(right, text="Beenden", command=self.destroy, width=22).pack(pady=12)

        status = ttk.Frame(self, padding=8)
        status.pack(fill=tk.X)
        self.status_var = tk.StringVar(value="")
        ttk.Label(status, textvariable=self.status_var).pack(side=tk.LEFT)

    def _choose_dir(self):
        d = filedialog.askdirectory(initialdir=self.project_dir.get() or os.getcwd(), title="Projektordner wählen")
        if d:
            self.project_dir.set(d)
            self._refresh_list()

    def _choose_out_dir(self):
        d = filedialog.askdirectory(initialdir=self.output_dir.get() or self.project_dir.get() or os.getcwd(),
                                    title="Ausgabeordner wählen")
        if d:
            self.output_dir.set(d)

    def _refresh_list(self):
        base = self.project_dir.get().strip()
        if not os.path.isdir(base):
            messagebox.showerror("Fehler", f"Ordner nicht gefunden:\n{base}")
            return
        files = list_geometry_files(base, self.pattern.get().strip() or None, self.recursive.get())
        self.listbox.delete(0, tk.END)
        for f in files:
            rel = os.path.relpath(f, base)
            self.listbox.insert(tk.END, rel)
        self.status_var.set(f"{len(files)} Dateien gefunden")
        self.files_abs = files  # parallele Liste in gleicher Reihenfolge

    def _current_selection_path(self) -> Optional[str]:
        sel = self.listbox.curselection()
        if not sel:
            return None
        idx = int(sel[0])
        return self.files_abs[idx]

    def _build_out_path(self, in_path: str) -> str:
        if self.inplace.get():
            return in_path
        out_dir = self.output_dir.get().strip() or os.path.dirname(in_path)
        os.makedirs(out_dir, exist_ok=True)
        base, ext = os.path.splitext(os.path.basename(in_path))
        return os.path.join(out_dir, base + self.suffix.get() + ext)

    def _open_one(self, in_path: str):
        try:
            mode = detect_mode_from_ext(in_path)
            out_path = self._build_out_path(in_path)
            keep_unselected = self.keep_unselected.get()

            ok = False
            if mode == "mesh":
                ok = edit_triangle_mesh(in_path, out_path, keep_unselected)
            elif mode == "pcd":
                ok = edit_point_cloud(in_path, out_path, keep_unselected)
            else:
                try:
                    ok = edit_triangle_mesh(in_path, out_path, keep_unselected)
                except Exception:
                    ok = edit_point_cloud(in_path, out_path, keep_unselected)

            if ok:
                messagebox.showinfo("Gespeichert", f"Gespeichert:\n{out_path}")
            else:
                messagebox.showwarning("Hinweis", "Keine Änderungen gespeichert.")
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Bearbeiten:\n{e}")

    def _open_selected(self):
        p = self._current_selection_path()
        if not p:
            messagebox.showwarning("Auswahl", "Bitte eine Datei auswählen.")
            return
        self._open_one(p)

    def _open_all(self):
        if not hasattr(self, "files_abs") or not self.files_abs:
            messagebox.showwarning("Auswahl", "Keine Dateien in der Liste.")
            return
        start = self.listbox.curselection()
        start_idx = int(start[0]) if start else 0
        for i in range(start_idx, len(self.files_abs)):
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(i)
            self.listbox.activate(i)
            self.listbox.see(i)
            self.update_idletasks()
            self._open_one(self.files_abs[i])


def main():
    # Startverzeichnis: <skript>/projects, falls vorhanden, sonst cwd
    here = os.path.abspath(os.path.dirname(__file__))
    default_projects = os.path.join(here, "projects")
    start_dir = default_projects if os.path.isdir(default_projects) else os.getcwd()

    app = MeshSelectorApp(start_dir=start_dir)
    app.mainloop()


if __name__ == "__main__":
    main()
