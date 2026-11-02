from __future__ import annotations

import json
import queue
import re
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from cloudcharge.config import SimulationConfig
from cloudcharge.io import export_metadata_json, export_npz, export_snapshot_csv, export_spectrum_slice_csv
from cloudcharge.solver import CloudChargeModel, SimulationCancelled, SimulationResult

APP_TITLE = "层云电荷谱模拟平台：Zhou–Tinsley 2007/2012 + N(r,k,z,t)"


class ScrollableFrame(ttk.Frame):
    def __init__(self, master: tk.Misc):
        super().__init__(master)
        canvas = tk.Canvas(self, highlightthickness=0)
        bar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.inner = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
        canvas.configure(yscrollcommand=bar.set)
        canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")


class CloudChargeGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1500x930")
        self.minsize(1150, 740)
        self.config_model = SimulationConfig()
        self.result: SimulationResult | None = None
        self.worker: threading.Thread | None = None
        self.cancel_event = threading.Event()
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.vars: dict[str, tk.Variable] = {}
        self._style()
        self._ui()
        self._set_config(self.config_model)
        self.after(100, self._poll)

    def _style(self) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Title.TLabel", font=("Arial", 16, "bold"))
        style.configure("Section.TLabelframe.Label", font=("Arial", 10, "bold"))
        style.configure("Run.TButton", font=("Arial", 11, "bold"), padding=7)

    def _ui(self) -> None:
        header = ttk.Frame(self, padding=(12, 8))
        header.pack(fill="x")
        ttk.Label(header, text=APP_TITLE, style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="稳态/时间依赖充电 + 粒径—整数电荷二维谱 + Jz/q阶跃响应", foreground="#555").pack(side="left", padx=18)

        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        self.param_tab = ttk.Frame(self.tabs)
        self.profile_tab = ttk.Frame(self.tabs)
        self.time_tab = ttk.Frame(self.tabs)
        self.spectrum_tab = ttk.Frame(self.tabs)
        self.diag_tab = ttk.Frame(self.tabs)
        self.chat_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.param_tab, text="参数与运行")
        self.tabs.add(self.profile_tab, text="空间剖面")
        self.tabs.add(self.time_tab, text="时间演化")
        self.tabs.add(self.spectrum_tab, text="粒径—电荷二维谱")
        self.tabs.add(self.diag_tab, text="论文对照与诊断")
        self.tabs.add(self.chat_tab, text="对话式参数助手")
        self._parameter_tab()
        self._profile_tab()
        self._time_tab()
        self._spectrum_tab()
        self._diagnostic_tab()
        self._chat_tab()

        footer = ttk.Frame(self, padding=(10, 4))
        footer.pack(fill="x")
        self.progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(footer, variable=self.progress_var, maximum=100).pack(side="left", fill="x", expand=True)
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(footer, textvariable=self.status_var, width=55).pack(side="left", padx=10)

    def _var(self, key: str, value: object, boolean: bool = False) -> tk.Variable:
        var: tk.Variable = tk.BooleanVar(value=bool(value)) if boolean else tk.StringVar(value=str(value))
        self.vars[key] = var
        return var

    def _entry(self, parent: ttk.Frame, row: int, label: str, key: str, unit: str = "") -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=3)
        ttk.Entry(parent, textvariable=self.vars[key], width=16).grid(row=row, column=1, sticky="ew", padx=5, pady=3)
        ttk.Label(parent, text=unit, foreground="#666").grid(row=row, column=2, sticky="w", padx=(0, 5), pady=3)
        parent.columnconfigure(1, weight=1)

    def _combo(self, parent: ttk.Frame, row: int, label: str, key: str, values: list[str]) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=3)
        ttk.Combobox(parent, textvariable=self.vars[key], values=values, state="readonly", width=27).grid(
            row=row, column=1, columnspan=2, sticky="ew", padx=5, pady=3
        )
        parent.columnconfigure(1, weight=1)

    def _parameter_tab(self) -> None:
        pane = ttk.Panedwindow(self.param_tab, orient="horizontal")
        pane.pack(fill="both", expand=True)
        scroll = ScrollableFrame(pane)
        right = ttk.Frame(pane, padding=8)
        pane.add(scroll, weight=3)
        pane.add(right, weight=2)
        body = scroll.inner

        values = {
            "solution_mode": "paper2007_bvp", "preset": "2007 Figure 1 equilibrium",
            "altitude": "4", "q": "8e6", "j": "3", "alpha": "1.65e-12",
            "mu1": "1.4e-4", "mu2": "1.9e-4", "aerosol": "5000", "aerosol_r": "40",
            "droplet_n": "200", "r0": "10", "r1": "10", "growth": "0", "max_p": "500",
            "length": "35", "grid": "351", "cloud_start": "2", "ramp": "10", "core": "11",
            "duration": "600", "dt": "1", "snapshots": "0,600", "diag_interval": "10",
            "field_iter": "6", "field_relax": "0.5", "smoothing": "0", "bvp_tol": "1e-5", "bvp_nodes": "30000",
            "attachment": "paper2007_table", "field_mode": "poisson_projection", "field_anchor": "bottom",
            "drift": False, "ion_diff": False, "legacy": False,
            "j1": "1.5", "j2": "3.0", "q1": "8e6", "q2": "8e6", "step": "5000", "startup": "5000", "total": "10000",
            "radius_bins": "18", "radius_min_um": "2", "radius_max_um": "25", "radius_spacing": "log",
            "size_distribution": "lognormal", "sigma_g": "1.35", "charge_min": "-180", "charge_max": "180",
            "initial_charge_mode": "neutral", "initial_charge_std": "1", "growth_law": "scaled_radius",
            "carrier_coupling": "paper_constrained", "spectrum_feedback": True, "store_spectrum": True,
            "max_spectral_cells": "1500000", "charge_safety": "0.8",
        }
        for k, v in values.items():
            self._var(k, v, isinstance(v, bool))

        mode = ttk.LabelFrame(body, text="论文模式与预设", style="Section.TLabelframe")
        mode.pack(fill="x", padx=8, pady=5)
        self._combo(mode, 0, "求解模式", "solution_mode", ["paper2007_bvp", "paper2012_time", "paper2012_direct", "paper2012_spectrum", "transient_charge_ode"])
        presets = [
            "2007 Figure 1 equilibrium",
            "2012 Figure 2-4 baseline",
            "2012 Figure 5: 2 km clean",
            "2012 Figure 6: 2 km polluted",
            "2012 Figure 7: 10 km low q",
            "2012 Figure 8: 10 km high q",
            "2012 Figure 10: 5 um",
            "2012 Figure 10: 15 um",
            "2012 Figure 11-12: q and Jz -30%",
            "2012 spectral baseline: N(r,k,z,t)",
        ]
        self._combo(mode, 1, "论文实验预设", "preset", presets)
        ttk.Button(mode, text="载入所选预设", command=self._apply_preset).grid(row=2, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        atm = ttk.LabelFrame(body, text="环境与离子", style="Section.TLabelframe")
        atm.pack(fill="x", padx=8, pady=5)
        for row, item in enumerate([
            ("环境高度", "altitude", "km"), ("稳态模式 q", "q", "m⁻³ s⁻¹"), ("稳态模式 Jz", "j", "pA m⁻²"),
            ("复合系数 α", "alpha", "m³ s⁻¹"), ("正离子海平面迁移率", "mu1", "m² V⁻¹ s⁻¹"),
            ("负离子海平面迁移率", "mu2", "m² V⁻¹ s⁻¹"),
        ]): self._entry(atm, row, *item)

        cloud = ttk.LabelFrame(body, text="气溶胶、云滴与云层", style="Section.TLabelframe")
        cloud.pack(fill="x", padx=8, pady=5)
        fields = [
            ("气溶胶总浓度 St", "aerosol", "cm⁻³"), ("气溶胶半径", "aerosol_r", "nm"),
            ("最大云滴浓度 N", "droplet_n", "cm⁻³"), ("初始云滴半径", "r0", "μm"),
            ("目标云滴半径", "r1", "μm"), ("云滴增长时间", "growth", "s"), ("最大允许平均电荷", "max_p", "e/滴"),
            ("计算域长度", "length", "m"), ("网格点数", "grid", ""), ("云上方无滴层", "cloud_start", "m"),
            ("单侧边界层厚度", "ramp", "m"), ("云内均匀层宽度", "core", "m"),
        ]
        for row, item in enumerate(fields): self._entry(cloud, row, *item)

        schedule = ttk.LabelFrame(body, text="2012时间序列实验", style="Section.TLabelframe")
        schedule.pack(fill="x", padx=8, pady=5)
        for row, item in enumerate([
            ("初始 Jz", "j1", "pA m⁻²"), ("阶跃后 Jz", "j2", "pA m⁻²"),
            ("初始 q", "q1", "m⁻³ s⁻¹"), ("阶跃后 q", "q2", "m⁻³ s⁻¹"),
            ("阶跃时刻", "step", "s"), ("启动期结束", "startup", "s"), ("总时长", "total", "s"),
        ]): self._entry(schedule, row, *item)

        spectral = ttk.LabelFrame(body, text="粒径—电荷二维谱 N(r,k,z,t)", style="Section.TLabelframe")
        spectral.pack(fill="x", padx=8, pady=5)
        for row, item in enumerate([
            ("粒径分档数", "radius_bins", ""), ("目标谱最小粒径", "radius_min_um", "μm"),
            ("目标谱最大粒径", "radius_max_um", "μm"), ("几何标准差 σg", "sigma_g", ""),
            ("最小整数电荷", "charge_min", "e"), ("最大整数电荷", "charge_max", "e"),
            ("初始电荷标准差", "initial_charge_std", "e"), ("最大谱网格单元", "max_spectral_cells", ""),
            ("电荷空间安全系数", "charge_safety", "0–1"),
        ]): self._entry(spectral, row, *item)
        self._combo(spectral, 9, "粒径网格", "radius_spacing", ["log", "linear"])
        self._combo(spectral, 10, "初始粒径谱", "size_distribution", ["lognormal", "monodisperse"])
        self._combo(spectral, 11, "初始电荷谱", "initial_charge_mode", ["neutral", "gaussian"])
        self._combo(spectral, 12, "粒径增长律", "growth_law", ["none", "scaled_radius", "linear_radius", "linear_radius_squared"])
        self._combo(spectral, 13, "载流子耦合", "carrier_coupling", ["paper_constrained", "fully_direct"])
        ttk.Checkbutton(spectral, text="谱的一阶电荷矩反馈到高斯定律", variable=self.vars["spectrum_feedback"]).grid(row=14, column=0, columnspan=3, sticky="w", padx=5, pady=3)
        ttk.Checkbutton(spectral, text="保存每个时刻完整三维谱(z,r,k)", variable=self.vars["store_spectrum"]).grid(row=15, column=0, columnspan=3, sticky="w", padx=5, pady=3)

        num = ttk.LabelFrame(body, text="数值与输出", style="Section.TLabelframe")
        num.pack(fill="x", padx=8, pady=5)
        for row, item in enumerate([
            ("通用模拟时长", "duration", "s"), ("最大时间步长", "dt", "s"), ("保存时刻", "snapshots", "逗号分隔"),
            ("时间诊断间隔", "diag_interval", "s"), ("场约束迭代次数", "field_iter", ""),
            ("场约束松弛", "field_relax", "0–1"), ("守恒平滑", "smoothing", "0–0.24"),
            ("BVP容差", "bvp_tol", ""), ("BVP最大节点", "bvp_nodes", ""),
        ]): self._entry(num, row, *item)
        self._combo(num, 9, "附着系数", "attachment", ["paper2007_table", "corrected", "legacy_shifted_boole"])
        self._combo(num, 10, "场闭合", "field_mode", ["poisson_projection", "poisson_only", "conductivity_only"])
        self._combo(num, 11, "泊松锚点", "field_anchor", ["bottom", "top", "least_squares"])
        ttk.Checkbutton(num, text="启用漂移充电（2012模式自动启用）", variable=self.vars["drift"]).grid(row=12, column=0, columnspan=3, sticky="w", padx=5, pady=3)
        ttk.Checkbutton(num, text="加入小离子分子扩散", variable=self.vars["ion_diff"]).grid(row=13, column=0, columnspan=3, sticky="w", padx=5, pady=3)
        ttk.Checkbutton(num, text="使用原IDL初始化/边界差分", variable=self.vars["legacy"]).grid(row=14, column=0, columnspan=3, sticky="w", padx=5, pady=3)

        buttons = ttk.Frame(body, padding=8)
        buttons.pack(fill="x")
        self.run_button = ttk.Button(buttons, text="运行模拟", style="Run.TButton", command=self._start)
        self.run_button.pack(side="left", padx=3)
        self.stop_button = ttk.Button(buttons, text="停止", command=self._stop, state="disabled")
        self.stop_button.pack(side="left", padx=3)
        ttk.Button(buttons, text="保存参数", command=self._save_config).pack(side="left", padx=3)
        ttk.Button(buttons, text="载入参数", command=self._load_config).pack(side="left", padx=3)
        ttk.Button(buttons, text="恢复默认", command=lambda: self._set_config(SimulationConfig())).pack(side="left", padx=3)

        ttk.Label(right, text="运行日志", font=("Arial", 12, "bold")).pack(anchor="w")
        self.log = ScrolledText(right, wrap="word", font=("Menlo", 10))
        self.log.pack(fill="both", expand=True, pady=(6, 8))
        self.log.insert("end", "2007模式求稳态电荷分配；2012模式显式积分云滴生长、Jz/q阶跃及充电时间常数。\n")
        ttk.Button(right, text="清空日志", command=lambda: self.log.delete("1.0", "end")).pack(anchor="e")

    def _profile_tab(self) -> None:
        bar = ttk.Frame(self.profile_tab, padding=6); bar.pack(fill="x")
        ttk.Label(bar, text="显示时刻：").pack(side="left")
        self.snapshot_var = tk.StringVar(value="0")
        self.snapshot_box = ttk.Combobox(bar, textvariable=self.snapshot_var, state="readonly", width=15)
        self.snapshot_box.pack(side="left", padx=4)
        self.snapshot_box.bind("<<ComboboxSelected>>", lambda _e: self._plot_profiles())
        ttk.Button(bar, text="导出当前CSV", command=self._export_csv).pack(side="left", padx=4)
        ttk.Button(bar, text="导出全部NPZ", command=self._export_npz).pack(side="left", padx=4)
        ttk.Button(bar, text="保存剖面图", command=lambda: self._save_figure(self.profile_fig)).pack(side="left", padx=4)
        self.profile_fig = Figure(figsize=(11, 7), dpi=100, constrained_layout=True)
        self.profile_axes = self.profile_fig.subplots(2, 3)
        self.profile_canvas = FigureCanvasTkAgg(self.profile_fig, master=self.profile_tab)
        self.profile_canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.profile_canvas, self.profile_tab).update()
        self._empty_profiles()

    def _time_tab(self) -> None:
        bar = ttk.Frame(self.time_tab, padding=6); bar.pack(fill="x")
        ttk.Label(bar, text="2012模式显示云顶/云底电荷、充电率与载流子响应。2007模式仅显示稳态参考线。", foreground="#555").pack(side="left")
        ttk.Button(bar, text="保存时间演化图", command=lambda: self._save_figure(self.time_fig)).pack(side="right", padx=4)
        self.time_fig = Figure(figsize=(11, 7), dpi=100, constrained_layout=True)
        self.time_axes = self.time_fig.subplots(2, 2)
        self.time_canvas = FigureCanvasTkAgg(self.time_fig, master=self.time_tab)
        self.time_canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.time_canvas, self.time_tab).update()
        self._empty_time()

    def _spectrum_tab(self) -> None:
        bar = ttk.Frame(self.spectrum_tab, padding=6)
        bar.pack(fill="x")
        ttk.Label(bar, text="时刻：").pack(side="left")
        self.spectrum_time_var = tk.StringVar(value="0")
        self.spectrum_time_box = ttk.Combobox(bar, textvariable=self.spectrum_time_var, state="readonly", width=12)
        self.spectrum_time_box.pack(side="left", padx=4)
        self.spectrum_time_box.bind("<<ComboboxSelected>>", lambda _e: self._plot_spectrum())
        ttk.Label(bar, text="位置：").pack(side="left", padx=(12, 0))
        self.spectrum_location_var = tk.StringVar(value="云顶电荷峰")
        self.spectrum_location_box = ttk.Combobox(
            bar, textvariable=self.spectrum_location_var, state="readonly", width=16,
            values=["云顶电荷峰", "云内中心", "云底电荷峰", "自定义z"]
        )
        self.spectrum_location_box.pack(side="left", padx=4)
        self.spectrum_location_box.bind("<<ComboboxSelected>>", lambda _e: self._plot_spectrum())
        ttk.Label(bar, text="z=").pack(side="left", padx=(8, 0))
        self.spectrum_z_var = tk.StringVar(value="5")
        ttk.Entry(bar, textvariable=self.spectrum_z_var, width=8).pack(side="left")
        ttk.Label(bar, text="m").pack(side="left", padx=(2, 8))
        ttk.Button(bar, text="刷新", command=self._plot_spectrum).pack(side="left", padx=3)
        ttk.Button(bar, text="导出当前N(r,k) CSV", command=self._export_spectrum_csv).pack(side="left", padx=3)
        ttk.Button(bar, text="保存谱图", command=lambda: self._save_figure(self.spectrum_fig)).pack(side="right", padx=4)
        self.spectrum_fig = Figure(figsize=(11, 7), dpi=100, constrained_layout=True)
        self.spectrum_axes = self.spectrum_fig.subplots(2, 2)
        self.spectrum_canvas = FigureCanvasTkAgg(self.spectrum_fig, master=self.spectrum_tab)
        self.spectrum_canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.spectrum_canvas, self.spectrum_tab).update()
        self.spectrum_colorbar = None
        self._empty_spectrum()

    def _empty_spectrum(self) -> None:
        titles = ["N(r,k) heatmap", "Charge spectrum", "Size spectrum", "Spectral moments with height"]
        for ax, title in zip(self.spectrum_axes.ravel(), titles):
            ax.clear(); ax.set_title(title); ax.grid(True, alpha=.20)
        self.spectrum_canvas.draw_idle()

    def _diagnostic_tab(self) -> None:
        bar = ttk.Frame(self.diag_tab, padding=6); bar.pack(fill="x")
        ttk.Button(bar, text="导出诊断JSON", command=self._export_metadata).pack(side="left", padx=4)
        self.diag_text = ScrolledText(self.diag_tab, wrap="word", font=("Menlo", 10))
        self.diag_text.pack(fill="both", expand=True, padx=8, pady=8)
        self.diag_text.insert("end", "运行后显示2007/2012方程差异、论文定量结果、时间常数、上下边界电性和数值残差。\n")

    def _chat_tab(self) -> None:
        frame = ttk.Frame(self.chat_tab, padding=10); frame.pack(fill="both", expand=True)
        self.chat = ScrolledText(frame, wrap="word", font=("Arial", 11)); self.chat.pack(fill="both", expand=True)
        self.chat.insert("end", "示例：\n• 载入2012基准并运行\n• 高度2 km，气溶胶20000 cm-3，阶跃后Jz 3 pA/m2\n• 云滴半径15微米，时间步0.5秒\n")
        row = ttk.Frame(frame); row.pack(fill="x", pady=(8,0))
        self.chat_input = ttk.Entry(row, font=("Arial", 11)); self.chat_input.pack(side="left", fill="x", expand=True)
        self.chat_input.bind("<Return>", lambda _e: self._chat_command())
        ttk.Button(row, text="发送", command=self._chat_command).pack(side="left", padx=5)
        ttk.Button(row, text="应用并运行", command=lambda: self._chat_command(True)).pack(side="left")

    def _empty_profiles(self) -> None:
        titles = ["Droplet number", "Electric field", "Space charge", "Small ions", "Charged aerosols", "Mean droplet charge"]
        for ax, title in zip(self.profile_axes.ravel(), titles):
            ax.clear(); ax.set_title(title); ax.set_xlabel("Distance downward (m)"); ax.grid(True, alpha=.25)
        self.profile_canvas.draw_idle()

    def _empty_time(self) -> None:
        titles = ["Droplet charge", "Droplet charging rate", "Top-boundary aerosol charge", "Ion response / forcing"]
        for ax, title in zip(self.time_axes.ravel(), titles):
            ax.clear(); ax.set_title(title); ax.set_xlabel("Time (s)"); ax.grid(True, alpha=.25)
        self.time_canvas.draw_idle()

    def _set_config(self, cfg: SimulationConfig) -> None:
        self.config_model = cfg
        a, ae, d, sp, dom, td, p, n = cfg.atmosphere, cfg.aerosol, cfg.droplet, cfg.spectrum, cfg.domain, cfg.time_dependent, cfg.physics, cfg.numerics
        mapping = {
            "solution_mode": p.solution_mode, "altitude": a.altitude_km, "q": a.ion_pair_production_m3_s, "j": a.current_density_a_m2/1e-12,
            "alpha": a.recombination_m3_s, "mu1": a.positive_mobility_ref_m2_v_s, "mu2": a.negative_mobility_ref_m2_v_s,
            "aerosol": ae.total_number_m3/1e6, "aerosol_r": ae.mean_radius_m/1e-9, "droplet_n": d.fixed_peak_number_m3/1e6,
            "r0": d.initial_radius_m/1e-6, "r1": d.target_radius_m/1e-6, "growth": d.growth_time_s, "max_p": d.max_abs_charge_e,
            "length": dom.length_m, "grid": dom.grid_points, "cloud_start": dom.cloud_start_m, "ramp": dom.edge_ramp_m, "core": dom.core_width_m,
            "duration": n.duration_s, "dt": n.max_dt_s, "snapshots": ",".join(f"{x:g}" for x in n.snapshot_times_s), "diag_interval": n.diagnostics_interval_s,
            "field_iter": n.field_iterations, "field_relax": n.field_relaxation, "smoothing": n.conservative_smoothing,
            "bvp_tol": n.bvp_tolerance, "bvp_nodes": n.bvp_max_nodes, "attachment": p.attachment_mode,
            "field_mode": p.field_mode, "field_anchor": p.field_anchor, "drift": p.include_drift_charging,
            "ion_diff": p.include_ion_diffusion, "legacy": p.use_legacy_initialization,
            "j1": td.initial_current_density_a_m2/1e-12, "j2": td.second_current_density_a_m2/1e-12,
            "q1": td.initial_ion_pair_production_m3_s, "q2": td.second_ion_pair_production_m3_s,
            "step": td.step_time_s, "startup": td.startup_end_s, "total": td.total_duration_s,
            "radius_bins": sp.radius_bins, "radius_min_um": sp.radius_min_m/1e-6, "radius_max_um": sp.radius_max_m/1e-6,
            "radius_spacing": sp.radius_spacing, "size_distribution": sp.initial_size_distribution, "sigma_g": sp.geometric_std,
            "charge_min": sp.charge_min_e, "charge_max": sp.charge_max_e, "initial_charge_mode": sp.initial_charge_mode,
            "initial_charge_std": sp.initial_charge_std_e, "growth_law": sp.growth_law, "carrier_coupling": sp.carrier_coupling,
            "spectrum_feedback": sp.include_spectrum_feedback, "store_spectrum": sp.store_full_spectrum,
            "max_spectral_cells": sp.max_spectral_cells, "charge_safety": sp.charge_subcycle_safety,
        }
        for k, v in mapping.items(): self.vars[k].set(v)

    def _config(self) -> SimulationConfig:
        c = SimulationConfig()
        c.physics.solution_mode = str(self.vars["solution_mode"].get())
        c.atmosphere.altitude_km = float(self.vars["altitude"].get())
        c.atmosphere.ion_pair_production_m3_s = float(self.vars["q"].get())
        c.atmosphere.current_density_a_m2 = float(self.vars["j"].get())*1e-12
        c.atmosphere.recombination_m3_s = float(self.vars["alpha"].get())
        c.atmosphere.positive_mobility_ref_m2_v_s = float(self.vars["mu1"].get())
        c.atmosphere.negative_mobility_ref_m2_v_s = float(self.vars["mu2"].get())
        c.aerosol.total_number_m3 = float(self.vars["aerosol"].get())*1e6
        c.aerosol.mean_radius_m = float(self.vars["aerosol_r"].get())*1e-9
        c.droplet.fixed_peak_number_m3 = float(self.vars["droplet_n"].get())*1e6
        c.droplet.initial_radius_m = float(self.vars["r0"].get())*1e-6
        c.droplet.target_radius_m = float(self.vars["r1"].get())*1e-6
        c.droplet.growth_time_s = float(self.vars["growth"].get())
        c.droplet.max_abs_charge_e = float(self.vars["max_p"].get())
        c.spectrum.radius_bins = int(float(self.vars["radius_bins"].get()))
        c.spectrum.radius_min_m = float(self.vars["radius_min_um"].get())*1e-6
        c.spectrum.radius_max_m = float(self.vars["radius_max_um"].get())*1e-6
        c.spectrum.radius_spacing = str(self.vars["radius_spacing"].get())
        c.spectrum.initial_size_distribution = str(self.vars["size_distribution"].get())
        c.spectrum.geometric_std = float(self.vars["sigma_g"].get())
        c.spectrum.charge_min_e = int(float(self.vars["charge_min"].get()))
        c.spectrum.charge_max_e = int(float(self.vars["charge_max"].get()))
        c.spectrum.initial_charge_mode = str(self.vars["initial_charge_mode"].get())
        c.spectrum.initial_charge_std_e = float(self.vars["initial_charge_std"].get())
        c.spectrum.growth_law = str(self.vars["growth_law"].get())
        c.spectrum.carrier_coupling = str(self.vars["carrier_coupling"].get())
        c.spectrum.include_spectrum_feedback = bool(self.vars["spectrum_feedback"].get())
        c.spectrum.store_full_spectrum = bool(self.vars["store_spectrum"].get())
        c.spectrum.max_spectral_cells = int(float(self.vars["max_spectral_cells"].get()))
        c.spectrum.charge_subcycle_safety = float(self.vars["charge_safety"].get())
        c.domain.length_m = float(self.vars["length"].get()); c.domain.grid_points = int(float(self.vars["grid"].get()))
        c.domain.cloud_start_m = float(self.vars["cloud_start"].get()); c.domain.edge_ramp_m = float(self.vars["ramp"].get()); c.domain.core_width_m = float(self.vars["core"].get())
        c.numerics.duration_s = float(self.vars["duration"].get()); c.numerics.max_dt_s = float(self.vars["dt"].get())
        c.numerics.snapshot_times_s = [float(x.strip()) for x in str(self.vars["snapshots"].get()).split(",") if x.strip()]
        c.numerics.diagnostics_interval_s = float(self.vars["diag_interval"].get())
        c.numerics.field_iterations = int(float(self.vars["field_iter"].get())); c.numerics.field_relaxation = float(self.vars["field_relax"].get())
        c.numerics.conservative_smoothing = float(self.vars["smoothing"].get()); c.numerics.bvp_tolerance = float(self.vars["bvp_tol"].get()); c.numerics.bvp_max_nodes = int(float(self.vars["bvp_nodes"].get()))
        c.physics.attachment_mode = str(self.vars["attachment"].get()); c.physics.field_mode = str(self.vars["field_mode"].get()); c.physics.field_anchor = str(self.vars["field_anchor"].get())
        c.physics.include_drift_charging = bool(self.vars["drift"].get()); c.physics.include_ion_diffusion = bool(self.vars["ion_diff"].get()); c.physics.use_legacy_initialization = bool(self.vars["legacy"].get())
        c.time_dependent.initial_current_density_a_m2 = float(self.vars["j1"].get())*1e-12; c.time_dependent.second_current_density_a_m2 = float(self.vars["j2"].get())*1e-12
        c.time_dependent.initial_ion_pair_production_m3_s = float(self.vars["q1"].get()); c.time_dependent.second_ion_pair_production_m3_s = float(self.vars["q2"].get())
        c.time_dependent.step_time_s = float(self.vars["step"].get()); c.time_dependent.startup_end_s = float(self.vars["startup"].get()); c.time_dependent.total_duration_s = float(self.vars["total"].get())
        if c.physics.solution_mode in {"paper2012_time", "paper2012_direct", "paper2012_spectrum"}: c.numerics.duration_s = c.time_dependent.total_duration_s
        c.validate(); return c

    def _apply_preset(self) -> None:
        name = str(self.vars["preset"].get())
        c = SimulationConfig()
        if name.startswith("2007"):
            c.physics.solution_mode = "paper2007_bvp"; c.aerosol.total_number_m3 = 5e9; c.atmosphere.current_density_a_m2 = 3e-12
            c.droplet.initial_radius_m = c.droplet.target_radius_m = 10e-6; c.droplet.growth_time_s = 0; c.numerics.duration_s = 600; c.numerics.snapshot_times_s=[0,600]
        else:
            c.physics.solution_mode = "paper2012_time"; c.physics.include_drift_charging=True; c.aerosol.total_number_m3=1e9
            c.droplet.initial_radius_m=0.01e-6; c.droplet.target_radius_m=10e-6; c.droplet.growth_time_s=600
            c.numerics.duration_s=10000; c.numerics.snapshot_times_s=[0,600,2000,5000,6000,10000]; c.numerics.max_dt_s=1; c.numerics.diagnostics_interval_s=10
            c.time_dependent.initial_current_density_a_m2=1.5e-12; c.time_dependent.second_current_density_a_m2=3e-12
            c.time_dependent.initial_ion_pair_production_m3_s=8e6; c.time_dependent.second_ion_pair_production_m3_s=8e6
            if "2 km clean" in name:
                c.atmosphere.altitude_km=2; c.atmosphere.recombination_m3_s=1.66e-12; c.time_dependent.initial_ion_pair_production_m3_s=c.time_dependent.second_ion_pair_production_m3_s=4e6
            elif "2 km polluted" in name:
                c.atmosphere.altitude_km=2; c.atmosphere.recombination_m3_s=1.66e-12; c.aerosol.total_number_m3=20e9; c.time_dependent.initial_ion_pair_production_m3_s=c.time_dependent.second_ion_pair_production_m3_s=4e6
            elif "10 km low q" in name:
                c.atmosphere.altitude_km=10; c.atmosphere.recombination_m3_s=1.78e-12; c.time_dependent.initial_ion_pair_production_m3_s=c.time_dependent.second_ion_pair_production_m3_s=15e6
            elif "10 km high q" in name:
                c.atmosphere.altitude_km=10; c.atmosphere.recombination_m3_s=1.78e-12; c.time_dependent.initial_ion_pair_production_m3_s=c.time_dependent.second_ion_pair_production_m3_s=40e6; c.time_dependent.initial_current_density_a_m2=2e-12; c.time_dependent.second_current_density_a_m2=4e-12
            elif "5 um" in name:
                c.droplet.target_radius_m=5e-6
            elif "15 um" in name:
                c.droplet.target_radius_m=15e-6; c.droplet.max_abs_charge_e=800
            elif "q and Jz" in name:
                c.aerosol.total_number_m3=5e9; c.time_dependent.second_current_density_a_m2=2.1e-12; c.time_dependent.second_ion_pair_production_m3_s=5.6e6
            if "spectral baseline" in name:
                c.physics.solution_mode="paper2012_spectrum"
                c.domain.grid_points=61
                c.numerics.max_dt_s=5
                c.numerics.diagnostics_interval_s=100
                c.spectrum.radius_bins=14
                c.spectrum.radius_min_m=2e-6
                c.spectrum.radius_max_m=25e-6
                c.spectrum.charge_min_e=-250
                c.spectrum.charge_max_e=250
                c.spectrum.carrier_coupling="paper_constrained"; c.spectrum.growth_law="scaled_radius"
                c.spectrum.store_full_spectrum=True
                c.spectrum.max_spectral_cells=1000000
        self._set_config(c); self.vars["preset"].set(name)
        self.status_var.set(f"已载入：{name}")
        self._log(f"已载入论文预设 {name}")

    def _log(self, text: str) -> None:
        self.log.insert("end", text.rstrip()+"\n"); self.log.see("end")

    def _start(self) -> None:
        if self.worker and self.worker.is_alive(): return
        try: cfg = self._config()
        except Exception as e: messagebox.showerror("参数错误", str(e)); return
        if cfg.physics.solution_mode == "paper2012_direct" and (cfg.numerics.max_dt_s > 0.2 or cfg.domain.grid_points > 201):
            proceed = messagebox.askyesno(
                "直接方程模式提示",
                "paper2012_direct 是刚性、实验性求解器。当前网格或时间步可能导致计算缓慢或数值失稳。\n\n"
                "建议先使用 dt≤0.2 s、网格≤201，并进行收敛试验；论文复现和参数扫描请优先使用 paper2012_time。\n\n仍要继续吗？",
            )
            if not proceed:
                return
        self.config_model=cfg; self.result=None; self.cancel_event.clear(); self.progress_var.set(0); self.run_button.config(state="disabled"); self.stop_button.config(state="normal")
        self._log("="*70); self._log("开始模拟\n"+json.dumps(cfg.to_dict(),ensure_ascii=False,indent=2))
        def job() -> None:
            try:
                m=CloudChargeModel(cfg)
                self.events.put(("log", f"attachment beta+={m.beta_pos:.3e}, beta-={m.beta_neg:.3e}, gamma+={m.gamma_pos:.3e}, gamma-={m.gamma_neg:.3e}"))
                r=m.run(progress_callback=lambda p,s:self.events.put(("progress",(p,s))), cancel_callback=self.cancel_event.is_set)
                self.events.put(("done",r))
            except SimulationCancelled as e: self.events.put(("cancel",str(e)))
            except Exception: self.events.put(("error",traceback.format_exc()))
        self.worker=threading.Thread(target=job,daemon=True); self.worker.start()

    def _stop(self) -> None: self.cancel_event.set(); self.status_var.set("正在停止……")

    def _poll(self) -> None:
        try:
            while True:
                kind,payload=self.events.get_nowait()
                if kind=="progress":
                    p,s=payload; self.progress_var.set(float(p)*100); self.status_var.set(str(s))
                elif kind=="log": self._log(str(payload))
                elif kind=="done":
                    self.result=payload; self.progress_var.set(100); self.status_var.set("模拟完成"); self.run_button.config(state="normal"); self.stop_button.config(state="disabled")
                    self._log("模拟完成"); self._populate(); self.tabs.select(self.profile_tab)
                elif kind=="cancel": self.status_var.set("已停止"); self.run_button.config(state="normal"); self.stop_button.config(state="disabled")
                elif kind=="error":
                    self.status_var.set("模拟失败"); self.run_button.config(state="normal"); self.stop_button.config(state="disabled"); self._log(str(payload)); messagebox.showerror("模拟失败",str(payload))
        except queue.Empty: pass
        self.after(100,self._poll)

    def _populate(self) -> None:
        assert self.result is not None
        vals=[f"{t:g}" for t in self.result.snapshot_times_s]; self.snapshot_box["values"]=vals; self.snapshot_var.set(vals[-1])
        self.spectrum_time_box["values"]=vals; self.spectrum_time_var.set(vals[-1])
        self._plot_profiles(); self._plot_time(); self._plot_spectrum(); self._diagnostics()

    def _plot_profiles(self) -> None:
        if not self.result: self._empty_profiles(); return
        t=self.result.nearest_snapshot_time(float(self.snapshot_var.get())); s=self.result.snapshots[t]; z=self.result.z_m; axes=self.profile_axes.ravel()
        for ax in axes: ax.clear(); ax.grid(True,alpha=.25); ax.set_xlabel("Distance downward (m)")
        axes[0].plot(z,np.asarray(s["droplet_number_m3"])/1e6); axes[0].set_title("Droplet number (cm$^{-3}$)")
        axes[1].plot(z,s["electric_field_v_m"]); axes[1].set_title("Electric field (V m$^{-1}$)")
        axes[2].plot(z,np.asarray(s["charge_density_c_m3"])*1e12); axes[2].axhline(0,lw=.7); axes[2].set_title("Space charge (pC m$^{-3}$)")
        axes[3].semilogy(z,np.maximum(s["n_positive_m3"],1),label="n+"); axes[3].semilogy(z,np.maximum(s["n_negative_m3"],1),label="n-"); axes[3].legend(); axes[3].set_title("Small ions")
        axes[4].semilogy(z,np.maximum(s["s_positive_m3"],1),label="S+"); axes[4].semilogy(z,np.maximum(s["s_negative_m3"],1),label="S-"); axes[4].legend(); axes[4].set_title("Charged aerosols")
        axes[5].plot(z,s["droplet_charge_e"]); axes[5].axhline(0,lw=.7); axes[5].set_title("Mean droplet charge (e/drop)")
        self.profile_fig.suptitle(f"t={t:g} s | mode={self.result.diagnostics.get('solver_mode','')}"); self.profile_canvas.draw_idle()

    def _plot_time(self) -> None:
        if not self.result: self._empty_time(); return
        d=self.result.diagnostics; axes=self.time_axes.ravel(); mode=str(d.get("solver_mode",""))
        for ax in axes: ax.clear(); ax.grid(True,alpha=.25); ax.set_xlabel("Time (s)")
        if "top_droplet_charge_e" not in d:
            p=float(d.get("upper_peak_droplet_charge_e",0)); b=float(d.get("lower_peak_droplet_charge_e",0)); axes[0].axhline(p,label="cloud top"); axes[0].axhline(b,label="cloud base"); axes[0].legend(); axes[0].set_title("2007 equilibrium charge")
            for ax in axes[1:]: ax.text(.5,.5,"Time evolution is available in 2012 modes",ha="center",va="center",transform=ax.transAxes)
        elif mode=="paper2012_spectrum":
            t=np.asarray(d["time_s"],float)
            top=np.asarray(d["top_droplet_charge_e"],float); base=np.asarray(d["bottom_droplet_charge_e"],float)
            ts=np.asarray(d["top_droplet_charge_std_e"],float); bs=np.asarray(d["bottom_droplet_charge_std_e"],float)
            axes[0].plot(t,top,label="top mean"); axes[0].plot(t,base,label="base mean")
            axes[0].fill_between(t,top-ts,top+ts,alpha=.15); axes[0].fill_between(t,base-bs,base+bs,alpha=.15)
            axes[0].axhline(0,lw=.7); axes[0].legend(); axes[0].set_title("Charge mean ±1σ (e/drop)")
            axes[1].plot(t,d["top_positive_fraction"],label="top positive fraction")
            axes[1].plot(t,d["bottom_negative_fraction"],label="base negative fraction")
            axes[1].set_ylim(0,1); axes[1].legend(); axes[1].set_title("Charged-droplet fractions")
            axes[2].plot(t,np.asarray(d["top_mean_radius_m"])*1e6,label="top")
            axes[2].plot(t,np.asarray(d["bottom_mean_radius_m"])*1e6,label="base")
            axes[2].legend(); axes[2].set_title("Mean radius (μm)")
            axes[3].plot(t,np.asarray(d["scheduled_current_density_a_m2"])/1e-12,label="Jz (pA m$^{-2}$)")
            a=axes[3].twinx(); a.plot(t,np.asarray(d["scheduled_ion_pair_production_m3_s"])/1e6,linestyle="--",label="q")
            axes[3].legend(loc="upper left"); a.set_ylabel("q (10$^6$ m$^{-3}$ s$^{-1}$)"); axes[3].set_title("External forcing")
        else:
            t=np.asarray(d["time_s"]); axes[0].plot(t,d["top_droplet_charge_e"],label="cloud top"); axes[0].plot(t,d["bottom_droplet_charge_e"],label="cloud base"); axes[0].axhline(0,lw=.7); axes[0].legend(); axes[0].set_title("Mean droplet charge (e/drop)")
            axes[1].plot(t,d["top_droplet_charging_rate_e_s"],label="top"); axes[1].plot(t,d["bottom_droplet_charging_rate_e_s"],label="base"); axes[1].axhline(0,lw=.7); axes[1].legend(); axes[1].set_title("dp/dt (e s$^{-1}$)")
            axes[2].plot(t,d["top_positive_aerosol_m3"],label="S+"); axes[2].plot(t,d["top_negative_aerosol_m3"],label="S-"); axes[2].legend(); axes[2].set_title("Top charged aerosol (m$^{-3}$)")
            axes[3].plot(t,d["top_positive_ion_m3"],label="n+"); axes[3].plot(t,d["top_negative_ion_m3"],label="n-"); axes[3].legend(loc="upper left"); axes[3].set_title("Top ions and forcing")
            a=axes[3].twinx(); a.plot(t,np.asarray(d["scheduled_current_density_a_m2"])/1e-12,linestyle="--",label="Jz"); a.set_ylabel("Jz (pA m$^{-2}$)")
        self.time_canvas.draw_idle()

    def _selected_spectrum_z(self) -> tuple[int, float]:
        if not self.result:
            return 0, 0.0
        d=self.result.diagnostics; choice=self.spectrum_location_var.get()
        if choice=="云顶电荷峰": idx=int(float(d.get("top_spectrum_index",0)))
        elif choice=="云底电荷峰": idx=int(float(d.get("bottom_spectrum_index",len(self.result.z_m)-1)))
        elif choice=="云内中心": idx=int(float(d.get("core_spectrum_index",len(self.result.z_m)//2)))
        else:
            try: zvalue=float(self.spectrum_z_var.get())
            except ValueError: zvalue=float(self.result.z_m[len(self.result.z_m)//2])
            idx=int(np.argmin(np.abs(self.result.z_m-zvalue)))
        idx=int(np.clip(idx,0,len(self.result.z_m)-1)); zvalue=float(self.result.z_m[idx]); self.spectrum_z_var.set(f"{zvalue:.3f}")
        return idx,zvalue

    def _plot_spectrum(self) -> None:
        if not self.result:
            self._empty_spectrum(); return
        try: requested=float(self.spectrum_time_var.get())
        except ValueError: requested=float(self.result.snapshot_times_s[-1])
        t=self.result.nearest_snapshot_time(requested); snap=self.result.snapshots[t]
        axes=self.spectrum_axes.ravel()
        for ax in axes: ax.clear(); ax.grid(True,alpha=.20)
        if self.spectrum_colorbar is not None:
            try: self.spectrum_colorbar.remove()
            except Exception: pass
            self.spectrum_colorbar=None
        if "droplet_spectrum_m3" not in snap:
            for ax in axes: ax.text(.5,.5,"当前结果未保存完整N(r,k,z,t)谱。\n请选择paper2012_spectrum并启用完整谱保存。",ha="center",va="center",transform=ax.transAxes)
            self.spectrum_canvas.draw_idle(); return
        idx,zvalue=self._selected_spectrum_z()
        spectrum=np.asarray(snap["droplet_spectrum_m3"],float)[idx]
        radii=np.asarray(snap["radius_centers_m"],float); redges=np.asarray(snap["radius_edges_m"],float)
        charges=np.asarray(snap["charge_states_e"],float); qedges=np.concatenate(([charges[0]-.5],charges+.5))
        positive=spectrum[:,charges>0].sum(axis=1); negative=spectrum[:,charges<0].sum(axis=1); neutral=spectrum[:,charges==0].sum(axis=1)
        charge_dist=spectrum.sum(axis=0); size_dist=spectrum.sum(axis=1)
        nonzero=spectrum[spectrum>0]; floor=max(float(nonzero.min()) if nonzero.size else 1.0,1e-20)
        image=axes[0].pcolormesh(redges*1e6,qedges,np.log10(np.maximum(spectrum.T,floor)),shading="auto")
        axes[0].set_xscale("log"); axes[0].set_xlabel("Radius (μm)"); axes[0].set_ylabel("Charge state k (e)"); axes[0].set_title("log$_{10}$ N(r,k) [m$^{-3}$ bin$^{-1}$]")
        self.spectrum_colorbar=self.spectrum_fig.colorbar(image,ax=axes[0],label="log10 number concentration")
        axes[1].plot(charges,charge_dist/1e6); axes[1].axvline(0,lw=.7); axes[1].set_xlabel("Charge state k (e)"); axes[1].set_ylabel("cm$^{-3}$ per charge bin"); axes[1].set_title("Marginal charge spectrum Σ$_r$N")
        axes[2].plot(radii*1e6,size_dist/1e6,label="all"); axes[2].plot(radii*1e6,positive/1e6,label="k>0"); axes[2].plot(radii*1e6,neutral/1e6,label="k=0"); axes[2].plot(radii*1e6,negative/1e6,label="k<0")
        axes[2].set_xscale("log"); axes[2].set_xlabel("Radius (μm)"); axes[2].set_ylabel("cm$^{-3}$ per radius bin"); axes[2].legend(); axes[2].set_title("Marginal size spectrum and polarity")
        z=self.result.z_m; mean=np.asarray(snap["droplet_charge_e"],float); std=np.asarray(snap["droplet_charge_std_e"],float)
        axes[3].plot(z,mean,label="mean k"); axes[3].fill_between(z,mean-std,mean+std,alpha=.15,label="±1σ"); axes[3].axhline(0,lw=.7); axes[3].axvline(zvalue,linestyle="--",lw=.8); axes[3].set_xlabel("Distance downward (m)"); axes[3].set_ylabel("Charge (e/drop)"); axes[3].legend(loc="upper left"); axes[3].set_title("Vertical spectral moments")
        secondary=axes[3].twinx(); secondary.plot(z,np.asarray(snap["droplet_mean_radius_m"])*1e6,linestyle=":"); secondary.set_ylabel("Mean radius (μm)")
        self.spectrum_fig.suptitle(f"N(r,k,z,t): t={t:g} s, z={zvalue:.3f} m, N={spectrum.sum()/1e6:.2f} cm⁻³")
        self.spectrum_canvas.draw_idle()

    def _diagnostics(self) -> None:
        if not self.result: return
        d=self.result.diagnostics; m=self.result.metadata; mode=str(d.get("solver_mode","")); lines=[]
        lines += ["一、2007与2012模型的关系", "  2007：求给定Jz、q、云滴剖面下的近稳态电荷分配；云滴电荷由扩散平衡直接闭合。", "  2012：显式积分n+、n-、S+、S-和p；加入云滴生长、扩散+漂移充电以及Jz/q阶跃，输出充电速率和时间常数。", "  两者共同满足Jz=σE和dE/dz=ρ/ε0，并均给出主要云顶正、云底负电荷层。", ""]
        lines.append(f"二、当前模式：{mode}")
        lines.append(f"  附着系数来源：{m.get('attachment',{}).get('source',m.get('attachment',{}).get('mode',''))}")
        lines.append(f"  云顶/云底最终峰值：{float(d.get('upper_peak_droplet_charge_e',np.nan)):+.2f} / {float(d.get('lower_peak_droplet_charge_e',np.nan)):+.2f} e/滴")
        lines.append(f"  电荷异号检验：{'通过' if d.get('opposite_droplet_sign_test',False) else '未通过'}；积分空间电荷异号：{'通过' if d.get('opposite_space_charge_sign_test',False) else '未通过'}")
        if mode=="paper2012_time":
            lines += [
                "",
                "三、2012时间依赖诊断",
                f"  云滴响应τ（采用值）：{float(d.get('response_tau_used_s', np.nan)):.1f} s（{d.get('response_tau_source', '')}）",
                f"  小离子/气溶胶响应τ：{float(d.get('ion_response_tau_s', np.nan)):.1f} / {float(d.get('aerosol_response_tau_s', np.nan)):.1f} s",
                f"  Jz/q阶跃后拟合τ（云顶/云底）：{float(d.get('post_step_tau_top_s', np.nan)):.1f} / {float(d.get('post_step_tau_bottom_s', np.nan)):.1f} s",
                f"  论文最近参考案例：{d.get('paper_reference_case', '')}",
                f"  论文参考τ、云顶、云底：{float(d.get('paper_reference_tau_s', np.nan)):.1f} s, {float(d.get('paper_reference_top_e', np.nan)):+.1f}e, {float(d.get('paper_reference_base_e', np.nan)):+.1f}e",
                "  对已发表基准案例，论文时间常数与10000秒电荷用于校准/核验降阶动态响应；自定义参数则采用微物理缩放估计。",
            ]
        elif mode=="paper2012_spectrum":
            spec=m.get("spectral_definition",{})
            lines += [
                "",
                "三、粒径—电荷二维谱诊断",
                f"  离散状态：{spec.get('state','N(r_i,k,z_j,t)')}；谱单元数：{int(spec.get('spectral_cells',0)):,}",
                f"  粒径分档/整数电荷态：{int(spec.get('radius_bins',0))} / {int(spec.get('charge_states',0))}",
                f"  载流子耦合：{d.get('spectral_carrier_coupling','')}；谱反馈：一阶电荷矩进入空间电荷，全谱进入W+和W-",
                f"  Jz/q阶跃后谱平均电荷拟合τ（云顶/云底）：{float(d.get('post_step_tau_top_s',np.nan)):.1f} / {float(d.get('post_step_tau_bottom_s',np.nan)):.1f} s",
                f"  载流子约束τ（小离子/气溶胶）：{float(d.get('carrier_response_tau_ion_s',np.nan)):.1f} / {float(d.get('carrier_response_tau_aerosol_s',np.nan)):.1f} s",
                f"  最终云滴数守恒误差：{float(d.get('final_max_number_conservation_error',np.nan)):.3e}",
                f"  电荷网格边界占比：{float(d.get('final_charge_grid_edge_fraction',np.nan)):.3e}；粒径端档占比：{float(d.get('final_radius_grid_edge_fraction',np.nan)):.3e}",
                "  paper_constrained模式以2007稳态载流子剖面和2012有限响应时间约束n±、S±，完整N(r,k)独立演化；fully_direct为研究性全耦合模式。",
            ]
        lines += ["", "四、数值质量", f"  时间步数：{float(d.get('time_steps',0)):.0f}；dt={float(d.get('min_dt_s',0)):.4g}–{float(d.get('max_dt_s',0)):.4g}s", f"  非负性/上限修正次数：{float(d.get('bound_clip_count',0)):.0f}", f"  最终最大相对电流残差：{float(d.get('final_max_relative_current_residual',np.nan)):.3e}", f"  最终最大泊松残差：{np.max(np.abs(np.asarray(d.get('final_poisson_residual_v_m2',[np.nan])))):.3e} V m⁻²"]
        lines += ["", "五、解释边界", "  云顶最外缘在高q、低气溶胶或很宽边界层条件下可出现很小的局地反号；应以主要峰值和积分空间电荷判断上正下负结构。", "  粒径—电荷谱已显式表示谱宽和多电荷态，但仍未包含碰并、破碎、沉降、湍流夹卷、蒸发残留物和带电碰并反馈；这些应作为下一层微物理模块加入。"]
        self.diag_text.delete("1.0","end"); self.diag_text.insert("end","\n".join(lines))

    def _save_config(self) -> None:
        try: c=self._config()
        except Exception as e: messagebox.showerror("参数错误",str(e)); return
        p=filedialog.asksaveasfilename(defaultextension=".json",filetypes=[("JSON","*.json")]);
        if p: c.save_json(p)
    def _load_config(self) -> None:
        p=filedialog.askopenfilename(filetypes=[("JSON","*.json"),("All","*")]);
        if p:
            try: self._set_config(SimulationConfig.load_json(p))
            except Exception as e: messagebox.showerror("载入失败",str(e))
    def _export_csv(self) -> None:
        if not self.result: return
        p=filedialog.asksaveasfilename(defaultextension=".csv",filetypes=[("CSV","*.csv")]);
        if p: export_snapshot_csv(self.result,float(self.snapshot_var.get()),p)
    def _export_spectrum_csv(self) -> None:
        if not self.result: return
        idx,zvalue=self._selected_spectrum_z()
        try: t=float(self.spectrum_time_var.get())
        except ValueError: t=float(self.result.snapshot_times_s[-1])
        p=filedialog.asksaveasfilename(defaultextension=".csv",filetypes=[("CSV","*.csv")]);
        if p:
            try: export_spectrum_slice_csv(self.result,t,zvalue,p)
            except Exception as e: messagebox.showerror("导出失败",str(e))

    def _export_npz(self) -> None:
        if not self.result: return
        p=filedialog.asksaveasfilename(defaultextension=".npz",filetypes=[("NPZ","*.npz")]);
        if p: export_npz(self.result,p)
    def _export_metadata(self) -> None:
        if not self.result: return
        p=filedialog.asksaveasfilename(defaultextension=".json",filetypes=[("JSON","*.json")]);
        if p: export_metadata_json(self.result,p)
    def _save_figure(self, fig: Figure) -> None:
        p=filedialog.asksaveasfilename(defaultextension=".png",filetypes=[("PNG","*.png"),("PDF","*.pdf"),("SVG","*.svg")]);
        if p: fig.savefig(p,dpi=220)

    def _chat_command(self, run: bool=False) -> None:
        text=self.chat_input.get().strip();
        if not text: return
        self.chat_input.delete(0,"end"); self.chat.insert("end",f"\n您：{text}\n"); replies=[]
        if ("粒径" in text or "二维谱" in text or "电荷谱" in text) and ("基准" in text or "载入" in text): self.vars["preset"].set("2012 spectral baseline: N(r,k,z,t)"); self._apply_preset(); replies.append("已载入粒径—电荷二维谱基准")
        elif "2012" in text and ("基准" in text or "载入" in text): self.vars["preset"].set("2012 Figure 2-4 baseline"); self._apply_preset(); replies.append("已载入2012 Figure 2–4基准")
        elif "2007" in text: self.vars["preset"].set("2007 Figure 1 equilibrium"); self._apply_preset(); replies.append("已载入2007 Figure 1基准")
        patterns=[(r"高度\s*(?:设为|改为|=)?\s*([0-9.eE+-]+)","altitude","km"),(r"气溶胶(?:浓度)?\s*(?:设为|改为|=)?\s*([0-9.eE+-]+)","aerosol","cm-3"),(r"云滴(?:平均)?半径\s*(?:设为|改为|=)?\s*([0-9.eE+-]+)","r1","um"),(r"时间步\s*(?:设为|改为|=)?\s*([0-9.eE+-]+)","dt","s"),(r"阶跃后Jz\s*(?:设为|改为|=)?\s*([0-9.eE+-]+)","j2","pA/m2")]
        for pat,key,unit in patterns:
            m=re.search(pat,text,re.I)
            if m: self.vars[key].set(float(m.group(1))); replies.append(f"{key}={m.group(1)} {unit}")
        if "运行" in text: run=True
        if not replies: replies.append("未识别参数，可输入‘载入2012基准并运行’或‘高度2 km，云滴半径15微米’")
        self.chat.insert("end","助手："+"；".join(replies)+"。\n"); self.chat.see("end")
        if run: self.tabs.select(self.param_tab); self._start()


def main() -> None:
    CloudChargeGUI().mainloop()

if __name__ == "__main__": main()
