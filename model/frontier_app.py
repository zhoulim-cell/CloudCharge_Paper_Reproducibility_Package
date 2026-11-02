from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np

from cloudcharge.frontier_config import FrontierConfig
from cloudcharge.frontier_solver import run_frontier_model
from cloudcharge.io import export_metadata_json, export_npz, export_snapshot_csv


class FrontierApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("云层电荷分布前沿模拟平台 v5.0")
        self.geometry("1450x900")
        self.minsize(1150, 720)
        self.cfg = FrontierConfig.load_json(Path(__file__).parent / "presets" / "frontier_baseline.json")
        self.result = None
        self.cancel = threading.Event()
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.vars: dict[str, tk.Variable] = {}
        self._build()
        self.after(100, self._poll)

    def _build(self) -> None:
        top = ttk.Frame(self); top.pack(fill="x", padx=8, pady=6)
        ttk.Button(top, text="载入参数", command=self.load_config).pack(side="left")
        ttk.Button(top, text="保存参数", command=self.save_config).pack(side="left", padx=4)
        self.run_btn = ttk.Button(top, text="运行", command=self.run); self.run_btn.pack(side="left", padx=(12, 4))
        ttk.Button(top, text="停止", command=self.cancel.set).pack(side="left")
        ttk.Button(top, text="导出结果", command=self.export).pack(side="left", padx=4)
        self.progress = ttk.Progressbar(top, length=300); self.progress.pack(side="left", padx=12)
        self.status = ttk.Label(top, text="就绪"); self.status.pack(side="left", fill="x", expand=True)

        nb = ttk.Notebook(self); nb.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.param_tab = ttk.Frame(nb); self.profile_tab = ttk.Frame(nb); self.spectrum_tab = ttk.Frame(nb); self.diag_tab = ttk.Frame(nb)
        nb.add(self.param_tab, text="研究参数")
        nb.add(self.profile_tab, text="剖面结果")
        nb.add(self.spectrum_tab, text="粒径—电荷谱")
        nb.add(self.diag_tab, text="守恒与诊断")
        self._build_params()
        self._build_profile_figure()
        self._build_spectrum_figure()
        self.diag_text = tk.Text(self.diag_tab, font=("Menlo", 11)); self.diag_text.pack(fill="both", expand=True)

    def _entry(self, parent, row, col, label, key, value, width=12):
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="e", padx=4, pady=3)
        var = tk.StringVar(value=str(value)); self.vars[key] = var
        ttk.Entry(parent, textvariable=var, width=width).grid(row=row, column=col+1, sticky="w", padx=4, pady=3)

    def _check(self, parent, row, col, label, key, value):
        var = tk.BooleanVar(value=value); self.vars[key] = var
        ttk.Checkbutton(parent, text=label, variable=var).grid(row=row, column=col, columnspan=2, sticky="w", padx=4, pady=3)

    def _build_params(self):
        canvas = tk.Canvas(self.param_tab); scrollbar = ttk.Scrollbar(self.param_tab, orient="vertical", command=canvas.yview)
        frame = ttk.Frame(canvas); frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=frame, anchor="nw"); canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True); scrollbar.pack(side="right", fill="y")
        groups = []
        for title in ["大气与云", "外部电流/电离强迫", "谱分辨率", "输运与电动力学", "电清除"]:
            lf = ttk.LabelFrame(frame, text=title); lf.pack(fill="x", padx=12, pady=7); groups.append(lf)
        g=groups[0]
        self._entry(g,0,0,"云高(km)","altitude",self.cfg.base.atmosphere.altitude_km)
        self._entry(g,0,2,"气溶胶浓度(m⁻³)","aerosol",self.cfg.base.aerosol.total_number_m3)
        self._entry(g,1,0,"云滴峰值(m⁻³)","droplet",self.cfg.base.droplet.fixed_peak_number_m3)
        self._entry(g,1,2,"云滴中值半径(µm)","radius_um",self.cfg.base.droplet.initial_radius_m*1e6)
        self._entry(g,2,0,"边界层厚度(m)","edge",self.cfg.base.domain.edge_ramp_m)
        self._entry(g,2,2,"垂直网格数","nz",self.cfg.base.domain.grid_points)
        g=groups[1]
        self._entry(g,0,0,"初始Jz(pA m⁻²)","j1",self.cfg.forcing.current_density_initial_a_m2/1e-12)
        self._entry(g,0,2,"第二阶段Jz(pA m⁻²)","j2",self.cfg.forcing.current_density_final_a_m2/1e-12)
        self._entry(g,1,0,"初始q(m⁻³ s⁻¹)","q1",self.cfg.forcing.ionization_initial_m3_s)
        self._entry(g,1,2,"第二阶段q(m⁻³ s⁻¹)","q2",self.cfg.forcing.ionization_final_m3_s)
        self._entry(g,2,0,"强迫切换时间(s)","step",self.cfg.forcing.step_time_s)
        self._entry(g,2,2,"总时长(s)","duration",self.cfg.numerics.duration_s)
        g=groups[2]
        self._entry(g,0,0,"云滴粒径档","drbins",self.cfg.base.spectrum.radius_bins)
        self._entry(g,0,2,"云滴最大电荷(e)","dqmax",self.cfg.base.spectrum.charge_max_e)
        self._entry(g,1,0,"气溶胶粒径档","arbins",self.cfg.aerosol_spectrum.radius_bins)
        self._entry(g,1,2,"气溶胶最大电荷(e)","aqmax",self.cfg.aerosol_spectrum.charge_max_e)
        g=groups[3]
        self._entry(g,0,0,"涡扩散Kz(m² s⁻¹)","kz",self.cfg.transport.eddy_diffusivity_m2_s)
        self._entry(g,0,2,"向下气流(m s⁻¹)","w",self.cfg.transport.air_velocity_down_m_s)
        self._entry(g,1,0,"Maxwell/Gauss权重","field_weight",self.cfg.electrodynamics.gauss_projection_relaxation)
        self._entry(g,1,2,"最大时间步(s)","dt",self.cfg.numerics.max_dt_s)
        self._check(g,2,0,"考虑位移电流","displacement",self.cfg.electrodynamics.include_displacement_current)
        self._check(g,2,2,"考虑液滴沉降","settling",self.cfg.transport.include_droplet_sedimentation)
        self._check(g,3,0,"考虑凝结粒径输运","condensation",self.cfg.transport.include_condensational_growth)
        g=groups[4]
        self._check(g,0,0,"启用带电气溶胶清除","scavenging",self.cfg.scavenging.enabled)
        self._check(g,0,2,"将气溶胶电荷转移到云滴","active_transfer",self.cfg.scavenging.active_transfer)
        self._entry(g,1,0,"清除核倍率","kernel_scale",self.cfg.scavenging.kernel_scale)

    def _build_profile_figure(self):
        self.profile_fig = Figure(figsize=(10,7), dpi=100); self.profile_axes = self.profile_fig.subplots(2,2)
        canvas = FigureCanvasTkAgg(self.profile_fig, master=self.profile_tab); canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(canvas, self.profile_tab); self.profile_canvas = canvas

    def _build_spectrum_figure(self):
        control=ttk.Frame(self.spectrum_tab); control.pack(fill="x")
        ttk.Label(control,text="高度(m)").pack(side="left",padx=4)
        self.zvar=tk.DoubleVar(value=7.0); ttk.Scale(control,from_=0,to=35,variable=self.zvar,command=lambda _: self.draw_spectra()).pack(side="left",fill="x",expand=True)
        self.spectrum_fig=Figure(figsize=(11,5),dpi=100); self.spectrum_axes=self.spectrum_fig.subplots(1,2)
        canvas=FigureCanvasTkAgg(self.spectrum_fig,master=self.spectrum_tab); canvas.get_tk_widget().pack(fill="both",expand=True)
        NavigationToolbar2Tk(canvas,self.spectrum_tab); self.spectrum_canvas=canvas

    def _make_config(self) -> FrontierConfig:
        c=FrontierConfig.from_dict(self.cfg.to_dict())
        f=lambda k: float(self.vars[k].get())
        c.base.atmosphere.altitude_km=f("altitude"); c.base.aerosol.total_number_m3=f("aerosol")
        c.base.droplet.fixed_peak_number_m3=f("droplet"); c.base.droplet.initial_radius_m=f("radius_um")*1e-6
        c.base.domain.edge_ramp_m=f("edge"); c.base.domain.grid_points=int(f("nz"))
        c.forcing.current_density_initial_a_m2=f("j1")*1e-12; c.forcing.current_density_final_a_m2=f("j2")*1e-12
        c.forcing.ionization_initial_m3_s=f("q1"); c.forcing.ionization_final_m3_s=f("q2")
        c.forcing.step_time_s=f("step"); c.numerics.duration_s=f("duration")
        c.base.spectrum.radius_bins=int(f("drbins")); qmax=int(f("dqmax")); c.base.spectrum.charge_min_e=-qmax; c.base.spectrum.charge_max_e=qmax
        c.aerosol_spectrum.radius_bins=int(f("arbins")); aqmax=int(f("aqmax")); c.aerosol_spectrum.charge_min_e=-aqmax; c.aerosol_spectrum.charge_max_e=aqmax
        c.transport.eddy_diffusivity_m2_s=f("kz"); c.transport.ion_eddy_diffusivity_m2_s=f("kz"); c.transport.air_velocity_down_m_s=f("w")
        c.electrodynamics.gauss_projection_relaxation=f("field_weight"); c.numerics.max_dt_s=f("dt")
        c.electrodynamics.include_displacement_current=bool(self.vars["displacement"].get())
        c.transport.include_droplet_sedimentation=bool(self.vars["settling"].get()); c.transport.include_condensational_growth=bool(self.vars["condensation"].get())
        c.scavenging.enabled=bool(self.vars["scavenging"].get()); c.scavenging.active_transfer=bool(self.vars["active_transfer"].get()); c.scavenging.kernel_scale=f("kernel_scale")
        c.numerics.snapshot_times_s=sorted(set([0.0,c.forcing.step_time_s,min(c.numerics.duration_s,c.forcing.step_time_s+60),c.numerics.duration_s]))
        c.validate(); return c

    def run(self):
        try: cfg=self._make_config()
        except Exception as e: messagebox.showerror("参数错误",str(e)); return
        self.cfg=cfg; self.cancel.clear(); self.run_btn.configure(state="disabled")
        def worker():
            try:
                r=run_frontier_model(cfg,progress_callback=lambda f,m:self.messages.put(("progress",(f,m))),cancel_callback=self.cancel.is_set)
                self.messages.put(("done",r))
            except Exception as e: self.messages.put(("error",e))
        threading.Thread(target=worker,daemon=True).start()

    def _poll(self):
        try:
            while True:
                kind,payload=self.messages.get_nowait()
                if kind=="progress": f,m=payload; self.progress["value"]=100*f; self.status.configure(text=m)
                elif kind=="done": self.result=payload; self.run_btn.configure(state="normal"); self.status.configure(text="完成"); self.draw_results()
                elif kind=="error": self.run_btn.configure(state="normal"); self.status.configure(text="失败"); messagebox.showerror("运行失败",str(payload))
        except queue.Empty: pass
        self.after(100,self._poll)

    def draw_results(self):
        if self.result is None:return
        t=float(self.result.snapshot_times_s[-1]); s=self.result.snapshots[t]; z=self.result.z_m
        for ax in self.profile_axes.ravel(): ax.clear()
        a=self.profile_axes
        a[0,0].plot(z,np.asarray(s["droplet_number_m3"])/1e6,label="droplet"); a[0,0].plot(z,np.asarray(s["aerosol_number_m3"])/1e6,label="aerosol"); a[0,0].legend(); a[0,0].set_title("粒子浓度 cm⁻³")
        a[0,1].plot(z,s["electric_field_v_m"]); a[0,1].set_title("向下电场 V m⁻¹")
        a[1,0].plot(z,s["droplet_charge_e"],label="droplet"); a[1,0].plot(z,s["aerosol_charge_e"],label="aerosol"); a[1,0].axhline(0,lw=.7); a[1,0].legend(); a[1,0].set_title("平均电荷 e")
        a[1,1].plot(z,np.asarray(s["charge_density_c_m3"])*1e12); a[1,1].axhline(0,lw=.7); a[1,1].set_title("空间电荷 pC m⁻³")
        for ax in a.ravel(): ax.set_xlabel("向下距离(m)"); ax.grid(alpha=.25)
        self.profile_canvas.draw_idle(); self.draw_spectra(); self.draw_diagnostics()

    def draw_spectra(self):
        if self.result is None:return
        t=float(self.result.snapshot_times_s[-1]); s=self.result.snapshots[t]; idx=int(np.argmin(np.abs(self.result.z_m-self.zvar.get())))
        for ax in self.spectrum_axes: ax.clear()
        ds=np.asarray(s["droplet_spectrum_m3"])[idx]; dr=np.asarray(s["droplet_radius_centers_m"])*1e6; dq=np.asarray(s["droplet_charge_states_e"])
        ass=np.asarray(s["aerosol_spectrum_m3"])[idx]; ar=np.asarray(s["aerosol_radius_centers_m"])*1e9; aq=np.asarray(s["aerosol_charge_states_e"])
        self.spectrum_axes[0].pcolormesh(dr,dq,np.log10(np.maximum(ds.T,1e-20)),shading="auto"); self.spectrum_axes[0].set_xscale("log"); self.spectrum_axes[0].set_title("云滴 N(r,k)"); self.spectrum_axes[0].set_xlabel("µm"); self.spectrum_axes[0].set_ylabel("e")
        self.spectrum_axes[1].pcolormesh(ar,aq,np.log10(np.maximum(ass.T,1e-20)),shading="auto"); self.spectrum_axes[1].set_xscale("log"); self.spectrum_axes[1].set_title("气溶胶 N(a,k)"); self.spectrum_axes[1].set_xlabel("nm"); self.spectrum_axes[1].set_ylabel("e")
        self.spectrum_fig.suptitle(f"t={t:g}s, z={self.result.z_m[idx]:.2f}m"); self.spectrum_canvas.draw_idle()

    def draw_diagnostics(self):
        d=self.result.diagnostics
        summary={k:d[k] for k in ["upper_peak_droplet_charge_e","lower_peak_droplet_charge_e","upper_integrated_space_charge_c_m2","lower_integrated_space_charge_c_m2","opposite_droplet_sign_test","opposite_space_charge_sign_test","final_max_gauss_residual_v_m2","final_max_relative_gauss_residual","final_max_boundary_gauss_residual_v_m2","final_max_relative_total_current_residual","time_steps"]}
        self.diag_text.delete("1.0","end"); self.diag_text.insert("end",json.dumps(summary,ensure_ascii=False,indent=2))

    def load_config(self):
        p=filedialog.askopenfilename(filetypes=[("JSON","*.json")]);
        if not p:return
        try:self.cfg=FrontierConfig.load_json(p); messagebox.showinfo("完成","参数已载入。请重启界面或手动调整当前字段。")
        except Exception as e:messagebox.showerror("错误",str(e))

    def save_config(self):
        try:c=self._make_config()
        except Exception as e:messagebox.showerror("错误",str(e));return
        p=filedialog.asksaveasfilename(defaultextension=".json",filetypes=[("JSON","*.json")]);
        if p:c.save_json(p)

    def export(self):
        if self.result is None:messagebox.showwarning("无结果","请先运行模拟。");return
        d=filedialog.askdirectory();
        if not d:return
        p=Path(d); export_npz(self.result,p/"frontier_result.npz"); export_metadata_json(self.result,p/"frontier_metadata.json"); export_snapshot_csv(self.result,float(self.result.snapshot_times_s[-1]),p/"frontier_final.csv"); messagebox.showinfo("完成",f"结果已导出到 {p}")


if __name__ == "__main__":
    FrontierApp().mainloop()
