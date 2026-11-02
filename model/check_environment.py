from __future__ import annotations

import os
import platform
import sys


def main() -> None:
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    problems: list[str] = []
    warnings: list[str] = []
    for name in ("numpy", "scipy", "matplotlib", "numba"):
        try:
            module = __import__(name)
            print(f"{name}: {getattr(module, '__version__', 'available')}")
        except Exception as exc:
            problems.append(f"{name}: {exc}")
    try:
        import tkinter as tk
        print("tkinter: import available")
        try:
            root = tk.Tk()
            root.withdraw()
            print("Tk patchlevel:", root.tk.call("info", "patchlevel"))
            root.destroy()
        except Exception as exc:
            message = str(exc).lower()
            if os.name != "nt" and ("display" in message or not os.environ.get("DISPLAY")):
                warnings.append(f"Tk display test skipped in headless/inaccessible display session: {exc}")
            else:
                problems.append(f"tkinter display: {exc}")
    except Exception as exc:
        problems.append(f"tkinter import: {exc}")
    if warnings:
        print("\nWarnings:")
        for item in warnings:
            print(" -", item)
    if problems:
        print("\nEnvironment check failed:")
        for item in problems:
            print(" -", item)
        raise SystemExit(1)
    print("\nEnvironment is ready for command-line simulations.")
    if warnings:
        print("Run the GUI in a desktop session with an active display.")
    else:
        print("Environment is ready for the GUI.")


if __name__ == "__main__":
    main()
