"""PyInstaller Standalone .EXE Executable Builder for Siren-VLC."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != "utf-8" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def build_executable(onefile: bool = False, debug_console: bool = True) -> str:
    """Build standalone Windows .exe using PyInstaller."""
    print(f"\n=======================================================", flush=True)
    print(f"📦 SIREN-VLC: Windows Standalone .EXE Builder", flush=True)
    print(f"   Bundle Mode   : {'Single File (.exe)' if onefile else 'Folder Bundle (Fast Startup)'}", flush=True)
    print(f"   Console Window: {'Enabled (Debugging)' if debug_console else 'Disabled (Windowed)'}", flush=True)
    print(f"=======================================================\n", flush=True)

    entrypoint = os.path.abspath("scripts/launch_vlc.py")
    dist_dir = os.path.abspath("dist")
    build_dir = os.path.abspath("build")

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name=Siren-VLC",
        "--clean",
        "--noconfirm",
        f"--distpath={dist_dir}",
        f"--workpath={build_dir}",
        "--add-data=src;src",
    ]

    if onefile:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")

    if not debug_console:
        cmd.append("--noconsole")

    cmd.append(entrypoint)

    print(f"Executing: {' '.join(cmd)}\n", flush=True)
    res = subprocess.run(cmd)

    if res.returncode != 0:
        raise RuntimeError("PyInstaller build failed!")

    if onefile:
        exe_path = os.path.join(dist_dir, "Siren-VLC.exe")
    else:
        exe_path = os.path.join(dist_dir, "Siren-VLC", "Siren-VLC.exe")

    print(f"\n=========================================================================================", flush=True)
    print(f"🎉 SIREN-VLC WINDOWS .EXE BUILD COMPLETE!", flush=True)
    print(f"-----------------------------------------------------------------------------------------", flush=True)
    print(f"   Executable Path : {exe_path}", flush=True)
    if os.path.exists(exe_path):
        print(f"   Executable Size : {os.path.getsize(exe_path) / (1024.0 * 1024.0):.2f} MB", flush=True)
    print(f"   Launch Command  : {exe_path}", flush=True)
    print(f"=========================================================================================\n", flush=True)

    return exe_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Siren-VLC Windows Standalone Executable.")
    parser.add_argument("--onefile", action="store_true", help="Build single standalone .exe")
    parser.add_argument("--noconsole", action="store_true", help="Disable console window")
    args = parser.parse_args()

    build_executable(onefile=args.onefile, debug_console=not args.noconsole)


if __name__ == "__main__":
    main()
