# hiprtc probe (PROXNUM-617)

Why does hiprtc return `HIPRTC_ERROR_COMPILATION` (6) with an empty log from native
processes but succeed from python3 ctypes? Each flag of the probe tests one way
python.exe differs from a plain exe.

Build (x64 Native Tools prompt) and run:

    cl /nologo /O2 hiprtc_probe.c
    hiprtc_probe.exe [--default-dirs <hip_bin_dir>] [--preload <hip_bin_dir>] [--bigstack]

`<hip_bin_dir>` is the SDK bin, e.g. `C:\Program Files\AMD\ROCm\6.2\bin`.

Set comgr's own diagnostics first:

    set AMD_COMGR_EMIT_VERBOSE_LOGS=1
    set AMD_COMGR_REDIRECT_LOGS=stderr
    set AMD_COMGR_SAVE_TEMPS=1
    set AMD_LOG_LEVEL=4

Run order: plain; `--default-dirs`; `--preload`; `--bigstack` (each alone); then all three.

Reading the result: the flag that flips compile rc 6 to 0 names the cause.
- `--default-dirs` (a): hiprtc's lazy LoadLibrary of comgr/builtins resolves differently under
  LOAD_LIBRARY_SEARCH_DEFAULT_DIRS. Fix in the Rust HIP loader: call SetDefaultDllDirectories and
  AddDllDirectory(hip bin) before loading hiprtc.
- `--preload` (c): a wrong amd_comgr (e.g. System32 from the display driver) was being picked. Fix:
  LoadLibraryExW the SDK's amd_comgr*.dll and hiprtc-builtins*.dll by full path before compiling.
- `--bigstack` (b): comgr overflows the 1 MB default stack. Fix: run hiprtc compile on a thread
  with a 64 MB reserve (or link the host with /STACK:67108864).
- Several flags needed: combine the fixes. None flips it: read the comgr logs from the env block.

The printed `bound:` lines show which hiprtc, amd_comgr and hiprtc-builtins DLLs got loaded. If none
match, run `listdlls <pid>` (Sysinternals) while the probe is paused or on a loop.

Python's stack reserve, for comparison with MSVC's 1 MB:

    dumpbin /headers python.exe | findstr /i stack
