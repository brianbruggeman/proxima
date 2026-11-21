// build: cl /nologo /O2 hiprtc_probe.c
// run:   hiprtc_probe.exe [--default-dirs <hip_bin_dir>] [--preload <hip_bin_dir>] [--bigstack]
// each flag isolates one difference from python.exe: --default-dirs is SetDefaultDllDirectories (a),
// --bigstack is a 64 MB main-thread-like stack (b), --preload pins the SDK comgr/builtins by full path (c).
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void *Program;
typedef int (*CreateFn)(Program *, const char *, const char *, int, const char **, const char **);
typedef int (*CompileFn)(Program, int, const char **);
typedef int (*SizeFn)(Program, size_t *);
typedef int (*LogFn)(Program, char *);

static const wchar_t *HIPRTC_NAMES[] = {L"hiprtc0602.dll", L"hiprtc0601.dll", L"hiprtc0507.dll"};
static const wchar_t *COMGR_NAMES[] = {L"amd_comgr.dll", L"amd_comgr_2.dll", L"amd_comgr_3.dll", L"amd_comgr0602.dll", L"amd_comgr0601.dll", L"amd_comgr0507.dll"};
static const wchar_t *BUILTIN_NAMES[] = {L"hiprtc-builtins.dll", L"hiprtc-builtins0602.dll", L"hiprtc-builtins0601.dll", L"hiprtc-builtins0507.dll"};
static const char *SMOKE_SRC = "extern \"C\" __global__ void smoke(float *out) { out[threadIdx.x] = 1.0f; }\n";
#define COUNT(table) (sizeof(table) / sizeof((table)[0]))

typedef struct {
    const wchar_t *default_dir;
    const wchar_t *preload_dir;
    int bigstack;
} Options;

static Options g_options;

static int parse_args(int argc, wchar_t **argv)
{
    for (int index = 1; index < argc; index++) {
        if (!wcscmp(argv[index], L"--bigstack")) {
            g_options.bigstack = 1;
        } else if (!wcscmp(argv[index], L"--default-dirs") && index + 1 < argc) {
            g_options.default_dir = argv[++index];
        } else if (!wcscmp(argv[index], L"--preload") && index + 1 < argc) {
            g_options.preload_dir = argv[++index];
        } else {
            fwprintf(stderr, L"usage: hiprtc_probe [--default-dirs <dir>] [--preload <dir>] [--bigstack]\n");
            return 0;
        }
    }
    return 1;
}

static void apply_default_dirs(const wchar_t *dir)
{
    BOOL set_ok = SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    DLL_DIRECTORY_COOKIE cookie = AddDllDirectory(dir);
    wprintf(L"default-dirs: SetDefaultDllDirectories=%d AddDllDirectory=%s (%s)\n", set_ok, cookie ? L"ok" : L"FAILED", dir);
}

static void preload_glob(const wchar_t *dir, const wchar_t *pattern)
{
    wchar_t glob[MAX_PATH], full[MAX_PATH];
    WIN32_FIND_DATAW found;
    swprintf(glob, MAX_PATH, L"%s\\%s", dir, pattern);
    HANDLE find = FindFirstFileW(glob, &found);
    if (find == INVALID_HANDLE_VALUE) {
        wprintf(L"preload: no match for %s\n", glob);
        return;
    }
    do {
        swprintf(full, MAX_PATH, L"%s\\%s", dir, found.cFileName);
        HMODULE module = LoadLibraryExW(full, NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        wprintf(L"preload: %s %s\n", module ? L"ok" : L"FAILED", full);
    } while (FindNextFileW(find, &found));
    FindClose(find);
}

static HMODULE load_hiprtc(void)
{
    for (size_t index = 0; index < COUNT(HIPRTC_NAMES); index++) {
        HMODULE module = LoadLibraryW(HIPRTC_NAMES[index]);
        if (module) {
            wprintf(L"loaded %s\n", HIPRTC_NAMES[index]);
            return module;
        }
    }
    wprintf(L"no hiprtc dll found (tried hiprtc0602/0601/0507)\n");
    return NULL;
}

static int report_bound(const wchar_t *const *names, size_t count)
{
    wchar_t path[MAX_PATH];
    int found = 0;
    for (size_t index = 0; index < count; index++) {
        HMODULE module = GetModuleHandleW(names[index]);
        if (module && GetModuleFileNameW(module, path, MAX_PATH)) {
            wprintf(L"bound: %s\n", path);
            found++;
        }
    }
    return found;
}

static void report_modules(void)
{
    int found = report_bound(HIPRTC_NAMES, COUNT(HIPRTC_NAMES));
    found += report_bound(COMGR_NAMES, COUNT(COMGR_NAMES));
    found += report_bound(BUILTIN_NAMES, COUNT(BUILTIN_NAMES));
    if (!found) {
        wprintf(L"no hiprtc/amd_comgr/hiprtc-builtins module matched; use listdlls <pid>\n");
    }
}

static void print_log(Program program, SizeFn log_size, LogFn log_text)
{
    size_t bytes = 0;
    int size_rc = log_size(program, &bytes);
    printf("log size rc=%d bytes=%zu\n", size_rc, bytes);
    char *text = (char *)calloc(bytes + 1, 1);
    if (text && bytes && log_text(program, text) == 0) {
        printf("log text:\n%s\n", text);
    }
    free(text);
}

static int compile_smoke(HMODULE module)
{
    CreateFn create = (CreateFn)GetProcAddress(module, "hiprtcCreateProgram");
    CompileFn compile = (CompileFn)GetProcAddress(module, "hiprtcCompileProgram");
    SizeFn log_size = (SizeFn)GetProcAddress(module, "hiprtcGetProgramLogSize");
    LogFn log_text = (LogFn)GetProcAddress(module, "hiprtcGetProgramLog");
    SizeFn code_size = (SizeFn)GetProcAddress(module, "hiprtcGetCodeSize");
    if (!create || !compile || !log_size || !log_text || !code_size) {
        printf("missing hiprtc export\n");
        return -1;
    }
    Program program = NULL;
    int create_rc = create(&program, SMOKE_SRC, "smoke.hip", 0, NULL, NULL);
    printf("create rc=%d\n", create_rc);
    int compile_rc = compile(program, 0, NULL);
    printf("compile rc=%d\n", compile_rc);
    print_log(program, log_size, log_text);
    if (compile_rc == 0) {
        size_t bytes = 0;
        int size_rc = code_size(program, &bytes);
        printf("code size rc=%d bytes=%zu\n", size_rc, bytes);
    }
    return compile_rc;
}

static int run_probe(void)
{
    if (g_options.default_dir) {
        apply_default_dirs(g_options.default_dir);
    }
    HMODULE module = load_hiprtc();
    if (!module) {
        return -1;
    }
    if (g_options.preload_dir) {
        preload_glob(g_options.preload_dir, L"amd_comgr*.dll");
        preload_glob(g_options.preload_dir, L"hiprtc-builtins*.dll");
    }
    int rc = compile_smoke(module);
    report_modules();
    return rc;
}

static DWORD WINAPI thread_main(LPVOID unused)
{
    (void)unused;
    return (DWORD)run_probe();
}

static int run_on_big_stack(void)
{
    HANDLE thread = CreateThread(NULL, 64u * 1024u * 1024u, thread_main, NULL, STACK_SIZE_PARAM_IS_A_RESERVATION, NULL);
    if (!thread) {
        printf("CreateThread failed: %lu\n", GetLastError());
        return -1;
    }
    WaitForSingleObject(thread, INFINITE);
    DWORD rc = 0;
    GetExitCodeThread(thread, &rc);
    CloseHandle(thread);
    printf("bigstack thread rc=%d\n", (int)rc);
    return (int)rc;
}

int wmain(int argc, wchar_t **argv)
{
    if (!parse_args(argc, argv)) {
        return 2;
    }
    return g_options.bigstack ? run_on_big_stack() : run_probe();
}
