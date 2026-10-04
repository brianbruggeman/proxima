# caller toolchains may include this policy again; do not recurse through them.
include_guard(GLOBAL)

if(NOT "$ENV{TARGET}" STREQUAL "x86_64-pc-windows-msvc")
    message(FATAL_ERROR "windows-msvc-runtime requires Cargo TARGET=x86_64-pc-windows-msvc")
endif()

get_filename_component(_proxima_runtime_policy "${CMAKE_CURRENT_LIST_FILE}" REALPATH)
if("$ENV{HOST}" STREQUAL "$ENV{TARGET}")
    set(_proxima_toolchain_kind HOST)
else()
    set(_proxima_toolchain_kind TARGET)
endif()

# match cmake-rs precedence while skipping the injected policy itself.
foreach(_proxima_toolchain_key IN ITEMS
        "CMAKE_TOOLCHAIN_FILE_x86_64-pc-windows-msvc"
        "CMAKE_TOOLCHAIN_FILE_x86_64_pc_windows_msvc"
        "${_proxima_toolchain_kind}_CMAKE_TOOLCHAIN_FILE"
        "CMAKE_TOOLCHAIN_FILE")
    if(DEFINED ENV{${_proxima_toolchain_key}})
        set(_proxima_caller_toolchain "$ENV{${_proxima_toolchain_key}}")
        if(_proxima_caller_toolchain STREQUAL "")
            message(FATAL_ERROR "${_proxima_toolchain_key} names an empty caller toolchain")
        endif()
        # cmake resolves relative toolchains against build then source directories.
        if(NOT IS_ABSOLUTE "${_proxima_caller_toolchain}")
            if(EXISTS "${CMAKE_BINARY_DIR}/${_proxima_caller_toolchain}")
                set(_proxima_caller_toolchain "${CMAKE_BINARY_DIR}/${_proxima_caller_toolchain}")
            else()
                set(_proxima_caller_toolchain "${CMAKE_SOURCE_DIR}/${_proxima_caller_toolchain}")
            endif()
        endif()
        get_filename_component(_proxima_caller_toolchain "${_proxima_caller_toolchain}" REALPATH)
        if(NOT _proxima_caller_toolchain STREQUAL _proxima_runtime_policy)
            include("${_proxima_caller_toolchain}")
            break()
        endif()
    endif()
endforeach()

set(_proxima_runtime MultiThreadedDLL)
if(",$ENV{CARGO_CFG_TARGET_FEATURE}," MATCHES ",crt-static,")
    set(_proxima_runtime MultiThreaded)
endif()

# an empty property delegates runtime selection; a conflicting ABI must be explicit.
if(DEFINED CMAKE_MSVC_RUNTIME_LIBRARY
        AND NOT CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL ""
        AND NOT CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL _proxima_runtime)
    message(FATAL_ERROR
        "CMAKE_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY} conflicts with Cargo CRT ${_proxima_runtime}")
endif()
set(CMAKE_MSVC_RUNTIME_LIBRARY "${_proxima_runtime}" CACHE STRING "Cargo target C runtime" FORCE)
set(CMAKE_MSVC_RUNTIME_LIBRARY "${_proxima_runtime}")
