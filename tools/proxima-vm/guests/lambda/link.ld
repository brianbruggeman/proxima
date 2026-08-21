ENTRY(_start)

MEMORY
{
    RAM (rwx) : ORIGIN = 0x0, LENGTH = 64M
}

SECTIONS
{
    . = ORIGIN(RAM);

    .text : {
        KEEP(*(.text._start))
        *(.text .text.*)
    } > RAM

    .rodata : {
        *(.rodata .rodata.*)
    } > RAM

    .data : {
        *(.data .data.*)
    } > RAM

    .bss (NOLOAD) : {
        __bss_start = .;
        *(.bss .bss.*)
        *(COMMON)
        __bss_end = .;
    } > RAM

    __stack_top = ORIGIN(RAM) + LENGTH(RAM);

    /DISCARD/ : {
        *(.comment)
        *(.eh_frame)
        *(.note.*)
        *(.ARM.attributes)
    }
}
