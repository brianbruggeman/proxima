ENTRY(_start)

MEMORY
{
    RAM (rwx) : ORIGIN = 0x0, LENGTH = 2M
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
        *(.got .got.*)
    } > RAM

    .bss (NOLOAD) : {
        __bss_start = .;
        *(.bss .bss.*)
        *(COMMON)
        __bss_end = .;
    } > RAM

    .stack (NOLOAD) : ALIGN(4096) {
        . += 64K;
        __stack_top = .;
    } > RAM

    /DISCARD/ : {
        *(.comment)
        *(.eh_frame)
        *(.note.*)
        *(.ARM.attributes)
    }
}
