# proxima-clock

Hardware-clock source polymorphism expressed as `proxima-primitives` pipes: tick sources, wall-clock anchoring, and a coarse shared-time cell. The crate defines no clock trait of its own.

Add it with `cargo add proxima-clock`.

## Why there is no `Clock` trait

A monotonic tick source (a memory-mapped register read, `rdtsc`, `mrs CNTVCT_EL0`) takes nothing and produces a reading. That is the source form of the `Pipe` trait, `In = ()`, `Out = Ticks`, so binding a new hardware backend is "implement `Pipe` for your type". The crate's lib docs work through the other shapes the same way:

- a sleeper is a `Pipe` with `In = Ticks, Out = ()` whose future pends until the deadline (the sink form); this crate ships no reference sleeper, because a correct one needs an interrupt or a runtime's timer wheel;
- converting ticks to wall-clock time is a `Pipe` with `In = Ticks, Out = UnixNanos`, which is `ToUnixNanos`, composed onto a source with `.and_then`;
- arbitrating between redundant tick sources is `proxima_primitives::pipe::fan_in::FanIn`.

## What it provides

- `Ticks` and `UnixNanos`: value types so tick-domain and wall-clock-domain arithmetic cannot be mixed by accident.
- `TickCell` (`coarse`): a shared coarse-time cell a hardware-facing pipe can hold.
- `AnchorCell` and `ToUnixNanos` (`anchor`): the seqlock-protected `(ticks, unix_nanos)` correlation point, re-anchorable by an NTP or PTP discipline loop, and the pipe that applies it.

## What it composes

`proxima-primitives` (the `Pipe` trait). `proxima-telemetry` uses it for its std `SystemClock` bridge, and `prime` uses `TickCell` for its virtual-clock timer wheel.

## Features and tiers

Default feature: `std`.

- Bare no_std, no alloc: `ticks`, `unix_nanos`, `coarse` and `anchor` contain no `alloc::` items. Per the lib docs, `--no-default-features` builds for `thumbv7m-none-eabi` and runs the crate's unit tests.
- `std` (default): turns off `no_std`; adds no API.
- `config` (opt-in, implies `std`): a config plus fluent-builder surface (`bon`, `conflaguration`, `serde`) for `AnchorCell`'s initial `(ticks, unix_nanos)` pair. It sits at the std boundary because `conflaguration` is std-only.

## License

MIT OR Apache-2.0.

Repository: <https://github.com/brianbruggeman/proxima>
