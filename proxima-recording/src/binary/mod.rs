#[cfg(feature = "std")]
pub mod bin_format;
#[cfg(feature = "alloc")]
pub mod frame;
#[cfg(feature = "std")]
pub mod index;
#[cfg(feature = "std")]
pub mod source;
#[cfg(feature = "alloc")]
mod wire;

#[cfg(feature = "std")]
pub use bin_format::BinFormat;
#[cfg(feature = "alloc")]
pub use frame::FrameEncoder;
#[cfg(feature = "std")]
pub use index::{INDEX_RECORD_BYTES, IndexReader, IndexRecord, IndexWriter};
#[cfg(feature = "std")]
pub use source::BinSource;
