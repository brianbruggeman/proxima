/// integers and pointers only; floats arrive later as proxima-numeric layouts
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    I8,
    I16,
    I32,
    I64,
    Ptr,
}
