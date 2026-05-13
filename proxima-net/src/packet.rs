//! Connectionless datagram primitive. Each packet is a `(src, dst, bytes)`
//! tuple sent/received atomically. Dispatched through
//! `Pipe::call(Request)` with method `"PACKET"`,
//! body bytes via `Response::ok(data)` / `request.body`.
//!
//! [`Packet`]'s `src` + `data` zip onto `proxima_codec::Datagram::decode`'s
//! `(peer, bytes)` parameters 1:1 — a protocol codec (`dns`,
//! `memcached-codec-trait` in `proxima-protocols`) decodes straight from
//! a received `Packet`, and a reply's `dst` is the `peer` a
//! `proxima_codec::Addressed` carries back out of `encode`.

use std::io;
use std::net::SocketAddr;
use std::task::{Context, Poll};

use bytes::Bytes;

#[derive(Debug, Clone)]
pub struct Packet {
    /// remote peer (received-from on rx, send-to on tx).
    pub src: SocketAddr,
    /// local-side address — usually the listener bind on rx.
    pub dst: SocketAddr,
    pub data: Bytes,
}

pub trait PacketListener: Send + Sync + 'static {
    fn poll_recv(&self, cx: &mut Context<'_>, buf: &mut [u8]) -> Poll<io::Result<Packet>>;

    fn poll_send(&self, cx: &mut Context<'_>, packet: &Packet) -> Poll<io::Result<()>>;

    fn local_addr(&self) -> Option<SocketAddr>;
}

pub trait PacketListenerExt: PacketListener {
    fn recv<'lifetime>(&'lifetime self, buf: &'lifetime mut [u8]) -> Recv<'lifetime, Self> {
        Recv {
            listener: self,
            buf,
        }
    }

    fn send<'lifetime>(&'lifetime self, packet: &'lifetime Packet) -> Send_<'lifetime, Self> {
        Send_ {
            listener: self,
            packet,
        }
    }
}

impl<T: PacketListener + ?Sized> PacketListenerExt for T {}

pub struct Recv<'lifetime, L: PacketListener + ?Sized> {
    listener: &'lifetime L,
    buf: &'lifetime mut [u8],
}

impl<L: PacketListener + ?Sized> std::future::Future for Recv<'_, L> {
    type Output = io::Result<Packet>;

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        this.listener.poll_recv(cx, this.buf)
    }
}

pub struct Send_<'lifetime, L: PacketListener + ?Sized> {
    listener: &'lifetime L,
    packet: &'lifetime Packet,
}

impl<L: PacketListener + ?Sized> std::future::Future for Send_<'_, L> {
    type Output = io::Result<()>;

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.into_ref().get_ref();
        this.listener.poll_send(cx, this.packet)
    }
}
