//! `impl Pipe` that spawns a [`CommandDescriptor`] as a subprocess with a
//! typed dispatch chain attached.
//!
//! # Composition
//!
//! `CommandPipe::call` composes:
//!
//! 1. [`spawn_and_dispatch`] — fork+exec the child with the
//!    dispatch chain wired to `extra_fd[7]` + `PROXIMA_DISPATCH_FD`
//!    in the child env. A sibling thread drives the chain over the
//!    parent end of a socketpair.
//! 2. [`FdPairPipe`] — wrap the (parent_write_fd, parent_read_fd)
//!    pair as a Pipe; its `call(request)` does the byte shuttle.
//! 3. Background reap — when the response body fully drains, a
//!    detached thread runs [`DispatchedChild::wait`] which joins
//!    the dispatch thread and `waitpid`s the child.
//!
//! # Required surface
//!
//! Construction is fluent and typestate-gated: both `command` and
//! `dispatch` must be set before `.build()` is callable. There is
//! no `new`; there is no "sandbox toggle" — for an un-shim'd
//! child, pass [`super::grounds::Empty`] or
//! [`super::grounds::Deny`] as the dispatch chain. The chain is
//! still live; it just never receives requests because nothing in
//! the child writes to `PROXIMA_DISPATCH_FD`.
use bytes::Bytes;

use std::ffi::CString;
use std::future::Future;
use std::marker::PhantomData;
use std::thread;

use futures::stream::{self, StreamExt};
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::{ProximaError, Request, Response, ResponseStream};

use super::descriptor::{CommandDescriptor, Stdio};
use super::dispatched::{DispatchedChild, spawn_and_dispatch};
use super::fd_pipe::FdPairPipe;
use super::libc_shim;
use super::protocol::{ChildRequest, ChildResponse};

/// A Pipe whose `call(request)` spawns `command` as a subprocess
/// with the dispatch chain `Chain` attached to `extra_fd[7]`.
pub struct CommandPipe<Chain> {
    command: CommandDescriptor,
    chain: Chain,
    libc_shim: bool,
}

impl CommandPipe<NotBuiltYet> {
    /// Entry to the fluent builder. Both `.command(..)` and
    /// `.dispatch(..)` must be called before `.build()` will
    /// type-check.
    #[must_use]
    pub fn builder() -> CommandPipeBuilder<Unset, Unset> {
        CommandPipeBuilder {
            command: Unset,
            chain: Unset,
            libc_shim: false,
            _marker: PhantomData,
        }
    }
}

/// Phantom placeholder used purely to give `CommandPipe::builder`
/// a stable receiver type. The real `CommandPipe<C>` is parameterised
/// by the user-supplied chain type.
pub enum NotBuiltYet {}

/// Typestate marker — corresponding field has not been set on the
/// builder.
pub struct Unset;

/// Typestate marker — corresponding field has been set on the
/// builder. Carries the supplied value.
pub struct Set<T>(T);

/// Fluent builder for [`CommandPipe`]. The two type parameters
/// track which required fields have been provided. `.build()` is
/// only defined when both are `Set<_>`, so forgetting either is a
/// compile error.
pub struct CommandPipeBuilder<CommandState, ChainState> {
    command: CommandState,
    chain: ChainState,
    libc_shim: bool,
    _marker: PhantomData<fn() -> (CommandState, ChainState)>,
}

impl<ChainState> CommandPipeBuilder<Unset, ChainState> {
    /// Supply the [`CommandDescriptor`] to spawn.
    #[must_use]
    pub fn command(
        self,
        command: CommandDescriptor,
    ) -> CommandPipeBuilder<Set<CommandDescriptor>, ChainState> {
        CommandPipeBuilder {
            command: Set(command),
            chain: self.chain,
            libc_shim: self.libc_shim,
            _marker: PhantomData,
        }
    }
}

impl<CommandState> CommandPipeBuilder<CommandState, Unset> {
    /// Supply the dispatch chain. Any [`Pipe`] mapping
    /// [`ChildRequest`] → [`ChildResponse`] works; e.g.
    /// [`super::grounds::Empty`], [`super::grounds::Deny`], or
    /// a [`super::operators::AndThen`] composition.
    #[must_use]
    pub fn dispatch<Chain>(self, chain: Chain) -> CommandPipeBuilder<CommandState, Set<Chain>>
    where
        Chain: SendPipe<In = ChildRequest, Out = ChildResponse, Err = ProximaError> + Clone,
    {
        CommandPipeBuilder {
            command: self.command,
            chain: Set(chain),
            libc_shim: self.libc_shim,
            _marker: PhantomData,
        }
    }
}

impl<CommandState, ChainState> CommandPipeBuilder<CommandState, ChainState> {
    /// Opt the spawned child into the libc-interpose shim. When
    /// set, [`Pipe::call`] adds the platform-correct preload env
    /// var (`DYLD_INSERT_LIBRARIES` on macOS, `LD_PRELOAD` on
    /// Linux) pointing at [`libc_shim::PATH`] to the child's env
    /// list. Default off; only the children of pipes built with
    /// this method see the shim. The parent process is unaffected.
    #[must_use]
    pub fn libc_shim(mut self) -> Self {
        self.libc_shim = true;
        self
    }
}

impl<Chain> CommandPipeBuilder<Set<CommandDescriptor>, Set<Chain>>
where
    Chain: SendPipe<In = ChildRequest, Out = ChildResponse, Err = ProximaError> + Clone,
{
    /// Finalise the builder. Only callable when both `command` and
    /// `dispatch` have been set — a missing field is a compile
    /// error, not a runtime panic.
    #[must_use]
    pub fn build(self) -> CommandPipe<Chain> {
        CommandPipe {
            command: self.command.0,
            chain: self.chain.0,
            libc_shim: self.libc_shim,
        }
    }
}

impl<Chain> SendPipe for CommandPipe<Chain>
where
    Chain: SendPipe<In = ChildRequest, Out = ChildResponse, Err = ProximaError> + Clone,
{
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let mut command = self.command.clone();
        command.stdin(Stdio::Piped).stdout(Stdio::Piped);
        let shim_result = if self.libc_shim {
            attach_libc_shim_env(&mut command)
        } else {
            Ok(())
        };
        let chain = self.chain.clone();

        async move {
            shim_result?;
            let mut dispatched = spawn_and_dispatch(&command, chain)?;

            let input_fd = dispatched
                .child
                .stdin
                .take()
                .ok_or_else(|| ProximaError::Body("spawn returned no Piped input fd".into()))?;
            let output_fd =
                dispatched.child.stdout.take().ok_or_else(|| {
                    ProximaError::Body("spawn returned no Piped output fd".into())
                })?;

            let inner_response =
                SendPipe::call(&FdPairPipe::with_child(input_fd, output_fd, 0), request).await?;

            let wrapped_body = wrap_body_with_dispatched(inner_response, dispatched);
            Ok(Response::streamed(wrapped_body))
        }
    }
}


// Stamp the platform-correct preload env var onto `command` so
// the dynamic loader links the interpose shim into the child.
// `CommandDescriptor::env` is the COMPLETE child env, so we overwrite any
// pre-existing entry for the same key — keeps behaviour stable
// when the caller has also called `inherit_current_env`.
fn attach_libc_shim_env(command: &mut CommandDescriptor) -> Result<(), ProximaError> {
    let key = CString::new(libc_shim::PRELOAD_ENV_VAR).map_err(|err| {
        ProximaError::Body(format!("libc-shim preload env var contains NUL: {err}"))
    })?;
    let value = CString::new(libc_shim::PATH)
        .map_err(|err| ProximaError::Body(format!("libc-shim path contains NUL: {err}")))?;
    command.env(key, value);
    Ok(())
}

// Wrap the response body so that when it fully drains, the
// DispatchedChild is `wait`ed on a background thread. The wait
// joins the dispatch thread and reaps the child pid; doing it
// from inside the async stream would block the executor since
// `libc::waitpid` is synchronous.
fn wrap_body_with_dispatched(
    response: Response<Bytes>,
    dispatched: DispatchedChild,
) -> ResponseStream {
    let stream = response.into_chunk_stream();
    ResponseStream::new(stream::unfold(
        (stream, Some(dispatched)),
        |(mut stream, mut handle)| async move {
            match stream.next().await {
                Some(item) => Some((item, (stream, handle))),
                None => {
                    if let Some(dispatched) = handle.take() {
                        thread::spawn(move || {
                            let _ = dispatched.wait();
                        });
                    }
                    None
                }
            }
        },
    ))
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::field_reassign_with_default,
        clippy::type_complexity,
        clippy::useless_vec,
        clippy::needless_range_loop,
        clippy::default_constructed_unit_structs
    )]

    use super::*;
    use crate::grounds::Empty;
    use bytes::Bytes;
    use futures::executor::block_on;
    use std::ffi::CString;

    fn cstr(text: &str) -> CString {
        CString::new(text).expect("test literal contains no interior NUL")
    }

    #[test]
    fn echo_hello_world_round_trip() {
        let mut command = CommandDescriptor::new(cstr("/bin/echo"));
        command.arg(cstr("hello world"));
        let pipe = CommandPipe::builder()
            .command(command)
            .dispatch(Empty)
            .build();

        let request = Request::builder()
            .method("POST")
            .path("/")
            .build()
            .expect("request builds with default empty body");

        let response = block_on(SendPipe::call(&pipe, request)).expect("echo runs");

        let mut body_stream = response.into_chunk_stream();
        let mut output = Vec::new();
        block_on(async {
            while let Some(chunk) = body_stream.next().await {
                output.extend_from_slice(&chunk.expect("body chunk ok"));
            }
        });

        assert_eq!(output, b"hello world\n");
    }

    #[test]
    fn tr_lowercases_request_body() {
        let mut command = CommandDescriptor::new(cstr("/usr/bin/tr"));
        command.arg(cstr("A-Z")).arg(cstr("a-z"));
        let pipe = CommandPipe::builder()
            .command(command)
            .dispatch(Empty)
            .build();

        let request_body = Bytes::from_static(b"HELLO PROXIMA");
        let request = Request::builder()
            .method("POST")
            .path("/")
            .body(request_body)
            .build()
            .expect("request builds with body");

        let response = block_on(SendPipe::call(&pipe, request)).expect("tr runs");

        let mut body_stream = response.into_chunk_stream();
        let mut output = Vec::new();
        block_on(async {
            while let Some(chunk) = body_stream.next().await {
                output.extend_from_slice(&chunk.expect("body chunk ok"));
            }
        });

        assert_eq!(output, b"hello proxima");
    }
}
