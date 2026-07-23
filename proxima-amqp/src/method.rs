//! AMQP 0-9-1 method arguments — typed decode/encode for the
//! `class_id`/`method_id`/`args` triple
//! `proxima_protocols::amqp::Frame::Method` carries. Field layouts are the
//! AMQP 0-9-1 spec (stable since 2008); only the subset a broker facade
//! needs is implemented — connection/channel lifecycle, exchange/queue
//! declare + bind, and `basic.{qos,consume,cancel,publish,deliver,ack,nack}`.
//! Any other `(class_id, method_id)` decodes to
//! [`MethodError::Unsupported`] — see the crate-level docs for the exact
//! excluded set (`tx.*`, `confirm.*`, `basic.get`, `channel.flow`,
//! `exchange.delete`, `queue.{purge,unbind,delete}`).
//!
//! `id` groups the `(class_id, method_id)` constants this module matches
//! against; [`decode`] and [`encode`] are the two directions.

use crate::wire::{
    FieldTable, WireError, read_bit_flags, read_field_table, read_long, read_longlong,
    read_longstr, read_octet, read_short, read_shortstr, write_bit_flags, write_field_table,
    write_long, write_longlong, write_longstr, write_octet, write_short, write_shortstr,
};

/// `(class_id, method_id)` constants for every method [`decode`]/[`encode`]
/// handle.
pub mod id {
    pub const CONNECTION: u16 = 10;
    pub const CONNECTION_START: u16 = 10;
    pub const CONNECTION_START_OK: u16 = 11;
    pub const CONNECTION_TUNE: u16 = 30;
    pub const CONNECTION_TUNE_OK: u16 = 31;
    pub const CONNECTION_OPEN: u16 = 40;
    pub const CONNECTION_OPEN_OK: u16 = 41;
    pub const CONNECTION_CLOSE: u16 = 50;
    pub const CONNECTION_CLOSE_OK: u16 = 51;

    pub const CHANNEL: u16 = 20;
    pub const CHANNEL_OPEN: u16 = 10;
    pub const CHANNEL_OPEN_OK: u16 = 11;
    pub const CHANNEL_CLOSE: u16 = 40;
    pub const CHANNEL_CLOSE_OK: u16 = 41;

    pub const EXCHANGE: u16 = 40;
    pub const EXCHANGE_DECLARE: u16 = 10;
    pub const EXCHANGE_DECLARE_OK: u16 = 11;

    pub const QUEUE: u16 = 50;
    pub const QUEUE_DECLARE: u16 = 10;
    pub const QUEUE_DECLARE_OK: u16 = 11;
    pub const QUEUE_BIND: u16 = 20;
    pub const QUEUE_BIND_OK: u16 = 21;

    pub const BASIC: u16 = 60;
    pub const BASIC_QOS: u16 = 10;
    pub const BASIC_QOS_OK: u16 = 11;
    pub const BASIC_CONSUME: u16 = 20;
    pub const BASIC_CONSUME_OK: u16 = 21;
    pub const BASIC_CANCEL: u16 = 30;
    pub const BASIC_CANCEL_OK: u16 = 31;
    pub const BASIC_PUBLISH: u16 = 40;
    pub const BASIC_DELIVER: u16 = 60;
    pub const BASIC_ACK: u16 = 80;
    pub const BASIC_NACK: u16 = 120;
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MethodError {
    #[error("method arguments: {0}")]
    Wire(#[from] WireError),
    #[error("unsupported method class={class_id} method={method_id}")]
    Unsupported { class_id: u16, method_id: u16 },
}

/// One decoded AMQP method, owned (every string/table is copied out of the
/// frame's borrowed payload — needed because a `basic.publish` method must
/// outlive the header/body frames that complete it, which arrive in later
/// reads).
#[derive(Debug, Clone, PartialEq)]
pub enum Method {
    ConnectionStart {
        version_major: u8,
        version_minor: u8,
        server_properties: FieldTable,
        mechanisms: Vec<u8>,
        locales: Vec<u8>,
    },
    ConnectionStartOk {
        client_properties: FieldTable,
        mechanism: Vec<u8>,
        response: Vec<u8>,
        locale: Vec<u8>,
    },
    ConnectionTune {
        channel_max: u16,
        frame_max: u32,
        heartbeat: u16,
    },
    ConnectionTuneOk {
        channel_max: u16,
        frame_max: u32,
        heartbeat: u16,
    },
    ConnectionOpen {
        virtual_host: Vec<u8>,
    },
    ConnectionOpenOk,
    ConnectionClose {
        reply_code: u16,
        reply_text: Vec<u8>,
        class_id: u16,
        method_id: u16,
    },
    ConnectionCloseOk,
    ChannelOpen,
    ChannelOpenOk,
    ChannelClose {
        reply_code: u16,
        reply_text: Vec<u8>,
        class_id: u16,
        method_id: u16,
    },
    ChannelCloseOk,
    ExchangeDeclare {
        exchange: Vec<u8>,
        kind: Vec<u8>,
        passive: bool,
        durable: bool,
        auto_delete: bool,
        internal: bool,
        no_wait: bool,
        arguments: FieldTable,
    },
    ExchangeDeclareOk,
    QueueDeclare {
        queue: Vec<u8>,
        passive: bool,
        durable: bool,
        exclusive: bool,
        auto_delete: bool,
        no_wait: bool,
        arguments: FieldTable,
    },
    QueueDeclareOk {
        queue: Vec<u8>,
        message_count: u32,
        consumer_count: u32,
    },
    QueueBind {
        queue: Vec<u8>,
        exchange: Vec<u8>,
        routing_key: Vec<u8>,
        no_wait: bool,
        arguments: FieldTable,
    },
    QueueBindOk,
    BasicQos {
        prefetch_size: u32,
        prefetch_count: u16,
        global: bool,
    },
    BasicQosOk,
    BasicConsume {
        queue: Vec<u8>,
        consumer_tag: Vec<u8>,
        no_local: bool,
        no_ack: bool,
        exclusive: bool,
        no_wait: bool,
        arguments: FieldTable,
    },
    BasicConsumeOk {
        consumer_tag: Vec<u8>,
    },
    BasicCancel {
        consumer_tag: Vec<u8>,
        no_wait: bool,
    },
    BasicCancelOk {
        consumer_tag: Vec<u8>,
    },
    BasicPublish {
        exchange: Vec<u8>,
        routing_key: Vec<u8>,
        mandatory: bool,
        immediate: bool,
    },
    BasicDeliver {
        consumer_tag: Vec<u8>,
        delivery_tag: u64,
        redelivered: bool,
        exchange: Vec<u8>,
        routing_key: Vec<u8>,
    },
    BasicAck {
        delivery_tag: u64,
        multiple: bool,
    },
    BasicNack {
        delivery_tag: u64,
        multiple: bool,
        requeue: bool,
    },
}

/// Decodes one method's arguments given its `class_id`/`method_id` (already
/// split off the frame payload by `proxima_protocols::amqp::parse_frame`).
///
/// # Errors
/// [`MethodError::Unsupported`] for a class/method pair outside the
/// implemented subset; [`MethodError::Wire`] on a truncated or malformed
/// field.
pub fn decode(class_id: u16, method_id: u16, args: &[u8]) -> Result<Method, MethodError> {
    match (class_id, method_id) {
        (id::CONNECTION, id::CONNECTION_START) => {
            let (version_major, rest) = read_octet(args)?;
            let (version_minor, rest) = read_octet(rest)?;
            let (server_properties, rest) = read_field_table(rest)?;
            let (mechanisms, rest) = read_longstr(rest)?;
            let (locales, _rest) = read_longstr(rest)?;
            Ok(Method::ConnectionStart {
                version_major,
                version_minor,
                server_properties,
                mechanisms: mechanisms.to_vec(),
                locales: locales.to_vec(),
            })
        }
        (id::CONNECTION, id::CONNECTION_START_OK) => {
            let (client_properties, rest) = read_field_table(args)?;
            let (mechanism, rest) = read_shortstr(rest)?;
            let (response, rest) = read_longstr(rest)?;
            let (locale, _rest) = read_shortstr(rest)?;
            Ok(Method::ConnectionStartOk {
                client_properties,
                mechanism: mechanism.to_vec(),
                response: response.to_vec(),
                locale: locale.to_vec(),
            })
        }
        (id::CONNECTION, id::CONNECTION_TUNE) => {
            let (channel_max, rest) = read_short(args)?;
            let (frame_max, rest) = read_long(rest)?;
            let (heartbeat, _rest) = read_short(rest)?;
            Ok(Method::ConnectionTune {
                channel_max,
                frame_max,
                heartbeat,
            })
        }
        (id::CONNECTION, id::CONNECTION_TUNE_OK) => {
            let (channel_max, rest) = read_short(args)?;
            let (frame_max, rest) = read_long(rest)?;
            let (heartbeat, _rest) = read_short(rest)?;
            Ok(Method::ConnectionTuneOk {
                channel_max,
                frame_max,
                heartbeat,
            })
        }
        (id::CONNECTION, id::CONNECTION_OPEN) => {
            let (virtual_host, rest) = read_shortstr(args)?;
            let (_reserved_capabilities, rest) = read_shortstr(rest)?;
            let (_reserved_insist, _rest) = read_bit_flags(rest)?;
            Ok(Method::ConnectionOpen {
                virtual_host: virtual_host.to_vec(),
            })
        }
        (id::CONNECTION, id::CONNECTION_OPEN_OK) => {
            let (_reserved_known_hosts, _rest) = read_shortstr(args)?;
            Ok(Method::ConnectionOpenOk)
        }
        (id::CONNECTION, id::CONNECTION_CLOSE) => {
            let (reply_code, reply_text, class_id, method_id) = decode_close(args)?;
            Ok(Method::ConnectionClose {
                reply_code,
                reply_text,
                class_id,
                method_id,
            })
        }
        (id::CONNECTION, id::CONNECTION_CLOSE_OK) => Ok(Method::ConnectionCloseOk),
        (id::CHANNEL, id::CHANNEL_OPEN) => {
            let (_reserved, _rest) = read_shortstr(args)?;
            Ok(Method::ChannelOpen)
        }
        (id::CHANNEL, id::CHANNEL_OPEN_OK) => {
            let (_reserved, _rest) = read_longstr(args)?;
            Ok(Method::ChannelOpenOk)
        }
        (id::CHANNEL, id::CHANNEL_CLOSE) => {
            let (reply_code, reply_text, class_id, method_id) = decode_close(args)?;
            Ok(Method::ChannelClose {
                reply_code,
                reply_text,
                class_id,
                method_id,
            })
        }
        (id::CHANNEL, id::CHANNEL_CLOSE_OK) => Ok(Method::ChannelCloseOk),
        (id::EXCHANGE, id::EXCHANGE_DECLARE) => {
            let (_reserved_ticket, rest) = read_short(args)?;
            let (exchange, rest) = read_shortstr(rest)?;
            let (kind, rest) = read_shortstr(rest)?;
            let (flags, rest) = read_bit_flags(rest)?;
            let (arguments, _rest) = read_field_table(rest)?;
            Ok(Method::ExchangeDeclare {
                exchange: exchange.to_vec(),
                kind: kind.to_vec(),
                passive: flags[0],
                durable: flags[1],
                auto_delete: flags[2],
                internal: flags[3],
                no_wait: flags[4],
                arguments,
            })
        }
        (id::EXCHANGE, id::EXCHANGE_DECLARE_OK) => Ok(Method::ExchangeDeclareOk),
        (id::QUEUE, id::QUEUE_DECLARE) => {
            let (_reserved_ticket, rest) = read_short(args)?;
            let (queue, rest) = read_shortstr(rest)?;
            let (flags, rest) = read_bit_flags(rest)?;
            let (arguments, _rest) = read_field_table(rest)?;
            Ok(Method::QueueDeclare {
                queue: queue.to_vec(),
                passive: flags[0],
                durable: flags[1],
                exclusive: flags[2],
                auto_delete: flags[3],
                no_wait: flags[4],
                arguments,
            })
        }
        (id::QUEUE, id::QUEUE_DECLARE_OK) => {
            let (queue, rest) = read_shortstr(args)?;
            let (message_count, rest) = read_long(rest)?;
            let (consumer_count, _rest) = read_long(rest)?;
            Ok(Method::QueueDeclareOk {
                queue: queue.to_vec(),
                message_count,
                consumer_count,
            })
        }
        (id::QUEUE, id::QUEUE_BIND) => {
            let (_reserved_ticket, rest) = read_short(args)?;
            let (queue, rest) = read_shortstr(rest)?;
            let (exchange, rest) = read_shortstr(rest)?;
            let (routing_key, rest) = read_shortstr(rest)?;
            let (flags, rest) = read_bit_flags(rest)?;
            let (arguments, _rest) = read_field_table(rest)?;
            Ok(Method::QueueBind {
                queue: queue.to_vec(),
                exchange: exchange.to_vec(),
                routing_key: routing_key.to_vec(),
                no_wait: flags[0],
                arguments,
            })
        }
        (id::QUEUE, id::QUEUE_BIND_OK) => Ok(Method::QueueBindOk),
        (id::BASIC, id::BASIC_QOS) => {
            let (prefetch_size, rest) = read_long(args)?;
            let (prefetch_count, rest) = read_short(rest)?;
            let (flags, _rest) = read_bit_flags(rest)?;
            Ok(Method::BasicQos {
                prefetch_size,
                prefetch_count,
                global: flags[0],
            })
        }
        (id::BASIC, id::BASIC_QOS_OK) => Ok(Method::BasicQosOk),
        (id::BASIC, id::BASIC_CONSUME) => {
            let (_reserved_ticket, rest) = read_short(args)?;
            let (queue, rest) = read_shortstr(rest)?;
            let (consumer_tag, rest) = read_shortstr(rest)?;
            let (flags, rest) = read_bit_flags(rest)?;
            let (arguments, _rest) = read_field_table(rest)?;
            Ok(Method::BasicConsume {
                queue: queue.to_vec(),
                consumer_tag: consumer_tag.to_vec(),
                no_local: flags[0],
                no_ack: flags[1],
                exclusive: flags[2],
                no_wait: flags[3],
                arguments,
            })
        }
        (id::BASIC, id::BASIC_CONSUME_OK) => {
            let (consumer_tag, _rest) = read_shortstr(args)?;
            Ok(Method::BasicConsumeOk {
                consumer_tag: consumer_tag.to_vec(),
            })
        }
        (id::BASIC, id::BASIC_CANCEL) => {
            let (consumer_tag, rest) = read_shortstr(args)?;
            let (flags, _rest) = read_bit_flags(rest)?;
            Ok(Method::BasicCancel {
                consumer_tag: consumer_tag.to_vec(),
                no_wait: flags[0],
            })
        }
        (id::BASIC, id::BASIC_CANCEL_OK) => {
            let (consumer_tag, _rest) = read_shortstr(args)?;
            Ok(Method::BasicCancelOk {
                consumer_tag: consumer_tag.to_vec(),
            })
        }
        (id::BASIC, id::BASIC_PUBLISH) => {
            let (_reserved_ticket, rest) = read_short(args)?;
            let (exchange, rest) = read_shortstr(rest)?;
            let (routing_key, rest) = read_shortstr(rest)?;
            let (flags, _rest) = read_bit_flags(rest)?;
            Ok(Method::BasicPublish {
                exchange: exchange.to_vec(),
                routing_key: routing_key.to_vec(),
                mandatory: flags[0],
                immediate: flags[1],
            })
        }
        (id::BASIC, id::BASIC_DELIVER) => {
            let (consumer_tag, rest) = read_shortstr(args)?;
            let (delivery_tag, rest) = read_longlong(rest)?;
            let (flags, rest) = read_bit_flags(rest)?;
            let (exchange, rest) = read_shortstr(rest)?;
            let (routing_key, _rest) = read_shortstr(rest)?;
            Ok(Method::BasicDeliver {
                consumer_tag: consumer_tag.to_vec(),
                delivery_tag,
                redelivered: flags[0],
                exchange: exchange.to_vec(),
                routing_key: routing_key.to_vec(),
            })
        }
        (id::BASIC, id::BASIC_ACK) => {
            let (delivery_tag, rest) = read_longlong(args)?;
            let (flags, _rest) = read_bit_flags(rest)?;
            Ok(Method::BasicAck {
                delivery_tag,
                multiple: flags[0],
            })
        }
        (id::BASIC, id::BASIC_NACK) => {
            let (delivery_tag, rest) = read_longlong(args)?;
            let (flags, _rest) = read_bit_flags(rest)?;
            Ok(Method::BasicNack {
                delivery_tag,
                multiple: flags[0],
                requeue: flags[1],
            })
        }
        (class_id, method_id) => Err(MethodError::Unsupported {
            class_id,
            method_id,
        }),
    }
}

fn decode_close(args: &[u8]) -> Result<(u16, Vec<u8>, u16, u16), WireError> {
    let (reply_code, rest) = read_short(args)?;
    let (reply_text, rest) = read_shortstr(rest)?;
    let (class_id, rest) = read_short(rest)?;
    let (method_id, _rest) = read_short(rest)?;
    Ok((reply_code, reply_text.to_vec(), class_id, method_id))
}

fn encode_close(
    out: &mut Vec<u8>,
    reply_code: u16,
    reply_text: &[u8],
    class_id: u16,
    method_id: u16,
) {
    write_short(out, reply_code);
    write_shortstr(out, reply_text);
    write_short(out, class_id);
    write_short(out, method_id);
}

/// Encodes one method's arguments, returning `(class_id, method_id, args)`
/// — the caller wraps `args` in a `Frame::Method` envelope.
#[must_use]
pub fn encode(method: &Method) -> (u16, u16, Vec<u8>) {
    let mut out = Vec::new();
    let ids = match method {
        Method::ConnectionStart {
            version_major,
            version_minor,
            server_properties,
            mechanisms,
            locales,
        } => {
            write_octet(&mut out, *version_major);
            write_octet(&mut out, *version_minor);
            write_field_table(&mut out, server_properties);
            write_longstr(&mut out, mechanisms);
            write_longstr(&mut out, locales);
            (id::CONNECTION, id::CONNECTION_START)
        }
        Method::ConnectionStartOk {
            client_properties,
            mechanism,
            response,
            locale,
        } => {
            write_field_table(&mut out, client_properties);
            write_shortstr(&mut out, mechanism);
            write_longstr(&mut out, response);
            write_shortstr(&mut out, locale);
            (id::CONNECTION, id::CONNECTION_START_OK)
        }
        Method::ConnectionTune {
            channel_max,
            frame_max,
            heartbeat,
        } => {
            write_short(&mut out, *channel_max);
            write_long(&mut out, *frame_max);
            write_short(&mut out, *heartbeat);
            (id::CONNECTION, id::CONNECTION_TUNE)
        }
        Method::ConnectionTuneOk {
            channel_max,
            frame_max,
            heartbeat,
        } => {
            write_short(&mut out, *channel_max);
            write_long(&mut out, *frame_max);
            write_short(&mut out, *heartbeat);
            (id::CONNECTION, id::CONNECTION_TUNE_OK)
        }
        Method::ConnectionOpen { virtual_host } => {
            write_shortstr(&mut out, virtual_host);
            write_shortstr(&mut out, b"");
            write_bit_flags(&mut out, &[false]);
            (id::CONNECTION, id::CONNECTION_OPEN)
        }
        Method::ConnectionOpenOk => {
            write_shortstr(&mut out, b"");
            (id::CONNECTION, id::CONNECTION_OPEN_OK)
        }
        Method::ConnectionClose {
            reply_code,
            reply_text,
            class_id,
            method_id,
        } => {
            encode_close(&mut out, *reply_code, reply_text, *class_id, *method_id);
            (id::CONNECTION, id::CONNECTION_CLOSE)
        }
        Method::ConnectionCloseOk => (id::CONNECTION, id::CONNECTION_CLOSE_OK),
        Method::ChannelOpen => {
            write_shortstr(&mut out, b"");
            (id::CHANNEL, id::CHANNEL_OPEN)
        }
        Method::ChannelOpenOk => {
            write_longstr(&mut out, b"");
            (id::CHANNEL, id::CHANNEL_OPEN_OK)
        }
        Method::ChannelClose {
            reply_code,
            reply_text,
            class_id,
            method_id,
        } => {
            encode_close(&mut out, *reply_code, reply_text, *class_id, *method_id);
            (id::CHANNEL, id::CHANNEL_CLOSE)
        }
        Method::ChannelCloseOk => (id::CHANNEL, id::CHANNEL_CLOSE_OK),
        Method::ExchangeDeclare {
            exchange,
            kind,
            passive,
            durable,
            auto_delete,
            internal,
            no_wait,
            arguments,
        } => {
            write_short(&mut out, 0);
            write_shortstr(&mut out, exchange);
            write_shortstr(&mut out, kind);
            write_bit_flags(
                &mut out,
                &[*passive, *durable, *auto_delete, *internal, *no_wait],
            );
            write_field_table(&mut out, arguments);
            (id::EXCHANGE, id::EXCHANGE_DECLARE)
        }
        Method::ExchangeDeclareOk => (id::EXCHANGE, id::EXCHANGE_DECLARE_OK),
        Method::QueueDeclare {
            queue,
            passive,
            durable,
            exclusive,
            auto_delete,
            no_wait,
            arguments,
        } => {
            write_short(&mut out, 0);
            write_shortstr(&mut out, queue);
            write_bit_flags(
                &mut out,
                &[*passive, *durable, *exclusive, *auto_delete, *no_wait],
            );
            write_field_table(&mut out, arguments);
            (id::QUEUE, id::QUEUE_DECLARE)
        }
        Method::QueueDeclareOk {
            queue,
            message_count,
            consumer_count,
        } => {
            write_shortstr(&mut out, queue);
            write_long(&mut out, *message_count);
            write_long(&mut out, *consumer_count);
            (id::QUEUE, id::QUEUE_DECLARE_OK)
        }
        Method::QueueBind {
            queue,
            exchange,
            routing_key,
            no_wait,
            arguments,
        } => {
            write_short(&mut out, 0);
            write_shortstr(&mut out, queue);
            write_shortstr(&mut out, exchange);
            write_shortstr(&mut out, routing_key);
            write_bit_flags(&mut out, &[*no_wait]);
            write_field_table(&mut out, arguments);
            (id::QUEUE, id::QUEUE_BIND)
        }
        Method::QueueBindOk => (id::QUEUE, id::QUEUE_BIND_OK),
        Method::BasicQos {
            prefetch_size,
            prefetch_count,
            global,
        } => {
            write_long(&mut out, *prefetch_size);
            write_short(&mut out, *prefetch_count);
            write_bit_flags(&mut out, &[*global]);
            (id::BASIC, id::BASIC_QOS)
        }
        Method::BasicQosOk => (id::BASIC, id::BASIC_QOS_OK),
        Method::BasicConsume {
            queue,
            consumer_tag,
            no_local,
            no_ack,
            exclusive,
            no_wait,
            arguments,
        } => {
            write_short(&mut out, 0);
            write_shortstr(&mut out, queue);
            write_shortstr(&mut out, consumer_tag);
            write_bit_flags(&mut out, &[*no_local, *no_ack, *exclusive, *no_wait]);
            write_field_table(&mut out, arguments);
            (id::BASIC, id::BASIC_CONSUME)
        }
        Method::BasicConsumeOk { consumer_tag } => {
            write_shortstr(&mut out, consumer_tag);
            (id::BASIC, id::BASIC_CONSUME_OK)
        }
        Method::BasicCancel {
            consumer_tag,
            no_wait,
        } => {
            write_shortstr(&mut out, consumer_tag);
            write_bit_flags(&mut out, &[*no_wait]);
            (id::BASIC, id::BASIC_CANCEL)
        }
        Method::BasicCancelOk { consumer_tag } => {
            write_shortstr(&mut out, consumer_tag);
            (id::BASIC, id::BASIC_CANCEL_OK)
        }
        Method::BasicPublish {
            exchange,
            routing_key,
            mandatory,
            immediate,
        } => {
            write_short(&mut out, 0);
            write_shortstr(&mut out, exchange);
            write_shortstr(&mut out, routing_key);
            write_bit_flags(&mut out, &[*mandatory, *immediate]);
            (id::BASIC, id::BASIC_PUBLISH)
        }
        Method::BasicDeliver {
            consumer_tag,
            delivery_tag,
            redelivered,
            exchange,
            routing_key,
        } => {
            write_shortstr(&mut out, consumer_tag);
            write_longlong(&mut out, *delivery_tag);
            write_bit_flags(&mut out, &[*redelivered]);
            write_shortstr(&mut out, exchange);
            write_shortstr(&mut out, routing_key);
            (id::BASIC, id::BASIC_DELIVER)
        }
        Method::BasicAck {
            delivery_tag,
            multiple,
        } => {
            write_longlong(&mut out, *delivery_tag);
            write_bit_flags(&mut out, &[*multiple]);
            (id::BASIC, id::BASIC_ACK)
        }
        Method::BasicNack {
            delivery_tag,
            multiple,
            requeue,
        } => {
            write_longlong(&mut out, *delivery_tag);
            write_bit_flags(&mut out, &[*multiple, *requeue]);
            (id::BASIC, id::BASIC_NACK)
        }
    };
    (ids.0, ids.1, out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::wire::FieldValue;

    fn round_trip(method: Method) {
        let (class_id, method_id, args) = encode(&method);
        let decoded = decode(class_id, method_id, &args).expect("decode");
        assert_eq!(decoded, method);
    }

    #[test]
    fn connection_start_round_trips() {
        let mut server_properties = FieldTable::new();
        server_properties.insert(
            "product".into(),
            FieldValue::LongString(b"proxima".to_vec()),
        );
        round_trip(Method::ConnectionStart {
            version_major: 0,
            version_minor: 9,
            server_properties,
            mechanisms: b"PLAIN".to_vec(),
            locales: b"en_US".to_vec(),
        });
    }

    #[test]
    fn connection_start_ok_round_trips() {
        round_trip(Method::ConnectionStartOk {
            client_properties: FieldTable::new(),
            mechanism: b"PLAIN".to_vec(),
            response: b"\0guest\0guest".to_vec(),
            locale: b"en_US".to_vec(),
        });
    }

    #[test]
    fn connection_tune_and_tune_ok_round_trip() {
        round_trip(Method::ConnectionTune {
            channel_max: 2047,
            frame_max: 131_072,
            heartbeat: 60,
        });
        round_trip(Method::ConnectionTuneOk {
            channel_max: 2047,
            frame_max: 131_072,
            heartbeat: 60,
        });
    }

    #[test]
    fn connection_open_round_trips() {
        round_trip(Method::ConnectionOpen {
            virtual_host: b"/".to_vec(),
        });
    }

    #[test]
    fn channel_open_and_open_ok_round_trip() {
        round_trip(Method::ChannelOpen);
        round_trip(Method::ChannelOpenOk);
    }

    #[test]
    fn exchange_declare_round_trips_with_bits_and_arguments() {
        let mut arguments = FieldTable::new();
        arguments.insert("x-alternate".into(), FieldValue::Boolean(true));
        round_trip(Method::ExchangeDeclare {
            exchange: b"orders".to_vec(),
            kind: b"topic".to_vec(),
            passive: false,
            durable: true,
            auto_delete: false,
            internal: false,
            no_wait: false,
            arguments,
        });
    }

    #[test]
    fn queue_declare_and_bind_round_trip() {
        round_trip(Method::QueueDeclare {
            queue: b"orders.eu".to_vec(),
            passive: false,
            durable: true,
            exclusive: false,
            auto_delete: false,
            no_wait: false,
            arguments: FieldTable::new(),
        });
        round_trip(Method::QueueDeclareOk {
            queue: b"orders.eu".to_vec(),
            message_count: 0,
            consumer_count: 0,
        });
        round_trip(Method::QueueBind {
            queue: b"orders.eu".to_vec(),
            exchange: b"orders".to_vec(),
            routing_key: b"orders.eu.*".to_vec(),
            no_wait: false,
            arguments: FieldTable::new(),
        });
    }

    #[test]
    fn basic_publish_round_trips_mandatory_and_immediate_bits() {
        round_trip(Method::BasicPublish {
            exchange: b"orders".to_vec(),
            routing_key: b"orders.eu.created".to_vec(),
            mandatory: true,
            immediate: false,
        });
    }

    #[test]
    fn basic_consume_and_deliver_round_trip() {
        round_trip(Method::BasicConsume {
            queue: b"orders.eu".to_vec(),
            consumer_tag: b"ctag-1".to_vec(),
            no_local: false,
            no_ack: true,
            exclusive: false,
            no_wait: false,
            arguments: FieldTable::new(),
        });
        round_trip(Method::BasicDeliver {
            consumer_tag: b"ctag-1".to_vec(),
            delivery_tag: 1,
            redelivered: false,
            exchange: b"orders".to_vec(),
            routing_key: b"orders.eu.created".to_vec(),
        });
    }

    #[test]
    fn basic_ack_and_nack_round_trip() {
        round_trip(Method::BasicAck {
            delivery_tag: 42,
            multiple: true,
        });
        round_trip(Method::BasicNack {
            delivery_tag: 42,
            multiple: false,
            requeue: true,
        });
    }

    #[test]
    fn unsupported_method_is_reported_precisely() {
        assert_eq!(
            decode(90, 10, &[]), // tx.select — deliberately out of scope
            Err(MethodError::Unsupported {
                class_id: 90,
                method_id: 10
            })
        );
    }

    #[test]
    fn truncated_args_surface_a_wire_error() {
        assert_eq!(
            decode(id::CONNECTION, id::CONNECTION_TUNE, &[0, 1]),
            Err(MethodError::Wire(WireError::Short("long")))
        );
    }
}
