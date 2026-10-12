use super::*;
use conflaguration::Validate;

/// prepares the existing [`Plan`] using explicit [`RuntimeConfig`] policy.
///
/// unlike [`plan_named`], this entry preserves a checked input contract and
/// enforces optional payload ceilings before execution allocations.
pub fn plan_named_with_config(
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, QuantizedBlock<'_>)],
    outputs: &[NodeId],
    config: &RuntimeConfig,
) -> Result<Plan, MetalError> {
    config
        .validate()
        .map_err(|error| MetalError::CompileFailed {
            log: error.to_string(),
        })?;
    let blocks = resolve_named_blocks(program, named)?;
    let prepared = prepare_with_options(
        program,
        symbols,
        &blocks,
        outputs,
        (config.numeric, true, config.allow_fusion),
        &[],
    )?;
    let mut plan = plan_from_prepared(program, symbols, prepared, config.numeric);
    plan.configured_input_contract = blocks
        .iter()
        .map(input_contract)
        .collect::<Result<_, _>>()?;
    plan.configured_constraints = Some(config.constraints.clone());
    if plan.has_constraints() {
        plan.preflight_constraints(&blocks)?;
    }
    Ok(plan)
}

fn input_contract(block: &QuantizedBlock<'_>) -> Result<(Option<Codec>, bool, usize), MetalError> {
    let codec = match block {
        QuantizedBlock::Packed { codec, .. } => Some(*codec),
        QuantizedBlock::Float32(_) => None,
        QuantizedBlock::Int32(_) => None,
    };
    Ok((
        codec,
        matches!(block, QuantizedBlock::Int32(_)),
        block.element_count()?,
    ))
}

fn checked_bytes(count: u64, width: u64) -> Result<u64, MetalError> {
    count
        .checked_mul(width)
        .ok_or(MetalError::ResourceOverflow {
            field: "payload_bytes",
        })
}

fn checked_sum(left: u64, right: u64) -> Result<u64, MetalError> {
    left.checked_add(right).ok_or(MetalError::ResourceOverflow {
        field: "payload_bytes",
    })
}

fn check_limit(field: &'static str, requested: u64, limit: Option<u64>) -> Result<(), MetalError> {
    if let Some(limit) = limit
        && requested > limit
    {
        return Err(MetalError::ResourceLimit {
            field,
            requested,
            limit,
        });
    }
    Ok(())
}

impl Plan {
    /// actual buffer roles, nodes, object identities and capacities from the last bounded call.
    /// repeated identities denote aliases and must be counted only once.
    #[must_use]
    pub fn scoped_resource_rows(&self) -> Vec<(&'static str, NodeId, usize, u64)> {
        self.scoped_resource_rows.borrow().clone()
    }

    /// the physical Metal device used by this thread's execution queue.
    pub fn device_identity(&self) -> Result<String, MetalError> {
        Ok(device_and_queue()?.0.name().to_string())
    }

    pub(super) fn record_scoped_buffer(
        &self,
        role: &'static str,
        node: NodeId,
        buffer: &MetalBuffer,
    ) {
        if self.has_constraints() {
            self.scoped_resource_rows.borrow_mut().push((
                role,
                node,
                Retained::as_ptr(buffer) as usize,
                buffer.length() as u64,
            ));
        }
    }

    pub(super) fn reject_bounded_extension(&self) -> Result<(), MetalError> {
        if self.has_constraints() {
            return Err(MetalError::UnsupportedConstraint {
                field: "constraints",
                reason: "expert or diagnostic execution requires workspace outside the bounded ordinary path",
            });
        }
        Ok(())
    }

    /// conservative host, device and largest-buffer payload reservations.
    ///
    /// [`plan_named_with_config`] supplies these for bounded execution;
    /// unrestricted plans have no census and return `None`.
    #[must_use]
    pub fn scoped_reservation(&self) -> Option<(u64, u64, u64)> {
        *self.scoped_reservation.borrow()
    }

    /// number of operations produced by the policy-sensitive tensor binder.
    #[must_use]
    pub fn resolved_op_count(&self) -> usize {
        self.prepared.resolved.len()
    }

    pub(super) fn has_constraints(&self) -> bool {
        self.configured_constraints.as_ref().is_some_and(|limits| {
            limits.max_host_tensor_bytes.is_some()
                || limits.max_device_buffer_bytes.is_some()
                || limits.max_single_device_buffer_bytes.is_some()
        })
    }

    pub(super) fn validate_configured_inputs(
        &self,
        blocks: &[QuantizedBlock<'_>],
    ) -> Result<(), MetalError> {
        if self.configured_constraints.is_none() {
            return Ok(());
        }
        if blocks.len() != self.configured_input_contract.len() {
            return Err(TensorError::InputCountMismatch {
                expected: self.configured_input_contract.len(),
                found: blocks.len(),
            }
            .into());
        }
        for ((node, expected), block) in self
            .prepared
            .block_nodes
            .iter()
            .zip(&self.configured_input_contract)
            .zip(blocks)
        {
            if input_contract(block)? != *expected {
                return Err(MetalError::InputContractChanged {
                    node: *node,
                    reason: "codec or element count differs from the compiled plan",
                });
            }
        }
        Ok(())
    }

    pub(super) fn preflight_constraints(
        &self,
        blocks: &[QuantizedBlock<'_>],
    ) -> Result<(), MetalError> {
        let Some(limits) = &self.configured_constraints else {
            return Ok(());
        };
        if !self.has_constraints() {
            return Ok(());
        }
        if !self.resident_nodes.is_empty() || !self.device_buffers.borrow().is_empty() {
            return Err(MetalError::UnsupportedConstraint {
                field: "constraints",
                reason: "resident and externally retained buffers require an explicit physical census",
            });
        }
        let (device, _) = device_and_queue()?;
        let hardware_limit = device.maxBufferLength() as u64;
        let mut device_bytes = 0_u64;
        let mut largest = 0_u64;
        let mut mappings = BTreeSet::new();
        for (((node, live), dtype), block) in self
            .prepared
            .block_nodes
            .iter()
            .zip(&self.prepared.live_block_inputs)
            .zip(&self.block_dtypes)
            .zip(blocks)
        {
            if !live {
                continue;
            }
            let QuantizedBlock::Float32(values) = block else {
                return Err(MetalError::UnsupportedConstraint {
                    field: "constraints",
                    reason: "bounded Metal uploads currently require Float32 payloads",
                });
            };
            if *dtype != DType::Float32 {
                return Err(MetalError::UnsupportedConstraint {
                    field: "max_host_tensor_bytes",
                    reason: "converted uploads have no bounded host workspace contract",
                });
            }
            let byte_length = checked_bytes(values.len() as u64, 4)?;
            let address = values.as_ptr() as usize;
            let (mapping, physical) = prospective_input_capacity(address, byte_length as usize)?;
            if let Some(mapping) = mapping
                && !mappings.insert(mapping)
            {
                continue;
            }
            check_limit("max_buffer_length", physical, Some(hardware_limit))?;
            check_limit(
                "max_single_device_buffer_bytes",
                physical,
                limits.max_single_device_buffer_bytes,
            )?;
            device_bytes = checked_sum(device_bytes, physical)?;
            largest = largest.max(physical);
            debug!(
                node = node.0,
                physical, "bounded input accounts for physical buffer capacity"
            );
        }
        for bound in &self.prepared.resolved {
            if bound.dtype != DType::Float32
                || gather_count(bound) != 0
                || !matches!(
                    &bound.kind,
                    BoundOpKind::Elementwise { .. }
                        | BoundOpKind::Reduce { .. }
                        | BoundOpKind::Iota
                        | BoundOpKind::Constant { .. }
                )
            {
                return Err(MetalError::UnsupportedConstraint {
                    field: "constraints",
                    reason: "specialized or gathered operations have no bounded Metal workspace contract",
                });
            }
            // validate the product before the allocator's usize calculation.
            let elements = bound
                .extents
                .iter()
                .try_fold(1_u64, |total, extent| checked_bytes(total, *extent))?;
            checked_bytes(elements.max(1), 4)?;
            let output = allocation_capacity(bound_output_len(bound), DType::Float32)? as u64;
            let uniform = pack_uniforms_byte_len(bound, self.numeric_policy).max(1) as u64;
            for physical in [output, uniform] {
                check_limit("max_buffer_length", physical, Some(hardware_limit))?;
                check_limit(
                    "max_single_device_buffer_bytes",
                    physical,
                    limits.max_single_device_buffer_bytes,
                )?;
                device_bytes = checked_sum(device_bytes, physical)?;
                largest = largest.max(physical);
            }
        }
        let host_bytes =
            self.prepared
                .effective_outputs
                .iter()
                .try_fold(0_u64, |total, node| {
                    let elements = self
                        .prepared
                        .shapes
                        .of(*node)
                        .iter()
                        .try_fold(1_u64, |count, extent| checked_bytes(count, *extent))?;
                    checked_sum(total, checked_bytes(elements, 4)?)
                })?;
        check_limit(
            "max_host_tensor_bytes",
            host_bytes,
            limits.max_host_tensor_bytes,
        )?;
        check_limit(
            "max_device_buffer_bytes",
            device_bytes,
            limits.max_device_buffer_bytes,
        )?;
        *self.scoped_reservation.borrow_mut() = Some((host_bytes, device_bytes, largest));
        debug!(
            host_bytes,
            device_bytes, largest, "bounded execution admitted before payload allocation"
        );
        Ok(())
    }
}

fn prospective_input_capacity(
    address: usize,
    byte_length: usize,
) -> Result<(Option<(usize, usize)>, u64), MetalError> {
    if byte_length == 0 {
        return Ok((None, allocation_capacity(0, DType::Float32)? as u64));
    }
    if expert_mapping_identity().is_some() {
        return Err(MetalError::UnsupportedConstraint {
            field: "max_device_buffer_bytes",
            reason: "concurrently replaceable expert mapping has no per-plan reservation owner",
        });
    }
    if let Some((base, length)) = CHECKPOINT_MAPPING.with(|mapping| *mapping.borrow()) {
        let end = address
            .checked_add(byte_length)
            .ok_or(MetalError::ResourceOverflow {
                field: "input_range",
            })?;
        let mapping_end = base
            .checked_add(length)
            .ok_or(MetalError::ResourceOverflow {
                field: "mapping_range",
            })?;
        if address >= base && end <= mapping_end {
            let page = page_size();
            let physical = length
                .checked_add(page - 1)
                .map(|length| length / page * page)
                .ok_or(MetalError::ResourceOverflow {
                    field: "mapping_capacity",
                })?;
            return Ok((Some((base, length)), physical as u64));
        }
    }
    Ok((None, byte_length as u64))
}
