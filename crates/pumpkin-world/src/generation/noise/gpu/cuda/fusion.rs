use std::sync::Arc;

use cudarc::driver::CudaSlice;
use pumpkin_data::noise_router::BinaryOperation;

use super::{
    super::super::router::density_function::NoiseFunctionComponentRange, Component, Dep, Leaf,
    Noise, Tail, abi::CudaLayer, sampler_branch,
};

pub(super) const MAX_PAIRS: usize = 4;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub(super) enum Operation {
    Add = 1,
    Sub = 2,
    Mul = 3,
}

pub(super) struct Operand {
    pub layers: Arc<CudaSlice<CudaLayer>>,
    pub tail: Tail,
    pub operation: Operation,
}

pub(super) struct Branch<'a> {
    pub left: (&'a Noise, Tail),
    pub right: (&'a Noise, Tail),
    pub operation: Operation,
}

pub(super) fn branch<'a>(stack: &'a [Component<'_>]) -> Option<Branch<'a>> {
    let (Component::Dependent(Dep::Binary(node)), prefix) = stack.split_last()? else {
        return None;
    };
    let lhs = prefix.get(..=node.input1_index)?;
    let rhs = prefix.get(..=node.input2_index)?;
    let operation = match node.data.operation {
        BinaryOperation::Add => Operation::Add,
        BinaryOperation::Sub => Operation::Sub,
        // CPU volume multiplication emits positive zero if the entire left volume is zero.
        BinaryOperation::Mul if lhs.last()?.min() > 0.0 || lhs.last()?.max() < 0.0 => {
            Operation::Mul
        }
        _ => return None,
    };
    let (Leaf::Noise(left), lhs_tail) = sampler_branch(lhs, true)? else {
        return None;
    };
    let (Leaf::Noise(right), rhs_tail) = sampler_branch(rhs, true)? else {
        return None;
    };
    Some(Branch {
        left: (left, lhs_tail),
        right: (right, rhs_tail),
        operation,
    })
}
