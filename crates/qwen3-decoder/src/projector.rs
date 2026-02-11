//! Audio Projector for mapping encoder outputs to LLM input space.

use candle_core::{Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

/// Audio Projector that maps encoder outputs to LLM hidden dimension.
#[derive(Debug, Clone)]
pub struct AudioProjector {
    proj: Linear,
}

impl AudioProjector {
    /// Create a new audio projector.
    ///
    /// # Arguments
    /// * `encoder_dim` - Dimension of encoder output (e.g., 896 or 1024)
    /// * `llm_dim` - Dimension of LLM hidden states
    /// * `vb` - Variable builder for loading weights
    pub fn new(encoder_dim: usize, llm_dim: usize, vb: VarBuilder) -> Result<Self> {
        let proj = linear(encoder_dim, llm_dim, vb.pp("proj"))?;
        Ok(Self { proj })
    }

    /// Project encoder outputs to LLM dimension.
    pub fn forward(&self, encoder_output: &Tensor) -> Result<Tensor> {
        self.proj.forward(encoder_output)
    }
}
