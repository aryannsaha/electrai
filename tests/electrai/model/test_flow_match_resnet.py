"""Tests for flow_match_resnet module."""

from __future__ import annotations

import pytest
import torch

from electrai.model.flow_match_resnet import (
    AdaLNResidualBlock,
    FlowMatchGeneratorResNet,
    PixelShuffle3d,
    TimeEmbedding,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_lr_input():
    """Small LR input tensor for fast tests (2x upscaling -> 16^3 HR)."""
    return torch.randn(2, 2, 8, 8, 8)


# =============================================================================
# TimeEmbedding Tests
# =============================================================================


class TestTimeEmbedding:
    """Tests for TimeEmbedding class."""

    def test_time_embedding_output_shape(self):
        """Output should be flat (B, dim), not (B, dim, 1, 1, 1)."""
        emb = TimeEmbedding(dim=32)
        t = torch.rand(4)
        out = emb(t)
        assert out.shape == (4, 32)

    def test_time_embedding_different_times(self):
        """Different timestep values produce different embeddings."""
        emb = TimeEmbedding(dim=32)
        t1 = torch.tensor([0.0])
        t2 = torch.tensor([0.5])
        t3 = torch.tensor([1.0])

        out1 = emb(t1)
        out2 = emb(t2)
        out3 = emb(t3)

        assert not torch.allclose(out1, out2)
        assert not torch.allclose(out2, out3)
        assert not torch.allclose(out1, out3)

    def test_time_embedding_batch_input(self):
        """Handles both (B,) and (B, 1) inputs."""
        emb = TimeEmbedding(dim=64)
        t_flat = torch.rand(3)
        t_col = t_flat.unsqueeze(-1)

        out_flat = emb(t_flat)
        out_col = emb(t_col)

        torch.testing.assert_close(out_flat, out_col)

    def test_time_embedding_deterministic(self):
        """Same input produces same output (no stochasticity)."""
        emb = TimeEmbedding(dim=32)
        emb.eval()
        t = torch.tensor([0.3, 0.7])
        out1 = emb(t)
        out2 = emb(t)
        torch.testing.assert_close(out1, out2)


# =============================================================================
# AdaLNResidualBlock Tests
# =============================================================================


class TestAdaLNResidualBlock:
    """Tests for AdaLNResidualBlock class."""

    @pytest.mark.parametrize(
        "shape", [(1, 32, 8, 8, 8), (2, 32, 16, 16, 16), (4, 32, 4, 8, 12)]
    )
    def test_adaln_block_output_shape(self, shape):
        """Input shape (B, C, D, H, W) -> output shape unchanged."""
        block = AdaLNResidualBlock(
            in_features=32, time_dim=32, K=3, use_checkpoint=False
        )
        block.eval()

        x = torch.randn(*shape)
        t_emb = torch.randn(shape[0], 32)
        output = block(x, t_emb)
        assert output.shape == x.shape

    def test_adaln_zero_init_starts_as_plain_residual(self):
        """With zero-initialized modulation, AdaLN reduces to standard norm."""
        block = AdaLNResidualBlock(
            in_features=32, time_dim=32, K=3, use_checkpoint=False
        )
        block.eval()

        # Verify the modulation linear layer is zero-initialized
        linear = block.modulation[1]  # nn.Linear after nn.SiLU
        assert torch.all(linear.weight == 0)
        assert torch.all(linear.bias == 0)

        # With zero modulation, scale=0 and shift=0, so:
        # h = norm(conv(x)) * (1 + 0) + 0 = norm(conv(x))
        # This is the same as a standard residual block with InstanceNorm.

    def test_adaln_time_sensitivity(self):
        """Different time embeddings produce different outputs."""
        block = AdaLNResidualBlock(
            in_features=32, time_dim=32, K=3, use_checkpoint=False
        )
        # Need non-zero modulation weights for time sensitivity
        # Initialize with small random values
        nn = torch.nn
        nn.init.normal_(block.modulation[1].weight, std=0.1)
        block.eval()

        x = torch.randn(1, 32, 8, 8, 8)
        t_emb1 = torch.randn(1, 32)
        t_emb2 = torch.randn(1, 32)

        out1 = block(x, t_emb1)
        out2 = block(x, t_emb2)

        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_adaln_checkpoint_eval_equivalence(self):
        """Checkpointed and non-checkpointed produce same output."""
        block_ckpt = AdaLNResidualBlock(
            in_features=32, time_dim=32, K=3, use_checkpoint=True
        )
        block_no_ckpt = AdaLNResidualBlock(
            in_features=32, time_dim=32, K=3, use_checkpoint=False
        )
        block_no_ckpt.load_state_dict(block_ckpt.state_dict())

        block_ckpt.eval()
        block_no_ckpt.eval()

        x = torch.randn(1, 32, 8, 8, 8)
        t_emb = torch.randn(1, 32)

        with torch.no_grad():
            out_ckpt = block_ckpt(x, t_emb)
            out_no_ckpt = block_no_ckpt(x, t_emb)

        torch.testing.assert_close(out_ckpt, out_no_ckpt)

    def test_adaln_residual_connection(self):
        """Output = input + block_fn(input, t_emb)."""
        block = AdaLNResidualBlock(
            in_features=32, time_dim=32, K=3, use_checkpoint=False
        )
        block.eval()

        x = torch.randn(1, 32, 8, 8, 8)
        t_emb = torch.randn(1, 32)

        block_output = block._block_fn(x, t_emb)
        expected = x + block_output
        actual = block(x, t_emb)

        torch.testing.assert_close(actual, expected)


# =============================================================================
# FlowMatchGeneratorResNet Tests
# =============================================================================


class TestFlowMatchGeneratorResNet:
    """Tests for FlowMatchGeneratorResNet class."""

    def test_flow_match_output_shape_with_upscale(self):
        """LR input with n_upscale_layers=1 produces 2x HR output."""
        gen = FlowMatchGeneratorResNet(
            n_residual_blocks=2,
            n_upscale_layers=1,
            n_channels=16,
            kernel_size1=3,
            kernel_size2=3,
            use_checkpoint=False,
        )
        gen.eval()

        # LR input: (B, 2, 8, 8, 8) -> HR output: (B, 1, 16, 16, 16)
        x = torch.randn(2, 2, 8, 8, 8)
        t = torch.rand(2)

        with torch.no_grad():
            output = gen(x, t)

        assert output.shape == (2, 1, 16, 16, 16)

    @pytest.mark.parametrize("size", [8, 12, 16])
    def test_flow_match_various_input_sizes(self, size):
        """Test with different spatial dimensions."""
        gen = FlowMatchGeneratorResNet(
            n_residual_blocks=2,
            n_upscale_layers=1,
            n_channels=16,
            kernel_size1=3,
            kernel_size2=3,
            use_checkpoint=False,
        )
        gen.eval()

        x = torch.randn(1, 2, size, size, size)
        t = torch.rand(1)

        with torch.no_grad():
            output = gen(x, t)

        expected_size = size * 2
        assert output.shape == (1, 1, expected_size, expected_size, expected_size)

    def test_flow_match_non_cubic_input(self):
        """Non-cubic input (B, 2, 4, 8, 12) -> (B, 1, 8, 16, 24)."""
        gen = FlowMatchGeneratorResNet(
            n_residual_blocks=2,
            n_upscale_layers=1,
            n_channels=16,
            kernel_size1=3,
            kernel_size2=3,
            use_checkpoint=False,
        )
        gen.eval()

        x = torch.randn(1, 2, 4, 8, 12)
        t = torch.rand(1)

        with torch.no_grad():
            output = gen(x, t)

        assert output.shape == (1, 1, 8, 16, 24)

    def test_flow_match_time_sensitivity(self):
        """Same spatial input with different t values produces different outputs."""
        gen = FlowMatchGeneratorResNet(
            n_residual_blocks=4,
            n_upscale_layers=1,
            n_channels=16,
            kernel_size1=3,
            kernel_size2=3,
            use_checkpoint=False,
        )
        gen.eval()

        x = torch.randn(1, 2, 8, 8, 8)
        t1 = torch.tensor([0.1])
        t2 = torch.tensor([0.9])

        with torch.no_grad():
            out1 = gen(x, t1)
            out2 = gen(x, t2)

        # Initially, AdaLN modulation is zero-initialized, so outputs are the same.
        # After any gradient update, they would differ. Test after one dummy step.
        # For a more meaningful test, manually set non-zero modulation weights.
        for block in gen.res_blocks:
            torch.nn.init.normal_(block.modulation[1].weight, std=0.1)

        with torch.no_grad():
            out1 = gen(x, t1)
            out2 = gen(x, t2)

        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_flow_match_gradient_flow(self):
        """Gradients flow through all layers including time embedding."""
        gen = FlowMatchGeneratorResNet(
            n_residual_blocks=2,
            n_upscale_layers=1,
            n_channels=16,
            kernel_size1=3,
            kernel_size2=3,
            use_checkpoint=False,
        )
        # Initialize modulation weights to non-zero so time signal has effect.
        # (At init, modulation is zero -> time has no effect -> t.grad == 0)
        for block in gen.res_blocks:
            torch.nn.init.normal_(block.modulation[1].weight, std=0.01)
        gen.train()

        x = torch.randn(1, 2, 8, 8, 8, requires_grad=True)
        t = torch.tensor([0.5], requires_grad=True)

        output = gen(x, t)
        loss = output.sum()
        loss.backward()

        # Input gradient exists
        assert x.grad is not None
        assert torch.any(x.grad != 0)

        # Time gradient flows (requires non-zero modulation weights)
        assert t.grad is not None
        assert torch.any(t.grad != 0)

        # Key layer gradients exist
        assert gen.conv1[0].weight.grad is not None
        assert gen.conv3.weight.grad is not None
        assert torch.any(gen.conv1[0].weight.grad != 0)

        # Time embedding gradients exist
        assert gen.time_emb.mlp[0].weight.grad is not None

        # AdaLN modulation gradients exist
        assert gen.res_blocks[0].modulation[1].weight.grad is not None

    def test_flow_match_list_input(self):
        """Variable-size list input works correctly."""
        gen = FlowMatchGeneratorResNet(
            n_residual_blocks=2,
            n_upscale_layers=1,
            n_channels=16,
            kernel_size1=3,
            kernel_size2=3,
            use_checkpoint=False,
        )
        gen.eval()

        # List of tensors with different spatial sizes
        x_list = [
            torch.randn(2, 6, 6, 6),
            torch.randn(2, 8, 8, 8),
        ]
        t = torch.rand(2)

        with torch.no_grad():
            outputs = gen(x_list, t)

        assert isinstance(outputs, list)
        assert len(outputs) == 2
        # Each output should be (1, D*2, H*2, W*2) (squeezed batch dim)
        assert outputs[0].shape == (1, 12, 12, 12)
        assert outputs[1].shape == (1, 16, 16, 16)

    def test_flow_match_no_upscale(self):
        """n_upscale_layers=0 preserves spatial dimensions."""
        gen = FlowMatchGeneratorResNet(
            n_residual_blocks=2,
            n_upscale_layers=0,
            n_channels=16,
            kernel_size1=3,
            kernel_size2=3,
            use_checkpoint=False,
        )
        gen.eval()

        x = torch.randn(1, 2, 8, 8, 8)
        t = torch.rand(1)

        with torch.no_grad():
            output = gen(x, t)

        assert output.shape == (1, 1, 8, 8, 8)

    def test_flow_match_velocity_can_be_negative(self):
        """Output contains negative values (no final ReLU)."""
        gen = FlowMatchGeneratorResNet(
            n_residual_blocks=4,
            n_upscale_layers=1,
            n_channels=16,
            kernel_size1=3,
            kernel_size2=3,
            use_checkpoint=False,
        )
        gen.eval()

        # Use input that's likely to produce negative velocity components
        torch.manual_seed(42)
        x = torch.randn(2, 2, 8, 8, 8)
        t = torch.rand(2)

        with torch.no_grad():
            output = gen(x, t)

        # At least some values should be negative (velocity, not charge density)
        assert torch.any(output < 0), "Velocity field should allow negative values"
