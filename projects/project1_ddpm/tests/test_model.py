import torch

from projects.project1_ddpm.model import ResBlock, SinusoidalPosEmb, UNet


def test_sinusoidal_embedding_shape_and_timestep_dependence():
    embedding = SinusoidalPosEmb(15)
    t = torch.tensor([0, 1, 10], dtype=torch.long)
    output = embedding(t)
    assert output.shape == (3, 15)
    assert not torch.equal(output[0], output[1])


def test_resblock_preserves_spatial_shape():
    block = ResBlock(8, 16, time_dim=32, dropout=0.0)
    x = torch.randn(2, 8, 8, 8)
    t_emb = torch.randn(2, 32)
    output = block(x, t_emb)
    assert output.shape == (2, 16, 8, 8)


def test_unet_output_matches_input_shape():
    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=16,
        channel_mult=(1, 2, 2),
        num_res_blocks=2,
        attn_resolutions=(8,),
        dropout=0.0,
        image_size=16,
    )
    x = torch.randn(2, 1, 16, 16)
    t = torch.tensor([0, 15], dtype=torch.long)
    assert model(x, t).shape == x.shape
