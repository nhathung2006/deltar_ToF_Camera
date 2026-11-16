"""Protect the stem semantics used to train the distributed DELTAR weights."""
import unittest

import torch

from src.models.encoder import ImageEncoder


class EncoderCheckpointTests(unittest.TestCase):
    def test_stem_normalization_does_not_apply_an_extra_activation(self):
        # timm 0.5.4 kept act1 separate. DELTAR intentionally selected bn1 only.
        # A modern fused BN+SiLU silently changes loaded checkpoint inference.
        encoder = ImageEncoder(pretrained=False).eval()
        norm = encoder.conv0[1]
        with torch.no_grad():
            norm.running_mean.zero_()
            norm.running_var.fill_(1)
            norm.weight.fill_(1)
            norm.bias.zero_()
            x = torch.full((1, norm.num_features, 2, 2), -2.0)
            result = norm(x)
        expected = torch.full_like(result, -1.9990007493755464)
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
