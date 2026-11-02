"""Regression checks for equivalent GPU filtering and the 3x4 comparison."""
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from loc_tool.dsm_patch import _SOBEL_X, _SOBEL_Y, _apply_deadzone_torch
from loc_tool.dsm_patch_batch import _median_filter_3x3_batch, compute_bev_layers_batch
from loc_tool.dsm_bev_score_vis import build_quad_patch_view
from loc_tool.dsm_bev_score import DsmBevScoreConfig, compute_dsm_bev_score
from loc_tool.dsm_bev_score_batch import compute_dsm_bev_score_batch


def reference_median(layer):
    n, h, w = layer.shape
    windows = F.pad(layer.unsqueeze(1), (1, 1, 1, 1), mode="replicate")
    windows = windows.unfold(2, 3, 1).unfold(3, 3, 1)
    return windows.reshape(n, 1, h, w, 9).median(dim=-1).values.squeeze(1)


class BevProcessingTest(unittest.TestCase):
    def check_median(self, device):
        generator = torch.Generator().manual_seed(42)
        for shape in ((3, 17, 23), (1, 1, 1), (2, 2, 3)):
            layer = torch.randn(shape, generator=generator).to(device)
            np.testing.assert_equal(
                _median_filter_3x3_batch(layer).cpu().numpy(),
                reference_median(layer).cpu().numpy(),
            )
        # Every position in the 3x3 neighborhood, including replicated borders.
        for y in range(3):
            for x in range(3):
                layer = torch.arange(25, dtype=torch.float32, device=device).reshape(1, 5, 5)
                layer[0, y, x] = float("nan")
                np.testing.assert_equal(
                    _median_filter_3x3_batch(layer).cpu().numpy(),
                    reference_median(layer).cpu().numpy(),
                )

    def test_cpu_median(self):
        self.check_median("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_median(self):
        self.check_median("cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_layers_match_original(self):
        torch.manual_seed(42)
        patch = torch.randn(4, 17, 23, device="cuda")
        patch[0, 0, 0] = float("nan")
        sx, sy = _SOBEL_X.cuda(), _SOBEL_Y.cuda()
        actual = compute_bev_layers_batch(patch, .2, sx, sy)
        rel = patch - patch[:, 8, 11, None, None]
        smooth = reference_median(rel).unsqueeze(1)
        expected = (
            rel,
            _apply_deadzone_torch(-F.conv2d(smooth, sy, padding=1).squeeze(1) / 1.6, .15).clamp(-3, 3),
            _apply_deadzone_torch(-F.conv2d(smooth, sx, padding=1).squeeze(1) / 1.6, .15).clamp(-3, 3),
        )
        for result, reference in zip(actual, expected):
            np.testing.assert_allclose(result.cpu().numpy(), reference.cpu().numpy(), atol=1e-6, equal_nan=True)

    def test_small_height_differences_are_preserved(self):
        from loc_tool.dsm_patch import compute_bev_layers
        # Two different center elevations model the GPS and PF references.
        offsets = np.array([[-.21, -.20, -.19], [-.01, 0., .01], [.19, .20, .21]], np.float32)
        patches = np.stack([offsets + 100., offsets + 150.])
        expected = patches - patches[:, 1:2, 1:2]
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            batch = compute_bev_layers_batch(torch.as_tensor(patches, device=device), .2,
                                             _SOBEL_X.to(device), _SOBEL_Y.to(device))[0]
            np.testing.assert_array_equal(batch.cpu().numpy(), expected)
            for patch, rel in zip(patches, expected):
                single = compute_bev_layers(patch, device=device)
                np.testing.assert_array_equal(single["H_rel_surf"], rel)

    def test_batch_scores_match_scalar_formula(self):
        rng = np.random.RandomState(42)
        shape = (17, 23)
        lidar = [rng.randn(*shape).astype(np.float32) for _ in range(3)]
        dsm = [rng.randn(4, *shape).astype(np.float32) for _ in range(3)]
        config = DsmBevScoreConfig()
        masks = [np.ones(shape, np.float32), (rng.rand(*shape) > .3).astype(np.float32), np.zeros(shape, np.float32)]
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            for mask in masks:
                tensors = [torch.as_tensor(a, device=device) for a in (*lidar, mask, *dsm)]
                actual = compute_dsm_bev_score_batch(*tensors, config=config, h_max=3.)
                for i in range(4):
                    expected = compute_dsm_bev_score(*lidar, mask, *(a[i] for a in dsm), config=config, h_max=3.)
                    for key in ("J_h", "J_gx", "J_gy", "J_total", "score", "valid_grad_pixel_num"):
                        self.assertAlmostEqual(float(actual[key][i].item()), expected[key], places=6)

    def test_chunked_particle_scores_match_full_batch(self):
        from types import SimpleNamespace
        from loc_tool.dem_tool import DemTiffData
        from loc_tool.dsm_patch import DsmPatchCropper
        from loc_tool.particle_filter import ParticleFilter, ParticleFilterConfig
        rng = np.random.RandomState(42)
        data = DemTiffData(raw_elevation_map=rng.randn(128, 128).astype(np.float32))
        cc = SimpleNamespace(tl_gauss_x=0., tl_gauss_y=25.6, res_gauss_x=.2, res_gauss_y=.2)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        crop = DsmPatchCropper(data, cc, map_size_x=4., map_size_y=4., device=device)
        cfg = ParticleFilterConfig(num_particles=40, save_debug=False, save_weight_vis=False, device=device)
        pf = ParticleFilter(cfg, DsmBevScoreConfig(), crop)
        pf.initialize(12.8, 12.8, 0.)
        shape = (20, 20)
        lidar = {key: rng.randn(*shape).astype(np.float32) for key in ("H_L", "Gx_L", "Gy_L")}
        lidar["M_obs"] = (rng.rand(*shape) > .2).astype(np.float32)
        pf._prepare_lidar(lidar, h_max=3.)
        with torch.inference_mode():
            expected = pf._score_particles(pf.particles)
        pf.update(lidar, h_max=3.)
        for key, actual in (("score", pf.scores), ("J_total", pf.j_totals), ("J_h", pf.j_h), ("J_gx", pf.j_gx), ("J_gy", pf.j_gy)):
            np.testing.assert_allclose(actual.cpu().numpy(), expected[key].cpu().numpy(), atol=1e-6, rtol=1e-6)
        self.assertEqual(pf.particles.shape[0], 40)
        self.assertAlmostEqual(float(pf.weights.sum().item()), 1., places=6)

    def test_large_utm_coordinates_do_not_quantize_patch(self):
        from types import SimpleNamespace
        from loc_tool.dem_tool import DemTiffData
        from loc_tool.dsm_patch import DsmPatchCropper
        from loc_tool.dsm_patch_batch import build_sample_grid_batch
        row, col = np.indices((128, 128))
        data = DemTiffData(raw_elevation_map=(row + .3 * col).astype(np.float32))
        cc = SimpleNamespace(tl_gauss_x=700000., tl_gauss_y=3173000., res_gauss_x=.2, res_gauss_y=.2)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        crop = DsmPatchCropper(data, cc, map_size_x=4., map_size_y=4., device=device)
        x, y = 700012.83, 3172987.17
        patch, _ = crop.crop_numpy(x, y, 0.)
        np.testing.assert_allclose(np.diff(patch, axis=0), -.3, atol=3e-5)
        np.testing.assert_allclose(np.diff(patch, axis=1), 1., atol=3e-5)
        poses = torch.tensor([[x, y, 0.], [x + .03, y + .02, .4]], dtype=torch.float64, device=device)
        batch = build_sample_grid_batch(crop, poses)
        for i, pose in enumerate(poses.cpu().tolist()):
            expected = crop._build_sample_grid(*pose)[0]
            np.testing.assert_allclose(batch[i].cpu().numpy(), expected.cpu().numpy(), atol=1e-6)

    def test_utm_particle_state_preserves_centimeter_motion(self):
        from types import SimpleNamespace
        from loc_tool.dem_tool import DemTiffData
        from loc_tool.dsm_patch import DsmPatchCropper
        from loc_tool.particle_filter import ParticleFilter, ParticleFilterConfig
        cc = SimpleNamespace(tl_gauss_x=700000., tl_gauss_y=3173000., res_gauss_x=.2, res_gauss_y=.2)
        data = DemTiffData(raw_elevation_map=np.zeros((32, 32), np.float32))
        crop = DsmPatchCropper(data, cc, map_size_x=4., map_size_y=4., device="cpu")
        cfg = ParticleFilterConfig(num_particles=8, save_debug=False, save_weight_vis=False, device="cpu",
            init_std_x=0., init_std_y=0., motion_std_x=0., motion_std_y=0.)
        pf = ParticleFilter(cfg, DsmBevScoreConfig(), crop)
        pf.initialize(700012.83, 3172987.17, 0.)
        pf.propagate(0., .01, 0., speed_mps=1.)
        np.testing.assert_allclose(pf.particles[:, 1].numpy(), 3172987.18, atol=1e-8, rtol=0.)
        self.assertAlmostEqual(pf.estimate_pose()["y_est"], 3172987.18, places=7)

    def test_three_rows_four_columns_and_hover_values(self):
        shape = (17, 23)
        height = np.ones(shape, dtype=np.float32)
        lidar = {"H_L": height, "Gy_L": height * 2, "Gx_L": height * 3, "M_obs": height}
        gps = {"H_rel_surf": height * 4, "G_long_L": height * 5, "G_lat_L": height * 6}
        pf = {"H_rel_surf": height * 7, "G_long_L": height * 8, "G_lat_L": height * 9}
        image, meta = build_quad_patch_view(lidar, gps, .7, pf, .8, pf_label="PF", return_meta=True)
        self.assertEqual(image.shape, (3 * (17 + 28), 4 * 23, 3))
        self.assertEqual(len(meta["panels"]), 12)
        for row, values in enumerate(((1, 2, 3, 1), (4, 5, 6, 1), (7, 8, 9, 1))):
            for col, value in enumerate(values):
                panel = meta["panels"][row * 4 + col]
                np.testing.assert_array_equal(panel["values"], height * value)
                self.assertEqual(panel["x0"], col * 23)
                self.assertEqual(panel["y0"], row * 45)


if __name__ == "__main__":
    unittest.main()
