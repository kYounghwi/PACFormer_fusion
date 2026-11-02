import json
import gc
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn

from PACFormer.exp import _OptimizerBundle, _build_optimizer
from PACFormer.modules.PACFormer import Model
from PACFormer.modules.tst_backbone import _MultiheadAttention
from src.data_factory import _split_sample_starts
from src.data_loader import ICONForecastArchiveStore, NWPMemmapStore, StandardScaler
from src.metrics import CapacityMetricAccumulator


class DataContractTests(unittest.TestCase):
    def _temporary_directory(self):
        return tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)

    def test_memmap_history_precedes_future(self):
        with self._temporary_directory() as tmp:
            root = Path(tmp)
            shape = (12, 2, 3, 1)
            values = np.memmap(root / "data.float16.memmap", mode="w+", dtype=np.float16, shape=shape)
            values[:] = np.arange(shape[0], dtype=np.float16)[:, None, None, None]
            values.flush()
            (root / "manifest.json").write_text(json.dumps({"shape": shape}), encoding="utf-8")
            np.save(root / "variables.npy", np.asarray(["x"]))
            dates = pd.date_range("2020-01-01", periods=shape[0], freq="h")
            store = NWPMemmapStore(
                root, dates, future_len=3, history_len=2, scale=False
            )
            result = store[dates[5]]
            self.assertEqual(result.shape, (3, 2, 5, 1))
            np.testing.assert_allclose(
                result[0, 0, :, 0], np.arange(3, 8), rtol=1e-5
            )
            del result
            del store.values
            del store
            del values
            gc.collect()

    def test_icon_uses_latest_available_cycle(self):
        initializations = pd.to_datetime(["2020-01-01 00:00", "2020-01-01 06:00"])
        fields = {}
        for cycle, base in zip(initializations, [0.0, 100.0]):
            field = np.zeros((2, 3, 12, 1), dtype=np.float32)
            field[..., 0] = base + np.arange(12)[None, None, :]
            fields[cycle] = field
        store = ICONForecastArchiveStore(fields, future_len=3, history_len=0)
        at_05 = store[pd.Timestamp("2020-01-01 05:00")]
        at_07 = store[pd.Timestamp("2020-01-01 07:00")]
        np.testing.assert_allclose(at_05[0, 0, :, 0], [5, 6, 7], rtol=1e-5)
        np.testing.assert_allclose(at_07[0, 0, :, 0], [101, 102, 103], rtol=1e-5)

    def test_icon_history_policy_is_explicit(self):
        field = np.zeros((2, 3, 12, 1), dtype=np.float32)
        store = ICONForecastArchiveStore(
            {pd.Timestamp("2020-01-01"): field}, future_len=2, history_len=1
        )
        with self.assertRaises(NotImplementedError):
            _ = store[pd.Timestamp("2020-01-01")]

    def test_missing_pv_excludes_complete_sample_window(self):
        dates = pd.date_range("2020-01-01", periods=30, freq="h")
        pv = np.ones((30, 2), dtype=np.float32)
        pv[7, 0] = np.nan
        available = {pd.Timestamp(date): True for date in dates}
        starts, _ = _split_sample_starts(
            dates, pv, available, seq_len=3, pred_len=2,
            train_ratio=0.7, val_ratio=0.1,
        )
        for split_starts in starts.values():
            for start in split_starts:
                self.assertTrue(np.isfinite(pv[start : start + 5]).all())

    def test_capacity_metrics_match_site_and_regional_definitions(self):
        accumulator = CapacityMetricAccumulator(np.asarray([2.0, 4.0]))
        target = np.asarray([[[1.0, 2.0]]])
        prediction = np.asarray([[[2.0, 0.0]]])
        accumulator.update(prediction, target, target > 1e-3)
        result = accumulator.compute()
        self.assertAlmostEqual(result["nMAE_percent"], 50.0)
        self.assertAlmostEqual(result["nRMSE_percent"], 50.0)
        self.assertAlmostEqual(result["regional_total_nMAE_percent"], 100.0 / 6.0)
        self.assertAlmostEqual(result["regional_total_nRMSE_percent"], 100.0 / 6.0)


class ModelContractTests(unittest.TestCase):
    def _temporary_directory(self):
        return tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)

    def _config(self, station_path, fusion_mode):
        return SimpleNamespace(
            pred_len=4, c_out=3, enc_in=3, seq_len=8, d_model=8, dropout=0.0,
            factor=1, output_attention=False, n_heads=2, d_ff=16, e_layers=1,
            d_layers=1, num_groups=2, patch_len=4, stride=2, norm="LayerNorm",
            padding_patch="end", stations_csv_path=str(station_path),
            q_event_mode="event", sites_per_group=2, diversity_weight=0.4,
            activation="gelu", time_len=6, nwp_n_vars=2,
            spatial_resolution=(6, 12), cube_patch=(2, 2, 2),
            nwp_vit_layers=1, nwp_use_stats=False, pooling=False,
            fusion_mode=fusion_mode, fusion_d_ff=16,
        )

    def test_both_fusion_modes_preserve_output_contract(self):
        with self._temporary_directory() as tmp:
            station_path = Path(tmp) / "stations.csv"
            station_path.write_text("site,lat,lon\na,0,0\nb,0,1\nc,1,0\n", encoding="utf-8")
            x = torch.randn(1, 8, 3)
            marks = torch.randn(1, 8, 4)
            nwp = torch.randn(1, 6, 12, 6, 2)
            for mode in ("decay", "swiglu"):
                model = Model(self._config(station_path, mode)).eval()
                group_attention = model.TST.backbone.encoder.layers[0].self_attn
                self.assertEqual(group_attention.sites_per_group, 2)
                self.assertEqual(group_attention.diversity_weight, 0.4)
                with torch.no_grad():
                    output = model(x, marks, None, None, nwp)
                self.assertEqual(tuple(output.shape), (1, 4, 3))
                self.assertTrue(torch.isfinite(output).all())

    def test_strict_masks_keep_probabilities_finite(self):
        with self._temporary_directory() as tmp:
            station_path = Path(tmp) / "stations.csv"
            station_path.write_text("site,lat,lon\na,0,0\nb,0,1\nc,1,0\n", encoding="utf-8")
            attention = _MultiheadAttention(
                num_groups=2, c_in=3, d_model=8, n_heads=2,
                stations_csv_path=str(station_path), sites_per_group=2,
                diversity_weight=0.4,
            )
            # Equal groups have no downward transition. Strict masking must
            # keep every forbidden off-diagonal link closed; the only allowed
            # fallback is the self-link used for an otherwise empty row.
            groups = torch.ones(2, 2, 4, 8)
            indices = torch.tensor([[0, 1], [1, 2]])
            propagated, weights = attention.cross_group_propagation(groups, indices)
            self.assertTrue(torch.isfinite(propagated).all())
            self.assertTrue(torch.isfinite(weights).all())
            torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
            expected = torch.eye(8).expand(2, -1, -1)
            torch.testing.assert_close(weights, expected)

    @unittest.skipUnless(hasattr(torch.optim, "Muon"), "torch.optim.Muon unavailable")
    def test_muon_steps_the_small_pacformer_contract(self):
        with self._temporary_directory() as tmp:
            station_path = Path(tmp) / "stations.csv"
            station_path.write_text(
                "site,lat,lon\na,0,0\nb,0,1\nc,1,0\n", encoding="utf-8"
            )
            model = Model(self._config(station_path, "swiglu")).train()
            args = SimpleNamespace(
                optimizer="muon", learning_rate=1e-4, weight_decay=0.0,
                muon_momentum=0.95, muon_ns_steps=5,
                muon_adjust_lr="match_rms_adamw",
            )
            optimizer = _build_optimizer(args, model)
            output = model(
                torch.randn(1, 8, 3), torch.randn(1, 8, 4), None, None,
                torch.randn(1, 6, 12, 6, 2),
            )
            loss = output.square().mean()
            loss.backward()
            optimizer.step()
            self.assertTrue(torch.isfinite(loss.detach()))


class OptimizerContractTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(torch.optim, "Muon"), "torch.optim.Muon unavailable")
    def test_muon_is_opt_in_and_uses_adamw_fallback(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden = nn.Linear(4, 4)
                self.enc_embedding = nn.Embedding(8, 4)
                self.nwp_branch = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
                self.output_projection = nn.Linear(4, 2)

        model = TinyModel()
        args = SimpleNamespace(
            optimizer="muon", learning_rate=1e-4, weight_decay=0.0,
            muon_momentum=0.95, muon_ns_steps=5,
            muon_adjust_lr="match_rms_adamw",
        )
        optimizer = _build_optimizer(args, model)
        self.assertIsInstance(optimizer, _OptimizerBundle)
        self.assertEqual(
            [item.__class__.__name__ for item in optimizer.optimizers],
            ["Muon", "AdamW"],
        )

        muon_ids = {
            id(parameter)
            for group in optimizer.optimizers[0].param_groups
            for parameter in group["params"]
        }
        adamw_ids = {
            id(parameter)
            for group in optimizer.optimizers[1].param_groups
            for parameter in group["params"]
        }
        all_ids = {id(parameter) for parameter in model.parameters()}
        self.assertFalse(muon_ids & adamw_ids)
        self.assertEqual(muon_ids | adamw_ids, all_ids)
        self.assertIn(id(model.hidden.weight), muon_ids)
        self.assertIn(id(model.hidden.bias), adamw_ids)
        self.assertIn(id(model.enc_embedding.weight), adamw_ids)
        self.assertIn(id(model.output_projection.weight), adamw_ids)


if __name__ == "__main__":
    unittest.main()
