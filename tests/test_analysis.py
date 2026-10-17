import unittest
import tempfile
from pathlib import Path
import torch
import pandas as pd
import sys
from types import SimpleNamespace
from unittest.mock import patch
from analyze_purity import main
from analyze_purity import compute_purity_metrics, plot_purity_vs_misalignment

class AnalysisTests(unittest.TestCase):
    def test_detached_projection(self):
        projection = torch.tensor([[1., 1.], [0., 1.]], requires_grad=True)
        metrics, matrix = compute_purity_metrics(projection)
        self.assertAlmostEqual(metrics['avg_abs_cos_sim'], 2**-.5 / 2, places=6)
        self.assertEqual(matrix.shape, (2,2))
    def test_invalid_projection(self):
        for p in (torch.zeros(2,2), torch.ones(2,1), torch.full((2,2), float('nan'))):
            with self.assertRaises(ValueError): compute_purity_metrics(p)
    def test_plot(self):
        df = pd.DataFrame({'Cs':[0,.3], 'avg_abs_cos_sim':[.1,.2], 'max_abs_cos_sim':[.3,.4]})
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'plot.png'
            plot_purity_vs_misalignment(df, output)
            self.assertGreater(output.stat().st_size, 0)

    def test_cli_all_projection_modes(self):
        p = torch.tensor([[1., 1.], [0., 1.]], requires_grad=True)
        model = SimpleNamespace(text_projection=p, image_projection=p)
        for mode in ('text', 'image', 'both'):
            with tempfile.TemporaryDirectory() as tmp:
                csv = Path(tmp)/'nested'/'result.csv'
                argv = ['analyze_purity.py','--checkpoints','Cs0:a','Cs30:b',
                        '--projection',mode,'--output-csv',str(csv),
                        '--output-plot',str(Path(tmp)/'curve.png'),
                        '--output-hist',str(Path(tmp)/'hist.png')]
                with patch.object(sys, 'argv', argv), patch('analyze_purity.load_model', return_value=model):
                    main()
                self.assertEqual(len(pd.read_csv(csv)),2)
                self.assertTrue((Path(tmp)/'curve.png').is_file())

if __name__ == '__main__': unittest.main()
