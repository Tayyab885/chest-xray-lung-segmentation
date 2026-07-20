import numpy as np
import pandas as pd
import pytest
from PIL import Image


@pytest.fixture
def fake_frame(tmp_path):
    """Factory for a manifest-shaped frame backed by synthetic PNGs on disk.

    The two blobs have deliberately different widths. An equal-width pair would
    be its own mirror image, so a horizontal flip anywhere in the pipeline would
    leave it unchanged and every test built on this fixture would miss it.
    """
    def _make(n, source="a", size=32):
        rows = []
        for i in range(n):
            img = np.random.RandomState(i).randint(0, 255, (size, size)).astype(np.uint8)
            mask = np.zeros((size, size), dtype=np.uint8)
            mask[size // 4:3 * size // 4, size // 8:3 * size // 8] = 255
            mask[size // 4:3 * size // 4, 5 * size // 8:13 * size // 16] = 255
            ip = tmp_path / f"{source}_{i}.png"
            mp = tmp_path / f"{source}_{i}_m.png"
            Image.fromarray(img).save(ip)
            Image.fromarray(mask).save(mp)
            rows.append({
                "image_path": str(ip),
                "mask_paths": [str(mp)],
                "source": source,
                "patient_id": f"{source}/{i}",
            })
        return pd.DataFrame(rows)
    return _make
