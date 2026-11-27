"""PR9 CLI never writes scans or exposes the source image in its output."""
import json
import cv2
import numpy as np
import pytest

from autodocscanner.tools.pr9_v3_boundary_cli import main


def test_cli_local_photo_diagnostics_only(tmp_path, capsys):
    photo = tmp_path / 'private_sample.png'
    cv2.imwrite(str(photo), np.full((120, 160, 3), 110, np.uint8))
    def fake(image):
        assert image.shape == (120, 160, 3)
        return {"status": "offline_diagnostic_only", "auto_crop_allowed": False}
    assert main([str(photo)], evaluator=fake) == 0
    content = capsys.readouterr().out
    assert str(photo) not in content
    assert "private_sample" not in content
    assert json.loads(content)["auto_crop_allowed"] is False
    assert list(tmp_path.iterdir()) == [photo]


def test_cli_invalid_path_fails_without_files(tmp_path):
    with pytest.raises(SystemExit):
        main([str(tmp_path / 'missing.png')])
    assert list(tmp_path.iterdir()) == []
