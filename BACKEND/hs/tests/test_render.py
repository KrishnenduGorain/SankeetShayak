"""OpenCV rendering utilities: video sink, codec fallback, offscreen drawing."""
import numpy as np
import pytest

from src.data.collector import hand_template
from src.vision.render import VideoSink, blank_canvas, read_all_frames, render_skeleton_canvas, save_image


def _frames(n=12, size=(64, 48)):
    return [np.full((size[1], size[0], 3), 10 * i, np.uint8) for i in range(n)]


def test_video_sink_roundtrip(tmp_path):
    with VideoSink(tmp_path / "o.mp4", 30.0, (64, 48)) as sink:
        for f in _frames():
            sink.write(f)
    back = read_all_frames(sink.path)
    assert sink.frames_written == 12 and len(back) == 12 and back[0].shape == (48, 64, 3)


def test_video_sink_codec_fallback(tmp_path):
    with VideoSink(tmp_path / "o.mp4", 30.0, (64, 48), codecs=[("ZZZZ", ".mp4"), ("MJPG", ".avi")]) as sink:
        sink.write(_frames(1)[0])
    assert sink.codec == "MJPG" and sink.path.suffix == ".avi" and sink.path.exists()


def test_video_sink_no_codec_raises(tmp_path):
    with pytest.raises(RuntimeError):
        VideoSink(tmp_path / "o.mp4", 30.0, (64, 48), codecs=[("ZZZZ", ".mp4")])


def test_video_sink_resizes_and_converts(tmp_path):
    with VideoSink(tmp_path / "o.avi", 25.0, (64, 48), codecs=[("MJPG", ".avi")]) as sink:
        sink.write(np.zeros((100, 120, 3), np.float64))
    assert len(read_all_frames(sink.path)) == 1


def test_invalid_size_rejected(tmp_path):
    with pytest.raises(ValueError):
        VideoSink(tmp_path / "x.mp4", 30.0, (0, 10))


def test_offscreen_skeleton_render_and_save(tmp_path):
    t = hand_template("open")
    norm = np.zeros((21, 3))
    norm[:, 0] = 0.5 + t[:, 0] * 2
    norm[:, 1] = 0.9 + t[:, 1] * 2
    img = render_skeleton_canvas(norm, 320, 240)
    assert img.shape == (240, 320, 3) and (img != blank_canvas(320, 240)).any()
    p = save_image(tmp_path / "sub" / "skel.png", img)
    assert p.exists() and p.stat().st_size > 100
