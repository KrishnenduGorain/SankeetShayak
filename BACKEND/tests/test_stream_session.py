"""Unit tests for WebSocket StreamSession buffering and error handling."""

import numpy as np
import pytest
from src.api.websocket_handler import StreamSession


def test_stream_session_warmup_and_stride():
    """Verify session warms up for 30 frames and runs inference on stride=3."""
    session = StreamSession(buffer_size=30, stride=3)

    # Frames 1 to 29: no inference
    for i in range(29):
        valid, err, should_infer, snap = session.push_frame(np.zeros((21, 3)))
        assert valid is True
        assert err is None
        assert should_infer is False
        assert snap is None

    # Frame 30: window full, 30 % 3 == 0 -> should infer!
    valid, err, should_infer, snap = session.push_frame(np.zeros((21, 3)))
    assert valid is True
    assert should_infer is True
    assert snap is not None
    assert snap.shape == (30, 21, 3)

    # Frame 31: 31 % 3 != 0 -> should not infer
    valid, err, should_infer, snap = session.push_frame(np.zeros((21, 3)))
    assert valid is True
    assert should_infer is False

    # Frame 32: 32 % 3 != 0 -> should not infer
    valid, err, should_infer, snap = session.push_frame(np.zeros((21, 3)))
    assert valid is True
    assert should_infer is False

    # Frame 33: 33 % 3 == 0 -> should infer!
    valid, err, should_infer, snap = session.push_frame(np.zeros((21, 3)))
    assert valid is True
    assert should_infer is True


def test_stream_session_bad_frame_nan_rejection():
    """Verify NaN and malformed frames are rejected without poisoning buffer."""
    session = StreamSession(buffer_size=30)

    # Push valid frame
    valid, err, _, _ = session.push_frame(np.zeros((21, 3)))
    assert valid is True
    assert len(session.buffer) == 1

    # Push NaN frame
    nan_frame = np.zeros((21, 3))
    nan_frame[0, 0] = np.nan
    valid, err, should_infer, snap = session.push_frame(nan_frame)
    assert valid is False
    assert "NaN or Infinite" in err
    # Buffer length did not increase!
    assert len(session.buffer) == 1

    # Push wrong shape
    wrong_shape = np.zeros((10, 3))
    valid, err, _, _ = session.push_frame(wrong_shape)
    assert valid is False
    assert "shape" in err
    assert len(session.buffer) == 1


def test_stream_session_reset():
    """Verify reset clears buffer and resets frame counter."""
    session = StreamSession(buffer_size=30)
    for _ in range(10):
        session.push_frame(np.zeros((21, 3)))
    assert len(session.buffer) == 10
    assert session.frame_count == 10

    session.reset()
    assert len(session.buffer) == 0
    assert session.frame_count == 0
