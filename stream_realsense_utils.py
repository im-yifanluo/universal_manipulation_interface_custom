import pyrealsense2 as rs
import numpy as np
from threading import Thread, Lock
from collections import deque
import time


class RealsenseVideoStream:
    """Latest-frame RealSense color grabber. read() returns the freshest frame
    immediately, without waiting for the next one. A short rolling history is
    kept so read_history() can also return a frame from a moment in the past."""

    def __init__(self, width=320, height=240, fps=60, serial=None, history_seconds=0.5):
        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial is not None:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        profile = self.pipeline.start(config)
        profile.get_device().first_color_sensor().set_option(rs.option.global_time_enabled, 1)

        # rolling buffer of (timestamp_ms, frame), oldest -> newest. Sized to cover
        # history_seconds with a couple frames of margin.
        self.history_seconds = history_seconds
        maxlen = max(2, int(round(history_seconds * fps)) + 2)
        self.buffer = deque(maxlen=maxlen)

        self.lock = Lock()
        self.running = True
        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            frames = self.pipeline.wait_for_frames()  # blocks until next frame; 0% CPU while waiting
            color = frames.get_color_frame()
            if color:
                # copy out of librealsense's frame pool: we retain many frames, so
                # storing raw views would exhaust the pool and get recycled/corrupted.
                frame = np.array(color.get_data())
                ts = frames.get_timestamp()
                with self.lock:
                    self.buffer.append((ts, frame))

    def read(self):
        """Return (latest_frame, latest_ts_ms), or None if no frame yet."""
        with self.lock:
            if not self.buffer:
                return None
            ts, frame = self.buffer[-1]
        return frame.copy(), ts

    def read_history(self, interval=0.1):
        """Return (past_frame, past_ts, latest_frame, latest_ts) ordered oldest ->
        newest to match the [history, current] obs ordering the policy expects
        (current is the last obs step). `past_frame` is the buffered frame closest
        to `interval` seconds before the latest. Returns None if no frame yet. If the
        buffer doesn't reach back that far (interval > history_seconds), the oldest
        available frame is returned."""
        target_offset_ms = interval * 1000.0
        with self.lock:
            if not self.buffer:
                return None
            latest_ts, latest_frame = self.buffer[-1]
            target = latest_ts - target_offset_ms
            past_ts, past_frame = min(self.buffer, key=lambda item: abs(item[0] - target))
        return past_frame.copy(), past_ts, latest_frame.copy(), latest_ts

    def stop(self):
        self.running = False
        self.thread.join()
        self.pipeline.stop()


# if __name__ == '__main__':
#     import cv2
#     stream = RealsenseVideoStream()
#     try:
#         while True:
#             out = stream.read_history(interval=0.1)
#             if out is None:
#                 continue
#             past_frame, past_ts, frame, ts = out
#             print(f'latest age ~{time.time() * 1000 - ts:.1f} ms | '
#                   f'gap latest->past {ts - past_ts:.1f} ms')
#             cv2.imshow('RealSense latest', frame)
#             cv2.imshow('RealSense ~0.1s ago', past_frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#     finally:
#         stream.stop()
#         cv2.destroyAllWindows()
