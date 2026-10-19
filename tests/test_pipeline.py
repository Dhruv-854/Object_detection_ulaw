import tempfile
import unittest
from pathlib import Path

from object_detection.domain import Detection
from object_detection.pipeline import RunOptions, run_detection


class _Frame:
    shape = (100, 200, 3)

    def copy(self):
        return _Frame()


class _Detector:
    model_id = "fake-model"
    family = "test"
    checkpoint = Path("fake.pt")
    checkpoint_sha256 = "abc123"
    checkpoint_size_bytes = 10
    parameter_count = 100
    device = "cpu"
    input_size = 640
    confidence = 0.25
    max_detections = 300

    def __init__(self) -> None:
        self.frames = []

    def predict(self, frame, frame_index, timestamp_ms):
        self.frames.append(frame_index)
        return [
            Detection(frame_index, timestamp_ms, -3, 2, 205, 90, 0, "bottle", 0.8),
            Detection(frame_index, timestamp_ms, 1, 1, 1, 5, 1, "chair", 0.9),
            Detection(frame_index, timestamp_ms, 2, 2, 5, 5, 2, "cup", 0.1),
        ]


class _Capture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.position = 0
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame

    def get(self, property_id):
        values = {3: 200, 4: 100, 5: 25.0, 6: self.position * 40.0}
        return values.get(property_id, 0)

    def release(self):
        self.released = True


class _Writer:
    def __init__(self, path):
        self.path = Path(path)
        self.path.write_bytes(b"video")
        self.frames = []
        self.released = False

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(frame)

    def release(self):
        self.released = True


class _Cv2:
    __version__ = "fake"
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_POS_MSEC = 6
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 0

    def __init__(self, frames=None):
        self.frames = frames or []
        self.capture = None
        self.writer = None

    def imread(self, path):
        return _Frame()

    def imwrite(self, path, frame):
        Path(path).write_bytes(b"annotated")
        return True

    def VideoCapture(self, source):
        self.capture = _Capture(self.frames)
        return self.capture

    def VideoWriter_fourcc(self, *codec):
        return 1234

    def VideoWriter(self, path, fourcc, fps, dimensions):
        self.writer = _Writer(path)
        return self.writer

    def rectangle(self, *args):
        return None

    def putText(self, *args):
        return None

    def imshow(self, *args):
        return None

    def waitKey(self, delay):
        return -1

    def destroyAllWindows(self):
        return None


class PipelineTests(unittest.TestCase):
    def test_image_pipeline_writes_media_detections_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "input.jpg"
            image.write_bytes(b"image")
            detector = _Detector()
            summary = run_detection(
                detector,
                RunOptions(source=str(image), output_root=root / "outputs"),
                cv2_module=_Cv2(),
            )
            self.assertEqual(summary.frames_processed, 1)
            self.assertEqual(summary.detections, 1)
            self.assertTrue(Path(summary.output_paths["media"]).is_file())
            detections = Path(summary.output_paths["detections"]).read_text(encoding="utf-8")
            self.assertEqual(len(detections.splitlines()), 1)
            self.assertTrue(Path(summary.output_paths["summary"]).is_file())

    def test_video_pipeline_preserves_order_and_excludes_warmup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "input.mp4"
            video.write_bytes(b"video")
            detector = _Detector()
            cv2 = _Cv2([_Frame(), _Frame(), _Frame(), _Frame()])
            summary = run_detection(
                detector,
                RunOptions(
                    source=str(video),
                    output_root=root / "outputs",
                    warmup_frames=1,
                    max_frames=2,
                ),
                cv2_module=cv2,
            )
            self.assertEqual(detector.frames, [0, 1, 2])
            self.assertEqual(summary.frames_processed, 2)
            self.assertEqual(summary.detections, 2)
            self.assertEqual(len(cv2.writer.frames), 2)
            self.assertTrue(cv2.capture.released)
            self.assertTrue(cv2.writer.released)


if __name__ == "__main__":
    unittest.main()
