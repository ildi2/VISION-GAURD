"""""
Experiment: fast YOLOv8n person detector with threaded camera.
Not part of core GaitGuard pipeline – used only for benchmarking.
"""
# fast live "person" detector (nano, FP16, low latency)
import cv2, time                  # camera + timing 
import threading, queue           # run camera capture in background thread
from ultralytics import YOLO      # YOLOv8 model
import torch                      # check CUDA and put model on GPU

 
# OPEN CAMERA SETUP
# Opens webcam with ID index (0 = default camera).
# Sets width, height, fps, MJPEG mode and buffer size
# (number of frames to keep in memory).
def open_camera(index=0, w=640, h=480, fps=30, mjpeg=True, buffersize=1):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    # If CAP_DSHOW ever fails, we could try cv2.CAP_MSMF instead.

    if mjpeg:
        # Force camera to send frames in MJPEG format (usually lower latency).
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    # Setting camera resolution and FPS targets (640x480 @ 30 fps).
    # 640x480 is a good balance between speed and enough detail.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # buffersize = 1 means OpenCV keeps only recent frames in memory,
    # reducing latency.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, buffersize)

    # Returns the configured VideoCapture object.
    return cap


# CameraSource – threaded, latest-frame-only reader
class CameraSource:
    # Wraps the camera in a class to manage it more cleanly.
    def __init__(self, cam_index=0, **kw):
        self.cap = open_camera(cam_index, **kw)
        if not self.cap.isOpened():
            raise RuntimeError("Camera not available")

        # Queue with maxsize=1 → we always keep only the most recent frame
        # from the background thread. This is memory-efficient and low latency.
        self.q = queue.Queue(maxsize=1)

        # Flag to control capture loop; when running=True we capture frames.
        self.running = False

    # Producer runs in a background thread to capture frames.
    # Reads frames continuously from camera.
    # If frame read fails → SKIP IT.
    # If queue already has a frame, remove it before putting the new one
    # (keeps only the most recent frame).
    def _producer(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                continue

            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass

            self.q.put(frame)

    # Sets running=True and starts daemon thread that runs _producer().
    def start(self):
        self.running = True
        # .start() at the end means thread starts running immediately.
        # daemon=True → thread will be killed when main program exits.
        threading.Thread(target=self._producer, daemon=True).start()

    def read_latest(self, timeout=1.0):
        # Main thread calls this to get the most recent frame.
        # timeout avoids blocking forever; if no frame arrives in 1 second,
        # returns None.
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    # Stops reading from camera and releases it cleanly.
    def stop(self):
        self.running = False
        self.cap.release()


# MAIN BLOCK – YOLO person detector loop
if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    if cuda:
        print("CUDA:", True, torch.cuda.get_device_name(0))

    # Use nano model for higher FPS.
    # NANO = smallest, fast variant; fits our goal for live detector.
    model = YOLO("yolov8n.pt")
    # For speed: fuse() merges convolution + batchnorm layers to reduce
    # overhead and speed up inference.
    model.fuse()

    # Move model to GPU and run one dummy prediction on a zero tensor.
    # This makes PyTorch/JIT and CUDA compile kernels and allocate memory
    # ahead of time, so the first real webcam frame won't have a long delay.
    # half=True → use FP16 (half precision), faster and uses less VRAM.
    if cuda:
        model.to("cuda")
        _ = model.predict(
            source=torch.zeros(1, 3, 480, 480).cuda(),
            imgsz=480,
            half=True,
            verbose=False,
        )

    # Create threaded camera source.
    src = CameraSource(0, w=640, h=480, fps=30, buffersize=1)
    src.start()

    t0, frames = time.time(), 0
    # We can try 480 → 416 → 384 if we want higher FPS.
    # Smaller → faster but less accuracy.
    IMG = 480
    # Confidence threshold: only detections above this value are kept.
    CONF = 0.25
    # Set to 2 to skip every other frame for higher FPS but fewer detections.
    STRIDE = 1

    try:
        # Infinite loop: read latest frame; if no frame → loop again.
        while True:
            frame = src.read_latest()
            if frame is None:
                continue

            results = model.predict(
                source=frame,
                device=0 if cuda else "cpu",
                imgsz=IMG,
                conf=CONF,
                iou=0.45,
                classes=[0],      # only person class
                half=cuda,        # use FP16 if using CUDA
                vid_stride=STRIDE,  # skip frames if > 1
                verbose=False,      # no console spam
            )

            # Take first result (this frame) and draw bounding boxes,
            # labels and scores on a copy of the frame.
            annotated = results[0].plot()

            # FPS overlay (smoothed).
            # Every 10 frames we recompute FPS = total frames / elapsed time.
            # Updating every 10 frames makes FPS more stable.
            frames += 1
            if frames % 10 == 0:
                fps = frames / (time.time() - t0)
                cv2.putText(
                    annotated,
                    f"FPS: {fps:.1f}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, 
                    (0, 255, 255),
                    2,
                )

            # Show window (live detector).
            cv2.imshow("GaitGuard 1.0  live detector", annotated)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break 

    finally:
        # finally block guarantees this runs even if there is an error
        # or ESC is pressed.
        src.stop()
        cv2.destroyAllWindows()
