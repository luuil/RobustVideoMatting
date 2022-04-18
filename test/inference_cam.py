import cv2
import torch
import time
from model import MattingNetwork
from model.model_rvm import MattingNetworkRVM
from PIL import Image
from torchvision import transforms
from threading import Thread, Lock


# ----------- Utility classes -------------


class Camera:
	"""
	A wrapper that reads data from cv2.VideoCapture in its own thread to optimize.
	Use .read() in a tight loop to get the newest frame.
	"""

	def __init__(self, device_id=0, width=1280, height=720):
		self.capture = cv2.VideoCapture(device_id)
		self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
		self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
		self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
		# self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
		self.success_reading, self.frame = self.capture.read()
		self.read_lock = Lock()
		self.thread = Thread(target=self.__update, args=())
		self.thread.daemon = True
		self.thread.start()

	def __update(self):
		while self.success_reading:
			grabbed, frame = self.capture.read()
			with self.read_lock:
				self.success_reading = grabbed
				self.frame = frame

	def read(self):
		with self.read_lock:
			frame = self.frame.copy()
		return frame

	def __exit__(self, exec_type, exc_value, traceback):
		self.capture.release()


class FPSTracker:
	"""
	An FPS tracker that computes exponentialy moving average FPS.
	"""

	def __init__(self, ratio=0.5):
		self._last_tick = None
		self._avg_fps = None
		self.ratio = ratio

	def tick(self):
		if self._last_tick is None:
			self._last_tick = time.time()
			return None
		t_new = time.time()
		fps_sample = 1.0 / (t_new - self._last_tick)
		self._avg_fps = self.ratio * fps_sample + (
					1 - self.ratio) * self._avg_fps if self._avg_fps is not None else fps_sample
		self._last_tick = t_new
		return self.get()

	def get(self):
		return self._avg_fps


class Displayer:
	"""
	Wrapper for playing a stream with cv2.imshow().
	It also tracks FPS and optionally overlays info onto the stream.
	"""

	def __init__(self, title, width=None, height=None, show_info=True):
		self.title, self.width, self.height = title, width, height
		self.show_info = show_info
		self.fps_tracker = FPSTracker()
		cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
		if width is not None and height is not None:
			cv2.resizeWindow(self.title, width, height)

	# Update the currently showing frame and return key press char code
	def step(self, image):
		fps_estimate = self.fps_tracker.tick()
		if self.show_info and fps_estimate is not None:
			message = f"{int(fps_estimate)} fps | {self.width}x{self.height}"
			cv2.putText(image, message, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
		cv2.imshow(self.title, image)
		return cv2.waitKey(1) & 0xFF


def cv2_frame_to_cuda(frame):
	"""
	convert cv2 frame to tensor.
	"""
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	loader = transforms.ToTensor()
	return loader(Image.fromarray(frame)).to(device, dtype, non_blocking=True).unsqueeze(0)


def auto_downsample_ratio(h, w):
	"""
	Automatically find a downsample ratio so that the largest side of the resolution be 512px.
	"""
	return min(512 / max(h, w), 1)

def init_model(m, w):
	model = m.to(device, dtype, non_blocking=True).eval()
	model.load_state_dict(torch.load(w))
	model = torch.jit.script(model)
	model = torch.jit.freeze(model)
	return model


# --------------- Main ---------------

if __name__ == '__main__':

	width, height = (640, 360)  # the show windows size.
	output_background = 'green'  # Options: ["green", "white", "image"].
	dtype = torch.float32
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# load mobilenetv3 model
	model = init_model(MattingNetwork('mobilenetv3'), r'test\exp38\stage1\epoch-29.pth')

	# load rvm mobilenetv3 model
	model_rvm = init_model(MattingNetworkRVM('mobilenetv3'), 'test/rvm_mobilenetv3/rvm_mobilenetv3.pth')

	cam = Camera(width=width, height=height)
	dsp_src = Displayer('Input', cam.width, cam.height, show_info=True)
	dsp = Displayer('Ours', cam.width, cam.height, show_info=True)
	dsp_rvm = Displayer('RVM', cam.width, cam.height, show_info=True)

	bgr = None
	if output_background == 'white':
		bgr = torch.tensor([255, 255, 255], device=device, dtype=dtype).div(255).view(3, 1, 1)  # white background
	elif output_background == 'green':
		bgr = torch.tensor([120, 255, 155], device=device, dtype=dtype).div(255).view(3, 1, 1)  # green background

	with torch.no_grad():
		while True:
			# matting
			frame = cam.read()
			src = cv2_frame_to_cuda(frame)
			downsample_ratio = auto_downsample_ratio(*src.shape[2:])
			
			rec_rvm = [None] * 4
			fgr_rvm, pha_rvm, *rec_rvm = model_rvm(src, *rec_rvm, downsample_ratio)
			
			rec = [None] * 4
			pha, *rec = model(src, *rec, downsample_ratio, segmentation_pass=True)     
			pha = pha.sigmoid()
			fgr = src * pha

			if bgr is None:
				h, w = src.shape[2:]
				# print(h, w)
				transform = transforms.Compose([
					transforms.Resize(size=(h, w)),
					transforms.ToTensor()
				])
				img = Image.open("work/background/background3.jpg")
				bgr = transform(img).to(device, dtype, non_blocking=True)

			com = fgr * pha + bgr * (1 - pha)
			com = com.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
			com = cv2.cvtColor(com, cv2.COLOR_RGB2BGR)

			com_rvm = fgr_rvm * pha_rvm + bgr * (1 - pha_rvm)
			com_rvm = com_rvm.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
			com_rvm = cv2.cvtColor(com_rvm, cv2.COLOR_RGB2BGR)

			_key = dsp_src.step(frame)
			_key = dsp_rvm.step(com_rvm)
			key = dsp.step(com)
			if key == ord('b'):
				break
			elif key == ord('q'):
				exit()