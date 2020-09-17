#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Code based on `https://elder.dev/posts/open-source-virtual-background/`

import os
import cv2
import numpy as np
import requests
import imutils
# from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
import pyfakewebcam

class FakeCam:
	def __init__(self,
			# input_device = '/dev/video1',
			output_device = '/dev/video20',
			virtual_background = '/home/dragon/Pictures/31m-Fondosparaelzoom/31m-Wallpaper-LinoLanaHouse.png',
			# virtual_background = 'data/background.jpg',
			width = 1280,
			height = 720,
			real_background = 'data/real_bg.jpg',
			bodypix_url='http://localhost:9000'):

		self.running = False
		self.bodypix_url = bodypix_url

		# setup access to the *real* webcam
		self.cap = cv2.VideoCapture(1)
		self.height = 720
		self.width = 1280
		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
		self.cap.set(cv2.CAP_PROP_FPS, 15)

		# setup the fake camera
		self.fake = pyfakewebcam.FakeWebcam('/dev/video20', width, height)

		# load the real background
		self.realbg = cv2.imread(real_background)
		self.realbg = cv2.resize(self.realbg, (width, height))
		# self.realbg = cv2.cvtColor(self.realbg, cv2.COLOR_BGR2GRAY)

		# load the virtual background
		self.background = cv2.imread(virtual_background)
		self.background_scaled = cv2.resize(self.background, (width, height))
		#end __init__

	def __del__(self):
		self.cap.release()
		cv2.destroyAllWindows()
	#end __del__

	def capture_background(self):
		frames = []
		tries = 0
		while len(frames) < 100 and tries < 1000:
			tries+= 1
			success, frame = self.cap.read()
			if success:
				frames.append(frame)
		avg_frame = np.mean(frames, axis=0)
		cv2.imwrite("data/real_bg.jpg", avg_frame)

	def run(self):
		self.running = True
		while self.running:
			frame = self._get_frame()
			# fake webcam expects RGB
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			self.fake.schedule_frame(frame)
	# end def

	def stop(self):
		self.running = False
	# end def

	def _get_frame(self):
		_, frame = self.cap.read()

		# fetch the mask
		frame = cv2.flip(frame, 1)
		mask = self._get_mask(frame)
		# cv2.imshow('mask', mask)
		# cv2.imshow('video', frame)

		# post-process mask and frame
		mask = post_process_mask(mask)
		# frame = hologram_effect(frame)

		# composite the foreground and background
		inv_mask = 1-mask
		for c in range(frame.shape[2]):
			frame[:,:,c] = frame[:,:,c] * mask + self.background_scaled[:,:,c] * inv_mask

		cv2.imshow('video', frame)
		cv2.waitKey(1)
		return frame
	#end def

	def _get_mask(self, frame):
		_, data = cv2.imencode(".jpg", frame)
		r = requests.post(
			url=self.bodypix_url,
			data=data.tobytes(),
			headers={'Content-Type': 'application/octet-stream'})
		mask = np.frombuffer(r.content, dtype=np.uint8)
		mask = mask.reshape((frame.shape[0], frame.shape[1]))
		return mask
	#end def

#end class


# def get_mask(frame, bodypix_url='http://localhost:9000'):
# 	_, data = cv2.imencode(".jpg", frame)
# 	r = requests.post(
# 		url=bodypix_url,
# 		data=data.tobytes(),
# 		headers={'Content-Type': 'application/octet-stream'})
# 	mask = np.frombuffer(r.content, dtype=np.uint8)
# 	mask = mask.reshape((frame.shape[0], frame.shape[1]))
# 	return mask


def post_process_mask(mask):
	mask = cv2.dilate(mask, np.ones((10,10), np.uint8) , iterations=1)
	mask = cv2.blur(mask.astype(float), (30,30))
	return mask


def shift_image(img, dx, dy):
	img = np.roll(img, dy, axis=0)
	img = np.roll(img, dx, axis=1)
	if dy>0:
		img[:dy, :] = 0
	elif dy<0:
		img[dy:, :] = 0
	if dx>0:
		img[:, :dx] = 0
	elif dx<0:
		img[:, dx:] = 0
	return img


def hologram_effect(img):
	# add a blue tint
	holo = cv2.applyColorMap(img, cv2.COLORMAP_WINTER)
	# add a halftone effect
	bandLength, bandGap = 2, 3
	for y in range(holo.shape[0]):
		if y % (bandLength+bandGap) < bandLength:
			holo[y,:,:] = holo[y,:,:] * np.random.uniform(0.1, 0.3)
	# add some ghosting
	holo_blur = cv2.addWeighted(holo, 0.2, shift_image(holo.copy(), 5, 5), 0.8, 0)
	holo_blur = cv2.addWeighted(holo_blur, 0.4, shift_image(holo.copy(), -5, -5), 0.6, 0)
	# combine with the original color, oversaturated
	out = cv2.addWeighted(img, 0.5, holo_blur, 0.6, 0)
	return out


def main():
	print('Starting FakeCam')
	fakecam = FakeCam(width=640, height=480)
	# fakecam.capture_background()
	# return
	print('Initialization complete!')
	fakecam.run()
	print('Running...')

if __name__ == '__main__':
	main()

	# capture = cv2.VideoCapture(1)
	# while(True):
	# 	ret, frame = capture.read()
	# 	cv2.imshow('video', frame)
	# 	if cv2.waitKey(1) == 27:
	# 		break
	# capture.release()
	# cv2.destroyAllWindows()

