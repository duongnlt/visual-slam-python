import cv2
import numpy as np



class Map:
    def __init__(self):
        self.points = []
        self.frames = []

    def add_point(self, point):
        self.points.append(point)

    def add_frame(self, frame):
        self.frames.append(frame)

    def get_length_frames(self):
        return len(self.frames)

    def get_current_frame(self):
        return self.frames[-1]
