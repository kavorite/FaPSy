import time

import cv2
import numpy as np

from common import Attenuator, EmbeddingObjective, Recognizer
from embed_tags import tag_hit_generator
from solve_recognizer import ensure_preview, preview_content

objective = EmbeddingObjective(np.load("./index/dictionary.npz"))
dictfile = np.load("./index/recognizer.npz")
recognizer = Recognizer(
    dictfile["thumb_dim"], dictfile["highfreq_factor"], dictfile["recognizer"]
)
attenuator = Attenuator(objective.vecs, np.load("./index/attenuator.npy"))


class MSE:
    def __init__(self):
        self.n = 0
        self.error = 0

    @staticmethod
    def compute(y, yhat):
        return np.sum((y - yhat) ** 2) / len(y)

    def update(self, y, yhat):
        error = self.compute(y, yhat)
        error += self.n * self.error
        self.n += 1
        error /= self.n
        self.error = error

    def value(self):
        return self.error

    def __str__(self):
        return str(self.value())


print("find post candidates...")
posts = list(ensure_preview(tag_hit_generator(objective.tags)))
last_print = 0
attenuator_error = MSE()
recognizer_error = MSE()
for post, img_str in preview_content(posts):
    tags = post["tag_string"].split()
    y = objective(tags)
    if y is None:
        continue
    attenuator_error.update(y, attenuator(tags))
    try:
        recognizer_error.update(y, recognizer(img_str))
    except cv2.error:
        continue
    now = time.time()
    if now - last_print >= 1.0:
        last_print = now
        print(f"attenuator MSE: {attenuator_error}")
        print(f"recognizer MSE: {recognizer_error}")
