import cv2
import numpy as np
from frame import Frame


def feature_matching(f1: Frame, f2: Frame):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(f1.descriptors, f2.descriptors, k=2)

    # idx1[i] in f1 matches idx2[i] in f2; the sets only guard against duplicates
    idx1, idx2 = [], []
    seen1, seen2 = set(), set()

    for pair in matches:
        if len(pair) < 2:
            continue
        m1, m2 = pair
        if m1.distance < 0.75 * m2.distance:
            pts1 = f1.pts[m1.queryIdx]
            pts2 = f2.pts[m1.trainIdx]

            # travel less than 10% of diagonal and be within orb distance 32
            if np.linalg.norm((pts1 - pts2)) < 0.1 * np.linalg.norm([f1.w, f1.h]) and m1.distance < 32:
                if m1.queryIdx not in seen1 and m1.trainIdx not in seen2:
                    seen1.add(m1.queryIdx)
                    seen2.add(m1.trainIdx)
                    idx1.append(m1.queryIdx)
                    idx2.append(m1.trainIdx)

    return np.array(idx1, dtype=int), np.array(idx2, dtype=int)

    