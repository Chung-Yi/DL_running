import numpy as np
import cv2


def order_points(pts):

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(s)]
    rect[3] = pts[np.argmax(s)]

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    h, w = image.shape[:2]

    widthA = np.sqrt(((br[0] - bl[0])**2) + ((br[1] - bl[1])**2))
    widthB = np.sqrt(((tl[0] - tl[0])**2) + ((tl[1] - tl[1])**2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0])**2) + ((tr[1] - br[1])**2))
    heightB = np.sqrt(((tl[0] - bl[0])**2) + ((tl[1] - bl[1])**2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]],
                   dtype="float32")
    dst = np.array([[0, 80], [w, 120], [w, 430], [0, 470]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (w, h))

    return warped


def main():

    img = cv2.imread('../data/lena.png')
    pts = np.array([[60, 40], [420, 40], [420, 510], [60, 510]],
                   dtype=np.float32)

    warped = four_point_transform(img, pts)

    cv2.imshow("Original", img)
    cv2.imshow("Warped", warped)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()