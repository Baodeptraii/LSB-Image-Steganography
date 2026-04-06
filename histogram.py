import cv2
import matplotlib.pyplot as plt

# -------- LOAD IMAGES --------
img_original = cv2.imread("ptit.jpg", cv2.IMREAD_GRAYSCALE)
img_stego = cv2.imread("hacked_ptit.jpg", cv2.IMREAD_GRAYSCALE)

# -------- COMPUTE HISTOGRAM --------
hist_orig = cv2.calcHist([img_original], [0], None, [256], [0,256])
hist_stego = cv2.calcHist([img_stego], [0], None, [256], [0,256])

# -------- PLOT --------
plt.figure()
plt.plot(hist_orig, label="Original Image")
plt.plot(hist_stego, label="Stego Image")
plt.xlabel("Gray level")
plt.ylabel("Number of pixels")
plt.title("Histogram comparison (Grayscale)")
plt.legend()
plt.grid(True)
plt.show()

