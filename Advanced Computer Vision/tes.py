import cv2
import numpy as np
import matplotlib.pyplot as plt

# Membaca citra
img = cv2.imread('D:\App\Kuliah\Advanced Computer Vision\gambar.jpg')
imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Rotasi objek - sudut rotasi 45 derajat
height, width = img.shape[:2]
center = (width//2, height//2)
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated_img = cv2.warpAffine(imgrgb, rotation_matrix, (width, height),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))

# Transformasi warna - konversi ke grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Menampilkan hasil
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(imgrgb)
axs[0].set_title('Gambar Asli')
axs[0].axis('off')

axs[1].imshow(rotated_img)
axs[1].set_title('Gambar Rotasi 45 Derajat')
axs[1].axis('off')

axs[2].imshow(gray_img, cmap='gray')
axs[2].set_title('Gambar Grayscale')
axs[2].axis('off')

plt.tight_layout()
plt.show()
