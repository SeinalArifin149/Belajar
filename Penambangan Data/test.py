import cv2

img = cv2.imread('/home/seinalmint/Semester 4 Mint/Penambangan Data/Dataset SEL/gambar1.png')

if img is None:
    print("Gambar tidak ditemukan! Cek path dan nama file.")
else:
    cv2.imshow('Gambar Saya', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
