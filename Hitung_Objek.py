import cv2
import numpy as np

image_path = 'gambar.jpg'
img = cv2.imread(image_path)
img = cv2.resize(img, (800, 800))
blurred = cv2.GaussianBlur(img, (5, 5), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

colors = {
    'biru': ([100, 100, 50], [130, 255, 255]),
    'merah': ([0, 100, 100], [10, 255, 255]),
    'kuning': ([20, 100, 100], [30, 255, 255]),
    'oranye': ([10, 100, 100], [20, 255, 255]),
}

shape_counts = {'persegi panjang': 0, 'lingkaran': 0, 'oval': 0, 'segitiga': 0}
color_counts = {'biru': 0, 'merah': 0, 'kuning': 0, 'oranye': 0}

for color_name, (lower, upper) in colors.items():
    lower = np.array(lower)
    upper = np.array(upper)
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        shape = ""
        if len(approx) == 3:
            shape = 'segitiga'
        elif len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ratio = float(w) / h
            shape = 'persegi panjang'
        elif len(approx) > 4:
            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
            if abs(MA - ma) < 10:
                shape = 'lingkaran'
            else:
                shape = 'oval'

        if shape:
            shape_counts[shape] += 1
            color_counts[color_name] += 1

print("Jumlah objek berdasarkan warna:")
for color, count in color_counts.items():
    print(f"- {color}: {count}")

print("\nJumlah objek berdasarkan bentuk:")
for shape, count in shape_counts.items():
    print(f"- {shape}: {count}")
