import cv2
import glob
import numpy as np

vid = cv2.VideoCapture(0)
  
while(True):
    ret, frame = vid.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_filtered = cv2.bilateralFilter(gray, 10, 40, 40)
    edges_filtered = cv2.Canny(gray_filtered, 30, 60)

    rows_rgb, cols_rgb, channels = frame.shape
    rows_gray, cols_gray = edges_filtered.shape

    rows_comb = max(rows_rgb, rows_gray)
    cols_comb = cols_rgb + cols_gray
    comb = np.zeros(shape=(rows_comb, cols_comb, channels), dtype=np.uint8)

    comb[:rows_rgb, :cols_rgb] = frame
    comb[:rows_gray, cols_rgb:] = edges_filtered[:, :, None]
    cv2.imshow('frame', comb)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
vid.release()
cv2.destroyAllWindows()