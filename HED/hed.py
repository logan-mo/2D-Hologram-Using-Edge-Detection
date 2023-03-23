import cv2
import time

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FPS, 5)
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "hed_pretrained_bsds.caffemodel")
ret, frame = vid.read()
(H, W) = frame.shape[:2]
print(H, W)


start = time.time()
while(True):
    ret, frame = vid.read()

    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(W, H),swapRB=False, crop=False)
    net.setInput(blob)
    hed = net.forward()
    hed = cv2.resize(hed[0, 0], (W, H))
    hed = (255 * hed).astype("uint8")

    cv2.imshow('frame', hed)
    print(time.time() - start)
    start = time.time()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
vid.release()
cv2.destroyAllWindows()