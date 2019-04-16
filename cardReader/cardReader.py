import numpy as np
import cv2
import time
import tensorflow as tf
execdir = None
model = None


#returns gray thresholded image
def threshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #img_w, img_h = np.shape(img)[:2]
    #bkg_level = gray[int(img_h/100)][int(img_w/2)]
    #thresh_level = bkg_level + 60
    ret, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
    #cv2.imshow("Thresh", thresh)
    return thresh


#returns contours in an image sorted by area
def getContours(thresh):
    contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return []
    index_sort = sorted(range(len(contours)), key=lambda i : cv2.contourArea(contours[i]),reverse=True)
    cnts_sort = []
    for i in index_sort:
        cnts_sort.append(contours[i])
    return contours


#given contour draw rotated rectangle and (currently commented out) straight bounding rectangle
def rectangles(cnt, img):
    rect = cv2.minAreaRect(cnt)
    box = np.int0(cv2.boxPoints(rect))
    cv2.drawContours(img, [box], -1, (255,0,0), 2)
    """
    x1 = min(point[0] for point in box)
    y1 = min(point[1] for point in box)
    x2 = max(point[0] for point in box)
    y2 = max(point[1] for point in box)
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0),2)
    """


#given contour, draw points on the corners
def corners(cnt, img, text=""):
    epsilon = 0.1*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    cv2.drawContours(img, approx, -1, (0,0,255), 8)
    if text:
        cv2.putText(img, text, (approx[0][0][0], approx[0][0][1]+15), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)


#personal attempt at card image extraction, replaced by new isolate function
"""
def isolate(cnt, img, draw=True):
    rect = cv2.minAreaRect(cnt)
    box = np.int0(cv2.boxPoints(rect))
    x1 = min(point[0] for point in box)
    y1 = min(point[1] for point in box)
    x2 = max(point[0] for point in box)
    y2 = max(point[1] for point in box)
    w = x2 - x1
    h = y2 - y1

    mask = np.zeros((img.shape[0], img.shape[1]))

    cv2.fillConvexPoly(mask, box, 1)
    mask = mask.astype(np.bool)

    out = np.zeros_like(img)
    out[mask] = img[mask]
    out = out[y1:y2, x1:x2]
    out = cv2.resize(out, (200,280))

    if draw:
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0),2)
        cv2.drawContours(img, [box], -1, (255,0,0), 2)
    #cv2.imshow("mask", out)
    return out
"""

#given a predetected card contour, extracts the image of the card in a 125x175 BGR image
def isolate(cnt, img):
    #https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    epsilon = 0.05*cv2.arcLength(cnt,True)
    approx = np.float32(cv2.approxPolyDP(cnt,epsilon,True))

    rect = np.zeros((4,2), dtype = "float32")
    s = np.sum(approx, axis=2)
    d = np.diff(approx, axis=2)
    #print("start", approx, "----------\n", s, "-----------\n", d, "end")

    rect[0] = approx[np.argmin(s)]
    rect[2] = approx[np.argmax(s)]
    rect[1] = approx[np.argmin(d)]
    rect[3] = approx[np.argmax(d)]
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    if maxWidth > maxHeight:
        rect = np.roll(rect, 2)
        maxWidth, maxHeight = maxHeight, maxWidth

    dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    width = 125 #125
    height = int(width*7/5)
    warped = cv2.resize(warped, (width,height))
    return warped


#takes a 125x175 gray image of card and returns neural net prediciton of card's rank/value
def getValue(img):
    num = img[0:40,0:35]
    gray = cv2.cvtColor(num, cv2.COLOR_BGR2GRAY)
    predictions = model.predict(np.array([gray]))
    return np.argmax(predictions[0])+1


# OPTIMIZE: Currenting calculating things twice with "check if card like" and with drawing
# find away to not do that
minArea = 20000 #specific to my camera setup
def getCards(img):
    thresh = threshold(img)
    cnts = getContours(thresh)
    #cv2.drawContours(img, cnts_sort, -1, (255,0,0), 3)

    card_cnts = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        epsilon = 0.1*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        (_,_), (width, height), angle = cv2.minAreaRect(cnt)
        try:
            aspectRatio = max(width, height) / min(width, height)
        except:
            aspectRatio = 0.0
        if area > minArea and 1.35 < aspectRatio < 1.45 and len(approx) == 4:
            card_cnts.append(cnt)

    #cv2.drawContours(img, card_cnts, -1, (255,255,255), 2)

    card_imgs = []
    card_value = []
    for i, card in enumerate(card_cnts):
        card_img = isolate(card, img)
        card_imgs.append(card_img)
        card_value.append(getValue(card_img))
        #cv2.imshow("card {}".format(i), card_img)


    for i, card in enumerate(card_cnts):
        rectangles(card, img)
        corners(card, img, format(card_value[i]))

    return card_imgs, card_value


#takes in network camera stream (raspberry pi right now) and shows feed with card shape and value overlay
def main():
    img_url = "http://192.168.1.2:8765/picture/1/current/"
    #img_url = "http://192.168.1.2:8081/"
    ret = True
    while ret:
        start = time.time()
        cap = cv2.VideoCapture(img_url)
        ret,img = cap.read()


        card_imgs,_ = getCards(img)
        cv2.imshow("image", img)

        print("FPS: {}".format(round(1/(time.time()-start), 1)))

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        """
        elif key == ord('w'): #for saving images that were used in neural net model for detecting card rank / value
            filepath = "images\\{}_{}.jpg".format("king", time.time())
            num = card_imgs[0][0:40,0:35]
            cv2.imwrite(filepath, num)
            print("Saved")
        """


    cap.release()
    cv2.destroyAllWindows()
    print("Done")


if __name__ == "__main__":
    execdir = "cardReader\\{}"
    model = tf.keras.models.load_model(execdir.format("cardID.model"))
    main()
