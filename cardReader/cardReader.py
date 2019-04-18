import numpy as np
import cv2
import time
import tensorflow as tf
execdir = None
model = None
img_url = None


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
    #return empty list if there are no contours
    if len(contours) == 0:
        return []
    #sort contours by area
    index_sort = sorted(range(len(contours)), key=lambda i : cv2.contourArea(contours[i]),reverse=True)
    cnts_sort = []
    for i in index_sort:
        cnts_sort.append(contours[i])

    return contours


#given contour draw rotated rectangle and (currently commented out) straight bounding rectangle
def rectangles(cnt, img):
    #draws rotated rectangle
    rect = cv2.minAreaRect(cnt)
    box = np.int0(cv2.boxPoints(rect))
    cv2.drawContours(img, [box], -1, (255,0,0), 2)
    #draws straight bouding rectangle based off of rotated rectangle
    """
    x1 = min(point[0] for point in box)
    y1 = min(point[1] for point in box)
    x2 = max(point[0] for point in box)
    y2 = max(point[1] for point in box)
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0),2)
    """


#given contour, draw points on the corners
def corners(cnt, img, text=""):
    #draws dots on the corners of the contour
    epsilon = 0.1*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    cv2.drawContours(img, approx, -1, (0,0,255), 8)
    #when passed text, draws text at the top most corner
    if text:
        cv2.putText(img, text, (approx[0][0][0], approx[0][0][1]+15), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)


#given a predetected card contour, extracts the image of the card in a 125x175 BGR image
def isolate(cnt, img):
    #https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    #calculates what the height of width should be from the rotated rectangle
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

    #orients the card vertically
    if maxWidth > maxHeight:
        rect = np.roll(rect, 2)
        maxWidth, maxHeight = maxHeight, maxWidth

    dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

    #transforms the contour image into a top down view
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    #resize to playing card aspect ratio ~ 7/5 or 1.4
    width = 125 #125
    height = int(width*7/5)
    warped = cv2.resize(warped, (width,height))
    return warped


#takes a 125x175 gray image of card and returns neural net prediciton of card's rank/value
def getValue(img):
    #[0:40,0:35] isolates the number on the card based off of 125x175 input image
    num = img[0:40,0:35]
    gray = cv2.cvtColor(num, cv2.COLOR_BGR2GRAY)
    predictions = model.predict(np.array([gray]))
    return np.argmax(predictions[0])+1


# OPTIMIZE: Currenting calculating things twice with "check if card like" and with drawing
# find away to not do that
minArea = 20000 #specific to my camera setup
def getCards(img):
    #threshhold image and return all contours sorted by area
    thresh = threshold(img)
    cnts = getContours(thresh)
    #cv2.drawContours(img, cnts_sort, -1, (255,0,0), 3)

    #tests all contours and checks if they have "card-like" aspects and adds those to card contours list
    card_cnts = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        epsilon = 0.1*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        (_,_), (width, height), angle = cv2.minAreaRect(cnt)
        try: #try except in case of division by zero error
            aspectRatio = max(width, height) / min(width, height)
        except:
            aspectRatio = 0.0
        #if meets minimum area, meets expected dimension aspect ratios, and has four corners
        if area > minArea and 1.3 < aspectRatio < 1.5 and len(approx) == 4:
            card_cnts.append(cnt)

    #cv2.drawContours(img, card_cnts, -1, (255,255,255), 2)

    #for each card contour, do something
    card_imgs = []
    card_value = []
    for i, card in enumerate(card_cnts):
        #extract image and rank / value of the card
        card_img = isolate(card, img)
        card_imgs.append(card_img)
        card_value.append(getValue(card_img))
        #cv2.imshow("card {}".format(i), card_img)
    #again for each card... (seperate for loop so previous loop doesn't have overlay drawn on it)
    for i, card in enumerate(card_cnts):
        #draw rectangles around the card
        rectangles(card, img)
        #draw corners and label the card with value
        corners(card, img, format(card_value[i]))

    return card_imgs, card_value


#method for writing white text on the image
def imgText(img, text, pos):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)


class contourProperties:
    def __init__(self, cnt, src):
        self.cnt = cnt
        self.src = src

    def getRectangle(self):
        if not self.rect:
            rect = cv2.minAreaRect(self.cnt)
            box = np.int0(cv2.boxPoints(rect))
            self.rect = box
        return self.rect

    def corners(self):
        if not self.corners:
            epsilon = 0.1*cv2.arcLength(self.cnt,True)
            approx = cv2.approxPolyDP(self.cnt,epsilon,True)
            sef.corners = approx
        return self.corners

    def isolateCard(self):
        if not self.img:
            #https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
            #calculates what the height of width should be from the rotated rectangle
            approx = np.float32(self.corners())

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

            #orients the card vertically
            if maxWidth > maxHeight:
                rect = np.roll(rect, 2)
                maxWidth, maxHeight = maxHeight, maxWidth

            dst = np.array([
        		[0, 0],
        		[maxWidth - 1, 0],
        		[maxWidth - 1, maxHeight - 1],
        		[0, maxHeight - 1]], dtype = "float32")

            #transforms the contour image into a top down view
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(self.img, M, (maxWidth, maxHeight))

            #resize to playing card aspect ratio ~ 7/5 or 1.4
            width = 125 #125
            height = int(width*7/5)
            self.img = cv2.resize(warped, (width,height))
        return self.img

    def getValue(self):
        if not self.value:
            #[0:40,0:35] isolates the number on the card based off of 125x175 input image
            num = img[0:40,0:35]
            gray = cv2.cvtColor(num, cv2.COLOR_BGR2GRAY)
            predictions = model.predict(np.array([gray]))
            self.value = np.argmax(predictions[0])+1
        return self.value





#takes in network camera stream (raspberry pi with motioneye right now) and shows feed with card shape and value overlay
def main():
    #img_url = "http://192.168.1.2:8765/picture/1/current/"

    ret = True
    while ret:
        start = time.time()
        #pulls image from url (could be video camera too)
        cap = cv2.VideoCapture(img_url)
        ret,img = cap.read()

        #get a list of extracted card images and their respective values
        #also adds overlay, possibly need to seperate into different function
        card_imgs,card_values = getCards(img)

        #prints (on image) interations of the for loop per second, calculated by 1/(time per interation)
        imgText(img, "FPS: {}".format(round(1/(time.time()-start))), (30, img.shape[0] - 10))
        imgText(img, "Hand: {}".format(np.sum(card_values)), (30, img.shape[0] - 40))

        cv2.imshow("Camera Image", img)

        #press q to close windows and program
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        #for saving images that were used in neural net model for detecting card rank / value
        """
        elif key == ord('w'):
            filepath = "images\\{}_{}.jpg".format("king", time.time())
            num = card_imgs[0][0:40,0:35]
            cv2.imwrite(filepath, num)
            print("Saved")
        """
    #releases cap in case of camera capture and close window
    cap.release()
    cv2.destroyAllWindows()
    print("Done")

#static variables are declared here and instantiated at the top of the program so functions can reach them
#this is likely the wrong way to do it, probably should pass as arguments but its probably fine right now
if __name__ == "__main__":
    img_url = "http://192.168.1.2:8765/picture/1/current/"
    execdir = "cardReader\\{}"
    model = tf.keras.models.load_model(execdir.format("cardID.model"))
    main()
