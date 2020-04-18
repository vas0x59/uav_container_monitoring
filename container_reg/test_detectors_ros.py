import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
cv = cv2
import string
import numpy as np 
from Detectors.YoloOpencvDetector import YoloOpencvDetetor
#from Detectors.YoloDarknetDetector import YoloDarknetDetector
from Detectors import Utils 
import pytesseract
import paho.mqtt.client as mqtt
import json 
import time
cont_ps = json.load(open("conts.json", "r"))
places_ps = json.load(open("places.json", "r"))

client = mqtt.Client()

# client.on_connect = on_connect
# client.on_message = on_message
# client.username_pw_set("gkxvfbbm", password="FVe4iUZQ8EHR")
# client.connect("m13.cloudmqtt.com", 18824, 60)


# detector = YoloOpencvDetetor("./Detectors/YOLO/yolov3.cfg", "./Detectors/YOLO/yolov3_320.weights")
# detector = YoloOpencvDetetor("./Detectors/YOLO/yolov3.cfg", "./Detectors/YOLO/yolov3.weights")
detector = YoloOpencvDetetor("/home/vasily/Projects/yolo_training/container/yolov3_cfg.cfg", "/home/vasily/Projects/yolo_training/container/backup/yolov3_cfg_10000.weights", CLASSESPath="/home/vasily/Documents/classes.txt")
# detector = YoloOpencvDetetor("./Detectors/YOLO/yolov2-voc.cfg", "./Detectors/YOLO/yolov2-voc.weights")
# detector = YoloOpencvDetetor("./Detectors/YOLO/yolov2-tiny.cfg", "./Detectors/YOLO/yolov2-tiny.weights")
# cap = cv2.VideoCapture("/home/vasily/Downloads/DJI_0002.MP4")
# cap = cv2.VideoCapture(2)
# out = cv2.VideoWriter()
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# cap.set(cv.CAP_PROP_FRAME_WIDTH, int(1920))
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, int(1200))
# out = cv2.VideoWriter('output_3.avi',fourcc, 20.0, (1920,1080))
frame_i = 0

def clip(a):
    if a <0:
        a = 0
    return a
bridge = CvBridge()

coords_now = [0, 0, 0]

def on_message(client, userdata, msg):
    global coords_now
    # print(msg.topic)
    if msg.topic == "/coords":
        s = msg.payload.decode("utf-8")
        # print("sssssssssssssssss", s)
        if s.split() != ["nan", "nan", "nan"]:
            coords_now = list(map(float,  s.split()))
        # print(";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;")
        # print("coooor", coords_now)
client.on_message = on_message
def aaa(client, userdata, flags, rc):
    client.subscribe("/coords")
client.on_connect = aaa
client.connect("localhost", 1883)

def get_dist(p1, p2):
    # print(((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5)
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def get_place_by_coords(ps):
    global cont_ps, places_ps
    # places_ps_list = list(dict(places_ps).items())
    # sorted(conts_list, key=lambda p: 
    # print(dict(places_ps).items())
    print(list((dict(places_ps).items())))
    place =  min(list(dict(places_ps).items()), key=lambda p: get_dist(ps, p[1]["xyz"]))[0]
    print(ps, "place", place)
    return place

def get_place_by_cont(cont_id):
    global cont_ps, places_ps
    conts_list = dict(cont_ps)
    # containers
    # sorted(conts_list, key=lambda p: 
    # container =  min(conts_list, key=lambda p: get_dist([x, y, z], p["xyz"]))
    if cont_id in  conts_list.keys():
        return conts_list[cont_id]["place_id"]
    else:
        return "-1"
def get_cont_by_place(place):
    global cont_ps, places_ps
    conts_list = dict(cont_ps)
    # containers
    # sorted(conts_list, key=lambda p: 
    # container =  min(conts_list, key=lambda p: get_dist([x, y, z], p["xyz"]))
    if place in  places_ps.keys():
        return places_ps[place]["cont"]
    else:
        return "-1"
def message_to_server(place_now, place_of_cont, cont_id, cont2):
    mes = {
        "time":time.time(),
        "cont_id":cont_id,
        "place1":place_now,
        "place2":place_of_cont,
        "cont2":cont2
    }
    return json.dumps(mes)

def callback(mes):
    global frame_i, coords_now
    print("Asd")
    frame = bridge.imgmsg_to_cv2(mes, "bgr8")
    frame = cv2.flip(cv2.flip(frame, 0), 1)
    # ret, frame = cap.read()
    
    frame_tonet = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    #boxes, classIDs, confidences = detector.detect(frame, s=(320, 320))
    # boxes, classIDs, confidences = detector.detect(frame, s=(320, 320))
    boxes, classIDs, confidences = detector.detect(frame_tonet, s=(416, 416))
    # boxes, classIDs, confidences = detector.detect(frame, s=(608, 608))
    # boxes, classIDs, confidences = detector.detect(frame, s=(700, 700))
    # print(detector.COLORS)= 2
    boxes = [[j * 2 for j in i]for i in boxes]
    #
    # for i in range(len(boxes)):
    # box = boxes[i]
    id_con = "-1"
    if len(boxes) > 0:
        box = min(boxes, key=lambda p: ((frame.shape[1]//2 - p[0])**2 + (frame.shape[0]//2 - p[1])**2)**0.5)
        box_i = boxes.index(box)
        print(box)
        croped = frame[clip(box[1]):clip(box[1] + box[3]), clip(box[0]):clip(box[0] + box[2])]
        croped = croped[0:croped.shape[0]//3, croped.shape[1]//2:]
        croped = cv2.cvtColor(croped, cv2.COLOR_BGR2GRAY)
        croped = cv2.bitwise_not(croped)
        _,croped = cv.threshold(croped,127,255,cv.THRESH_TRUNC)
        # cv2.imshow("1croped" + str(i), croped)
        croped = cv2.adaptiveThreshold(croped,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY,11,2)
        # croped = cv.morphologyEx(croped, cv.MORPH_OPEN, np.ones((3,3),np.uint8))
        # croped = cv.erode(croped,np.ones((3,3),np.uint8),iterations = 1)
        res = pytesseract.image_to_string(croped)
        res2 = "".join([i for i in res if i in string.digits or i == " " or i == "\n"])
        res2 = res2.split()
        id_con = "-1"
        res2 = [i for i in res2 if len(i) == 6 or len(i) == 7]
        if len(res2) > 0:
            id_con = res2[0][:6]
        print(id_con)
        cv2.imshow("croped", croped)
        boxes = [box]
        confidences = [id_con]
    # confidences[i] = id_con
    

    place_now = get_place_by_coords(coords_now)
    place_of_cont = get_place_by_cont(id_con)
    print(place_now, place_of_cont)
    boxes = [[j // 2 for j in i]for i in boxes]

    img_out = Utils.draw_boxes(frame_tonet, boxes, classIDs, confidences, detector.CLASSES, COLORS=[[0, 200, 255]])
    # out.write(frame)
    cv2.imshow("img_out", cv2.resize(img_out, (0, 0), fx=1.2, fy=1.2))
    cv2.waitKey(1)
    if id_con != "-1":
        mes_s = message_to_server(place_now, place_of_cont, id_con, get_cont_by_place(place_now))

        print(mes_s)
        client.publish("/conts/" + id_con, mes_s)
        client.publish("/places/" + place_now, mes_s)
        # client.publish("/conts/" + id_con, mes_s)
    # frame_i +=1 
    print("Asd2")
# cv2.namedWindow("img_out", 1)
rospy.init_node('image_conasdsverter', anonymous=True)
image_sub = rospy.Subscriber("/cv_camera/image_raw",Image,callback)
# rospy.Subscriber("telem", String, callback2)
client.loop_start()
while not rospy.is_shutdown():
    rospy.spin()
client.loop_stop(force=False)
# cap.release()
# out.release()
