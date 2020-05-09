#!/usr/bin/env python3
import os
import sys
import rospy
import cv2
import numpy as np
import threading
import time
from sensor_msgs.msg import Image
from std_msgs.msg import String, Int8, Header
from mrcnn.config import Config
from mrcnn import model as modellib, utils


class InferenceConfig(Config):
    NAME = "SEGMENT"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 2 + 1
    DETECTION_MIN_CONFIDENCE = 0.70

class MaskRCNNNode(object):
    def __init__(self):
        ### initialize maskrcnn parameters and model file
        os.chdir("/home/siasun/Desktop/RobGrab/src/object_segment/scripts")
        config = InferenceConfig()
        self.model = modellib.MaskRCNN(mode="inference", config=config, model_dir="")
        self.model.load_weights("./yuanpan_model.h5", by_name=True)

        ### ros initial and message subscribe
        self.sub1 = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)
        self.sub2 = rospy.Subscriber('/command/recog_command', Int8, self.command)
        self.sub3 = rospy.Subscriber('/camera/depth_registered/sw_registered/image_rect_raw', Image, self.depth_callback)
        self.sub4 = rospy.Subscriber('/TCPServer/Ready', String, self.ProcessStart)
        self.sub5 = rospy.Subscriber('/PointCloudError', Int8, self.return_info)
        self.binary_pub = rospy.Publisher('/segment/segment_image', Image, queue_size=1)
        self.segment_pub = rospy.Publisher('/segment/classify_image', Image, queue_size=1)
        self.cls_pub = rospy.Publisher('/segment/segment_class', String, queue_size=10)
        self.img_msg = Image()
        self.seg_msg = Image()

        ### multi-thread
        self.cmd_lock = threading.Lock()
        self.msg_lock = threading.Lock()
        self.proAgain_lock = threading.Lock()
        self.depth_lock = threading.Lock()
        self.start_process = True
        self.img_rgb = np.zeros((480, 640, 3), np.uint8)
        self.img_depth = np.zeros((480, 640, 1), np.uint16)
        self.str_class = ""
        self.min_label = -1

        ### control the processing
        self.proc_again = False
        self.test = True
        self.firstInit = True
        self.result = []
        self.get_out = []

    def pickBest(self, result, depth, get_out):
        self.min_label = -1
        cts = result['rois'].shape[0]
        min_distance = 99999
        for x in range(cts):
            if x in get_out:
                continue

            single_mask = result['masks'][:, :, x]
            i, j = np.nonzero(single_mask)
            sum = 0.0
            counts = 0
            for k in range(len(i)):
                if depth[i[k]][j[k]] > 0:
                    counts = counts+1
                    sum += depth[i[k]][j[k]] * 1.0 / 1000

            # 当前物品一定发生了什么问题，最有可能的情况是误识别的背景，然后没有深度数据
            if counts == 0:
                get_out.append(x)
                continue

            distance = sum * 1.0 / counts
            if (distance < min_distance):
                min_distance = distance
                self.min_label = x
        
    def run(self):
        while not rospy.is_shutdown():
            ### wait for processing signal
            flag = False
            if self.cmd_lock.acquire(False):
                flag = self.start_process
                self.start_process = False
                self.cmd_lock.release()

            if flag:
                print("jinru once!")
                ##########################################################
                ### Directly process the image 
                ##########################################################
                if self.proc_again == False:
                                        
                    ### copy global image to local, get image at the third time
                    for x in range(3):
                        self.msg_lock.acquire(True)
                        img_local = self.img_rgb.copy()
                        self.msg_lock.release()
                        self.depth_lock.acquire(True)
                        depth = self.img_depth.copy()
                        self.depth_lock.release()

                    ### Display depth image
                    #maxVal = np.max(depth)
                    #depth = depth / maxVal * 256
                    #cv2.imshow("depth", depth)
                    #cv2.waitKey(100)
                                    
                    ### segment the image
                    #img_local = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    t1 = cv2.getTickCount()
                    self.result = self.model.detect([img_local])[0]                    
                    t2 = cv2.getTickCount()
                    print("image segment costs about ", (t2-t1)/cv2.getTickFrequency(), " s")
                    ### send the best segmentation
                    counts = self.result['rois'].shape[0]
                    if counts > 0:
                        ### find the uppest object
                        self.get_out = []
                        for i in range(counts):
                            self.pickBest(self.result, self.img_depth, self.get_out)
                            if self.min_label == -1:    # 进入此分支只有识别在所有数据都不满足条件的状况下存在
                                print('warning.01: 当前没有适合抓取的目标或存在异常')
                                break
                            if self.result['scores'][self.min_label] < 0.75:
                                print('warning.02:距离最近的目标得分过低，更换抓取目标')
                                self.get_out.append(self.min_label)
                                continue
                        self.display(img_local)
                    else:
                        if self.firstInit == True:
                            self.firstInit = False
                            print("程序初始化完成!")
                        else:
                            print("warning.03:当前没有识别到目标物品!更换拍照示教点!")
                        if self.test == False:
                            print("send change!")
                            self.cls_pub.publish("change")
                        else:
                            print("not send")
                            self.test = False
                ###########################################################
                ### Pick result from last time
                ###########################################################
                else:   
                    self.proc_again = False
                    if len(self.result) == 0:
                        print("warning.04:未曾识别任何物品!")
                        continue
                        
                    total = self.result['rois'].shape[0]
                    self.get_out.append(self.min_label)    #将上一次pose_estimate认为不好的去除
                    bad_list = len(self.get_out)
                    print("已经pass过的工件：", bad_list)
                    if total > bad_list:   # 仍存在可选的工件
                        for i in range(total):
                            self.pickBest(self.result, self.img_depth, self.get_out)
                            if self.min_label == -1:    # 进入此分支只有识别在所有数据都不满足条件的状况下存在
                                print('warning.05:当前没有适合抓取的目标')
                                break
                            if self.result['scores'][self.min_label] < 0.75:
                                print('warning.06:距离最近的目标得分过低，更换抓取目标')
                                self.get_out.append(self.min_label)
                                continue
                            else:
                                break
                        self.display(img_local)
                    else:
                        print("warning.07:已没有可选的工件候选，更换机器人拍照点!")
                        if self.test == False:
                            self.cls_pub.publish("change")
                        else:
                            self.test = False                                              

    def display(self, img):      
        ### judge again
        if self.min_label >= 0:
            max_label = self.min_label
            
            ### get the name of result (max score object)
            if self.result['class_ids'][max_label] == 1:
                self.str_class = 'yp1'
            elif self.result['class_ids'][max_label] == 2:
                self.str_class = 'yp2'
            
            print ("class is ", self.str_class)
                
            # send the results of recognition
            if self.test == False:
                self.cls_pub.publish(self.str_class)  
                tmp_mask = self.result['masks'][:, :, max_label]    
                best_mask = tmp_mask.astype(np.uint8)
                best_mask[best_mask > 0] = 255  
                print(best_mask.shape)
                self.send_image(best_mask)
                #print("00000000000000000000000000")
            else:
                self.test = False
                #print("9999999999999999999999999")
            
            ## display the results
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            merged = cv2.merge([gray,gray,gray])
            display = merged
            
            if self.result['rois'].shape[0] > 0:
                a = self.result['rois'].shape[0]
                print (self.result['class_ids'])
                
                for x in range(a):
                    tmp_mask = self.result['masks'][:,:,x]
                    binary_mask = tmp_mask.astype(np.uint8)
                    binary_mask[binary_mask > 0] = 255
                    merged_mask = cv2.merge([binary_mask,binary_mask,binary_mask])
                    
                    if self.result['class_ids'][x] == 1:
                        str_disp = 'yp1' + '.' + str(round(self.result['scores'][x], 2))
                        img_blue = np.zeros([480, 640, 3], np.uint8)
                        img_blue[:, :, 0] = np.zeros([480, 640]) + 255
                        img_blue_plus_org = img_blue*0.5 + merged*0.5
                        display = np.where(merged_mask, img_blue_plus_org, display).astype(np.uint8)                                
                    elif self.result['class_ids'][x] == 2:
                        str_disp = 'yp2' + '.' + str(round(self.result['scores'][x], 2))
                        img_other = np.zeros([480, 640, 3], np.uint8)
                        img_other[:, :, 1] = np.zeros([480, 640]) + 255
                        #img_other[:, :, 2] = np.zeros([480, 640]) + 140
                        img_other_plus_org = img_other*0.5 + merged*0.5
                        display = np.where(merged_mask, img_other_plus_org, display).astype(np.uint8)                    

                for x in range(a):
                    if self.result['class_ids'][x] == 1:
                        str_disp = 'yp1' + '.' + str(round(self.result['scores'][x], 2))
                        cv2.rectangle(display, (self.result['rois'][x][1], self.result['rois'][x][0]), (self.result['rois'][x][3], self.result['rois'][x][2]), (255, 0, 0), 1)
                        cv2.putText(display, str_disp, (self.result['rois'][x][1], self.result['rois'][x][0]), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1)
                    elif self.result['class_ids'][x] == 2:
                        str_disp = 'yp2' + '.' + str(round(self.result['scores'][x], 2))
                        cv2.rectangle(display, (self.result['rois'][x][1], self.result['rois'][x][0]), (self.result['rois'][x][3], self.result['rois'][x][2]), (0, 255, 0), 1)
                        cv2.putText(display, str_disp, (self.result['rois'][x][1], self.result['rois'][x][0]), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1)                
                    
    
            #在rviz上显示语义分割结果
            self.send_segImg(display)
        else:
            print("warning.08:出现异常!")
            if self.test == False:
                self.cls_pub.publish("change")
            else:
                self.test = False            

    def return_info(self, cmd):
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        #self.cmd_lock.acquire(True)
        self.proc_again = True
        self.start_process = True
        #self.cmd_lock.release()

    def send_segImg(self, img):
        header = Header(stamp=rospy.Time.now())
        self.img_msg.header = header
        self.img_msg.height = img.shape[0]
        self.img_msg.width = img.shape[1]
        self.img_msg.encoding = "bgr8"
        self.img_msg.data = np.array(img).tostring()
        self.img_msg.step = 640*3 #len(self.img_msg.data)
        self.segment_pub.publish(self.img_msg)
                                    
    def send_image(self, img):
        header = Header(stamp=rospy.Time.now())
        self.img_msg.header = header
        self.img_msg.height = img.shape[0]
        self.img_msg.width = img.shape[1]
        self.img_msg.encoding = "mono8"
        self.img_msg.data = img.tostring()
        self.img_msg.step = 640 #len(self.img_msg.data)
        self.binary_pub.publish(self.img_msg)

    def ProcessStart(self, cmd):
        print("hhhhhhhhhhhhhhhhhhhhh")
        self.cmd_lock.acquire(True)
        self.start_process = True
        self.cmd_lock.release()

    def command(self, cmd):
        self.cmd_lock.acquire(True)
        if cmd.data == 1:
            self.start_process = True
        elif cmd.data == 2:
            self.proc_again = True
            self.start_process = True
        elif cmd.data == 0:
            self.test = True
            self.start_process = True            
        self.cmd_lock.release()

    def image_callback(self, data):
        image_raw = np.ndarray(shape=(data.height, data.width, 3), dtype=np.uint8, buffer=data.data)
        self.msg_lock.acquire(True)
        self.img_rgb = image_raw.copy()
        self.msg_lock.release()	
        
    def depth_callback(self, data):
        depth_raw = np.ndarray(shape=(data.height, data.width, 1), dtype=np.uint16, buffer=data.data)
        self.depth_lock.acquire(True)
        self.img_depth = depth_raw.copy()
        self.depth_lock.release()


def main():
    rospy.init_node('segment', anonymous=True)
    node = MaskRCNNNode()
    node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
            
            
            
        
        
        


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
