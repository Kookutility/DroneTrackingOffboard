import cv2
import rospy
import numpy as np
import torch
import threading
import time
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from sensor_msgs.msg import Image
from std_msgs.msg import String
from pynput import keyboard
import tkinter as tk
from tkinter import ttk
from PIL import Image as PILImage
from PIL import ImageTk

class RealTimeObjectDetection(threading.Thread):
    def __init__(self):
        super(RealTimeObjectDetection, self).__init__()
        self.daemon = True
        self.average_bounding_box_size = 0
        self.bounding_box_size = 0
        self.initialized = False
        self.frame_count = 0
        self.start_time = time.time()
        self.bridge = CvBridge()
        weights = './drone_yolov5.pt'
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=False)
        self.tracker = cv2.TrackerGOTURN_create()
        self.subscriber = rospy.Subscriber("iris0/flow_camera/image_raw", Image, self.callback)
        self.detection_pub = rospy.Publisher("detection_info", String, queue_size=10)
        self.cv_image = None
        self.cv_image_resized = None
    def callback(self, data):
        object_position = [0, 0]
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        self.frame_count += 1
        if not self.initialized or self.frame_count % 30 == 0:
            results = self.model(self.cv_image)
            if len(results.xyxy[0]) > 0:
                bounding_box_sizes = []
                for i in range(len(results.xyxy[0])):
                    x1, y1, x2, y2 = results.xyxy[0][i][:4]
                    width = x2 - x1
                    height = y2 - y1
                    self.bounding_box_size = width * height
                    bounding_box_sizes.append(self.bounding_box_size)
                
                if bounding_box_sizes:
                    self.average_bounding_box_size = sum(bounding_box_sizes) / len(bounding_box_sizes)
                else:
                    self.average_bounding_box_size = 0

                min_x = results.xyxy[0][:, 0].min().item()
                min_y = results.xyxy[0][:, 1].min().item()
                max_x = results.xyxy[0][:, 2].max().item()
                max_y = results.xyxy[0][:, 3].max().item()
                init_bb = (min_x, min_y, max_x - min_x, max_y - min_y)
                self.tracker.init(self.cv_image, tuple(map(int, init_bb)))
                self.initialized = True
            else:
                self.initialized = False
        else:
            success, box = self.tracker.update(self.cv_image)
            if success:
                x, y, w, h = map(int, box)
                cv2.rectangle(self.cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
                object_position = (x + w / 2, y + h / 2)
                detection_info = f"{object_position[0]} {object_position[1]} {self.average_bounding_box_size}"
                self.detection_pub.publish(detection_info)
            else:
                self.initialized = False

        cv2.putText(self.cv_image, "x : {:.2f}".format(object_position[0]), (140, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 170, 50), 1)
        cv2.putText(self.cv_image, "y : {:.2f}".format(object_position[1]), (140, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 170, 50), 1)
        cv2.putText(self.cv_image, "avg size : {:.2f}".format(self.average_bounding_box_size), (120, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 170, 50), 1)
        self.cv_image_resized = cv2.resize(self.cv_image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    def get_frame(self):
            if self.cv_image_resized is not None:
                return cv2.cvtColor(self.cv_image_resized, cv2.COLOR_BGR2RGB)
            else:
                return np.zeros((480, 640, 3), dtype=np.uint8)  # 기본 검은색 배경

    def run(self):
        rospy.spin()

class DroneController:
    def __init__(self, uav_id):
        self.detection_info = ""
        self.uav_id = uav_id
        self.current_state = State()
        self.pose = PoseStamped()
        self.tracking_enabled = False

        rospy.Subscriber("detection_info", String, self.detection_cb)
        rospy.Subscriber(f"/{self.uav_id}/mavros/state", State, self.state_cb)
        self.local_pos_pub = rospy.Publisher(f"/{self.uav_id}/mavros/setpoint_position/local", PoseStamped, queue_size=10)
        self.arming_client = rospy.ServiceProxy(f"/{self.uav_id}/mavros/cmd/arming", CommandBool)
        self.set_mode_client = rospy.ServiceProxy(f"/{self.uav_id}/mavros/set_mode", SetMode)
        self.rate = rospy.Rate(20)

        keyboard_thread = threading.Thread(target=self.keyboard_listener)
        keyboard_thread.daemon = True
        keyboard_thread.start()

    def state_cb(self, msg):
        self.current_state = msg

    def detection_cb(self, msg):
        self.detection_info = msg.data

    def on_press(self, key):
        try:
            if key.char == 't':
                self.tracking_enabled = not self.tracking_enabled
                rospy.loginfo(f"Tracking mode: {'Enabled' if self.tracking_enabled else 'Disabled'}")
        except AttributeError:
            pass

    def keyboard_listener(self):
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()
        listener.join()

    def connect_and_takeoff(self, altitude):
        while not rospy.is_shutdown() and not self.current_state.connected:
            self.rate.sleep()

        self.pose.pose.position.z = altitude
        for _ in range(100):
            if rospy.is_shutdown():
                break
            self.local_pos_pub.publish(self.pose)
            self.rate.sleep()

        offb_set_mode = SetModeRequest()
        offb_set_mode.custom_mode = 'OFFBOARD'
        arm_cmd = CommandBoolRequest()
        arm_cmd.value = True
        last_req = rospy.Time.now()

        while not rospy.is_shutdown():
            if self.current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(5.0):
                if self.set_mode_client.call(offb_set_mode).mode_sent:
                    rospy.loginfo(f"{self.uav_id} OFFBOARD enabled")
                last_req = rospy.Time.now()
            elif not self.current_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(5.0):
                if self.arming_client.call(arm_cmd).success:
                    rospy.loginfo(f"{self.uav_id} Vehicle armed")
                last_req = rospy.Time.now()

            self.local_pos_pub.publish(self.pose)
            self.rate.sleep()
            if self.tracking_enabled:
                self.tracking()

    def tracking(self):
        try:
            cx, cy, s = map(float, self.detection_info.split())

            y_pose = -0.07 if cx > 200 else (0.07 if 130 < cx else 0) 
            z_pose = 0 if s > 400 else (0 if 100 < s else 0)
            x_pose = -0.1 if cy > 200 else (0 if 70 < cy else 0.1)

            self.pose.pose.position.x += x_pose
            self.pose.pose.position.y += y_pose
            self.pose.pose.position.z += z_pose

            rospy.loginfo("추적 제어: x좌표 이동={}, y좌표 이동 ={}, 고도 (z좌표) 상승/하강={}".format(x_pose, y_pose, z_pose))
            
            rospy.sleep(0.1)
        except ValueError:
            rospy.loginfo("Invalid detection info received")

    def move_forward(self, distance):
        self.pose.pose.position.x += distance
        rospy.loginfo(f"{self.uav_id} 이동: {distance}m 앞으로")

class FollowerController(DroneController):
    def __init__(self, uav_id, leader_id):
        super().__init__(uav_id)
        self.leader_id = leader_id
        self.tracking_enabled = False
        self.follow_enabled = True  # 리더를 계속 따라갈지 여부 결정

        rospy.Subscriber(f"/{self.leader_id}/mavros/local_position/pose", PoseStamped, self.follow_leader)

    def follow_leader(self, msg):
        if self.follow_enabled:
            self.pose.pose.position.x = msg.pose.position.x
            self.pose.pose.position.y = msg.pose.position.y
            self.pose.pose.position.z = msg.pose.position.z - 10

    def update_pose(self):
        while not rospy.is_shutdown():
            self.local_pos_pub.publish(self.pose)
            self.rate.sleep()

    def disable_following(self):
        self.follow_enabled = False
        rospy.loginfo(f"{self.uav_id} 자폭 드론 출격, 더 이상 리더를 따라가지 않습니다.")

class DroneControlInterface:
    def __init__(self, root, rtod, uav0, uav1):
        self.root = root
        self.uav0 = uav0
        self.uav1 = uav1
        self.rtod = rtod

        self.root.title("Drone Control Interface")
        self.root.geometry("1200x800")

        self.video_frame = ttk.Label(self.root)
        self.video_frame.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

        ttk.Button(self.root, text="추적 시작", command=self.start_tracking).grid(row=1, column=0, padx=10, pady=10)
        ttk.Button(self.root, text="추적 중지", command=self.stop_tracking).grid(row=1, column=1, padx=10, pady=10)
        ttk.Button(self.root, text="자폭 드론 출격", command=self.deploy_kamikaze_drone).grid(row=1, column=2, padx=10, pady=10)
        ttk.Button(self.root, text="재밍 실시", command=self.jamming).grid(row=1, column=3, padx=10, pady=10)
        ttk.Button(self.root, text="리더 드론 복귀", command=self.return_to_home).grid(row=1, column=4, padx=10, pady=10)

        self.update_video()

    def update_video(self):
        frame = self.rtod.get_frame()
        img = PILImage.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_frame.imgtk = imgtk
        self.video_frame.configure(image=imgtk)
        self.root.after(10, self.update_video)

    def start_tracking(self):
        self.uav0.tracking_enabled = True
        rospy.loginfo("추적 시작")

    def stop_tracking(self):
        self.uav0.tracking_enabled = False
        rospy.loginfo("추적 중지")

    def jamming(self):
        rospy.loginfo("재밍 실시")

    def return_to_home(self):
        self.uav0.pose.pose.position.x = 0
        self.uav0.pose.pose.position.y = 0
        self.uav0.pose.pose.position.z = 20
        rospy.loginfo("리더 드론 복귀")

    def deploy_kamikaze_drone(self):
        self.uav1.disable_following()
        self.uav1.move_forward(10)
        rospy.loginfo("자폭 드론 출격: 10m 앞으로 이동")

if __name__ == "__main__":
    rospy.init_node('multi_drone_control', anonymous=True)

    rtod = RealTimeObjectDetection()
    uav0 = DroneController("uav0")
    uav1 = FollowerController("uav1", "uav0")

    def run_drone(drone, altitude):
        drone.connect_and_takeoff(altitude)
        if drone.tracking_enabled:
            while not rospy.is_shutdown():
                drone.tracking()
        else:
            drone.update_pose()

    t0 = threading.Thread(target=run_drone, args=(uav0, 20))
    t1 = threading.Thread(target=run_drone, args=(uav1, 10))
    t0.start()
    t1.start()

    root = tk.Tk()
    interface = DroneControlInterface(root, rtod, uav0, uav1)

    rtod.start()
    root.mainloop()

    t0.join()
    t1.join()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
