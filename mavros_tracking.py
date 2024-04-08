import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from std_msgs.msg import String

class DroneController:
    def __init__(self):
        self.current_state = State()
        self.pose = PoseStamped()
        rospy.init_node("tracking")
        rospy.Subscriber("/uav0/mavros/state", State, self.state_cb)
        # detection_info 구독자 추가
        rospy.Subscriber("detection_info", String, self.detection_cb)
        self.local_pos_pub = rospy.Publisher("/uav0/mavros/setpoint_position/local", PoseStamped, queue_size=10)
        self.arming_client = rospy.ServiceProxy("/uav0/mavros/cmd/arming", CommandBool)
        self.set_mode_client = rospy.ServiceProxy("/uav0/mavros/set_mode", SetMode)
        self.rate = rospy.Rate(20)  # Setpoint publishing MUST be faster than 2Hz

    def state_cb(self, msg):
        self.current_state = msg
    def detection_cb(self, msg):
        self.detection_info = msg.data
    def connect_and_takeoff(self):
        # Wait for Flight Controller connection
        while not rospy.is_shutdown() and not self.current_state.connected:
            self.rate.sleep()

        self.pose.pose.position.x = 0
        self.pose.pose.position.y = 0
        self.pose.pose.position.z = 10
        rospy.sleep(3)
        # Send a few setpoints before starting
        for i in range(100):
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
                    rospy.loginfo("OFFBOARD enabled")

                last_req = rospy.Time.now()
            else:
                if not self.current_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(5.0):
                    if self.arming_client.call(arm_cmd).success:
                        rospy.loginfo("Vehicle armed")

                    last_req = rospy.Time.now()

            self.local_pos_pub.publish(self.pose)
            self.rate.sleep()
            
            if self.current_state.armed:  # 'drone.armed' 대신 'self.current_state.armed' 사용
                tracking(self)

def tracking(self):
    # 추적 정보 업데이트
    cx, cy, w = map(float, self.detection_info.split())
    
    # 좌/우(x), 높이(y), 직진/후진(z) 제어 로직 조정
    y_pose = 0 #-0.03 if cx < 30 else (0.03 if cx > 200 else 0)
    z_pose = 0 # 0.03 if cy < 30 else (-0.03 if cy > 200 else 0)
    x_pose = 0.05 if w < 20 else (-0.05 if 100 > w > 30 else 0)
    
    # 추적 제어 적용
    self.pose.pose.position.x += x_pose
    self.pose.pose.position.y += y_pose
    self.pose.pose.position.z += z_pose
    
    # 로그 출력으로 추적 제어 확인
    rospy.loginfo("추적 제어: x={}, y={}, z={}".format(x_pose, y_pose, z_pose))
    
    # 더 짧은 대기 시간
    rospy.sleep(0.1)  # 추적 반응성 향상을 위해 대기 시간 축소
            

if __name__ == "__main__":
    drone = DroneController()
    drone.connect_and_takeoff()

    
