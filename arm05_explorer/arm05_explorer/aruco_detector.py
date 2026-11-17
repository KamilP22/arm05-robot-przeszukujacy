import cv2
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs
 
from arm05_interfaces.msg import CubeDetection
 
 
class ArucoDetector(Node):
 
    def __init__(self):
        super().__init__('aruco_detector')
 
     
        self.declare_parameter('marker_size', 0.07)
        self.declare_parameter('cube_size', 0.09)
        self.declare_parameter('camera_frame', 'camera_rgb_optical_frame')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('max_detection_range', 3.0)
        self.declare_parameter('min_marker_perimeter_px', 60.0)
        self.declare_parameter('max_z_error', 0.10)
        self.declare_parameter('process_every_n', 1)
        self.declare_parameter('debug_image_scale', 0.5)
 
        gp = self.get_parameter
        self.marker_size = float(gp('marker_size').value)
        self.cube_size = float(gp('cube_size').value)
        self.camera_frame = gp('camera_frame').value
        self.map_frame = gp('map_frame').value
        self.max_range = float(gp('max_detection_range').value)
        self.min_perimeter = float(gp('min_marker_perimeter_px').value)
        self.max_z_error = float(gp('max_z_error').value)
        self.every_n = max(1, int(gp('process_every_n').value))
        self.debug_scale = float(gp('debug_image_scale').value)
 
        self.K = None
        self.D = None

        self.bridge = CvBridge()
  
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
 

        s = self.marker_size / 2.0
        self.obj_pts = np.array([[-s, s, 0.0],
                                 [s, s, 0.0],
                                 [s, -s, 0.0],
                                 [-s, -s, 0.0]], dtype=np.float32)

        self.center_offset = np.array([0.0, 0.0, -self.cube_size / 2.0])

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)

        self.create_subscription(CameraInfo, gp('camera_info_topic').value,
                                 self.camera_info_callback, 10)
        self.create_subscription(Image, gp('image_topic').value, self.image_callback, 1)
        self.obs_pub = self.create_publisher(CubeDetection, 'cube_observations', 10)
        self.debug_pub = self.create_publisher(Image, 'aruco_debug_image', 1)
 
        self.frame_counter = 0
        self.get_logger().info(
            f'aruco_detector gotowy (marker {self.marker_size} m, kamera {self.camera_frame})')
 
    def camera_info_callback(self, msg):
        if self.K is not None:
            return  # wystarczy raz
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.D = np.array(msg.d, dtype=np.float64)
        self.get_logger().info(
            f'Mam parametry kamery: fx={self.K[0, 0]:.1f} fy={self.K[1, 1]:.1f} '
            f'cx={self.K[0, 2]:.1f} cy={self.K[1, 2]:.1f}')

    def image_callback(self, msg):
        if self.K is None:
            return
        self.frame_counter += 1
        if self.frame_counter % self.every_n != 0:
            return
 
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict,
                                                  parameters=self.aruco_params)
 
        want_debug = self.debug_pub.get_subscription_count() > 0
        if ids is None:
            if want_debug:
                self.publish_debug(img, msg.header)
            return
 
        if want_debug:
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
 
        for corner, marker_id in zip(corners, ids.flatten()):
            pts2d = corner.reshape(4, 2).astype(np.float32)

            if cv2.arcLength(pts2d, True) < self.min_perimeter:
                continue
 
            ok, rvec, tvec = cv2.solvePnP(self.obj_pts, pts2d, self.K, self.D,
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            tvec = tvec.reshape(3)
            distance = float(np.linalg.norm(tvec))
            if distance > self.max_range:
                continue
 
            R, _ = cv2.Rodrigues(rvec)
            center_cam = tvec + R @ self.center_offset
 
            p = PointStamped()
            p.header.frame_id = self.camera_frame
            p.header.stamp = msg.header.stamp
            p.point.x = float(center_cam[0])
            p.point.y = float(center_cam[1])
            p.point.z = float(center_cam[2])
            try:
                p_map = self.tf_buffer.transform(p, self.map_frame,
                                                 timeout=Duration(seconds=0.2))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                self.get_logger().warning(f'TF: {e}', throttle_duration_sec=2.0)
                continue
 
            if abs(p_map.point.z - self.cube_size / 2.0) > self.max_z_error:
                self.get_logger().warning(
                    f'ID {marker_id}: odrzucono (z={p_map.point.z:.3f})',
                    throttle_duration_sec=2.0)
                continue
 
            det = CubeDetection()
            det.id = int(marker_id)
            det.position = p_map.point
            det.num_observations = 1
            self.obs_pub.publish(det)
 
            self.get_logger().info(
                f'ID {det.id}: x={p_map.point.x:.2f} y={p_map.point.y:.2f} '
                f'z={p_map.point.z:.3f} dist={distance:.2f} m',
                throttle_duration_sec=1.0)
 
            if want_debug:
                cv2.drawFrameAxes(img, self.K, self.D, rvec, tvec.reshape(3, 1),
                                  self.marker_size / 2.0)
 
        if want_debug:
            self.publish_debug(img, msg.header)

    def publish_debug(self, img, header):
        if self.debug_scale != 1.0:
            img = cv2.resize(img, None, fx=self.debug_scale, fy=self.debug_scale)
        out = self.bridge.cv2_to_imgmsg(img, 'bgr8')
        out.header = header
        self.debug_pub.publish(out)
 
 
def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
 
 
if __name__ == '__main__':
    main()