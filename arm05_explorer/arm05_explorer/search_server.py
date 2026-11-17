import math
import os
import threading
import time
 
import numpy as np
import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
 
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration as DurationMsg
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose, Spin
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
 
from arm05_interfaces.action import FindCubes
from arm05_interfaces.msg import CubeDetection
from arm05_explorer.coverage_planner import plan_viewpoints
from arm05_explorer.grid_utils import quaternion_from_yaw
 
 
class SearchServer(Node):
 
    def __init__(self):
        super().__init__('search_server')

        defaults = {
            'map_frame': 'map',
            'base_frame': 'base_footprint',
            'coarse_resolution': 0.2,
            'robot_clearance': 0.35,
            'target_wall_margin': 0.25,
            'sensor_range': 2.5,
            'sensor_min_range': 0.3,
            'candidate_step': 0.4,
            'coverage_target': 0.97,
            'min_gain_cells': 5,
            'n_rays': 180,
            'nav_timeout': 120.0,
            'spin_angle': 2.0 * math.pi,
            'spin_time_allowance': 40,
            'min_observations': 3,
            'default_timeout': 900.0,
            'result_file': '~/.ros/arm05_explorer/last_result.txt',
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.p = {name: self.get_parameter(name).value for name in defaults}
 
        self.cb = ReentrantCallbackGroup()
        self.lock = threading.Lock()
 
        self.map_data = None
        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos,
                                 callback_group=self.cb)
 
        self.observations = {}
        self.create_subscription(CubeDetection, 'cube_observations', self.obs_callback, 50,
                                 callback_group=self.cb)
 
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)
 
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose',
                                       callback_group=self.cb)
        self.spin_client = ActionClient(self, Spin, 'spin', callback_group=self.cb)
 
        self.viewpoint_pub = self.create_publisher(MarkerArray, 'viewpoints', 1)
        self.cube_marker_pub = self.create_publisher(MarkerArray, 'found_cubes', 1)
        self.create_timer(1.0, self.publish_cube_markers, callback_group=self.cb)
 
        self.action_server = ActionServer(
            self, FindCubes, 'find_cubes',
            execute_callback=self.execute_callback,
            goal_callback=lambda goal: GoalResponse.ACCEPT,
            cancel_callback=lambda goal_handle: CancelResponse.ACCEPT,
            callback_group=self.cb)
 
        self.get_logger().info('search_server gotowy, czekanie na akcje /find_cubes')

    def map_callback(self, msg):
        grid = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        self.map_data = (grid, msg.info.resolution,
                         msg.info.origin.position.x, msg.info.origin.position.y)
        self.get_logger().info(
            f'Mapa: {msg.info.width}x{msg.info.height} kratek, {msg.info.resolution} m/kratke')
 
    def obs_callback(self, msg):
        with self.lock:
            self.observations.setdefault(msg.id, []).append(
                (msg.position.x, msg.position.y, msg.position.z))
 
    def cube_estimates(self):
        with self.lock:
            items = [(i, np.median(np.array(pts), axis=0), len(pts))
                     for i, pts in self.observations.items()]
        return sorted(items, key=lambda t: t[0])
 
    def confirmed(self):
        return [e for e in self.cube_estimates() if e[2] >= self.p['min_observations']]

    def execute_callback(self, goal_handle):
        goal = goal_handle.request
        n_wanted = goal.cubes_to_find if goal.cubes_to_find > 0 else 5
        timeout = goal.timeout_sec if goal.timeout_sec > 0 else self.p['default_timeout']
        t_start = time.monotonic()
        self.get_logger().info(f'Start: szukam {n_wanted} kostek, limit {timeout:.0f} s')
 
        with self.lock:
            self.observations.clear()

        if not self.wait_until(lambda: self.map_data is not None, 15.0):
            self.get_logger().error('Brak mapy na /map')
            goal_handle.abort()
            return self.make_result(n_wanted)

        start_xy = self.robot_xy()
        if start_xy is None:
            self.get_logger().warning('Brak TF robota')
            start_xy = (0.0, 0.0)

        grid, res, ox, oy = self.map_data
        try:
            plan = plan_viewpoints(
                grid, res, ox, oy, start_xy,
                coarse_resolution=self.p['coarse_resolution'],
                robot_clearance=self.p['robot_clearance'],
                target_wall_margin=self.p['target_wall_margin'],
                sensor_range=self.p['sensor_range'],
                sensor_min_range=self.p['sensor_min_range'],
                candidate_step=self.p['candidate_step'],
                coverage_target=self.p['coverage_target'],
                min_gain_cells=self.p['min_gain_cells'],
                n_rays=self.p['n_rays'])
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'Planowanie nie powiodlo sie: {e}')
            goal_handle.abort()
            return self.make_result(n_wanted)
 
        viewpoints = plan['viewpoints']
        self.get_logger().info(
            f'Plan: {len(viewpoints)} punktow, pokrycie {plan["coverage"] * 100:.1f}%')
        self.publish_viewpoints(viewpoints)
 
        feedback = FindCubes.Feedback()
        feedback.viewpoints_total = len(viewpoints)
        for i, (x, y, yaw) in enumerate(viewpoints):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Anulowano')
                return self.make_result(n_wanted)
            if time.monotonic() - t_start > timeout:
                self.get_logger().warning('Minal limit czasu')
                break
 
            self.get_logger().info(f'Punkt {i + 1}/{len(viewpoints)}: ({x:.2f}, {y:.2f})')
            if not self.navigate(x, y, yaw, goal_handle):
                self.get_logger().warning('Robot nie dojechal')
            if goal_handle.is_cancel_requested:
                continue
            if not self.spin(goal_handle):
                self.get_logger().warning('Obrot nie powiodl sie')
 
            found = self.confirmed()
            feedback.cubes_found = len(found)
            feedback.found_ids = [int(e[0]) for e in found]
            feedback.viewpoints_done = i + 1
            goal_handle.publish_feedback(feedback)
            self.get_logger().info(f'Potwierdzone kostki: {feedback.found_ids}')
 
            if len(found) >= n_wanted:
                self.get_logger().info('Znaleziono wszystkie kostki')
                break
 
        result = self.make_result(n_wanted)
        goal_handle.succeed()
        self.save_result(result)
        for c in result.cubes:
            self.get_logger().info(
                f'Kostka {c.id}: x={c.position.x:.3f} y={c.position.y:.3f} '
                f'(pomiarow: {c.num_observations})')
        self.get_logger().info(f'Koniec po {time.monotonic() - t_start:.0f} s, '
                               f'success={result.success}')
        return result
 
    def make_result(self, n_wanted):
        estimates = self.cube_estimates()
        confirmed = [e for e in estimates if e[2] >= self.p['min_observations']]
        chosen = confirmed if len(confirmed) >= n_wanted else estimates
        result = FindCubes.Result()
        for cube_id, xyz, n in chosen:
            c = CubeDetection()
            c.id = int(cube_id)
            c.position.x, c.position.y, c.position.z = (float(v) for v in xyz)
            c.num_observations = int(n)
            result.cubes.append(c)
        result.success = len(confirmed) >= n_wanted
        return result
 
    def save_result(self, result):
        path = os.path.expanduser(self.p['result_file'])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            for c in result.cubes:
                f.write(f'Aruco {c.id}: x: {c.position.x} y: {c.position.y} '
                        f'z: {c.position.z} n: {c.num_observations}\n')
        self.get_logger().info(f'Wynik zapisany do {path}')

    def robot_xy(self):
        try:
            t = self.tf_buffer.lookup_transform(self.p['map_frame'], self.p['base_frame'],
                                                Time(), timeout=Duration(seconds=2.0))
            return t.transform.translation.x, t.transform.translation.y
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
 
    def wait_until(self, condition, timeout, goal_handle=None):
        t0 = time.monotonic()
        while rclpy.ok() and not condition():
            if time.monotonic() - t0 > timeout:
                return False
            if goal_handle is not None and goal_handle.is_cancel_requested:
                return False
            time.sleep(0.05)
        return condition()
 
    def run_action(self, client, goal_msg, timeout, goal_handle):
        if not client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Brak serwera akcji Nav2')
            return False
        send_future = client.send_goal_async(goal_msg)
        if not self.wait_until(send_future.done, 10.0):
            return False
        nav_goal = send_future.result()
        if not nav_goal.accepted:
            return False
        result_future = nav_goal.get_result_async()
        if not self.wait_until(result_future.done, timeout, goal_handle):
            nav_goal.cancel_goal_async()
            return False
        return result_future.result().status == GoalStatus.STATUS_SUCCEEDED
 
    def navigate(self, x, y, yaw, goal_handle):
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = self.p['map_frame']
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        _, _, qz, qw = quaternion_from_yaw(yaw)
        goal.pose.pose.orientation.z = qz
        goal.pose.pose.orientation.w = qw
        return self.run_action(self.nav_client, goal, self.p['nav_timeout'], goal_handle)
 
    def spin(self, goal_handle):
        goal = Spin.Goal()
        goal.target_yaw = float(self.p['spin_angle'])
        goal.time_allowance = DurationMsg(sec=int(self.p['spin_time_allowance']))
        return self.run_action(self.spin_client, goal,
                               self.p['spin_time_allowance'] + 10.0, goal_handle)
 
    def publish_viewpoints(self, viewpoints):
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for i, (x, y, _) in enumerate(viewpoints):
            for kind in ('sphere', 'text'):
                m = Marker()
                m.header.frame_id = self.p['map_frame']
                m.header.stamp = stamp
                m.ns = 'viewpoints_' + kind
                m.id = i
                m.action = Marker.ADD
                m.pose.position.x = float(x)
                m.pose.position.y = float(y)
                m.pose.orientation.w = 1.0
                if kind == 'sphere':
                    m.type = Marker.SPHERE
                    m.scale.x = m.scale.y = m.scale.z = 0.15
                    m.color.b, m.color.a = 1.0, 0.8
                else:
                    m.type = Marker.TEXT_VIEW_FACING
                    m.pose.position.z = 0.3
                    m.scale.z = 0.25
                    m.text = str(i + 1)
                    m.color.r = m.color.g = m.color.b = m.color.a = 1.0
                arr.markers.append(m)
        self.viewpoint_pub.publish(arr)
 
    def publish_cube_markers(self):
        estimates = self.cube_estimates()
        if not estimates:
            return
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for cube_id, xyz, n in estimates:
            ok = n >= self.p['min_observations']
            for kind in ('cube', 'text'):
                m = Marker()
                m.header.frame_id = self.p['map_frame']
                m.header.stamp = stamp
                m.ns = 'cubes_' + kind
                m.id = int(cube_id)
                m.action = Marker.ADD
                m.pose.position.x = float(xyz[0])
                m.pose.position.y = float(xyz[1])
                m.pose.orientation.w = 1.0
                if kind == 'cube':
                    m.type = Marker.CUBE
                    m.pose.position.z = float(xyz[2])
                    m.scale.x = m.scale.y = m.scale.z = 0.09
                    m.color.g = 1.0
                    m.color.r = 0.0 if ok else 1.0
                    m.color.a = 1.0
                else:
                    m.type = Marker.TEXT_VIEW_FACING
                    m.pose.position.z = 0.35
                    m.scale.z = 0.2
                    m.text = f'ID {cube_id} ({n})'
                    m.color.r = m.color.g = m.color.b = m.color.a = 1.0
                arr.markers.append(m)
        self.cube_marker_pub.publish(arr)
 
 
def main(args=None):
    rclpy.init(args=args)
    node = SearchServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
 
 
if __name__ == '__main__':
    main()