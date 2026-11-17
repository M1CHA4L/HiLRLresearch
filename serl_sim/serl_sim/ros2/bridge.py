"""Optional ROS2 (Humble, Ubuntu 22.04) bridge for the HIL-SERL simulator.

Ros2BridgeWrapper           publishes sim state so you can watch training in rqt / RViz / foxglove,
                            and exposes the attention / alert status on ROS2 topics.
Ros2SpacenavIntervention    a real human intervening in sim through the ROS2 `spacenav` node
                            (sudo apt install ros-humble-spacenav; ros2 run spacenav spacenav_node).
                            Drop-in replacement for franka_env SpacemouseIntervention.

Topics (default prefix /hil_sim):
  pub  joint_states            sensor_msgs/JointState
  pub  tcp_pose                geometry_msgs/PoseStamped      (frame: base)
  pub  tcp_wrench              geometry_msgs/WrenchStamped
  pub  camera/<name>/image_raw sensor_msgs/Image (rgb8)
  pub  attention               std_msgs/Bool       human is attending this step
  pub  intervening             std_msgs/Bool       human is in control this step
  pub  human_state             std_msgs/String     away | reacting | attending
  pub  alert                   std_msgs/Bool       risk module raised an alert this step
  sub  request_attention       std_msgs/Empty      external alert -> SimHumanIntervention

rclpy is imported lazily so the simulator works without ROS2. Source ROS2 before activating the
conda env:  source /opt/ros/humble/setup.bash && conda activate hilserl
"""
import threading
from typing import Optional

import numpy as np
import gymnasium as gym

from serl_sim.envs.ur7e_pick_place_env import ARM_JOINTS

_node = None
_node_lock = threading.Lock()


def get_node(name: str = "hil_sim"):
    """One shared rclpy node per process, spun in a background thread."""
    global _node
    with _node_lock:
        if _node is not None:
            return _node
        import rclpy
        from rclpy.executors import SingleThreadedExecutor

        if not rclpy.ok():
            rclpy.init()
        _node = rclpy.create_node(name)
        ex = SingleThreadedExecutor()
        ex.add_node(_node)
        threading.Thread(target=ex.spin, daemon=True).start()
        return _node


def _find_wrapper(env, cls_name: str):
    while env is not None:
        if type(env).__name__ == cls_name:
            return env
        env = getattr(env, "env", None)
    return None


class Ros2BridgeWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, prefix: str = "/hil_sim", publish_images: bool = True,
                 image_every: int = 1):
        super().__init__(env)
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState, Image
        from geometry_msgs.msg import PoseStamped, WrenchStamped
        from std_msgs.msg import Bool, String, Empty

        self._msgs = dict(JointState=JointState, Image=Image, PoseStamped=PoseStamped,
                          WrenchStamped=WrenchStamped, Bool=Bool, String=String)
        self.node = get_node()
        n, p = self.node, prefix.rstrip("/")
        self.pub_js = n.create_publisher(JointState, f"{p}/joint_states", 10)
        self.pub_pose = n.create_publisher(PoseStamped, f"{p}/tcp_pose", 10)
        self.pub_wrench = n.create_publisher(WrenchStamped, f"{p}/tcp_wrench", 10)
        self.pub_att = n.create_publisher(Bool, f"{p}/attention", 10)
        self.pub_int = n.create_publisher(Bool, f"{p}/intervening", 10)
        self.pub_hs = n.create_publisher(String, f"{p}/human_state", 10)
        self.pub_alert = n.create_publisher(Bool, f"{p}/alert", 10)
        self.publish_images = publish_images
        self.image_every = max(1, image_every)
        self.pub_img = {
            k: n.create_publisher(Image, f"{p}/camera/{k}/image_raw", qos_profile_sensor_data)
            for k in env.unwrapped.config.IMAGE_KEYS
        } if publish_images else {}
        self._human = _find_wrapper(env, "SimHumanIntervention")
        if self._human is not None:
            n.create_subscription(Empty, f"{p}/request_attention",
                                  lambda _m: self._human.request_attention(), 10)
        self._alert_flag = False
        self._t = 0

    def mark_alert(self, alert: bool):
        """Called by the actor loop so the alert is visible on ROS2."""
        self._alert_flag = bool(alert)

    def _publish(self, obs_images: Optional[dict], info: dict):
        M, base = self._msgs, self.env.unwrapped
        stamp = self.node.get_clock().now().to_msg()

        js = M["JointState"]()
        js.header.stamp = stamp
        js.name = list(ARM_JOINTS)
        js.position = base.q.tolist()
        js.velocity = base.dq.tolist()
        self.pub_js.publish(js)

        ps = M["PoseStamped"]()
        ps.header.stamp, ps.header.frame_id = stamp, "base"
        x, y, z, qx, qy, qz, qw = base.currpos.tolist()
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = x, y, z
        ps.pose.orientation.x, ps.pose.orientation.y = qx, qy
        ps.pose.orientation.z, ps.pose.orientation.w = qz, qw
        self.pub_pose.publish(ps)

        ws = M["WrenchStamped"]()
        ws.header.stamp, ws.header.frame_id = stamp, "base"
        f, t = base.currforce, base.currtorque
        ws.wrench.force.x, ws.wrench.force.y, ws.wrench.force.z = map(float, f)
        ws.wrench.torque.x, ws.wrench.torque.y, ws.wrench.torque.z = map(float, t)
        self.pub_wrench.publish(ws)

        if obs_images is not None and self._t % self.image_every == 0:
            for k, pub in self.pub_img.items():
                img = np.ascontiguousarray(obs_images[k])
                msg = M["Image"]()
                msg.header.stamp, msg.header.frame_id = stamp, k
                msg.height, msg.width = img.shape[:2]
                msg.encoding, msg.step = "rgb8", img.shape[1] * 3
                msg.data = img.tobytes()
                pub.publish(msg)

        self.pub_att.publish(M["Bool"](data=bool(info.get("attention", 0))))
        self.pub_int.publish(M["Bool"](data="intervene_action" in info))
        self.pub_hs.publish(M["String"](data=str(info.get("human_state", ""))))
        self.pub_alert.publish(M["Bool"](data=self._alert_flag))
        self._alert_flag = False

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        self._t += 1
        try:
            self._publish(obs.get("images"), info)
        except Exception as e:  # never let visualization kill training
            self.node.get_logger().warn(f"publish failed: {e}", throttle_duration_sec=10.0)
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._publish(obs.get("images"), info)
        return obs, info


class Ros2SpacenavIntervention(gym.ActionWrapper):
    """Human teleop in sim via ROS2 spacenav. Same output convention as SpacemouseIntervention."""

    def __init__(self, env, twist_topic="/spacenav/twist", joy_topic="/spacenav/joy",
                 scale: float = 1.0, deadband: float = 0.05, action_indices=None):
        super().__init__(env)
        from geometry_msgs.msg import Twist
        from sensor_msgs.msg import Joy

        self.node = get_node()
        self._lock = threading.Lock()
        self._twist = np.zeros(6)
        self._buttons = [0, 0]
        self.scale, self.deadband = scale, deadband
        self.action_indices = action_indices
        self.gripper_enabled = self.action_space.shape != (6,)
        self.node.create_subscription(Twist, twist_topic, self._on_twist, 10)
        self.node.create_subscription(Joy, joy_topic, self._on_joy, 10)

    def _on_twist(self, m):
        with self._lock:
            self._twist = np.array([m.linear.x, m.linear.y, m.linear.z,
                                    m.angular.x, m.angular.y, m.angular.z])

    def _on_joy(self, m):
        with self._lock:
            self._buttons = list(m.buttons[:2]) + [0] * max(0, 2 - len(m.buttons))

    def action(self, action):
        with self._lock:
            tw, (left, right) = self._twist.copy(), self._buttons[:2]
        expert_a = np.clip(tw * self.scale, -1, 1)
        expert_a[np.abs(expert_a) < self.deadband] = 0.0
        intervened = bool(np.linalg.norm(expert_a) > 1e-3)
        if self.gripper_enabled:
            g = 0.0
            if left:
                g, intervened = -1.0, True
            elif right:
                g, intervened = 1.0, True
            expert_a = np.concatenate([expert_a, [g]])
        if self.action_indices is not None:
            filt = np.zeros_like(expert_a)
            filt[self.action_indices] = expert_a[self.action_indices]
            expert_a = filt
        return (expert_a, True) if intervened else (action, False)

    def step(self, action):
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["attention"] = 1  # a real human watching is full attention
        return obs, rew, done, truncated, info
