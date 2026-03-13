#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

import cv2
import numpy as np
import math


class PaperPose(Node):
    def __init__(self):
        super().__init__("paper_pose_strict")

        # --------- YOUR SPEC (meters) ----------
        self.page_w = 0.223
        self.page_h = 0.297
        self.marker_size = 0.050
        self.inset = 0.020

        # Marker IDs
        self.id_tl = 5
        self.id_tr = 12
        self.id_bl = 23
        self.id_br = 34
        self.want_ids = {self.id_tl, self.id_tr, self.id_bl, self.id_br}

        self.image_topic = "/camera/camera/color/image_rect_raw"
        self.info_topic  = "/camera/camera/color/camera_info"

        self.bridge = CvBridge()

        self.K = None
        self.D = None
        self.camera_frame = None

        self.sub_info = self.create_subscription(CameraInfo, self.info_topic, self.on_info, 10)
        self.sub_img  = self.create_subscription(Image, self.image_topic, self.on_image, 10)

        self.pub_pose = self.create_publisher(PoseStamped, "/paper_pose", 10)

        # ArUco (older OpenCV API)
        self.dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.params = cv2.aruco.DetectorParameters_create()

        self.get_logger().info("STRICT mode: publish only when ALL 4 markers visible.")
        self.get_logger().info(f"Required IDs: {sorted(self.want_ids)}")
        self.get_logger().info("Origin = CENTER of page")

    def on_info(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.D = np.array(msg.d, dtype=np.float64).reshape(-1, 1) if len(msg.d) else None
            self.camera_frame = msg.header.frame_id
            self.get_logger().info("Camera intrinsics received.")

    # Paper frame origin at CENTER: x right, y up, z=0 plane
    def marker_center_xy(self, which: str):
        s = self.marker_size
        i = self.inset
        W = self.page_w
        H = self.page_h

        # marker center positions in a top-left-origin frame (x right, y down)
        if which == "TL":
            x_tl, y_tl = (i + s/2, i + s/2)
        elif which == "TR":
            x_tl, y_tl = (W - i - s/2, i + s/2)
        elif which == "BL":
            x_tl, y_tl = (i + s/2, H - i - s/2)
        elif which == "BR":
            x_tl, y_tl = (W - i - s/2, H - i - s/2)
        else:
            raise ValueError("Invalid marker label")

        # convert to center-origin frame and flip y so +y is up
        x = x_tl - W/2
        y = (H/2) - y_tl
        return (x, y)

    def marker_object_corners(self, label: str):
        cx, cy = self.marker_center_xy(label)
        hs = self.marker_size / 2.0

        # Order must match OpenCV marker corner order: TL, TR, BR, BL
        TL = (cx - hs, cy + hs, 0.0)
        TR = (cx + hs, cy + hs, 0.0)
        BR = (cx + hs, cy - hs, 0.0)
        BL = (cx - hs, cy - hs, 0.0)

        return np.array([TL, TR, BR, BL], dtype=np.float64)

    def rvec_to_quaternion(self, rvec):
        R, _ = cv2.Rodrigues(rvec)

        qw = math.sqrt(max(0, 1 + R[0,0] + R[1,1] + R[2,2])) / 2
        qx = math.sqrt(max(0, 1 + R[0,0] - R[1,1] - R[2,2])) / 2
        qy = math.sqrt(max(0, 1 - R[0,0] + R[1,1] - R[2,2])) / 2
        qz = math.sqrt(max(0, 1 - R[0,0] - R[1,1] + R[2,2])) / 2

        qx = math.copysign(qx, R[2,1] - R[1,2])
        qy = math.copysign(qy, R[0,2] - R[2,0])
        qz = math.copysign(qz, R[1,0] - R[0,1])

        return qx, qy, qz, qw

    def on_image(self, msg: Image):
        if self.K is None:
            return

        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dict, parameters=self.params)
        if ids is None:
            return

        ids_list = [int(x) for x in ids.flatten()]
        present = set(ids_list)

        # STRICT: require all 4 markers
        if not self.want_ids.issubset(present):
            return

        # Build correspondences using ONLY the 4 required markers
        obj_pts = []
        img_pts = []

        # Map ID -> label
        id_to_label = {
            self.id_tl: "TL",
            self.id_tr: "TR",
            self.id_bl: "BL",
            self.id_br: "BR",
        }

        for i, mid in enumerate(ids_list):
            if mid not in self.want_ids:
                continue
            label = id_to_label[mid]
            obj = self.marker_object_corners(label)          # (4,3)
            img_c = corners[i][0].astype(np.float64)         # (4,2)

            obj_pts.append(obj)
            img_pts.append(img_c)

        obj_pts = np.vstack(obj_pts)  # (16,3)
        img_pts = np.vstack(img_pts)  # (16,2)

        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, self.K, self.D,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return

        ps = PoseStamped()
        ps.header = msg.header
        ps.header.frame_id = self.camera_frame

        ps.pose.position.x = float(tvec[0,0])
        ps.pose.position.y = float(tvec[1,0])
        ps.pose.position.z = float(tvec[2,0])

        qx, qy, qz, qw = self.rvec_to_quaternion(rvec)
        ps.pose.orientation.x = float(qx)
        ps.pose.orientation.y = float(qy)
        ps.pose.orientation.z = float(qz)
        ps.pose.orientation.w = float(qw)

        self.pub_pose.publish(ps)

        dist = float(np.linalg.norm(tvec))
        self.get_logger().info(f"ALL 4 visible {sorted(self.want_ids)} | distance: {dist:.3f} m")


def main():
    rclpy.init()
    node = PaperPose()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
