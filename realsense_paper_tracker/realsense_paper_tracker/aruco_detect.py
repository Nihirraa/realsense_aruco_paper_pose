#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np


class ArucoDetect(Node):
    def __init__(self):
        super().__init__("aruco_detect")

        self.bridge = CvBridge()
        self.image_topic = "/camera/camera/color/image_rect_raw"

        self.sub = self.create_subscription(Image, self.image_topic, self.on_image, 10)

        # Older OpenCV-compatible ArUco setup
        self.dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.params = cv2.aruco.DetectorParameters_create()

        self.want_ids = {5, 12, 23, 34}

        self.get_logger().info(f"Listening on {self.image_topic}")
        self.get_logger().info("Looking for ArUco IDs: 5, 12, 23, 34 (DICT_4X4_50)")

    def on_image(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge convert failed: {e}")
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray,
            self.dict,
            parameters=self.params
        )

        if ids is None or len(ids) == 0:
            return

        ids_list = [int(x) for x in ids.flatten()]
        found = sorted(set(ids_list) & self.want_ids)

        self.get_logger().info(f"Found IDs: {ids_list} | target present: {found}")

        for i, mid in enumerate(ids_list):
            if mid not in self.want_ids:
                continue

            pts = corners[i][0]  # 4 corner points
            tl, tr, br, bl = pts

            self.get_logger().info(
                f"  ID {mid}: TL={tl.round(1)} TR={tr.round(1)} "
                f"BR={br.round(1)} BL={bl.round(1)}"
            )


def main():
    rclpy.init()
    node = ArucoDetect()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
