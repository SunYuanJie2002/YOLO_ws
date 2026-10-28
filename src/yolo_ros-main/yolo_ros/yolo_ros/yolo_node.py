# Copyright (C) 2023 Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from typing import List, Dict, Optional
import cv2
import numpy as np
import zmq

import rclpy
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

import torch
from ultralytics import YOLO, YOLOWorld, YOLOE
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes
from ultralytics.engine.results import Masks
from ultralytics.engine.results import Keypoints

from rcl_interfaces.msg import SetParametersResult
from std_srvs.srv import SetBool
from std_msgs.msg import Header
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from yolo_msgs.msg import Point2D
from yolo_msgs.msg import BoundingBox2D
from yolo_msgs.msg import Mask
from yolo_msgs.msg import KeyPoint2D
from yolo_msgs.msg import KeyPoint2DArray
from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray
from yolo_msgs.srv import SetClasses
from yolo_msgs.srv import SetContainmentRule


class YoloNode(LifecycleNode):
    """
    ROS 2 Lifecycle Node for YOLO object detection.

    This node provides real-time object detection using various YOLO models
    (YOLO, YOLOWorld, YOLOE). It subscribes to image topics and publishes
    detection results including bounding boxes, masks, and keypoints.
    """

    def __init__(self) -> None:
        """
        Initialize the YOLO node.

        Declares all ROS parameters for model configuration and inference settings.
        """
        super().__init__("yolo_node")

        # Params
        self.declare_parameter("model_type", "YOLO")
        self.declare_parameter("model", "yolov8m.pt")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("fuse_model", False)
        self.declare_parameter("yolo_encoding", "bgr8")
        self.declare_parameter("enable", True)
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter("zmq_host", "192.168.123.164")
        self.declare_parameter("zmq_port", 55555)
        self.declare_parameter("zmq_timeout_ms", 5)
        self.declare_parameter("zmq_poll_period", 0.01)
        self.declare_parameter("source_height", 480)
        self.declare_parameter("source_width", 1280)

        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("iou", 0.5)
        self.declare_parameter("imgsz_height", 640)
        self.declare_parameter("imgsz_width", 640)
        self.declare_parameter("half", False)
        self.declare_parameter("max_det", 300)
        self.declare_parameter("augment", False)
        self.declare_parameter("agnostic_nms", False)
        self.declare_parameter("retina_masks", False)

        # Containment rules: dict mapping inner_class -> outer_class
        self.containment_rules: Dict[str, str] = {}
        # Minimum overlap ratio for containment (0.0~1.0, default 1.0 = fully contained)
        self.declare_parameter("containment_threshold", 1.0)

        self.type_to_model = {"YOLO": YOLO, "World": YOLOWorld, "YOLOE": YOLOE}

        # Hot-update runtime parameters via `ros2 param set`
        self.add_on_set_parameters_callback(self.params_cb)

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Configure lifecycle callback.

        Retrieves all parameters and sets up publishers and QoS profiles.

        @param state Current lifecycle state
        @return Transition callback return status
        """
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        # Model params
        self.model_type = (
            self.get_parameter("model_type").get_parameter_value().string_value
        )
        self.model = self.get_parameter("model").get_parameter_value().string_value
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.fuse_model = (
            self.get_parameter("fuse_model").get_parameter_value().bool_value
        )
        self.yolo_encoding = (
            self.get_parameter("yolo_encoding").get_parameter_value().string_value
        )

        # Inference params
        self.threshold = (
            self.get_parameter("threshold").get_parameter_value().double_value
        )
        self.iou = self.get_parameter("iou").get_parameter_value().double_value
        self.imgsz_height = (
            self.get_parameter("imgsz_height").get_parameter_value().integer_value
        )
        self.imgsz_width = (
            self.get_parameter("imgsz_width").get_parameter_value().integer_value
        )
        self.half = self.get_parameter("half").get_parameter_value().bool_value
        self.max_det = self.get_parameter("max_det").get_parameter_value().integer_value
        self.augment = self.get_parameter("augment").get_parameter_value().bool_value
        self.agnostic_nms = (
            self.get_parameter("agnostic_nms").get_parameter_value().bool_value
        )
        self.retina_masks = (
            self.get_parameter("retina_masks").get_parameter_value().bool_value
        )
        self.containment_threshold = (
            self.get_parameter("containment_threshold").get_parameter_value().double_value
        )

        # ROS params
        self.enable = self.get_parameter("enable").get_parameter_value().bool_value
        self.reliability = (
            self.get_parameter("image_reliability").get_parameter_value().integer_value
        )
        self.zmq_host = self.get_parameter("zmq_host").get_parameter_value().string_value
        self.zmq_port = self.get_parameter("zmq_port").get_parameter_value().integer_value
        self.zmq_timeout_ms = (
            self.get_parameter("zmq_timeout_ms").get_parameter_value().integer_value
        )
        self.zmq_poll_period = (
            self.get_parameter("zmq_poll_period").get_parameter_value().double_value
        )
        self.source_height = (
            self.get_parameter("source_height").get_parameter_value().integer_value
        )
        self.source_width = (
            self.get_parameter("source_width").get_parameter_value().integer_value
        )

        # Detection pub
        self.image_qos_profile = QoSProfile(
            reliability=self.reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self._pub = self.create_lifecycle_publisher(DetectionArray, "detections", 10)
        self._image_pub = self.create_publisher(
            Image, "image_raw", qos_profile=self.image_qos_profile
        )
        self.cv_bridge = CvBridge()
        self._zmq_context = None
        self._zmq_socket = None
        self._zmq_timer = None

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Activate lifecycle callback.

        Loads the YOLO model, optionally fuses it, and creates subscriptions and services.

        @param state Current lifecycle state
        @return Transition callback return status
        """
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        try:
            self.yolo = self.type_to_model[self.model_type](self.model)
            self.yolo.to(self.device)
        except FileNotFoundError:
            self.get_logger().error(f"Model file '{self.model}' does not exists")
            return TransitionCallbackReturn.ERROR

        # YOLOE does not support fusing
        if self.fuse_model and (
            isinstance(self.yolo, YOLO) or isinstance(self.yolo, YOLOWorld)
        ):
            try:
                self.get_logger().info("Trying to fuse model...")
                self.yolo.fuse()
            except TypeError as e:
                self.get_logger().warn(f"Error while fuse: {e}")

        self._enable_srv = self.create_service(SetBool, "enable", self.enable_cb)

        self._has_set_classes = hasattr(self.yolo, "set_classes")
        if self._has_set_classes:
            self._set_classes_srv = self.create_service(
                SetClasses, "set_classes", self.set_classes_cb
            )

        self._set_containment_srv = self.create_service(
            SetContainmentRule, "set_containment_rule", self.set_containment_rule_cb
        )

        self._zmq_context = zmq.Context()
        self._zmq_socket = self._zmq_context.socket(zmq.SUB)
        self._zmq_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._zmq_socket.setsockopt(zmq.LINGER, 0)
        # Keep only the latest frame in socket queue to reduce latency.
        self._zmq_socket.setsockopt(zmq.CONFLATE, 1)
        self._zmq_socket.setsockopt(zmq.RCVHWM, 1)
        self._zmq_socket.setsockopt(zmq.RCVTIMEO, self.zmq_timeout_ms)
        endpoint = f"tcp://{self.zmq_host}:{self.zmq_port}"
        self._zmq_socket.connect(endpoint)
        self.get_logger().info(f"Connected to ZMQ image source: {endpoint}")
        self._zmq_timer = self.create_timer(self.zmq_poll_period, self.zmq_image_cb)

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Deactivate lifecycle callback.

        Destroys the YOLO model, clears CUDA cache if applicable, and cleans up services and subscriptions.

        @param state Current lifecycle state
        @return Transition callback return status
        """
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        del self.yolo
        if "cuda" in self.device:
            self.get_logger().info("Clearing CUDA cache")
            torch.cuda.empty_cache()

        self.destroy_service(self._enable_srv)
        self._enable_srv = None

        if self._has_set_classes:
            self.destroy_service(self._set_classes_srv)
            self._set_classes_srv = None

        self.destroy_service(self._set_containment_srv)
        self._set_containment_srv = None

        if self._zmq_timer is not None:
            self.destroy_timer(self._zmq_timer)
            self._zmq_timer = None

        if self._zmq_socket is not None:
            self._zmq_socket.close()
            self._zmq_socket = None

        if self._zmq_context is not None:
            self._zmq_context.term()
            self._zmq_context = None

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Cleanup lifecycle callback.

        Destroys publishers and cleans up resources.

        @param state Current lifecycle state
        @return Transition callback return status
        """
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")

        self.destroy_publisher(self._pub)
        self.destroy_publisher(self._image_pub)
        del self.image_qos_profile

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Shutdown lifecycle callback.

        Performs final cleanup before node shutdown.

        @param state Current lifecycle state
        @return Transition callback return status
        """
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_shutdown(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS

    def enable_cb(
        self,
        request: SetBool.Request,
        response: SetBool.Response,
    ) -> SetBool.Response:
        """
        Service callback to enable or disable detection.

        @param request Service request containing enable/disable flag
        @param response Service response
        @return Service response with success status
        """
        self.enable = request.data
        response.success = True
        return response

    def parse_hypothesis(self, results: Results) -> List[Dict]:
        """
        Parse detection hypotheses from YOLO results.

        Extracts class IDs, class names, and confidence scores from detection results.

        @param results YOLO detection results
        @return List of dictionaries containing class information and scores
        """

        hypothesis_list = []

        if results.boxes:
            box_data: Boxes
            for box_data in results.boxes:
                hypothesis = {
                    "class_id": int(box_data.cls),
                    "class_name": self.yolo.names[int(box_data.cls)],
                    "score": float(box_data.conf),
                }
                hypothesis_list.append(hypothesis)

        elif results.obb:
            for i in range(results.obb.cls.shape[0]):
                hypothesis = {
                    "class_id": int(results.obb.cls[i]),
                    "class_name": self.yolo.names[int(results.obb.cls[i])],
                    "score": float(results.obb.conf[i]),
                }
                hypothesis_list.append(hypothesis)

        return hypothesis_list

    def parse_boxes(self, results: Results) -> List[BoundingBox2D]:
        """
        Parse bounding boxes from YOLO results.

        Converts YOLO bounding box format to ROS BoundingBox2D messages.
        Supports both regular boxes and oriented bounding boxes (OBB).

        @param results YOLO detection results
        @return List of BoundingBox2D messages
        """

        boxes_list = []

        if results.boxes:
            box_data: Boxes
            for box_data in results.boxes:

                msg = BoundingBox2D()

                # Get boxes values
                box = box_data.xywh[0]
                msg.center.position.x = float(box[0])
                msg.center.position.y = float(box[1])
                msg.size.x = float(box[2])
                msg.size.y = float(box[3])

                # Append msg
                boxes_list.append(msg)

        elif results.obb:
            for i in range(results.obb.cls.shape[0]):
                msg = BoundingBox2D()

                # Get boxes values
                box = results.obb.xywhr[i]
                msg.center.position.x = float(box[0])
                msg.center.position.y = float(box[1])
                msg.center.theta = float(box[4])
                msg.size.x = float(box[2])
                msg.size.y = float(box[3])

                # Append msg
                boxes_list.append(msg)

        return boxes_list

    def parse_masks(self, results: Results) -> List[Mask]:
        """
        Parse segmentation masks from YOLO results.

        Converts YOLO mask format to ROS Mask messages containing polygon points.

        @param results YOLO detection results
        @return List of Mask messages
        """

        masks_list = []

        def create_point2d(x: float, y: float) -> Point2D:
            p = Point2D()
            p.x = x
            p.y = y
            return p

        mask: Masks
        for mask in results.masks:

            msg = Mask()

            msg.data = [
                create_point2d(float(ele[0]), float(ele[1]))
                for ele in mask.xy[0].tolist()
            ]
            msg.height = results.orig_img.shape[0]
            msg.width = results.orig_img.shape[1]

            masks_list.append(msg)

        return masks_list

    def parse_keypoints(self, results: Results) -> List[KeyPoint2DArray]:
        """
        Parse keypoints from YOLO results.

        Extracts keypoint positions and confidence scores, filtering by threshold.

        @param results YOLO detection results
        @return List of KeyPoint2DArray messages
        """

        keypoints_list = []

        points: Keypoints
        for points in results.keypoints:

            msg_array = KeyPoint2DArray()

            if points.conf is None:
                continue

            for kp_id, (p, conf) in enumerate(zip(points.xy[0], points.conf[0])):

                if conf >= self.threshold:
                    msg = KeyPoint2D()

                    msg.id = kp_id + 1
                    msg.point.x = float(p[0])
                    msg.point.y = float(p[1])
                    msg.score = float(conf)

                    msg_array.data.append(msg)

            keypoints_list.append(msg_array)

        return keypoints_list

    def _decode_zmq_image(self, msg_parts: List[bytes]) -> Optional[np.ndarray]:
        """
        Decode ZMQ payload into an OpenCV BGR image.

        Supports common image-encoded payloads and raw uint8 arrays.
        """
        if not msg_parts:
            return None

        payload = msg_parts[-1]
        image_array = np.frombuffer(payload, dtype=np.uint8)
        cv_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if cv_image is not None:
            return cv_image

        rgb_size = self.source_height * self.source_width * 3
        gray_size = self.source_height * self.source_width

        if image_array.size == rgb_size:
            return image_array.reshape(self.source_height, self.source_width, 3)

        if image_array.size == gray_size:
            gray = image_array.reshape(self.source_height, self.source_width)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return None

    def zmq_image_cb(self) -> None:
        """
        Poll one image from ZMQ and run detection.
        """
        if not self.enable or self._zmq_socket is None:
            return

        try:
            msg_parts = self._zmq_socket.recv_multipart(flags=zmq.NOBLOCK)
            # Drain queued messages and keep only the newest frame.
            while True:
                msg_parts = self._zmq_socket.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            if "msg_parts" not in locals():
                return
        except Exception as exc:
            self.get_logger().warn(f"Failed to receive ZMQ image: {exc}")
            return

        cv_image = self._decode_zmq_image(msg_parts)
        if cv_image is None:
            self.get_logger().warn("Received undecodable ZMQ image payload")
            return

        if cv_image.shape[0] < self.source_height or cv_image.shape[1] < self.source_width:
            self.get_logger().warn(
                f"Unexpected frame shape {cv_image.shape}, expected at least "
                f"({self.source_height}, {self.source_width})"
            )
            return

        # Input frame is 480x1280, keep left half 480x640 for inference.
        cv_image = cv_image[: self.source_height, : self.source_width // 2]
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "zmq_left_camera"
        image_msg = self.cv_bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        image_msg.header = header
        self._image_pub.publish(image_msg)
        self.process_image(cv_image, header)

    def process_image(self, cv_image: np.ndarray, header: Header) -> None:
        """
        Run YOLO inference and publish detections.
        """
        results = self.yolo.predict(
            source=cv_image,
            verbose=False,
            stream=False,
            conf=self.threshold,
            iou=self.iou,
            imgsz=(self.imgsz_height, self.imgsz_width),
            half=self.half,
            max_det=self.max_det,
            augment=self.augment,
            agnostic_nms=self.agnostic_nms,
            retina_masks=self.retina_masks,
            device=self.device,
        )
        results: Results = results[0].cpu()

        hypothesis = []
        boxes = []
        masks = []
        keypoints = []

        if results.boxes or results.obb:
            hypothesis = self.parse_hypothesis(results)
            boxes = self.parse_boxes(results)

        # Apply containment filter (e.g. clothes must be inside blue box)
        if hypothesis and boxes and self.containment_rules:
            hypothesis, boxes = self._apply_containment_filter(hypothesis, boxes)

        if results.masks:
            masks = self.parse_masks(results)

        if results.keypoints:
            keypoints = self.parse_keypoints(results)

        detections_msg = DetectionArray()

        for i in range(len(hypothesis)):
            aux_msg = Detection()

            if (results.boxes or results.obb) and hypothesis and boxes:
                aux_msg.class_id = hypothesis[i]["class_id"]
                aux_msg.class_name = hypothesis[i]["class_name"]
                aux_msg.score = hypothesis[i]["score"]
                aux_msg.bbox = boxes[i]

            if results.masks and masks and i < len(masks):
                aux_msg.mask = masks[i]

            if results.keypoints and keypoints and i < len(keypoints):
                aux_msg.keypoints = keypoints[i]

            detections_msg.detections.append(aux_msg)

        detections_msg.header = header
        self._pub.publish(detections_msg)

        del results

    def set_classes_cb(
        self,
        req: SetClasses.Request,
        res: SetClasses.Response,
    ) -> SetClasses.Response:
        """
        Service callback to set detection classes (YOLOWorld only).

        Updates the classes that the model should detect.

        @param req Service request containing list of class names
        @param res Service response
        @return Service response
        """

        try:
            self.get_logger().info(f"Setting classes: {req.classes}")
            self.yolo.set_classes(req.classes)
            self.get_logger().info(f"New classes: {self.yolo.names}")
        except Exception as e:
            self.get_logger().error(f"Error setting classes: {e}")
        return res

    def set_containment_rule_cb(
        self,
        req: SetContainmentRule.Request,
        res: SetContainmentRule.Response,
    ) -> SetContainmentRule.Response:
        """
        Service callback to set or clear containment rules for detection filtering.

        - Both outer_class and inner_class non-empty: set a rule.
          inner_class detections are only published if they are contained
          within an outer_class bounding box.
        - inner_class non-empty, outer_class empty: remove the rule for that
          inner_class only.
        - Both empty: clear ALL rules (disable containment filtering entirely).

        @param req outer_class (container) and inner_class (contained object)
        @param res success and message
        """
        if not req.outer_class and not req.inner_class:
            # Clear all rules -> disable containment filtering
            self.containment_rules.clear()
            self.get_logger().info(
                "Containment rules cleared: filtering disabled"
            )
            res.success = True
            res.message = "All containment rules cleared (filtering disabled)"
        elif req.inner_class and not req.outer_class:
            # Remove only the rule for this inner_class
            removed = self.containment_rules.pop(req.inner_class, None)
            self.get_logger().info(
                f"Containment rule removed for '{req.inner_class}'"
                if removed
                else f"No containment rule found for '{req.inner_class}'"
            )
            res.success = True
            res.message = f"Rule removed for {req.inner_class}"
        else:
            # Set a new rule
            self.containment_rules[req.inner_class] = req.outer_class
            self.get_logger().info(
                f"Containment rule: '{req.inner_class}' must be inside '{req.outer_class}'"
            )
            res.success = True
            res.message = f"Rule set: {req.inner_class} inside {req.outer_class}"
        return res

    def params_cb(self, params: List[Parameter]) -> SetParametersResult:
        """
        Callback for hot-updating runtime parameters via `ros2 param set`.

        Supported now:
        - ``containment_threshold``: containment filter strictness (0.0~1.0).
        - ``threshold``: detection confidence threshold.

        Parameters read only during ``on_configure`` (e.g. zmq_*, source_*, model)
        are intentionally NOT updated here and still require a node restart.

        @param params List of parameters being set.
        @return SetParametersResult success flag.
        """
        result = SetParametersResult()
        result.successful = True
        for param in params:
            if param.name == "containment_threshold":
                self.containment_threshold = float(param.value)
                self.get_logger().info(
                    f"containment_threshold updated to {self.containment_threshold} "
                    "(hot reload, no restart needed)"
                )
            elif param.name == "threshold":
                self.threshold = float(param.value)
                self.get_logger().info(
                    f"threshold updated to {self.threshold} "
                    "(hot reload, no restart needed)"
                )
        return result

    @staticmethod
    def _compute_containment_ratio(
        inner_box: BoundingBox2D, outer_box: BoundingBox2D
    ) -> float:
        """
        Compute what fraction of inner_box's area lies inside outer_box.

        Returns 1.0 for fully contained, 0.0 for no overlap, and
        intermediate values for partial containment.

        @param inner_box The bounding box that should be inside.
        @param outer_box The bounding box that should contain the inner box.
        @return Ratio of inner_box area inside outer_box (0.0 ~ 1.0).
        """
        inner_left = inner_box.center.position.x - inner_box.size.x / 2.0
        inner_right = inner_box.center.position.x + inner_box.size.x / 2.0
        inner_top = inner_box.center.position.y - inner_box.size.y / 2.0
        inner_bottom = inner_box.center.position.y + inner_box.size.y / 2.0

        outer_left = outer_box.center.position.x - outer_box.size.x / 2.0
        outer_right = outer_box.center.position.x + outer_box.size.x / 2.0
        outer_top = outer_box.center.position.y - outer_box.size.y / 2.0
        outer_bottom = outer_box.center.position.y + outer_box.size.y / 2.0

        # Intersection area
        inter_left = max(inner_left, outer_left)
        inter_right = min(inner_right, outer_right)
        inter_top = max(inner_top, outer_top)
        inter_bottom = min(inner_bottom, outer_bottom)

        inter_w = max(0.0, inter_right - inter_left)
        inter_h = max(0.0, inter_bottom - inter_top)
        inter_area = inter_w * inter_h

        # Inner box area
        inner_area = inner_box.size.x * inner_box.size.y
        if inner_area <= 0.0:
            return 0.0

        return inter_area / inner_area

    def _apply_containment_filter(
        self,
        hypothesis: List[Dict],
        boxes: List[BoundingBox2D],
    ) -> tuple:
        """
        Filter detections based on containment rules.

        For each containment rule (inner_class must be inside outer_class),
        keep only those inner_class detections whose overlap ratio with
        an outer_class detection meets containment_threshold.

        @param hypothesis List of detection hypothesis dicts.
        @param boxes List of BoundingBox2D messages.
        @return Filtered (hypothesis, boxes) tuple.
        """
        if not self.containment_rules:
            return hypothesis, boxes

        # Find outer-class boxes for each rule
        # outer_class_name -> list of box indices
        outer_indices: Dict[str, List[int]] = {}
        for i, hyp in enumerate(hypothesis):
            class_name = hyp["class_name"]
            for inner_cls, outer_cls in self.containment_rules.items():
                if class_name == outer_cls:
                    outer_indices.setdefault(outer_cls, []).append(i)

        # Filter: keep inner_class detections only if overlap ratio >= threshold
        keep_mask = [True] * len(hypothesis)
        for i, hyp in enumerate(hypothesis):
            class_name = hyp["class_name"]
            if class_name in self.containment_rules:
                outer_cls = self.containment_rules[class_name]
                outer_idxs = outer_indices.get(outer_cls, [])
                best_ratio = 0.0
                for oi in outer_idxs:
                    ratio = self._compute_containment_ratio(boxes[i], boxes[oi])
                    if ratio > best_ratio:
                        best_ratio = ratio
                if best_ratio < self.containment_threshold:
                    keep_mask[i] = False
                    self.get_logger().debug(
                        f"Filtered out '{class_name}' "
                        f"(best containment ratio {best_ratio:.2f} < {self.containment_threshold})"
                    )

        filtered_hypothesis = [h for h, k in zip(hypothesis, keep_mask) if k]
        filtered_boxes = [b for b, k in zip(boxes, keep_mask) if k]
        return filtered_hypothesis, filtered_boxes


def main():
    rclpy.init()
    node = YoloNode()
    node.trigger_configure()
    node.trigger_activate()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
