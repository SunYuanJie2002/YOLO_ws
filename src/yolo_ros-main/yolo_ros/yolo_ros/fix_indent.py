with open("/home/unitree/YOLO_ws/src/yolo_ros-main/yolo_ros/yolo_ros/tracking_node.py", "r") as f:
    lines = f.readlines()
with open("/home/unitree/YOLO_ws/src/yolo_ros-main/yolo_ros/yolo_ros/tracking_node.py", "w") as f:
    for line in lines:
        if "self.tracker = self.create_tracker(tracker_name)" in line:
            f.write("        self.tracker = self.create_tracker(tracker_name)\n")
        else:
            f.write(line)
