#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

import time
import numpy as np
from pyquaternion import Quaternion

import argparse
import glob
from pathlib import Path

import mayavi.mlab as mlab
import numpy as np
import torch
import scipy.linalg as linalg

import sys
sys.path.append("/home/dyn/project_test/pointpillars_ros/src/pointpillars_ros")

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path: 根目录
            dataset_cfg: 数据集配置
            class_names: 类别名称
            training: 训练模式
            logger: 日志
            ext: 扩展名
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        


class Pointpillars_ROS:
    def __init__(self):
        config_path, ckpt_path = self.init_ros()
        self.init_pointpillars(config_path, ckpt_path)


    def init_ros(self):
        """ Initialize ros parameters """
        config_path = rospy.get_param("/config_path", "/home/dyn/project_test/pointpillars_ros/src/pointpillars_ros/tools/cfgs/kitti_models/pointpillar.yaml")
        ckpt_path = rospy.get_param("/ckpt_path", "/home/dyn/project_test/pointpillars_ros/src/pointpillars_ros/tools/models/pointpillar_7728.pth")
        self.sub_velo = rospy.Subscriber("/rslidar_points", PointCloud2, self.lidar_callback, queue_size=1,  buff_size=2**12)
        self.pub_bbox = rospy.Publisher("/detections", BoundingBoxArray, queue_size=1)
        return config_path, ckpt_path


    def init_pointpillars(self, config_path, ckpt_path):
        """ Initialize second model """
        logger = common_utils.create_logger() # 创建日志
        logger.info('-----------------Quick Demo of Pointpillars-------------------------')
        cfg_from_yaml_file(config_path, cfg)  # 加载配置文件
        
        self.demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            ext='.bin', logger=logger
        )
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.demo_dataset)
        # 加载权重文件
        self.model.load_params_from_file(filename=ckpt_path, logger=logger, to_cpu=True)
        self.model.cuda() # 将网络放到GPU上
        self.model.eval() # 开启评估模式


    def rotate_mat(self, axis, radian):
        rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
        return rot_matrix


    def lidar_callback(self, msg):
        """ Captures pointcloud data and feed into second model for inference """
        pcl_msg = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z","intensity","ring"))
        np_p = np.array(list(pcl_msg), dtype=np.float32)
        # 旋转轴
        rand_axis = [0,1,0]
        #旋转角度
        yaw = 0.1047
        #返回旋转矩阵
        rot_matrix = self.rotate_mat(rand_axis, yaw)
        np_p_rot = np.dot(rot_matrix, np_p[:,:3].T).T
        
        # convert to xyzi point cloud
        x = np_p_rot[:, 0].reshape(-1)
        y = np_p_rot[:, 1].reshape(-1)
        z = np_p_rot[:, 2].reshape(-1)
        if np_p.shape[1] == 4: # if intensity field exists
            i = np_p[:, 3].reshape(-1)
        else:
            i = np.zeros((np_p.shape[0], 1)).reshape(-1)
        points = np.stack((x, y, z, i)).T
        print(points.shape)
        # 组装数组字典
        input_dict = {
            'points': points,
            'frame_id': 0,
        }
        data_dict = self.demo_dataset.prepare_data(data_dict=input_dict) # 数据预处理
        data_dict = self.demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict) # 将数据放到GPU上
        pred_dicts, _ = self.model.forward(data_dict) # 模型前向传播
        boxes_lidar = pred_dicts[0]['pred_boxes'].detach().cpu().numpy()
        scores = pred_dicts[0]['pred_scores'].detach().cpu().numpy()
        label = pred_dicts[0]['pred_labels'].detach().cpu().numpy()
        num_detections = boxes_lidar.shape[0]
        rospy.loginfo("The num is: %d ", num_detections)

        # print(boxes_lidar)
        # print(scores)
        # print(label)

        arr_bbox = BoundingBoxArray()
        for i in range(num_detections):
            bbox = BoundingBox()

            bbox.header.frame_id = msg.header.frame_id
            bbox.header.stamp = rospy.Time.now()
            bbox.pose.position.x = float(boxes_lidar[i][0])
            bbox.pose.position.y = float(boxes_lidar[i][1])
            bbox.pose.position.z = float(boxes_lidar[i][2]) + float(boxes_lidar[i][5]) / 2
            bbox.dimensions.x = float(boxes_lidar[i][3])  # width
            bbox.dimensions.y = float(boxes_lidar[i][4])  # length
            bbox.dimensions.z = float(boxes_lidar[i][5])  # height
            q = Quaternion(axis=(0, 0, 1), radians=float(boxes_lidar[i][6]))
            bbox.pose.orientation.x = q.x
            bbox.pose.orientation.y = q.y
            bbox.pose.orientation.z = q.z
            bbox.pose.orientation.w = q.w
            bbox.value = scores[i]
            bbox.label = label[i]

            arr_bbox.boxes.append(bbox)
        
        arr_bbox.header.frame_id = msg.header.frame_id
        arr_bbox.header.stamp = rospy.Time.now()
        
        self.pub_bbox.publish(arr_bbox)


if __name__ == '__main__':
    sec = Pointpillars_ROS()
    rospy.init_node('pointpillars_ros_node', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        del sec
        print("Shutting down")
