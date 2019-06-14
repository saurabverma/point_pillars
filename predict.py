'''
Description: Use this program to predict a single 

TODO:
[ ] Input to be set as a regular pointcloud
[ ] Output yet to be decided
[ ] The config file path input to be set
[ ] Include ROS support as a service

Author: Dr. Saurab VERMA (saurab_verma@i2r.a-star.edu.sg)
'''


import os
import pathlib
import pickle
import shutil
import time
from functools import partial

# import rospy
# from visualization_msgs.msg import MarkerArray
# from sensor_msgs.msg import PointCloud2

import pprint
import fire
import numpy as np
import torch
from tqdm import tqdm
from google.protobuf import text_format
from tensorboardX import SummaryWriter

import torchplus
import second.data.kitti_common as kitti
from second.builder import target_assigner_builder, voxel_builder
from second.data.preprocess import merge_second_batch
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
									  lr_scheduler_builder, optimizer_builder,
									  second_builder)
from second.utils.eval import get_coco_eval_result, get_official_eval_result
from second.utils.progress_bar import ProgressBar

from second.pytorch.train import example_convert_to_torch



def _process_output(predictions_dicts,
                   batch_image_shape,
				   class_names,
				   center_limit_range=None,
				   lidar_input=False,
				   global_set=None):
	'''Predict net output, reformat output, return'''

	annos = []

	# For each lidar/camera scan, perform
	for i, preds_dict in enumerate(predictions_dicts):
		image_shape = batch_image_shape[i]
		img_idx = preds_dict["image_idx"]

		# If atleast one prediction is made by the net, process output
		if preds_dict["bbox"] is not None:

			# Detach from Grad, GPU and tensor
			bbox = preds_dict["bbox"].detach().cpu().numpy()
			box3d_camera = preds_dict["box3d_camera"].detach().cpu().numpy()
			box3d_lidar = preds_dict["box3d_lidar"].detach().cpu().numpy()
			label_preds = preds_dict["label_preds"].detach().cpu().numpy()
			scores = preds_dict["scores"].detach().cpu().numpy()

			# Setup initial variable values
			anno = kitti.get_start_result_anno()
			num_example = 0 # Number of bounding boxes found by the net

			# Append annotations for each bounding box detection
			for bbox_camera, bbox_lidar, bbox_2d, score, label in zip(
                                box3d_camera, box3d_lidar, bbox, scores,
                                label_preds):
				if not lidar_input: # If camera data is available along with lidar input, then
					if bbox_2d[0] > image_shape[1] or bbox_2d[1] > image_shape[0]: # If bbox_2d length/breadth > camera image size, then
						continue # Stop further processing of this specific 'for' loop
					if bbox_2d[2] < 0 or bbox_2d[3] < 0: # If bbox_2d length/breadth < 0, then
						continue # Stop further processing of this specific 'for' loop

				# # DEBUG:
				# print(f'image: {image_shape[::-1]}')  # NOTE: image_shape is shape of camera images
				# print(f'bbox:  {bbox_2d}')
				print(f'bbox_lidar: {bbox_lidar}')
				print(f'score:      {score}')

				if center_limit_range is not None:
					limit_range = np.array(center_limit_range)
					if (np.any(bbox_lidar[:3] < limit_range[:3])
                                                or np.any(bbox_lidar[:3] > limit_range[3:])): # If out of limit range, then
						continue # Stop further processing of this specific 'for' loop

				bbox_2d[2:] = np.minimum(bbox_2d[2:], image_shape[::-1]) # Location must be within image boundaries
				bbox_2d[:2] = np.maximum(bbox_2d[:2], [0, 0]) # Size must >= 0

				anno["name"].append(class_names[int(label)]) # label name such as 'car'
				anno["truncated"].append(0.0) # FIXME: Not sure what is the point
				anno["occluded"].append(0)  # FIXME: Not sure what is the point
				anno["alpha"].append(-np.arctan2(-bbox_lidar[1], bbox_lidar[0]) + bbox_camera[6])
				anno["bbox"].append(bbox_2d) # 2D bounding box: x, y, length, breadth
				anno["location"].append(bbox_camera[:3]) # x, y, z
				anno["dimensions"].append(bbox_camera[3:6]) # length, breadth
				anno["rotation_y"].append(bbox_camera[6]) # angle

				# FIXME: Not sure but looks like previous scores based update can be used here
				if global_set is not None:
					for i in range(100000):
						if score in global_set:
							score -= 1 / 100000
						else:
							global_set.add(score)
							break
				anno["score"].append(score)

				num_example += 1
				print(num_example) # DEBUG:
			if num_example != 0:
				anno = {n: np.stack(v) for n, v in anno.items()}
				annos.append(anno)
			else:
				annos.append(kitti.empty_result_anno())
		else:
			# Simply an empty set of annotations
			annos.append(kitti.empty_result_anno())

		num_example = annos[-1]["name"].shape[0]
		annos[-1]["image_idx"] = np.array([img_idx] * num_example, dtype=np.int64)

	return annos



def predict(config_path,
			model_dir,
			result_path=None,
			predict_test=False,
			ckpt_path=None,
			ref_detfile=None,
			pickle_result=True,
			pub_bb=None,
            pub_lidar=None):
	''' Setup network and provide useful output '''

	####################
	# SETUP PARAMETERS #
	####################
	model_dir = pathlib.Path(model_dir)
	if predict_test:
		result_name = 'predict_test'
	else:
		result_name = 'eval_results'
	if result_path is None:
		result_path = model_dir / result_name
	else:
		result_path = pathlib.Path(result_path)
	config = pipeline_pb2.TrainEvalPipelineConfig()
	with open(config_path, "r") as f:
		proto_str = f.read()
		text_format.Merge(proto_str, config)

	# TODO: use whole pointcloud data instead of reduced pointcloud
	# TODO: store data in respective pcd and bounding box (csv) files
	# TODO: create a cpp file to read and show (n number of) pcd files with respective bounding boxes
	input_cfg = config.eval_input_reader # Read the config file data into useful structures
	model_cfg = config.model.second # Read the config file data into useful structures
	train_cfg = config.train_config # Read the config file data into useful structures
	class_names = list(input_cfg.class_names)
	center_limit_range = model_cfg.post_center_limit_range

	#########################
	# BUILD VOXEL GENERATOR #
	#########################
	voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
	bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
	box_coder = box_coder_builder.build(model_cfg.box_coder)
	target_assigner_cfg = model_cfg.target_assigner
	target_assigner = target_assigner_builder.build(target_assigner_cfg, bv_range, box_coder)

	#####################
	# NETWORK GENERATOR #
	#####################
	# Build the NN in GPU mode
	net = second_builder.build(model_cfg, voxel_generator, target_assigner)
	net.cuda()

	# Standard conversion approach if using FloatingPoint16 instead of FloatingPoint32 type of tensor
	if train_cfg.enable_mixed_precision:
		net.half()
		net.metrics_to_float()
		net.convert_norm_to_float(net)
		float_dtype = torch.float16
	else:
		float_dtype = torch.float32

	# Restore old checkpoint if possible
	if ckpt_path is None:
		torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
	else:
		torchplus.train.restore(ckpt_path, net)

	# Setup network for evaluation mode
	net.eval()

	#####################
	# DATASET GENERATOR #
	#####################
	# Dataset build for easy usage
	eval_dataset = input_reader_builder.build(
		input_cfg,
		model_cfg,
		training=False,
		voxel_generator=voxel_generator,
		target_assigner=target_assigner)
	eval_dataloader = torch.utils.data.DataLoader(
		eval_dataset,
		batch_size=input_cfg.batch_size,
		shuffle=False,
		num_workers=input_cfg.num_workers,
		pin_memory=False,
		collate_fn=merge_second_batch)

	# Further variable setup
	result_path_step = result_path / f"step_{net.get_global_step()}"
	result_path_step.mkdir(parents=True, exist_ok=True)
	t = time.time()
	dt_annos = []
	global_set = None
	print()
	print("Generate output labels...")
	bar = ProgressBar()
	bar.start(len(eval_dataset) // input_cfg.batch_size + 1)

	#################
	# NETWORK USAGE #
	#################
	# Predict a set of 'num_workers'  samples, get info and reformat data as needed
	temp_count = 0
	for example in iter(eval_dataloader):
		# pprint.pprint(example, width=1)
		# for key, value in example.items():
		# 	print(key)
		# 	print(np.shape(value))
		example = example_convert_to_torch(example, float_dtype)
		print(example['image_idx'][0])
		# pprint.pprint(example, width=1)
		# for key, value in example.items():
		# 	print(key)
		# 	print(np.shape(value))
		# # # # if pickle_result:

		# NOTE: Predict network output
		# start_time = time.time()
		predictions_dicts = net(example)

		# Publish original data
		if pub_lidar:
			data=PointCloud2()
			# TODO: Extract pointclound info from 'example' (use original kitti data file if needed) > publish
			pub_lidar.publish(data)

		# Publish network output
		if pub_bb:
			data = MarkerArray()
			# TODO: Create a wireframe 3D bounding box and, if possible, a transluscent 3D cuboid as well > publish
			pub_bb.publish(data)

		# print('Network predict time: {}'.format(time.time()-start_time))
		# pprint.pprint(predictions_dicts[0])
		# for key, value in predictions_dicts[0].items():
		# 	print(key)
		# 	print(np.shape(value))

		# dt_annos += _process_output(
		# 	predictions_dicts, example['image_shape'], class_names, center_limit_range,
		# 	model_cfg.lidar_input, global_set)
		# pprint.pprint(dt_annos[0], width=1)
		# for key, value in dt_annos[0].items():
		# 	print(key)
		# 	print(np.shape(value))
		# # # # else:
		# # # # 	predict_kitti_to_file(net, example, result_path_step, class_names,
		# # # # 						   center_limit_range, model_cfg.lidar_input)
		temp_count += 1
		if temp_count > 5:
			break
		# bar.print_bar() # Update progress

	# sec_per_example = len(eval_dataset) / (time.time() - t)
	# print(f'generate label finished({sec_per_example:.2f}/s). start eval:')

	# print(f"avg forward time per example: {net.avg_forward_time:.3f}")
	# print(f"avg postprocess time per example: {net.avg_postprocess_time:.3f}")

	# # Store the data (in a format specified by user)
	# if not predict_test:
	# 	gt_annos = [info["annos"] for info in eval_dataset.dataset.kitti_infos]
	# 	if not pickle_result:
	# 		dt_annos = kitti.get_label_annos(result_path_step) # FIXME: Not sure what is this step

	# 	result = get_official_eval_result(gt_annos, dt_annos, class_names)
	# 	print(result)

	# 	result = get_coco_eval_result(gt_annos, dt_annos, class_names)
	# 	print(result)

	# 	if pickle_result:
	# 		with open(result_path_step / "result.pkl", 'wb') as f:
	# 			pickle.dump(dt_annos, f)



# def ros_predict(config_path,
#                 model_dir,
#                 result_path=None,
#                 predict_test=False,
#                 ckpt_path=None,
#                 ref_detfile=None,
#                 pickle_result=True):

# 	rospy.init_node('PointPillars')
# 	pub_bb = rospy.Publisher('lidar_segments', MarkerArray, queue_size=1)
# 	pub_lidar = rospy.Publisher('lidar_segments', PointCloud2, queue_size=1)

# 	predict(config_path, model_dir, result_path, predict_test,
# 	        ckpt_path, ref_detfile, pickle_result, pub_bb, pub_lidar)



if __name__ == '__main__':
	fire.Fire()
