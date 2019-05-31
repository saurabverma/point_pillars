'''
Description: Use this program to predict a single 

TODO:
[ ] Input to be set as a regular pointcloud
[ ] Output yet to be decided
[ ] The config file path input to be set
[ ] Include ROS support as a service

Author: Dr. Saurab VERMA (saurab_verma@i2r.a-star.edu.sg)
'''


# import os
# import pathlib
# import pickle
# import shutil
# import time
# from functools import partial

# import fire
# import numpy as np
# import torch
# from tqdm import tqdm
# from google.protobuf import text_format
# from tensorboardX import SummaryWriter

# import torchplus
# import second.data.kitti_common as kitti
# from second.builder import target_assigner_builder, voxel_builder
# from second.data.preprocess import merge_second_batch
# from second.protos import pipeline_pb2
# from second.pytorch.builder import (box_coder_builder, input_reader_builder,
# 									  lr_scheduler_builder, optimizer_builder,
# 									  second_builder)
# from second.utils.eval import get_coco_eval_result, get_official_eval_result
# from second.utils.progress_bar import ProgressBar

from second.pytorch.train import predict_kitti_to_anno

def evaluate(config_path,
			model_dir,
			result_path=None,
			predict_test=False,
			ckpt_path=None,
			ref_detfile=None,
			pickle_result=True):
	# Evaluate on a subset of kitti dataset

	# Setup parameters
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

	input_cfg = config.eval_input_reader
	model_cfg = config.model.second
	train_cfg = config.train_config
	class_names = list(input_cfg.class_names)
	center_limit_range = model_cfg.post_center_limit_range

	######################
	# BUILD VOXEL GENERATOR
	######################
	voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
	bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
	box_coder = box_coder_builder.build(model_cfg.box_coder)
	target_assigner_cfg = model_cfg.target_assigner
	target_assigner = target_assigner_builder.build(target_assigner_cfg, bv_range, box_coder)

	# Build the NN in GPU mode
	net = second_builder.build(model_cfg, voxel_generator, target_assigner)
	net.cuda()

	# Further net settings
	if train_cfg.enable_mixed_precision:
		net.half()
		net.metrics_to_float()
		net.convert_norm_to_float(net)

	# Restore old checkpoint if possible
	if ckpt_path is None:
		torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
	else:
		torchplus.train.restore(ckpt_path, net)

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
	if train_cfg.enable_mixed_precision:
		float_dtype = torch.float16
	else:
		float_dtype = torch.float32

	# Setup network for evaluation
	net.eval()

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

	# Predict each sample info and reformat data as needed
	for example in iter(eval_dataloader):
		example = _example_convert_to_torch(example, float_dtype)
		if pickle_result:
			dt_annos += predict_kitti_to_anno(
				net, example, class_names, center_limit_range,
				model_cfg.lidar_input, global_set)
		else:
			_predict_kitti_to_file(net, example, result_path_step, class_names,
								   center_limit_range, model_cfg.lidar_input)
		bar.print_bar() # Update progress
		break

	sec_per_example = len(eval_dataset) / (time.time() - t)
	print(f'generate label finished({sec_per_example:.2f}/s). start eval:')

	print(f"avg forward time per example: {net.avg_forward_time:.3f}")
	print(f"avg postprocess time per example: {net.avg_postprocess_time:.3f}")

	# Store the data (in a format specified by user)
	if not predict_test:
		gt_annos = [info["annos"] for info in eval_dataset.dataset.kitti_infos]
		if not pickle_result:
			dt_annos = kitti.get_label_annos(result_path_step) # FIXME: Not sure what is this step

		result = get_official_eval_result(gt_annos, dt_annos, class_names)
		print(result)

		result = get_coco_eval_result(gt_annos, dt_annos, class_names)
		print(result)

		if pickle_result:
			with open(result_path_step / "result.pkl", 'wb') as f:
				pickle.dump(dt_annos, f)


if __name__ == '__main__':
	fire.Fire()
