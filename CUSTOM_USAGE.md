# Custom Usage

This fork keeps the official UMI workflow, but adds custom data conversion, training environments, inference scripts, and debugging tools.

## Main Differences

Here are the main scripts on top of the original UMI repository:
- HDF5-to-zarr converters for lab datasets.
- Conda environments for H200, Blackwell and RTX 50-series, ROS Humble inference, and offline analysis.
- Direct inference scripts for Allen key and knob-turn experiments.
- `inference_umi.py` for ROS 2 inference and logging.
- Debug scripts that save prediction logs and exact policy input `.npz` files.
- Offline scripts that replay dataset samples or saved live observations.

## Data Conversion

Custom converters expect HDF5 files with:

```text
/obs/wrist_camera
/obs/eef_pos
/obs/eef_quat
/obs/gripper_qpos
```

And then they write UMI-style zarr keys as:

```text
camera0_rgb
robot0_eef_pos
robot0_eef_rot_axis_angle
robot0_gripper_width
robot0_demo_start_pose
robot0_demo_end_pose
meta/episode_ends
```

You can find these custom converting scripts in the `custom_data_pipeline` directory, and you can run them with the following commands:

Allen key:

```bash
python custom_data_pipeline/allen_key/hdf5_to_zarr_conversion_script.py \
  -i path/to/hdf5_folder \
  -o path/to/dataset.zarr.zip \
  --overwrite
```

Knob turn:

```bash
python custom_data_pipeline/knob_turn/hdf5_to_zarr_conversion_script.py \
  -i path/to/hdf5_or_folder \
  -o path/to/dataset.zarr.zip \
  --overwrite
```

Inspection helpers:

```text
custom_data_pipeline/inspect_hdf5.py
custom_data_pipeline/inspect_zarr.py
```

## Training

Training uses the official `train.py`. I created these new conda environments depending on the GPU used:

```bash
# H200 from NCSA
conda env create -f conda_environment_hopper_train.yaml
conda activate hopper-train

# RTX 50-series
conda env create -f conda_environment_blackwell_train.yaml
conda activate blackwell-train

# single GPU
python train.py --config-name=train_diffusion_unet_timm_umi_workspace task.dataset_path=path/to/dataset.zarr.zip
```

The training parameters can be changed in the `diffusion_policy/config/task/umi.yaml` file. Currently, the downsampling is kept at the original value of 3. However, if you are using an already downsampled dataset, change that value to 1 in `obs_down_sample_steps: 1 # 3, 1`.

Also note that the epochs are changed to 400, and the batch size is kept at 64.

## Inference

### ROS 2

`inference_umi.py` is the custom ROS 2 deployment script. It has been modified to match the new execution scripts after the parallel threads change to inference. Additionally, the `stream_realsense_utils.py` utility script is used, and the `inference_umi_reference.py` is the unmodified reference script used to create `inference_umi.py`.

To build the environment for inference, run:

```bash
conda env create -f conda_environment_ros_humble_inference.yaml
conda activate umi-ros-humble
source /opt/ros/humble/setup.bash
source ~/umi/ros2_ws/install/setup.bash
```

Then, start the low-level ROS control scripts, and you can run inference with:

```bash
python inference_umi.py --ckpt_path checkpoints/latest.ckpt --results_path data/turn_knob_ros_logs
```

Note that the gripper conversion uses `MAX_GRIPPER_WIDTH = 85.0` and `DATASET_MAX_GRIPPER_WIDTH_M = 0.09`. If another dataset is used with a different max width, then these values need to be changed.

### Direct Scripts

This repository contains inference scripts that do not require ROS 2. These scripts are in the `custom_inference_script` directory, and have the following uses:

- `custom_inference_script/inference_simple_allen_key.py`: Direct Allen key inference script
- `custom_inference_script/inference_simple_knob.py`: Direct knob-turn inference script
- `custom_inference_script/inference_simple.py`: Reference script which the above two scripts are created from.

There is also a script that moves to an initial position. The values are hard coded in the script, in `execute_command.py`.

Before running direct scripts, check checkpoint paths.

## Debugging Scripts

Certain debugging scripts that might be useful are in the `umi_debug_scripts`. The `eval_*.py` scripts are adapted from `eval_real.py` and will execute actions on the robot, while the `offline_*.py` are offline scripts.

- `umi_debug_scripts/eval_once.py`: After you press c to start, it waits for you to press n for each inference step. Each time you you press n, it does one prediction, logs it, and executes that one action horizon.

- `umi_debug_scripts/eval_pred_only.py`: Live hardware script that logs prediction summaries and can execute actions. Specifically, it runs exactly one prediction pass and executes one action horizon.

- `umi_debug_scripts/test_inference/offline_inference.py`: Offline script that runs checkpoint inference on dataset samples.

- `umi_debug_scripts/test_inference/offline_prediction.py`: Offline script that runs closed-loop-style prediction on one dataset episode, meaning it starts from the end of the last prediction horizon.

- `umi_debug_scripts/test_inference/offline_inference_robot_data.py`: Offline script that replays saved `eval_once` `.npz` inputs through the checkpoint.

## Logs And Git

| Folder | Generated by |
| --- | --- |
| `data/turn_knob_simple_logs/` | `custom_inference_script/inference_simple_knob.py` |
| `data/turn_knob_ros_logs/` | `inference_umi.py` |
| `data/eval_once_logs/` | `umi_debug_scripts/eval_once.py` |
| `data/eval_once_inference_inputs/` | `umi_debug_scripts/eval_once.py` |
| `data/eval_pred_only_logs/` | `umi_debug_scripts/eval_pred_only.py` |
| `data/offline_inference_robot_data_logs/` | `offline_inference_robot_data.py` |
