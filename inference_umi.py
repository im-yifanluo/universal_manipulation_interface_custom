"""
CUSTOM: This is the custom ROS 2 inference script
"""

import time
import argparse
import json
import cv2
import numpy as np
import torch
import dill
import hydra
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import os
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.real_world.real_inference_util import get_real_umi_obs_dict, get_real_umi_action

import threading
from stream_realsense_utils import RealsenseVideoStream

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from std_msgs.msg import Float32MultiArray
from rocky_scripts_msgs.msg import StampedFloat32MultiArray


H = 240
W = 320
FPS = 60
MAX_GRIPPER_WIDTH = 85.
DATASET_MAX_GRIPPER_WIDTH_M = 0.09
CONTROL_FREQUENCY = 60.
DATA_FREQUENCY = 10.
OBS_HISTORY_INTERVAL = 1./DATA_FREQUENCY

# how often inference_callback polls for the "run inference" signal
SUB_POLL_FREQUENCY = 60.

INFERENCE_LATENCY_STEPS = 1
HARDWARE_LATENCY_STEPS = 2


def interpolate_between_actions (start_action, end_action, interpolation_factor):
    '''
    This function interpolate between a start action and an end action to generate a trajectory
    with length interpolation_factor, including the end action but not the start action.

    Action has format [x, y, z, rx, ry, rz, gripper_width].
    '''

    # decode
    start_pos = start_action[0:3]
    start_rotvec = start_action[3:6]
    start_width = start_action[6]
    end_pos = end_action[0:3]
    end_rotvec = end_action[3:6]
    end_width = end_action[6]

    # rotation interpolation
    rotations = R.from_rotvec([start_rotvec, end_rotvec])
    key_times = [0, 1]
    slerp = Slerp(key_times, rotations)

    action_interpolated = []

    for i in range (1, int(interpolation_factor)):
        next_pos = (end_pos - start_pos) * i / interpolation_factor + start_pos
        next_rotvec = slerp(float(i) / interpolation_factor).as_rotvec()
        next_width = (end_width - start_width) * i / interpolation_factor + start_width

        next_action = next_pos.tolist() + next_rotvec.tolist() + [next_width]
        action_interpolated.append(next_action.copy())

    action_interpolated.append(end_action.copy())

    return action_interpolated


def wsg_width_mm_to_policy_m(width_mm):
    # Training zarr was built from normalized HDF5 qpos:
    #   robot0_gripper_width = gripper_qpos * DATASET_MAX_GRIPPER_WIDTH_M
    # where gripper_qpos = wsg_width_mm / MAX_GRIPPER_WIDTH.
    width_normalized = width_mm / MAX_GRIPPER_WIDTH
    width_m = width_normalized * DATASET_MAX_GRIPPER_WIDTH_M
    return np.float32(np.clip(width_m, 0.0, DATASET_MAX_GRIPPER_WIDTH_M))


def policy_width_m_to_wsg_mm(width_m):
    width_normalized = width_m / DATASET_MAX_GRIPPER_WIDTH_M
    width_mm = width_normalized * MAX_GRIPPER_WIDTH
    return float(np.clip(width_mm, 0.0, MAX_GRIPPER_WIDTH))


class DiffusionPolicyInference (Node):
    
    def __init__ (self, ckpt_path, results_path, show_wrist=False):

        super().__init__('diffusion_policy_inference')

        self.show_wrist = show_wrist

        # Dedicated callback groups so the MultiThreadedExecutor runs these callbacks
        # concurrently; in particular the heavy inference_callback runs alongside
        # (instead of blocking) the CONTROL_FREQUENCY publishing in callback_pub.
        self.inference_cb_group = MutuallyExclusiveCallbackGroup()
        self.interpolation_cb_group = MutuallyExclusiveCallbackGroup()
        self.sub_cb_group = MutuallyExclusiveCallbackGroup()

        # Guards the inference <-> interpolation handshake (cur/next traj, the ready and
        # finished flags, last_action, traj_idx).
        self._state_lock = threading.Lock()

        if not os.path.exists(results_path):
            os.makedirs(results_path)

        # to prevent overwriting
        episode_id = 0
        while os.path.exists('{}/{}'.format(results_path, episode_id)):
            episode_id += 1
        self.episode_path = '{}/{}'.format(results_path, episode_id)
        os.makedirs(self.episode_path)
        self.prediction_debug_path = '{}/prediction_debug.jsonl'.format(self.episode_path)
        open(self.prediction_debug_path, 'w').close()


        # ===== realsense initialization =====

        # umi setup only has wrist camera. Latest-frame grabber on its own thread:
        # the blocking wait_for_frames() happens inside the stream, so callbacks only
        # do non-blocking read()s. NOTE: the stream delivers BGR frames; convert to
        # RGB before feeding the policy (camera0_rgb is RGB).
        self.realsense_stream = RealsenseVideoStream(width=W*2, height=H*2, fps=FPS)

        # latest wrist frame (BGR) for the live view, rendered by the main thread (see
        # get_vis_frame / main); highgui is not thread-safe so callbacks never draw.
        self.vis_frame = None


        # ===== diffusion policy init =====

        # load checkpoint
        payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill, weights_only=False)
        cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)

        # if need to overwrite
        # cfg['policy']['num_inference_steps'] = 10
        self.n_action_steps = 8
        self.n_latency_steps = INFERENCE_LATENCY_STEPS  # about 0.1 seconds, should be more than enough
        # chunk ends at default + latency, matching inference_umi_reference.py
        # (cfg['policy']['n_action_steps'] there); HARDWARE_LATENCY_STEPS only moves the
        # stitch entry point, it does not extend the chunk.
        self.model_action_steps = self.n_action_steps + self.n_latency_steps
        self.action_down_sample_steps = int(
            cfg.task.shape_meta.action.get('down_sample_steps', 1)
        )
        self.policy_action_frequency = DATA_FREQUENCY
        # DiffusionUnetTimmPolicy returns the full action horizon and has no
        # policy.n_action_steps config key; crop to model_action_steps after inference.

        self.cfg = cfg
        self.obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
        self.action_pose_repr = cfg.task.pose_repr.action_pose_repr
        self.episode_start_pose = None

        # for interpolation and latency handling
        self.counter = 0
        self.traj_idx = 0
        self.finished_cur_action_chunk = False
        self.finished_cur_traj = False
        self.next_traj_ready = False
        self.last_action = None  # this is the last action in the previous action chunk
        self.interpolation_factor = CONTROL_FREQUENCY / DATA_FREQUENCY
        self.cur_traj = []
        self.next_traj = []

        self.get_logger().info(
            'Using policy action frequency {:.3f} Hz from DATA_FREQUENCY {} Hz. '
            'Checkpoint action.down_sample_steps is {}. Interpolation factor: {}.'.format(
                self.policy_action_frequency,
                DATA_FREQUENCY,
                self.action_down_sample_steps,
                self.interpolation_factor,
            )
        )

        # for trajectory logging
        self.trajectory_received = []
        self.trajectory_executed = []
        self.cur_robot_pose = None
        self.cur_gripper_width = None
        self.timestep_limit = 1200
        self.eef_positions = []
        self.predicted_action_chunks = []
        self.wrist_images = []

        workspace = cls(cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # get policy from workspace
        self.policy = workspace.model
        if cfg.training.use_ema:
            self.policy = workspace.ema_model
        
        self.policy.to(torch.device('cuda:0'))
        self.policy.eval()
        self.device = self.policy.device


        # ===== ros init =====

        # Proprioception messages carry [history, current] stacked (see the callbacks).
        # Their own group keeps pose/gripper updates flowing while inference runs.
        self.pose_sub = self.create_subscription(
            StampedFloat32MultiArray, 'robot_pose_pair', self.robot_pose_callback, 1,
            callback_group=self.sub_cb_group)
        self.gripper_width_sub = self.create_subscription(
            StampedFloat32MultiArray, 'gripper_width_pair', self.gripper_width_callback, 1,
            callback_group=self.sub_cb_group)

        self.robot_trajectory_pub = self.create_publisher(Float32MultiArray, 'robot_target_pose', 1)
        self.gripper_trajectory_pub = self.create_publisher(Float32MultiArray, 'gripper_target_width', 1)

        # inference_callback polls for the "run inference" signal and does the heavy
        # policy prediction; callback_pub publishes the interpolated trajectory at
        # CONTROL_FREQUENCY and must never be blocked by inference.
        self.inference_timer = self.create_timer(
            1./SUB_POLL_FREQUENCY, self.inference_callback, callback_group=self.inference_cb_group)
        self.timer = self.create_timer(
            1./CONTROL_FREQUENCY, self.callback_pub, callback_group=self.interpolation_cb_group)

        self.last_inference_timestamp = float(self.get_clock().now().nanoseconds) / 1e9


    def robot_pose_callback (self, pose_msg):
        # data is [history, current] flattened: [x,y,z,rx,ry,rz] x 2 -> length 12
        self.cur_robot_pose = np.array(pose_msg.data, dtype=np.float32)


    def gripper_width_callback (self, gripper_width_msg):
        # data is [width_history, width_current] -> length 2, WSG mm
        self.cur_gripper_width = np.array(gripper_width_msg.data, dtype=np.float32)


    def get_vis_frame (self):
        # Latest wrist frame for the main thread's GUI loop. The attribute is only ever
        # rebound (never mutated in place), so this needs no lock.
        return self.vis_frame


    def inference_callback (self):
        # Polls at SUB_POLL_FREQUENCY and runs the heavy policy prediction when the
        # current action chunk is done. Runs in its own callback group so callback_pub
        # keeps publishing while this is computing.

        if self.cur_robot_pose is None or self.cur_gripper_width is None:
            self.get_logger().warn('Waiting for robot proprioception messages to arrive')
            return

        # ===== compose the live view + save the frame (cheap, every tick) =====
        # Only store the frame here; main() does the imshow/waitKey.
        latest = self.realsense_stream.read()
        if latest is not None:
            wrist_latest, _ = latest
            self.vis_frame = wrist_latest
            # BGR, which is what cv2.VideoWriter expects
            self.wrist_images.append(wrist_latest.copy())

        # ===== decide whether we need to compute the next trajectory =====
        # Only infer when no next traj is already queued AND either inference has not
        # started yet or callback_pub signalled the chunk is done.
        with self._state_lock:
            need_inference = (
                not self.next_traj_ready
                and (self.finished_cur_action_chunk or len(self.cur_traj) == 0)
            )
            # callback_pub cannot swap while next_traj_ready is False, so cur_traj
            # (and hence this decision) is stable for the rest of the call.
            is_first_chunk = len(self.cur_traj) == 0
        if not need_inference:
            return

        # ===== gather the [history, current] obs from the stream + messages =====
        realsense_history = self.realsense_stream.read_history(interval=OBS_HISTORY_INTERVAL)
        if realsense_history is None:
            # not enough buffered camera data yet
            return
        past_frame, _, latest_frame, _ = realsense_history

        # snapshot the proprioception messages ([history, current] stacked)
        robot_pose_stacked = self.cur_robot_pose.reshape(2, 6)          # [hist, cur]
        gripper_width_stacked = self.cur_gripper_width.reshape(2)       # [hist, cur], WSG mm

        # stream frames are BGR; policy obs are RGB. env_obs images stay uint8 HWC —
        # get_real_umi_obs_dict handles resize/normalization.
        wrist_images_rgb = np.stack((
            cv2.cvtColor(past_frame, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB),
        ), axis=0)

        gripper_width_m = np.array(
            [[wsg_width_mm_to_policy_m(width_mm)] for width_mm in gripper_width_stacked],
            dtype=np.float32,
        )  # (2, 1)

        env_obs = {
            "camera0_rgb": wrist_images_rgb,
            "robot0_eef_pos": robot_pose_stacked[:, 0:3].astype(np.float32),
            "robot0_eef_rot_axis_angle": robot_pose_stacked[:, 3:6].astype(np.float32),
            "robot0_gripper_width": gripper_width_m,
        }

        # current obs values (for logging / seam verification)
        eef_pos = env_obs["robot0_eef_pos"][-1]
        robot_pose = robot_pose_stacked[-1].astype(np.float32)
        gripper_width = float(gripper_width_stacked[-1])

        if self.episode_start_pose is None:
            self.episode_start_pose = [
                np.concatenate([
                    env_obs["robot0_eef_pos"][-1],
                    env_obs["robot0_eef_rot_axis_angle"][-1],
                ])
            ]

        # need_inference was already checked at the top; finished_cur_action_chunk is
        # reset by callback_pub when it swaps in the new trajectory.

        # # get frequency for debug
        # # note: this frequency is how often inference is run, which only happens once per action chunk execution
        # # it is not the same as how fast the model can predict
        # cur_timestamp = float(self.get_clock().now().nanoseconds) / 1e9
        # cur_fps = 1./(cur_timestamp - self.last_inference_timestamp)
        # self.last_inference_timestamp = float(self.get_clock().now().nanoseconds) / 1e9
        # self.get_logger().info("Running inference at {} fps".format(cur_fps))

        obs_dict_np = get_real_umi_obs_dict(
            env_obs=env_obs,
            shape_meta=self.cfg.task.shape_meta,
            obs_pose_repr=self.obs_pose_rep,
            tx_robot1_robot0=None,
            episode_start_pose=self.episode_start_pose,
        )

        # device transfer
        obs_dict = dict_apply(
            obs_dict_np,
            lambda x: torch.from_numpy(x).unsqueeze(0).to(device=self.device),
        )

        # run policy
        with torch.no_grad():
            action_dict = self.policy.predict_action(obs_dict)

        raw_action = action_dict["action"][0].detach().to("cpu").numpy()
        action_chunk = get_real_umi_action(raw_action, env_obs, self.action_pose_repr)
        if len(action_chunk) > self.model_action_steps:
            self.get_logger().warning(
                "Policy returned {} action steps after n_action_steps override; "
                "cropping to {} to match inference_umi_reference.py.".format(
                    len(action_chunk), self.model_action_steps
                ),
                once=True,
            )
            action_chunk = action_chunk[:self.model_action_steps]
        action_chunk = action_chunk.copy()

        if not np.all(np.isfinite(action_chunk)):
            raise RuntimeError('Nan or Inf action')

        # # debug
        # print(np.round(robot_pose, 4))
        # print(np.round(action_chunk, 4))
        # rclpy.shutdown()
        # eef_pos_vis = vedo.Point(robot_pose[0:3], r=10, c='red')
        # traj_vis = vedo.Points(action_chunk[:, 0:3], r=5, c='green')
        # plotter = vedo.Plotter(axes=1)
        # plotter.show(eef_pos_vis, traj_vis)

        # logging
        self.eef_positions.append(eef_pos)
        self.predicted_action_chunks.append(action_chunk.copy())

        # Convert UMI policy gripper width in meters back to the WSG command in mm.
        action_chunk[:, -1] = np.array([
            policy_width_m_to_wsg_mm(width_m) for width_m in action_chunk[:, -1]
        ], dtype=np.float32)

        # interpolate trajectory
        self.next_traj = []

        # if this is the first action chunk, there is no latency
        if is_first_chunk:
            # interpolation starts from the start of the first action chunk
            self.next_traj.append(action_chunk[0].tolist())

            for i in range (len(action_chunk)-1):
                action_interpolated = interpolate_between_actions(action_chunk[i], action_chunk[i+1], self.interpolation_factor)
                self.next_traj += action_interpolated

        # otherwise, interpolate between the last action and cur_action_chunk[n_latency_steps]
        else:
            # self.last_action is not included as part of the current traj because it was already included in the last traj
            # enter at n_latency_steps + HARDWARE_LATENCY_STEPS so the entry lands ahead of the lagging robot
            entry_idx = self.n_latency_steps + HARDWARE_LATENCY_STEPS
            action_interpolated = interpolate_between_actions(self.last_action, action_chunk[entry_idx], self.interpolation_factor)
            self.next_traj += action_interpolated

            # # to make actions smoother, interpolate from current robot pose
            # interp_start = np.array(self.cur_robot_pose.tolist() + [self.cur_gripper_width])
            # action_interpolated = interpolate_between_actions(interp_start, action_chunk[self.n_latency_steps], self.interpolation_factor)
            # self.next_traj += action_interpolated

            # interpolate the rest of cur_action_chunk
            for i in range (entry_idx, len(action_chunk)-1):
                action_interpolated = interpolate_between_actions(action_chunk[i], action_chunk[i+1], self.interpolation_factor)
                self.next_traj += action_interpolated

            # self.get_logger().info('eef_pos: {}; 1st point in action_chunk: {}; 1st point in next_traj: {}; last_action: {}'.format(
            #     np.round(eef_pos, 4),
            #     np.round(action_chunk[0, 0:3], 4),
            #     np.round(self.next_traj[0][0:3], 4),
            #     np.round(self.last_action[0:3], 4)
            # ))

        # self.get_logger().info('next_traj length: {}'.format(len(self.next_traj)))
        # self.get_logger().info('eef pos: {}'.format(eef_pos))
        # self.get_logger().info('action chunk: {}'.format(action_chunk[:, 0:3]))

        try:
            action_pos_steps = np.linalg.norm(np.diff(action_chunk[:, :3], axis=0), axis=1)
            traj_arr = np.asarray(self.next_traj, dtype=np.float32)
            traj_pos_steps = (
                np.linalg.norm(np.diff(traj_arr[:, :3], axis=0), axis=1)
                if len(traj_arr) > 1
                else np.asarray([], dtype=np.float32)
            )
            debug_record = {
                "time": time.time(),
                "counter": int(self.counter),
                "raw_action_steps": int(raw_action.shape[0]),
                "selected_action_steps": int(action_chunk.shape[0]),
                "base_data_frequency": float(DATA_FREQUENCY),
                "action_down_sample_steps": int(self.action_down_sample_steps),
                "policy_action_frequency": float(self.policy_action_frequency),
                "interpolation_factor": float(self.interpolation_factor),
                "eef_pos": eef_pos.astype(float).tolist(),
                # full observed state at this inference (for rot/gripper seam verification)
                "obs_robot_pose": robot_pose.astype(float).tolist(),
                "obs_gripper_width_mm": float(gripper_width),
                "first_target_pos": action_chunk[0, :3].astype(float).tolist(),
                "first_target_dist_m": float(np.linalg.norm(action_chunk[0, :3] - eef_pos)),
                "last_action_dist_m": (
                    None
                    if self.last_action is None
                    else float(np.linalg.norm(self.last_action[:3] - eef_pos))
                ),
                "action_step_mean_m": float(np.mean(action_pos_steps)) if len(action_pos_steps) else 0.0,
                "action_step_max_m": float(np.max(action_pos_steps)) if len(action_pos_steps) else 0.0,
                "traj_len": int(len(self.next_traj)),
                "traj_step_mean_m": float(np.mean(traj_pos_steps)) if len(traj_pos_steps) else 0.0,
                "traj_step_max_m": float(np.max(traj_pos_steps)) if len(traj_pos_steps) else 0.0,
                "action_chunk_pos": action_chunk[:, :3].astype(float).tolist(),
                # full predicted action chunk, exactly as fed to the stitch/interpolation.
                # columns: [0:3]=eef pos (m), [3:6]=eef rot axis-angle (rad), [6]=gripper (WSG mm).
                # consecutive records reconstruct the stitch: prev["action_chunk_full"][-1]
                # (last_action) -> this record's action_chunk_full[n_latency_steps + HARDWARE_LATENCY_STEPS].
                "action_chunk_full": action_chunk.astype(float).tolist(),
            }
            with open(self.prediction_debug_path, 'a') as f:
                f.write(json.dumps(debug_record) + '\n')
        except Exception as e:
            self.get_logger().warning('Prediction debug logging failed: {}'.format(e), once=True)

        # publish the freshly computed trajectory for callback_pub to pick up
        with self._state_lock:
            self.last_action = action_chunk[-1].copy()  # this would be the last latency step
            self.next_traj_ready = True

        self.get_logger().info('Predicted next action chunk.')

    
    def callback_pub (self):

        # determine the trajectory to execute
        # if finished the current trajectory and the next trajectory is ready, update cur_traj to be next_traj
        # this should always be the case, since n_latency_steps * 1./DATA_FREQUENCY > inference time
        # Must be atomic w.r.t. inference_callback, which publishes next_traj.
        with self._state_lock:
            if (self.finished_cur_traj or len(self.cur_traj) == 0) and self.next_traj_ready:
                self.cur_traj = self.next_traj
                # reset
                self.next_traj = []
                self.finished_cur_traj = False
                self.next_traj_ready = False
                self.finished_cur_action_chunk = False
                self.traj_idx = 0

            # snapshot handshake state for the decisions below
            cur_traj_empty = len(self.cur_traj) == 0
            finished_cur_traj = self.finished_cur_traj
            next_traj_ready = self.next_traj_ready

        # conditions for the robot to not do anything:
        # 1. cur_traj is empty (inference has not started) OR
        # 2. finished cur traj but next traj is not ready
        if cur_traj_empty:
            self.get_logger().warning('Inference has not started.', once=True)

        # if finished executing current traj but the next traj is not ready yet, give warning
        elif finished_cur_traj and not next_traj_ready:
            self.get_logger().warning('Finished current trajectory, but the next trajectory is not ready yet!')

        # otherwise, execute trajectory as usual
        # cur_traj / traj_idx are only rebound during the swap above, and callback_pub
        # is mutually exclusive with itself, so they are stable here.
        else:  # would not enter this statement if self.finished_cur_traj == True
            cur_action = self.cur_traj[self.traj_idx]
            robot_target_pose = cur_action[0:6]
            gripper_target_width = cur_action[6]

            robot_target_pose_msg = Float32MultiArray()
            robot_target_pose_msg.data = list(map(float, robot_target_pose)) + [float(self.counter)]  # add counter as part of the message to check delay
            self.robot_trajectory_pub.publish(robot_target_pose_msg)

            gripper_target_width_msg = Float32MultiArray()
            gripper_target_width_msg.data = [float(gripper_target_width), float(self.counter)]
            self.gripper_trajectory_pub.publish(gripper_target_width_msg)

            cur_timestamp = float(self.get_clock().now().nanoseconds) / 1e9
            self.get_logger().info("Published trajectory point {} at time {}".format(self.counter, cur_timestamp))
            
            # log trajectory
            # append received trajectory
            self.trajectory_received.append(cur_action)
            
            # append executed trajectory (messages are [history, current] stacked; log current)
            self.trajectory_executed.append(self.cur_robot_pose[6:12].tolist() + [float(self.cur_gripper_width[-1])])

            if len(self.trajectory_received) >= self.timestep_limit:
                # create episode folder
                if not os.path.exists(self.episode_path):
                    os.makedirs(self.episode_path)

                np.savez(
                    '{}/trajectory.npz'.format(self.episode_path), 
                    trajectory_received = self.trajectory_received,
                    trajectory_executed = self.trajectory_executed,
                )
                np.savez(
                    '{}/predictions.npz'.format(self.episode_path),
                    eef_positions = self.eef_positions,
                    predicted_action_chunks = self.predicted_action_chunks,
                )

                # write video
                wrist_cam_writer = cv2.VideoWriter(
                    '{}/wrist.mp4'.format(self.episode_path), 
                    cv2.VideoWriter_fourcc(*'mp4v') , 
                    FPS, 
                    (W*2, H*2),
                )
                self.get_logger().info('Saving images to mp4...')
                for wrist in self.wrist_images:
                    wrist_cam_writer.write(wrist)
                wrist_cam_writer.release()

                self.get_logger().info('Saved episode data.')
                rclpy.shutdown()
            
            # updates
            self.traj_idx += 1
            self.counter += 1

            # check status
            # if this is the first trajectory, the length of the trajectory should be:
            #   (n_action_steps + n_latency_steps - 1) * interpolation_factor + 1;
            # and the number of steps excluding the latency steps should be:
            #   (n_action_steps - 1) * interpolation_factor + 1.
            # if this is not the first trajectory, the length of the trajectory should be:
            #   (n_action_steps - 1) * interpolation_factor (the -1 is because of 0.1s robot lookahead time)
            # and the number of steps exluding the latency steps should be:
            #   (n_action_steps - 1 - n_latency_steps) * interpolation_factor

            if len(self.cur_traj) == (self.n_action_steps + self.n_latency_steps - 1) * self.interpolation_factor + 1:  # first trajectory
                if self.traj_idx == (self.n_action_steps - 1) * self.interpolation_factor + 1:  # use == and not >= to trigger this only once
                    with self._state_lock:
                        self.finished_cur_action_chunk = True  # this is the signal that inference should run again
                    self.get_logger().info('Signaling next inference at trajectory index {}.'.format(self.traj_idx))
            else:
                if self.traj_idx == (self.n_action_steps - HARDWARE_LATENCY_STEPS - self.n_latency_steps) * self.interpolation_factor:
                    with self._state_lock:
                        self.finished_cur_action_chunk = True
                    self.get_logger().info('Signaling next inference at trajectory index {}.'.format(self.traj_idx))

            if self.traj_idx >= len(self.cur_traj):
                with self._state_lock:
                    self.finished_cur_traj = True


def main (args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', '--ckpt-path', dest='ckpt_path', default='models/latest(1).ckpt')
    parser.add_argument('--results_path', '--results-path', dest='results_path', default='data/turn_knob_ros_logs')
    parser.add_argument('--show_wrist', '--show-wrist', dest='show_wrist', action='store_true')
    parsed_args, ros_args = parser.parse_known_args(args)

    rclpy.init(args=ros_args)
    dp_inference_node = DiffusionPolicyInference(parsed_args.ckpt_path, parsed_args.results_path, parsed_args.show_wrist)

    # MultiThreadedExecutor so the heavy inference_callback does not block the
    # callback_pub publishing (each lives in its own group).
    #
    # It spins on a background thread so the main thread can own the OpenCV GUI.
    # highgui is not thread-safe: the executor hands each callback to whichever worker
    # is free, so calling imshow/waitKey from a callback deadlocks as soon as the
    # callback lands on a different thread than the one that created the window.
    executor = MultiThreadedExecutor()
    executor.add_node(dp_inference_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            if dp_inference_node.show_wrist:
                vis_frame = dp_inference_node.get_vis_frame()
                if vis_frame is not None:
                    # stream frames are BGR, which is what imshow expects
                    cv2.imshow('wrist', cv2.resize(vis_frame, (W, H)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        dp_inference_node.realsense_stream.stop()
        dp_inference_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
