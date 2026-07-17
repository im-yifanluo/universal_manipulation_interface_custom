import cv2
import numpy as np
import torch
import dill
import hydra
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import copy
import os
from diffusion_policy.common.pose_repr_util import compute_relative_pose
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.model.common.rotation_transformer import RotationTransformer, RotationTransformerUMI

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
MAX_GRIPPER_WIDTH = 82.
CONTROL_FREQUENCY = 24.
INFERENCE_FREQUENCY = 12.
OBS_HISTORY_INTERVAL = 1./INFERENCE_FREQUENCY

# how often inference_callback polls for the "run inference" signal
SUB_POLL_FREQUENCY = 60.

INFERENCE_LATENCY_STEPS = 1
HARDWARE_LATENCY_STEPS = 3


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


class DiffusionPolicyInference (Node):
    
    def __init__ (self, ckpt_path, results_path):

        super().__init__('diffusion_policy_inference')

        # Dedicated callback groups so the MultiThreadedExecutor runs these callbacks
        # concurrently; in particular the heavy inference_callback runs alongside
        # (instead of blocking) the CONTROL_FREQUENCY publishing in interpolation_callback.
        self.inference_cb_group = MutuallyExclusiveCallbackGroup()
        self.interpolation_cb_group = MutuallyExclusiveCallbackGroup()
        self.sub_cb_group = MutuallyExclusiveCallbackGroup()

        # Guards the inference <-> interpolation handshake (cur/next traj, the ready and
        # finished flags, last_action, traj_idx).
        self._state_lock = threading.Lock()


        # results are only logged / saved when a results_path is configured
        self.results_path = results_path
        if results_path is not None:
            if not os.path.exists(results_path):
                os.makedirs(results_path)

            # to prevent overwriting
            episode_id = len(os.listdir(results_path))
            self.episode_path = '{}/{}'.format(results_path, episode_id)


        # ===== realsense initialization =====

        # umi setup only has wrist camera. Latest-frame grabber with a short rolling
        # history buffer: read_history() returns [past, latest] for the two obs steps,
        # and the blocking wait_for_frames() happens on the stream's own thread instead
        # of inside a callback.
        self.realsense_stream = RealsenseVideoStream(width=W, height=H, fps=FPS)


        # ===== diffusion policy init =====

        # load checkpoint
        payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)

        # if need to overwrite
        # cfg['policy']['num_inference_steps'] = 10
        # self.n_action_steps = cfg['n_action_steps']
        self.n_action_steps = 8
        self.n_latency_steps = INFERENCE_LATENCY_STEPS  # about 0.1 seconds, should be more than enough
        cfg['policy']['n_action_steps'] = self.n_action_steps + self.n_latency_steps  # overwrite

        # for interpolation and latency handling
        self.counter = 0
        self.traj_idx = 0
        self.finished_cur_action_chunk = False
        self.finished_cur_traj = False
        self.next_traj_ready = False
        self.last_action = None  # this is the last action in the previous action chunk
        self.interpolation_factor = CONTROL_FREQUENCY / INFERENCE_FREQUENCY
        self.cur_traj = []
        self.next_traj = []

        # composed frame for the live view, rendered by the main thread (see
        # get_vis_frame / main); highgui is not thread-safe so callbacks never draw.
        self.vis_frame = None

        # for trajectory logging
        self.trajectory_received = []
        self.trajectory_executed = []
        self.cur_robot_pose = None
        self.cur_gripper_width = None
        self.timestep_limit = 120
        self.eef_positions = []
        self.predicted_action_chunks = []
        # self.wrist_images = []

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

        self.rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        self.rot_quat2mat = RotationTransformerUMI(
            from_rep='quaternion',
            to_rep='matrix')
        self.rot_aa2mat = RotationTransformerUMI(
            from_rep='axis_angle',
            to_rep='matrix')
        self.rot_mat2rot6d = RotationTransformerUMI(
            from_rep='matrix',
            to_rep='rotation_6d')


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
        # policy prediction; interpolation_callback publishes the interpolated
        # trajectory at CONTROL_FREQUENCY and must never be blocked by inference.
        self.inference_timer = self.create_timer(
            1./SUB_POLL_FREQUENCY, self.inference_callback, callback_group=self.inference_cb_group)
        self.interpolation_timer = self.create_timer(
            1./CONTROL_FREQUENCY, self.interpolation_callback, callback_group=self.interpolation_cb_group)

        self.last_inference_timestamp = float(self.get_clock().now().nanoseconds) / 1e9


    def robot_pose_callback (self, pose_msg):
        # data is [history, current] flattened: [x,y,z,rx,ry,rz] x 2 -> length 12
        self.cur_robot_pose = np.array(pose_msg.data)


    def gripper_width_callback (self, gripper_width_msg):
        # data is [width_history, width_current] -> length 2
        self.cur_gripper_width = np.array(gripper_width_msg.data)


    def get_vis_frame (self):
        # Latest wrist frame for the main thread's GUI loop. The attribute is only ever
        # rebound (never mutated in place), so this needs no lock.
        return self.vis_frame


    def inference_callback (self):
        # Polls at SUB_POLL_FREQUENCY and runs the heavy policy prediction when the
        # current action chunk is done. Runs in its own callback group so
        # interpolation_callback keeps publishing while this is computing.

        if self.cur_robot_pose is None or self.cur_gripper_width is None:
            self.get_logger().warn('Waiting for robot proprioception messages to arrive')
            return

        # ===== compose the live view + save the frame (cheap, every tick) =====
        # Only build the frame here; main() does the imshow/waitKey.
        latest = self.realsense_stream.read()
        if latest is not None:
            wrist_latest, _ = latest
            self.vis_frame = wrist_latest
            # if self.results_path is not None:
            #     self.wrist_images.append(wrist_latest.copy())

        # ===== decide whether we need to compute the next trajectory =====
        # Only infer when no next traj is already queued AND either inference has not
        # started yet or interpolation_callback signalled the chunk is done.
        with self._state_lock:
            need_inference = (
                not self.next_traj_ready
                and (self.finished_cur_action_chunk or len(self.cur_traj) == 0)
            )
            # interpolation_callback cannot swap while next_traj_ready is False, so
            # cur_traj (and hence this branch decision) is stable for the rest of the call.
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
        gripper_width_stacked = self.cur_gripper_width.reshape(2)       # [hist, cur]

        # to account for the two steps of observations
        # shapes to match:
        # - wrist_camera: torch.Size([1, 2, 3, 240, 320])
        # - eef_pos: torch.Size([1, 2, 3])
        # - eef_quat: torch.Size([1, 2, 4])
        # - gripper_qpos: torch.Size([1, 2, 2])

        # the stream already captures at the policy resolution (W, H);
        # swap axis and normalize -> (2, 3, H, W)
        wrist_images = np.stack((past_frame, latest_frame), axis=0)
        wrist_images = np.swapaxes(np.swapaxes(wrist_images, 2, 3), 1, 2).astype(np.float32)/255.

        # robot_pose is [x, y, z, rx, ry, rz] per step
        eef_pos = robot_pose_stacked[:, 0:3].copy()                     # (2, 3)
        eef_quat = R.from_rotvec(robot_pose_stacked[:, 3:6]).as_quat()  # (2, 4)

        # gripper_width is not normalized
        gripper_qpos_normalized = gripper_width_stacked / MAX_GRIPPER_WIDTH
        gripper_qpos = np.stack((gripper_qpos_normalized, gripper_qpos_normalized), axis=-1)  # (2, 2)

        np_obs_dict_stacked = {
            'wrist_camera': wrist_images,
            'eef_pos': eef_pos,
            'eef_quat': eef_quat,
            'gripper_qpos': gripper_qpos,
        }

        # # get frequency for debug
        # # note: this frequency is how often inference is run, which only happens once per action chunk execution
        # # it is not the same as how fast the model can predict
        # cur_timestamp = float(self.get_clock().now().nanoseconds) / 1e9
        # cur_fps = 1./(cur_timestamp - self.last_inference_timestamp)
        # self.last_inference_timestamp = float(self.get_clock().now().nanoseconds) / 1e9
        # self.get_logger().info("Running inference at {} fps".format(cur_fps))

        # using relative action
        current_pos = copy.copy(np_obs_dict_stacked['eef_pos'][-1])
        current_rot_mat = copy.copy(self.rot_quat2mat.forward(np_obs_dict_stacked['eef_quat'][-1]))
        T_world_baseframe = np.eye(4)
        T_world_baseframe[:3, :3] = current_rot_mat.copy()
        T_world_baseframe[:3, 3] = current_pos.copy()

        # for key in np_obs_dict_stacked:
        #     print(key, np_obs_dict_stacked[key].shape)

        np_obs_dict_stacked['eef_pos'], np_obs_dict_stacked['eef_quat'] = compute_relative_pose(
            pos=np_obs_dict_stacked['eef_pos'],
            rot=np_obs_dict_stacked['eef_quat'],
            base_pos=current_pos,
            base_rot_mat=current_rot_mat,
            rot_transformer_to_mat=self.rot_quat2mat,
            rot_transformer_to_target=self.rot_mat2rot6d
        )

        np_obs_dict_stacked = {
            key: value[np.newaxis, :] for key, value in np_obs_dict_stacked.items()
        }

        # device transfer
        obs_dict = dict_apply(np_obs_dict_stacked,
            lambda x: torch.from_numpy(x).to(
                device=self.device))

        # run policy
        with torch.no_grad():
            action_dict = self.policy.predict_action(obs_dict)

        # device_transfer
        np_action_dict = dict_apply(action_dict,
            lambda x: x.detach().to('cpu').numpy())

        action_chunk = np_action_dict['action'][0]
        if not np.all(np.isfinite(action_chunk)):
            raise RuntimeError('Nan or Inf action')

        # action rotation transformer
        action_pos, action_rot = compute_relative_pose(
            pos=action_chunk[..., :3],
            rot=action_chunk[..., 3: -1],
            base_pos=current_pos,
            base_rot_mat=current_rot_mat,
            rot_transformer_to_mat=self.rot_aa2mat,
            rot_transformer_to_target=self.rot_mat2rot6d,
            backward=True
        )
        action_gripper = action_chunk[..., -1:]
        action_chunk = np.concatenate([action_pos, action_rot, action_gripper], axis=-1)

        # logging
        if self.results_path is not None:
            self.eef_positions.append(eef_pos[-1])  # the current step
            self.predicted_action_chunks.append(action_chunk.copy())

        # convert gripper width back
        # print('Predicted raw gripper width: {}'.format(action_chunk[:, -1]))
        action_chunk[:, -1] *= MAX_GRIPPER_WIDTH
        # print('Converted gripper width: {}'.format(action_chunk[:, -1]))

        # ===== interpolate trajectory =====
        # built into a local list, then published under the lock below
        next_traj = []

        # if this is the first action chunk, there is no latency
        if is_first_chunk:
            # interpolation starts from the start of the first action chunk
            next_traj.append(action_chunk[0].tolist())

            for i in range (len(action_chunk)-1):
                action_interpolated = interpolate_between_actions(action_chunk[i], action_chunk[i+1], self.interpolation_factor)
                next_traj += action_interpolated

        # otherwise, interpolate between the last action and cur_action_chunk[n_latency_steps]
        else:
            # self.last_action is not included as part of the current traj because it was already included in the last traj
            # using action_chunk[self.n_latency_steps + 1] instead of self.n_latency_steps because robot has 0.1s lookahead time
            action_interpolated = interpolate_between_actions(self.last_action, action_chunk[self.n_latency_steps + HARDWARE_LATENCY_STEPS], self.interpolation_factor)
            next_traj += action_interpolated

            # interpolate the rest of cur_action_chunk
            for i in range (self.n_latency_steps + HARDWARE_LATENCY_STEPS, len(action_chunk)-1):
                action_interpolated = interpolate_between_actions(action_chunk[i], action_chunk[i+1], self.interpolation_factor)
                next_traj += action_interpolated

        # self.get_logger().info('next_traj length: {}'.format(len(next_traj)))

        # publish the freshly computed trajectory for interpolation_callback to pick up
        with self._state_lock:
            self.next_traj = next_traj
            self.last_action = action_chunk[-1].copy()  # this would be the last latency step
            self.next_traj_ready = True

        self.get_logger().info('Predicted next action chunk.')

    
    def interpolation_callback (self):

        # ===== determine the trajectory to execute =====
        # If finished the current trajectory and the next one is ready, promote next_traj
        # to cur_traj. This should always be the case, since
        # n_latency_steps * 1./INFERENCE_FREQUENCY > inference time.
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
            return

        # if finished executing current traj but the next traj is not ready yet, give warning
        if finished_cur_traj and not next_traj_ready:
            self.get_logger().warning('Finished current trajectory, but the next trajectory is not ready yet!', once=True)
            return

        # ===== execute trajectory as usual (outside the lock) =====
        # cur_traj / traj_idx are only rebound during the swap above, and
        # interpolation_callback is mutually exclusive with itself, so they are stable here.
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

        # log trajectory and save the episode once the timestep limit is reached
        if self.results_path is not None:
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

                # # write video
                # wrist_cam_writer = cv2.VideoWriter(
                #     '{}/wrist.mp4'.format(self.episode_path),
                #     cv2.VideoWriter_fourcc(*'mp4v') ,
                #     FPS,
                #     (W, H),
                # )
                # self.get_logger().info('Saving images to mp4...')
                # for wrist in self.wrist_images:
                #     wrist_cam_writer.write(wrist)
                # wrist_cam_writer.release()

                self.get_logger().info('Saved episode data.')
                rclpy.shutdown()

        # updates
        self.traj_idx += 1
        self.counter += 1

        # ===== check status / signal inference_callback =====
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

    # ckpt_path = '/home/jingyixiang/diffusion_policy/data/outputs/grasp_cube_without_tactile/checkpoints/latest.ckpt'
    ckpt_path = '/home/jingyixiang/diffusion_policy/data/outputs/turn_knob_umi/latest.ckpt'
    # set to None to disable result logging / saving (and the timestep-limit shutdown)
    results_path = '/home/jingyixiang/diffusion_policy/data/inference/turn_konb_umi'
    # results_path = None

    rclpy.init(args=args)
    dp_inference_node = DiffusionPolicyInference(ckpt_path, results_path)

    # MultiThreadedExecutor so the heavy inference_callback does not block the
    # interpolation_callback publishing (each lives in its own group).
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
            vis_frame = dp_inference_node.get_vis_frame()
            if vis_frame is not None:
                cv2.imshow('wrist', vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
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