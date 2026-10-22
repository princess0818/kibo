#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Kibo-RPC 7th user node for the ROS1 Int-Ball2 simulator."""

import json
import math
import threading
import time

import actionlib
import rospy
from geometry_msgs.msg import PoseStamped
from ib2_msgs.msg import (
    CtlCommandAction,
    CtlCommandGoal,
    CtlCommandResult,
    CtlStatusType,
    Navigation,
)
from krpc_ib2_msgs.msg import CheckpointPassedAction, CheckpointPassedGoal
from krpc_ib2_msgs.srv import CheckPoints, MissionFinished, MissionStart
from platform_msgs.msg import UserLogic
from platform_msgs.srv import (
    StopProcessingUserNode,
    StopProcessingUserNodeResponse,
)


class KiboRpcUserNode:
    """Visit every non-goal checkpoint, then finish while at the GoalPoint."""

    def __init__(self):
        rospy.init_node("krpc_user")

        self._service_timeout = float(rospy.get_param("~service_timeout", 10.0))
        self._action_server_timeout = float(
            rospy.get_param("~action_server_timeout", 10.0)
        )
        self._move_timeout = float(rospy.get_param("~move_timeout", 35.0))
        self._checkpoint_timeout = float(
            rospy.get_param("~checkpoint_timeout", 15.0)
        )
        self._navigation_wait_timeout = float(
            rospy.get_param("~navigation_wait_timeout", 10.0)
        )
        self._navigation_max_age = float(
            rospy.get_param("~navigation_max_age", 0.5)
        )
        self._stability_timeout = float(
            rospy.get_param("~stability_timeout", 15.0)
        )
        # The rule requires the scoring conditions to hold continuously for 1 s.
        self._stability_duration = max(
            1.0, float(rospy.get_param("~stability_duration", 1.0))
        )
        self._mission_timeout = float(rospy.get_param("~mission_timeout", 300.0))
        self._frame_id = rospy.get_param("~frame_id", "iss_body")

        self._command = actionlib.SimpleActionClient(
            "/ctl/command", CtlCommandAction
        )
        self._checkpoint_client = actionlib.SimpleActionClient(
            "/event/checkpoint_passed", CheckpointPassedAction
        )
        self._mission_start = rospy.ServiceProxy(
            "/event/mission_start", MissionStart
        )
        self._mission_finished = rospy.ServiceProxy(
            "/event/mission_finished", MissionFinished
        )
        self._checkpoints = rospy.ServiceProxy(
            "/mission/get_checkpoints", CheckPoints
        )

        self._start_lock = threading.Lock()
        self._action_lock = threading.Lock()
        self._navigation_lock = threading.Lock()
        self._started = False
        self._stop_requested = threading.Event()
        self._active_action = None
        self._mission_deadline = None
        self._navigation = None
        self._navigation_sequence = 0

        # Required lifecycle interfaces: keep these names unchanged.
        self._subscriber_start = rospy.Subscriber(
            "/ib2_user/start", UserLogic, self._callback_start
        )
        self._subscriber_navigation = rospy.Subscriber(
            "/sensor_fusion/navigation",
            Navigation,
            self._callback_navigation,
            queue_size=1,
        )
        self._stop_processing_server = rospy.Service(
            "/ib2_user/stop", StopProcessingUserNode, self._stop_processing
        )
        rospy.set_param("/ib2_user/ready", True)
        rospy.loginfo("Kibo-RPC user node ready")

    def _remaining_time(self):
        if self._mission_deadline is None:
            return None
        return self._mission_deadline - time.monotonic()

    def _bounded_timeout(self, timeout):
        remaining = self._remaining_time()
        if remaining is None:
            return max(0.0, timeout)
        return max(0.0, min(timeout, remaining))

    def _wait_for_service(self, name):
        timeout = self._bounded_timeout(self._service_timeout)
        if timeout <= 0.0:
            rospy.logerr("Mission time expired before waiting for %s", name)
            return False
        try:
            rospy.wait_for_service(name, timeout=timeout)
            return True
        except rospy.ROSException:
            rospy.logerr("Timed out waiting for service %s", name)
            return False

    def _call_mission_service(self, name, proxy):
        if not self._wait_for_service(name):
            return False
        try:
            response = proxy()
        except rospy.ServiceException as exc:
            rospy.logerr("Service %s failed: %s", name, exc)
            return False
        rospy.loginfo(
            "%s result: success=%s message=%s",
            name,
            response.success,
            response.message,
        )
        return bool(response.success)

    @staticmethod
    def _as_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _vector3(value):
        if isinstance(value, dict):
            try:
                vector = [float(value[key]) for key in ("x", "y", "z")]
            except (KeyError, TypeError, ValueError):
                return None
        elif isinstance(value, (list, tuple)) and len(value) >= 3:
            try:
                vector = [float(value[index]) for index in range(3)]
            except (TypeError, ValueError):
                return None
        else:
            return None
        if not all(math.isfinite(component) for component in vector):
            return None
        return vector

    @staticmethod
    def _normalise_quaternion(value):
        if isinstance(value, dict):
            try:
                quaternion = [
                    float(value[key]) for key in ("x", "y", "z", "w")
                ]
            except (KeyError, TypeError, ValueError):
                return None
        elif isinstance(value, (list, tuple)) and len(value) >= 4:
            try:
                quaternion = [float(value[index]) for index in range(4)]
            except (TypeError, ValueError):
                return None
        else:
            return None

        if not all(math.isfinite(component) for component in quaternion):
            return None
        norm = math.sqrt(sum(component * component for component in quaternion))
        if norm <= 1e-9:
            return None
        return [component / norm for component in quaternion]

    @staticmethod
    def _parse_bands(value, threshold_key):
        if not isinstance(value, list):
            return []
        bands = []
        for band in value:
            if not isinstance(band, dict) or threshold_key not in band:
                continue
            try:
                limit = float(band[threshold_key])
                level = int(band.get("level", 0))
            except (TypeError, ValueError):
                continue
            if math.isfinite(limit) and limit >= 0.0:
                bands.append({"limit": limit, "level": level})
        return sorted(bands, key=lambda item: item["limit"])

    @staticmethod
    def _widest_band(bands):
        if not bands:
            return None
        return max(item["limit"] for item in bands)

    def _parse_checkpoint(self, item):
        checkpoint_id = None
        for key in ("checkpoint_id", "checkpointid", "id"):
            if key in item:
                checkpoint_id = self._as_int(item[key])
                break
        if checkpoint_id is None:
            raise ValueError("checkpoint has no valid ID")

        position = self._vector3(item.get("position"))
        if position is None:
            raise ValueError("checkpoint %d has an invalid position" % checkpoint_id)

        score_bands = self._parse_bands(
            item.get("score_bands", item.get("bands")), "radius"
        )
        if not score_bands:
            raise ValueError("checkpoint %d has no position score bands" % checkpoint_id)

        marker_pose = item.get("marker_pose")
        if not isinstance(marker_pose, dict):
            marker_pose = {}
        marker_position = self._vector3(marker_pose.get("position"))

        attitude = item.get("attitude")
        if not isinstance(attitude, dict):
            attitude = {}
        quaternion = self._normalise_quaternion(attitude.get("quaternion"))
        centering_bands = self._parse_bands(
            attitude.get("centering_bands"), "angle"
        )
        roll_bands = self._parse_bands(attitude.get("roll_bands"), "angle")
        if (centering_bands or roll_bands) and quaternion is None:
            raise ValueError(
                "checkpoint %d has attitude bands but no target quaternion"
                % checkpoint_id
            )
        if centering_bands and marker_position is None:
            raise ValueError(
                "checkpoint %d has centering bands but no marker position"
                % checkpoint_id
            )

        return {
            "checkpoint_id": checkpoint_id,
            "position": position,
            "quaternion": quaternion,
            "marker_position": marker_position,
            "score_bands": score_bands,
            "centering_bands": centering_bands,
            "roll_bands": roll_bands,
        }

    def _decode_checkpoint_payload(self, message):
        payload = json.loads(message)
        if not isinstance(payload, dict):
            raise ValueError("checkpoint response must be a JSON object")

        goalpoint_id = self._as_int(
            payload.get("goalpoint_id", payload.get("goal_point_id"))
        )
        if goalpoint_id is None:
            raise ValueError("response has no valid goalpoint_id")

        raw_checkpoints = payload.get(
            "checkpoints", payload.get("Checkpoints", [])
        )
        if not isinstance(raw_checkpoints, list) or not raw_checkpoints:
            raise ValueError("response has no checkpoints")

        checkpoints = []
        ids = set()
        for item in raw_checkpoints:
            if not isinstance(item, dict):
                raise ValueError("checkpoint list contains a non-object value")
            checkpoint = self._parse_checkpoint(item)
            checkpoint_id = checkpoint["checkpoint_id"]
            if checkpoint_id in ids:
                raise ValueError("duplicate checkpoint ID %d" % checkpoint_id)
            ids.add(checkpoint_id)
            checkpoints.append(checkpoint)

        goals = [item for item in checkpoints if item["checkpoint_id"] == goalpoint_id]
        if len(goals) != 1:
            raise ValueError("goalpoint_id is not present exactly once in checkpoints")

        goalpoint = goals[0]
        regular_checkpoints = [
            item for item in checkpoints if item["checkpoint_id"] != goalpoint_id
        ]
        return regular_checkpoints, goalpoint

    def _get_checkpoints(self):
        name = "/mission/get_checkpoints"
        if not self._wait_for_service(name):
            return None
        try:
            response = self._checkpoints()
        except rospy.ServiceException as exc:
            rospy.logerr("Service %s failed: %s", name, exc)
            return None

        rospy.loginfo("get_checkpoints: success=%s", response.success)
        if not response.success:
            rospy.logerr("get_checkpoints failed: %s", response.message)
            return None
        try:
            regular, goalpoint = self._decode_checkpoint_payload(response.message)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            rospy.logerr("Could not parse checkpoint response: %s", exc)
            return None

        rospy.loginfo(
            "Loaded %d scored checkpoints and GoalPoint %d",
            len(regular),
            goalpoint["checkpoint_id"],
        )
        return regular, goalpoint

    def _callback_navigation(self, message):
        pose = message.pose.pose
        position = [pose.position.x, pose.position.y, pose.position.z]
        quaternion = self._normalise_quaternion(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ]
        )
        if quaternion is None or not all(
            math.isfinite(value) for value in position
        ):
            return

        with self._navigation_lock:
            self._navigation_sequence += 1
            self._navigation = {
                "position": list(position),
                "quaternion": quaternion,
                "received_at": time.monotonic(),
                "sequence": self._navigation_sequence,
            }

    def _navigation_snapshot(self):
        with self._navigation_lock:
            if self._navigation is None:
                return None
            snapshot = dict(self._navigation)
        snapshot["age"] = time.monotonic() - snapshot["received_at"]
        return snapshot

    def _wait_for_navigation(self, timeout):
        deadline = time.monotonic() + self._bounded_timeout(timeout)
        while not rospy.is_shutdown() and not self._stop_requested.is_set():
            snapshot = self._navigation_snapshot()
            if snapshot is not None and snapshot["age"] <= self._navigation_max_age:
                return snapshot
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        rospy.logerr("No fresh /sensor_fusion/navigation message")
        return None

    @staticmethod
    def _distance(left, right):
        return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))

    @staticmethod
    def _normalise_vector(vector):
        norm = math.sqrt(sum(component * component for component in vector))
        if norm <= 1e-9:
            return None
        return [component / norm for component in vector]

    @staticmethod
    def _rotate_vector_by_quaternion(vector, quaternion):
        qx, qy, qz, qw = quaternion
        vx, vy, vz = vector
        cross1 = [
            qy * vz - qz * vy,
            qz * vx - qx * vz,
            qx * vy - qy * vx,
        ]
        cross2 = [
            qy * cross1[2] - qz * cross1[1],
            qz * cross1[0] - qx * cross1[2],
            qx * cross1[1] - qy * cross1[0],
        ]
        return [
            vector[index] + 2.0 * (qw * cross1[index] + cross2[index])
            for index in range(3)
        ]

    @staticmethod
    def _vector_angle_degrees(left, right):
        dot = sum(left[index] * right[index] for index in range(3))
        dot = max(-1.0, min(1.0, dot))
        return math.degrees(math.acos(dot))

    @staticmethod
    def _twist_angle_degrees(quaternion, axis):
        qx, qy, qz, qw = quaternion
        if qw < 0.0:
            qx, qy, qz, qw = -qx, -qy, -qz, -qw
        dot = qx * axis[0] + qy * axis[1] + qz * axis[2]
        return math.degrees(2.0 * math.atan2(dot, qw))

    @staticmethod
    def _wrap_angle_degrees(angle):
        return (angle + 180.0) % 360.0 - 180.0

    def _inside_scoring_bands(self, navigation, checkpoint):
        position_error = self._distance(
            navigation["position"], checkpoint["position"]
        )
        position_limit = self._widest_band(checkpoint["score_bands"])
        if position_limit is None or position_error > position_limit:
            return False

        marker_position = checkpoint["marker_position"]
        if checkpoint["centering_bands"] or checkpoint["roll_bands"]:
            if marker_position is None:
                return False
            view_axis = self._normalise_vector(
                [
                    marker_position[index] - navigation["position"][index]
                    for index in range(3)
                ]
            )
            if view_axis is None:
                return False

            if checkpoint["centering_bands"]:
                camera_forward = self._normalise_vector(
                    self._rotate_vector_by_quaternion(
                        [1.0, 0.0, 0.0], navigation["quaternion"]
                    )
                )
                centering_error = self._vector_angle_degrees(
                    camera_forward, view_axis
                )
                centering_limit = self._widest_band(
                    checkpoint["centering_bands"]
                )
                if centering_error > centering_limit:
                    return False

            if checkpoint["roll_bands"]:
                target_quaternion = checkpoint["quaternion"]
                if target_quaternion is None:
                    return False
                actual_roll = self._twist_angle_degrees(
                    navigation["quaternion"], view_axis
                )
                target_roll = self._twist_angle_degrees(
                    target_quaternion, view_axis
                )
                roll_error = abs(
                    self._wrap_angle_degrees(actual_roll - target_roll)
                )
                roll_limit = self._widest_band(checkpoint["roll_bands"])
                if roll_error > roll_limit:
                    return False
        return True

    def _wait_for_scoring_window(self, checkpoint):
        timeout = self._bounded_timeout(self._stability_timeout)
        deadline = time.monotonic() + timeout
        stable_since = None
        last_sequence = None
        last_sample_time = None

        while not rospy.is_shutdown() and not self._stop_requested.is_set():
            if time.monotonic() >= deadline:
                break
            navigation = self._navigation_snapshot()
            if navigation is None or navigation["age"] > self._navigation_max_age:
                stable_since = None
                last_sample_time = None
                time.sleep(0.02)
                continue

            if navigation["sequence"] == last_sequence:
                time.sleep(0.02)
                continue
            last_sequence = navigation["sequence"]
            sample_time = navigation["received_at"]

            if (
                last_sample_time is not None
                and sample_time - last_sample_time > self._navigation_max_age
            ):
                stable_since = None
            last_sample_time = sample_time

            if self._inside_scoring_bands(navigation, checkpoint):
                if stable_since is None:
                    stable_since = sample_time
                if sample_time - stable_since >= self._stability_duration:
                    rospy.loginfo(
                        "ID %d held inside scoring bands for %.1f s",
                        checkpoint["checkpoint_id"],
                        self._stability_duration,
                    )
                    return True
            else:
                stable_since = None

            time.sleep(0.02)

        rospy.logwarn(
            "ID %d did not remain inside its returned scoring bands",
            checkpoint["checkpoint_id"],
        )
        return False

    def _wait_for_action_server(self, client, name):
        timeout = self._bounded_timeout(self._action_server_timeout)
        if timeout > 0.0 and client.wait_for_server(rospy.Duration(timeout)):
            return True
        rospy.logerr("Action server unavailable: %s", name)
        return False

    def _cancel_and_wait(self, client, timeout=5.0):
        client.cancel_goal()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if client.wait_for_result(rospy.Duration(0.2)):
                return True
        return False

    def _make_pose(self, checkpoint):
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = self._frame_id
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = (
            checkpoint["position"]
        )

        quaternion = checkpoint["quaternion"]
        if quaternion is None:
            navigation = self._navigation_snapshot()
            quaternion = (
                navigation["quaternion"]
                if navigation is not None
                else [0.0, 0.0, 0.0, 1.0]
            )
        (
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        ) = quaternion
        return pose

    def _move_to_checkpoint(self, checkpoint):
        name = "/ctl/command"
        if not self._wait_for_action_server(self._command, name):
            raise RuntimeError("/ctl/command action server is unavailable")

        timeout = self._bounded_timeout(self._move_timeout)
        if timeout <= 0.0:
            return False

        command_type = CtlStatusType(type=CtlStatusType.MOVE_TO_ABSOLUTE_TARGET)
        goal = CtlCommandGoal(target=self._make_pose(checkpoint), type=command_type)
        position = checkpoint["position"]
        rospy.loginfo(
            "Moving to ID %d at (%.3f, %.3f, %.3f)",
            checkpoint["checkpoint_id"],
            position[0],
            position[1],
            position[2],
        )

        with self._action_lock:
            self._active_action = self._command
        try:
            self._command.send_goal(goal)
            deadline = time.monotonic() + timeout
            while not rospy.is_shutdown() and not self._stop_requested.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                if self._command.wait_for_result(
                    rospy.Duration(min(0.2, remaining))
                ):
                    result = self._command.get_result()
                    rospy.loginfo(
                        "Move result for ID %d: %s",
                        checkpoint["checkpoint_id"],
                        result,
                    )
                    return bool(
                        result
                        and result.type == CtlCommandResult.TERMINATE_SUCCESS
                    )

            if not self._cancel_and_wait(self._command):
                raise RuntimeError("/ctl/command did not stop after cancellation")
            rospy.logwarn(
                "Move cancelled or timed out for ID %d",
                checkpoint["checkpoint_id"],
            )
            return False
        finally:
            with self._action_lock:
                if self._active_action is self._command:
                    self._active_action = None

    def _notify_checkpoint_passed(self, checkpoint_id):
        name = "/event/checkpoint_passed"
        if not self._wait_for_action_server(self._checkpoint_client, name):
            rospy.logerr(
                "Cannot score checkpoint %d without the action server",
                checkpoint_id,
            )
            return False

        goal = CheckpointPassedGoal()
        goal.checkpoint_id = checkpoint_id
        with self._action_lock:
            self._active_action = self._checkpoint_client
        try:
            self._checkpoint_client.send_goal(goal)
            timeout = self._bounded_timeout(self._checkpoint_timeout)
            deadline = time.monotonic() + timeout
            while not rospy.is_shutdown() and not self._stop_requested.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                if self._checkpoint_client.wait_for_result(
                    rospy.Duration(min(0.2, remaining))
                ):
                    result = self._checkpoint_client.get_result()
                    if result is None:
                        rospy.logerr("Checkpoint %d returned no result", checkpoint_id)
                        return False
                    rospy.loginfo(
                        "Checkpoint %d result: success=%s message=%s",
                        checkpoint_id,
                        result.success,
                        result.message,
                    )
                    return bool(result.success)

            if not self._cancel_and_wait(self._checkpoint_client):
                raise RuntimeError("checkpoint action did not stop after cancellation")
            rospy.logerr("Checkpoint %d scoring timed out", checkpoint_id)
            return False
        finally:
            with self._action_lock:
                if self._active_action is self._checkpoint_client:
                    self._active_action = None

    @staticmethod
    def _euclidean_distance(left, right):
        return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))

    def _plan_route(self, start_position, checkpoints, goalpoint):
        """Held-Karp route using Euclidean distance as a movement-time proxy."""
        count = len(checkpoints)
        if count <= 1:
            return list(checkpoints)

        # Normal difficulty has at most 11 scored checkpoints plus the goal.
        # Use a bounded nearest-neighbour fallback if an API version adds more.
        if count > 14:
            remaining = list(checkpoints)
            route = []
            current = start_position
            while remaining:
                if len(remaining) <= 2:
                    tail_options = [list(remaining)]
                    if len(remaining) == 2:
                        tail_options.append(list(reversed(remaining)))
                    tail = min(
                        tail_options,
                        key=lambda order: self._euclidean_distance(
                            current, order[0]["position"]
                        )
                        + self._euclidean_distance(
                            order[0]["position"], order[-1]["position"]
                        )
                        + self._euclidean_distance(
                            order[-1]["position"], goalpoint["position"]
                        ),
                    )
                    route.extend(tail)
                    break
                next_checkpoint = min(
                    remaining,
                    key=lambda item: self._euclidean_distance(
                        current, item["position"]
                    ),
                )
                route.append(next_checkpoint)
                remaining.remove(next_checkpoint)
                current = next_checkpoint["position"]
            return route

        costs = {}
        parents = {}
        for index, checkpoint in enumerate(checkpoints):
            state = (1 << index, index)
            costs[state] = self._euclidean_distance(
                start_position, checkpoint["position"]
            )
            parents[state] = None

        for mask in range(1, 1 << count):
            for last in range(count):
                state = (mask, last)
                if state not in costs:
                    continue
                for next_index in range(count):
                    bit = 1 << next_index
                    if mask & bit:
                        continue
                    next_state = (mask | bit, next_index)
                    candidate_cost = costs[state] + self._euclidean_distance(
                        checkpoints[last]["position"],
                        checkpoints[next_index]["position"],
                    )
                    if (
                        next_state not in costs
                        or candidate_cost < costs[next_state]
                    ):
                        costs[next_state] = candidate_cost
                        parents[next_state] = last

        full_mask = (1 << count) - 1
        last = min(
            range(count),
            key=lambda index: costs[(full_mask, index)]
            + self._euclidean_distance(
                checkpoints[index]["position"], goalpoint["position"]
            ),
        )
        reverse_route = []
        mask = full_mask
        while mask:
            reverse_route.append(checkpoints[last])
            previous = parents[(mask, last)]
            mask ^= 1 << last
            last = previous
        return list(reversed(reverse_route))

    def _run_user_flow(self, message):
        self._mission_deadline = time.monotonic() + self._mission_timeout
        try:
            rospy.loginfo("Start event received: %s", message)
            if not self._call_mission_service(
                "/event/mission_start", self._mission_start
            ):
                return

            payload = self._get_checkpoints()
            if payload is None:
                return
            checkpoints, goalpoint = payload

            navigation = self._wait_for_navigation(self._navigation_wait_timeout)
            if navigation is None:
                return

            route = self._plan_route(
                navigation["position"], checkpoints, goalpoint
            )
            rospy.loginfo(
                "Route IDs: %s, then GoalPoint %d",
                [item["checkpoint_id"] for item in route],
                goalpoint["checkpoint_id"],
            )

            for checkpoint in route:
                if (
                    rospy.is_shutdown()
                    or self._stop_requested.is_set()
                    or self._bounded_timeout(0.1) <= 0.0
                ):
                    return
                checkpoint_id = checkpoint["checkpoint_id"]
                if not self._move_to_checkpoint(checkpoint):
                    rospy.logwarn(
                        "Skipping score request for checkpoint %d after move failure",
                        checkpoint_id,
                    )
                    continue
                if not self._wait_for_scoring_window(checkpoint):
                    rospy.logwarn(
                        "Skipping score request for checkpoint %d",
                        checkpoint_id,
                    )
                    continue
                if not self._notify_checkpoint_passed(checkpoint_id):
                    # Do not submit the same checkpoint repeatedly; continue the route.
                    rospy.logwarn("Checkpoint %d was not accepted", checkpoint_id)

            # The goalpoint is included in the API's checkpoint list. Do not call
            # checkpoint_passed for it: mission_finished evaluates it in place.
            goal_reached = False
            for attempt in range(2):
                if not self._move_to_checkpoint(goalpoint):
                    rospy.logwarn(
                        "Could not move to GoalPoint (attempt %d)", attempt + 1
                    )
                    continue
                if self._wait_for_scoring_window(goalpoint):
                    goal_reached = True
                    break
                rospy.logwarn("GoalPoint not stable; correcting once")

            if not goal_reached:
                rospy.logerr("Not calling mission_finished outside the GoalPoint")
                return

            if not rospy.is_shutdown() and not self._stop_requested.is_set():
                self._call_mission_service(
                    "/event/mission_finished", self._mission_finished
                )
        except Exception as exc:
            rospy.logerr("User mission flow failed: %s", exc)

    def _callback_start(self, message):
        with self._start_lock:
            if self._started:
                rospy.logwarn("Start event already processed; ignoring duplicate")
                return
            self._started = True
            self._stop_requested.clear()

        worker = threading.Thread(
            target=self._run_user_flow, args=(message,), daemon=True
        )
        worker.start()

    def _stop_processing(self, request):
        del request
        self._stop_requested.set()
        with self._action_lock:
            active_action = self._active_action
        if active_action is not None:
            if not self._cancel_and_wait(active_action):
                rospy.logerr("Active action did not stop after cancellation")
                return StopProcessingUserNodeResponse(
                    result=StopProcessingUserNodeResponse.ERROR
                )
        rospy.loginfo("User-node processing stopped")
        return StopProcessingUserNodeResponse(
            result=StopProcessingUserNodeResponse.SUCCESS
        )

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    KiboRpcUserNode().run()
