# Kibo-RPC 7th user program

This repository contains the team's ROS1 Python user node for the 7th Kibo-RPC simulator. The simulator and its API/message definitions remain in the official `krpc7th-simulator` resources.

## What the program does

- Uses the official mission services and actions: `/event/mission_start`, `/mission/get_checkpoints`, `/ctl/command`, `/event/checkpoint_passed`, and `/event/mission_finished`.
- Reads this run's checkpoints, scoring bands, and `goalpoint_id` from the API. It does not substitute an origin target when the response is invalid.
- Plans an exact Euclidean-distance visit order for the regular checkpoints, including the current position as the start and GoalPoint as the finish.
- Waits until the returned position, camera-centering, and roll bands are continuously satisfied for at least one second before requesting checkpoint scoring.
- Treats GoalPoint as the final destination and calls `mission_finished` there; it does not call `checkpoint_passed` for GoalPoint.
- Uses `iss_body`, the absolute movement frame in the simulator manual, and includes finite service, action, navigation, and mission timeouts.

## Use with the local simulator

Copy `user_template.py` to the official simulator's `users/catkin_ws/src/krpc_user/scripts/user_template.py`, replacing the sample user script. The default `krpc_user.launch` selects that filename. Then run the official simulator from its Ubuntu/WSL2 environment:

```bash
bash ./build.sh
bash ./simulation.sh
```

This repository does not include the simulator infrastructure. Do not edit the official API/message definitions to use this node. The local simulator is intended for iterative checks; confirm the final source on the Web Simulator as well.
