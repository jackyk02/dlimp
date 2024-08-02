from typing import Any, Dict

import tensorflow as tf


def add_next_obs(traj: Dict[str, Any], pad: bool = True) -> Dict[str, Any]:
    """
    Given a trajectory with a key "observations", add the key "next_observations". If pad is False, discards the last
    value of all other keys. Otherwise, the last transition will have "observations" == "next_observations".
    """
    if not pad:
        traj_truncated = tf.nest.map_structure(lambda x: x[:-1], traj)
        traj_truncated["next_obs"] = tf.nest.map_structure(
            lambda x: x[1:], traj["obs"]
        )
        return traj_truncated
    else:
        traj["next_obs"] = tf.nest.map_structure(
            lambda x: tf.concat((x[1:], x[-1:]), axis=0), traj["obs"]
        )
        return traj


def process_obs_actions(traj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given cur state and previous action, predict the next state
States : [1]  [2]   [3]  [4] 
Actions: [a1] [a2] [a3] [null]

->
    Obs: [2][3]   (remove first and last)
Actions: [a1][a2] (remove last two actions)
Next_obs: [3] [4] (remove first two)
    """
    traj_truncated = {}
    traj_truncated["obs"] = tf.nest.map_structure(
        lambda x: x[1:-1], traj["obs"]
    )

    traj_truncated["actions"] = tf.nest.map_structure(
        lambda x: x[:-2], traj["actions"]
    )

    traj_truncated["next_obs"] = tf.nest.map_structure(
        lambda x: x[2:], traj["obs"]
    )

    traj_truncated["lang"] = tf.nest.map_structure(
        lambda x: x[:-2], traj["lang"]
    )

    return traj_truncated
