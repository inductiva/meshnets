"""This file uses the inductiva API to sumbit tasks"""
from absl import app
from absl import flags
from absl import logging

import os
import shutil
import json

import random

import inductiva

FLAGS = flags.FLAGS

flags.DEFINE_string("input_dataset", None, "Path to the dataset of objects.")

flags.DEFINE_list("flow_velocity_range_x", None,
                  "Range of flow velocity in the x-direction.")
flags.DEFINE_list("flow_velocity_range_y", None,
                  "Range of flow velocity in the y-direction.")
flags.DEFINE_list("flow_velocity_range_z", None,
                  "Range of flow velocity in the z-direction.")
flags.DEFINE_list("x_geometry", [-5, 20], "X geometry of the domain.")
flags.DEFINE_list("y_geometry", [-5, 5], "Y geometry of the domain.")
flags.DEFINE_list("z_geometry", [-2, 10], "Z geometry of the domain.")
flags.DEFINE_integer("num_iterations", 100, "Number of iterations to run.")

flags.DEFINE_string("machine_type", "c2-standard-16", "Machine type.")
flags.DEFINE_integer("num_machines", 1, "Number of machines.")
flags.DEFINE_integer("disk_size_gb", 40, "Disk size in GB.")

flags.DEFINE_string("output_dataset", None, "Path to the output dataset.")

flags.mark_flag_as_required("input_dataset")
flags.mark_flag_as_required("output_dataset")
flags.mark_flag_as_required("flow_velocity_range_x")
flags.mark_flag_as_required("flow_velocity_range_y")
flags.mark_flag_as_required("flow_velocity_range_z")


def simulate_wind_tunnel_scenario(obj_path, flow_velocity, x_geometry,
                                  y_geometry, z_geometry, num_iterations,
                                  machine_group):
    domain_geometry = {"x": x_geometry, "y": y_geometry, "z": z_geometry}

    scenario = inductiva.fluids.WindTunnel(flow_velocity=flow_velocity,
                                           domain=domain_geometry)

    task = scenario.simulate(object_path=obj_path,
                             num_iterations=num_iterations,
                             resolution="low",
                             run_async=True,
                             machine_group=machine_group)
    return task


def main(_):
    object_paths = [
        os.path.join(FLAGS.input_dataset, path)
        for path in os.listdir(FLAGS.input_dataset)
    ]

    flow_velocity_range_x = list(map(float, FLAGS.flow_velocity_range_x))
    flow_velocity_range_y = list(map(float, FLAGS.flow_velocity_range_y))
    flow_velocity_range_z = list(map(float, FLAGS.flow_velocity_range_z))

    x_geometry = list(map(float, FLAGS.x_geometry))
    y_geometry = list(map(float, FLAGS.y_geometry))
    z_geometry = list(map(float, FLAGS.z_geometry))

    try:
        # Start the machine group with the requested parameters
        machine_group = inductiva.resources.MachineGroup(
            machine_type=FLAGS.machine_type,
            num_machines=FLAGS.num_machines,
            disk_size_gb=FLAGS.disk_size_gb)
        machine_group.start()

        obj_task_velocities = []
        for object_path in object_paths:
            flow_velocity_x = random.uniform(*flow_velocity_range_x)
            flow_velocity_y = random.uniform(*flow_velocity_range_y)
            flow_velocity_z = random.uniform(*flow_velocity_range_z)

            task = simulate_wind_tunnel_scenario(
                object_path,
                [flow_velocity_x, flow_velocity_y, flow_velocity_z], x_geometry,
                y_geometry, z_geometry, FLAGS.num_iterations, machine_group)

            obj_task_velocities.append(
                (object_path, task,
                 [flow_velocity_x, flow_velocity_y, flow_velocity_z]))

        # Copy the object files to a folder with path
        # FLAGS.output_dataset/task_id. This keeps track of the obj file
        # for the given file.
        for object_path, task, flow_velocity in obj_task_velocities:
            os.makedirs(os.path.join(FLAGS.output_dataset, task.id))
            shutil.copy(
                object_path,
                os.path.join(FLAGS.output_dataset, task.id, "object.obj"))

            # Save the flow velocity with the simulation.
            with open(os.path.join(FLAGS.output_dataset, task.id,
                                   "flow_velocity.json"),
                      "w",
                      encoding="utf-8") as f:
                json.dump(flow_velocity, f)

        # Make a json with the task ids and the machine group name.
        with open(os.path.join(FLAGS.output_dataset, "sim_info.json"),
                  "w",
                  encoding="utf-8") as f:
            dict_to_save = {
                "task_ids": [task.id for _, task, _ in obj_task_velocities],
                "machine_group": machine_group.name
            }
            json.dump(dict_to_save, f)

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Terminating machine group.")
        machine_group.terminate()
    # pylint: disable=broad-except
    except Exception as e:
        logging.error("Error occurred: %s", e)
        machine_group.terminate()


if __name__ == "__main__":
    app.run(main)
