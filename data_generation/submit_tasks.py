"""This file uses the inductiva API to sumbit tasks"""
from absl import app
from absl import flags

import os
import json

import inductiva

FLAGS = flags.FLAGS

flags.DEFINE_string("input_dataset", None, "Path to the dataset of objects.")

flags.DEFINE_list("flow_velocity", [30, 0, 0], "Flow velocity in.")
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


def simulate_wind_tunnel_scenario(obj_path, flow_velocity, x_geometry,
                                  y_geometry, z_geometry, num_iterations,
                                  machine_group):
    domain_geometry = {"x": x_geometry, "y": y_geometry, "z": z_geometry}

    scenario = inductiva.fluids.WindTunnel(flow_velocity=flow_velocity,
                                           domain=domain_geometry)

    task_id = scenario.simulate(object_path=obj_path,
                                num_iterations=num_iterations,
                                resolution="low",
                                run_async=True,
                                machine_group=machine_group)
    return task_id


def main(_):
    object_paths = [
        os.path.join(FLAGS.input_dataset, path)
        for path in os.listdir(FLAGS.input_dataset)
    ]

    flow_velocity = list(map(float, FLAGS.flow_velocity))
    x_geometry = list(map(float, FLAGS.x_geometry))
    y_geometry = list(map(float, FLAGS.y_geometry))
    z_geometry = list(map(float, FLAGS.z_geometry))

    # Start the machine group with the requested parameters
    machine_group = inductiva.resources.MachineGroup(
        machine_type=FLAGS.machine_type,
        num_machines=FLAGS.num_machines,
        disk_size_gb=FLAGS.disk_size_gb)
    machine_group.start()

    task_ids = [
        simulate_wind_tunnel_scenario(object_path, flow_velocity, x_geometry,
                                      y_geometry, z_geometry,
                                      FLAGS.num_iterations, machine_group)
        for object_path in object_paths
    ]

    # Make a json with the sim ids and the machine group name.
    with open(os.path.join(FLAGS.output_dataset, "sim_info.json"),
              "w",
              encoding="utf-8") as f:
        dict_to_save = {
            "task_ids": [task_id.id for task_id in task_ids],
            "machine_group": machine_group.name
        }
        json.dump(dict_to_save, f)


if __name__ == "__main__":
    app.run(main)
