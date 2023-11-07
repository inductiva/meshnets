"""This script estimates the costs of the API."""
from absl import app
from absl import flags
from absl import logging

import time
import random

import inductiva

FLAGS = flags.FLAGS

flags.DEFINE_string("obj_path", None, "Path to the input object.")

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

flags.mark_flag_as_required("flow_velocity_range_x")
flags.mark_flag_as_required("flow_velocity_range_y")
flags.mark_flag_as_required("flow_velocity_range_z")


def main(_):
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

        domain_geometry = {"x": x_geometry, "y": y_geometry, "z": z_geometry}

        flow_velocity_x = random.uniform(*flow_velocity_range_x)
        flow_velocity_y = random.uniform(*flow_velocity_range_y)
        flow_velocity_z = random.uniform(*flow_velocity_range_z)

        flow_velocity = [flow_velocity_x, flow_velocity_y, flow_velocity_z]

        scenario = inductiva.fluids.WindTunnel(flow_velocity=flow_velocity,
                                               domain=domain_geometry)

        task = scenario.simulate(object_path=FLAGS.obj_path,
                                 num_iterations=FLAGS.num_iterations,
                                 resolution="low",
                                 run_async=True,
                                 machine_group=machine_group)

        # every 10 seconds, test if the simulation is done
        task_status = task.get_status()
        while task_status in ["submitted", "started"]:
            time.sleep(10)
            task_status = task.get_status()
        # Compute the cost of the simulation
        price_per_hour = machine_group.estimate_cloud_cost()
        duration = task.get_execution_time()
        # logg simulation time with 3 decimal places
        logging.info("Simulation duration: %.3f seconds", duration)
        cost = price_per_hour * duration / 3600
        logging.info("Estimated cost per task: %.3f $", cost)

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Terminating machine group.")
        machine_group.terminate()
    # pylint: disable=broad-except
    except Exception as e:
        logging.error("Error occurred: %s", e)
        machine_group.terminate()
    machine_group.terminate()


if __name__ == "__main__":
    app.run(main)
