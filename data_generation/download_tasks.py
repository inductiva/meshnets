"""This file downloads the requested tasks"""
from absl import app
from absl import flags
from absl import logging

import os
import json
import warnings

import inductiva

FLAGS = flags.FLAGS

flags.DEFINE_string("path_to_sim_info", None,
                    "The path to the simulation json info files.")


def main(_):
    # Read the simulation ids
    with open(FLAGS.path_to_sim_info, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    task_ids = json_data["task_ids"]

    logging.info("Number of tasks: %s", len(task_ids))

    tasks_successfully_completed = []
    for task_id in task_ids:
        task = inductiva.tasks.Task(task_id)
        status = task.get_status()
        if status == "success":
            tasks_successfully_completed.append(task)

    logging.info("Tasks successfully completed: %s",
                 len(tasks_successfully_completed))

    download_dir = os.path.dirname(FLAGS.path_to_sim_info)
    for task in tasks_successfully_completed:
        save_path = os.path.join(download_dir, task.id)
        output = task.get_output()
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            warnings.warn(
                f"Download task {task.id} for which no directory existed")
        output.get_object_pressure_field(
            save_path=os.path.join(save_path, "pressure_field.vtk"))


if __name__ == "__main__":
    app.run(main)
