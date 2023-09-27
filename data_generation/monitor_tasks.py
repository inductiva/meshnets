"""This script monitors the submited tasks"""
from absl import app
from absl import flags
from absl import logging

import json

import inductiva

FLAGS = flags.FLAGS

flags.DEFINE_string("path_to_sim_info", None,
                    "The path to the simulation info json file.")

flags.DEFINE_bool("terminate_when_done", True,
                  "Terminates the machine group when no tasks are running.")


def main(_):
    # Read the simulation ids
    with open(FLAGS.path_to_sim_info, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    task_ids = json_data["task_ids"]
    machine_group_name = json_data["machine_group"]

    logging.info("Number of tasks: %s", len(task_ids))

    tasks_still_running = []
    tasks_successfully_completed = []
    tasks_failed = []
    tasks_killed = []
    other_tasks = []
    for task_id in task_ids:
        task = inductiva.tasks.Task(task_id)
        status = task.get_status()
        if status == "success":
            tasks_successfully_completed.append(task_id)
        elif status in ["submitted", "started"]:
            tasks_still_running.append(task_id)
        elif status == "failed":
            tasks_failed.append(task_id)
        elif status == "killed":
            tasks_killed.append(task_id)
        else:
            other_tasks.append(task_id)

    logging.info("Tasks still running: %s", len(tasks_still_running))
    logging.info("Tasks successfully completed: %s",
                 len(tasks_successfully_completed))
    logging.info("Tasks failed: %s", len(tasks_failed))
    logging.info("Tasks killed: %s", len(tasks_killed))
    logging.info("Other tasks: %s", len(other_tasks))

    if len(tasks_still_running) == 0 and FLAGS.terminate_when_done:
        machine_groups = inductiva.resources.machine_groups.get()
        for machine_group in machine_groups:
            if machine_group.name == machine_group_name:
                machine_group.terminate()
                logging.info("Machine group %s deleted.", machine_group_name)
                break


if __name__ == "__main__":
    app.run(main)
