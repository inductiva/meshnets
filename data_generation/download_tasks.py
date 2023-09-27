"""This file downloads the requested tasks"""
import os

from absl import app
from absl import flags
from absl import logging

import inductiva

FLAGS = flags.FLAGS

flags.DEFINE_string('path_to_sim_ids', None,
                    'The path to the simulation ids txt files.')

flags.DEFINE_string(
    'download_dir', None,
    'Where to download the data. If None, the data is not downloaded.')


def main(_):
    inductiva.api_key = os.environ['INDUCTIVA_API_KEY']

    # Read the simulation ids
    with open(FLAGS.path_to_sim_ids, 'r') as f:
        task_ids = f.read().splitlines()
    logging.info('Number of tasks: %s', len(task_ids))

    tasks_still_running = []
    tasks_successfully_completed = []
    tasks_failed = []
    tasks_killed = []
    other_tasks = []
    for task_id in task_ids:
        task = inductiva.tasks.Task(task_id)
        status = task.get_status()
        if status == 'success':
            tasks_successfully_completed.append(task_id)
        elif status in ['submitted', 'started']:
            tasks_still_running.append(task_id)
        elif status == 'failed':
            tasks_failed.append(task_id)
        elif status == 'killed':
            tasks_killed.append(task_id)
        else:
            other_tasks.append(task_id)

    logging.info('Tasks still running: %s', len(tasks_still_running))
    logging.info('Tasks successfully completed: %s',
                 len(tasks_successfully_completed))
    logging.info('Tasks failed: %s', len(tasks_failed))
    logging.info('Tasks killed: %s', len(tasks_killed))
    logging.info('Other tasks: %s', len(other_tasks))

    if FLAGS.download_dir is not None:
        for task_id in tasks_successfully_completed:
            task = inductiva.tasks.Task(task_id)
            output = task.get_output()
            save_path = f'{FLAGS.download_dir}_{task_id}'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            output.get_object_pressure_field(
                save_path=os.path.join(save_path, 'pressure_field.vtk'))


if __name__ == '__main__':
    app.run(main)
