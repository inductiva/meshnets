# Generate data

Use these scripts to generate data with the Inductiva API.

The first thing that you need to do is export your api key as an
environment variable:

```bash
export INDUCTIVA_API_KEY = <Your API Key>
```

To submit the tasks use the script `submit_tasks.py`. This will submit
the tasks to the API and save a `.json` file with their IDs and the
machine group where they ran. To run the script you will need `.obj`
objects over which to run the simulations. Once you have the `.obj`
objects over which to run the simulation run `python submit_tasks.py
--input_dataset=<path to .obj objects>`.

To monitor the tasks status use the script `monitor_tasks.py` with
`python monitor_tasks.py --path_to_sim_info=<json with ids and machien
group>`. It will output something like:

```bash
I0927 10:09:20.328027 140566996619328 download_tasks.py:26] Number of tasks: 30
I0927 10:09:23.044083 140566996619328 download_tasks.py:47] Tasks still running: 0
I0927 10:09:23.044179 140566996619328 download_tasks.py:48] Tasks successfully completed: 30
I0927 10:09:23.044209 140566996619328 download_tasks.py:50] Tasks failed: 0
I0927 10:09:23.044236 140566996619328 download_tasks.py:51] Tasks killed: 0
I0927 10:09:23.044259 140566996619328 download_tasks.py:52] Other tasks: 0
```

This script also terminates the machine group when all tasks are done.

To then fetch the output of these tasks you can run the script
`download_tasks.py`.
