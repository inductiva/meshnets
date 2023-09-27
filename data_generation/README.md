# Generate data

Use this scripts to generate data with the Inductiva API.

The first thing that you need to do is export your api key as an
environment variable:

```bash
export INDUCTIVA_API_KEY = <Your API Key>
```

This will prevent accidentally committing it to github since now we
only need to do:

```
inductiva.api_key = os.environ['INDUCTIVA_API_KEY']
```


To submit the tasks use the script `submit_tasks.py`. This will submit
the tasks to the API and save a `.txt` file with their IDs. To run the
script you will `.vtk` objects over which to run the simulations. Once
you have the `.vtk` objects over which to run the simulation run
`python submit_tasks.py --input_dataset=<path to .vtk objects>`.

To then fetch the output of these tasks you can run the script
`download_tasks.py`. If no `--download_dir` is given the script will
only monitor the tasks and output something like:

```bash
I0927 10:09:20.328027 140566996619328 download_tasks.py:26] Number of tasks: 30
I0927 10:09:23.044083 140566996619328 download_tasks.py:47] Tasks still running: 0
I0927 10:09:23.044179 140566996619328 download_tasks.py:48] Tasks successfully completed: 30
I0927 10:09:23.044209 140566996619328 download_tasks.py:50] Tasks failed: 0
I0927 10:09:23.044236 140566996619328 download_tasks.py:51] Tasks killed: 0
I0927 10:09:23.044259 140566996619328 download_tasks.py:52] Other tasks: 0
```
