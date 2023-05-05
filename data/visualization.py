from absl import app, flags, logging
import pyvista as pv

FLAGS = flags.FLAGS

flags.DEFINE_string("input_object", "data/blackbird.obj",
                    "File path of the object to be visualized")

def main(_):

    logging.info("Loading the object from %s", FLAGS.input_object)
    # Load the object
    mesh = pv.read(FLAGS.input_object)

    logging.info("Print the points of the object.")
    logging.info(mesh.points)

    logging.info("Plotting the object")
    # Plot the object
    mesh.plot(show_edges=True)

if __name__ == "__main__":
    app.run(main)

