""" Runs notebooks to check they still work """
from pytest import mark


@mark.parametrize(
    "filename", ["Creating a Job Folder", "IPython high-throughput interface"])
def test_notebooks(tmpdir, filename):
    """ Runs a given notebook in a tmpdir """
    import pylada
    from os.path import join, dirname
    from nbformat import read
    from nbconvert.preprocessors import ExecutePreprocessor
    from sys import version_info
    directory = join(dirname(pylada.__file__), 'notebooks')
    with open(join(directory, filename + ".ipynb")) as notebook_file:
        notebook = read(notebook_file, as_version=4)
    preprocessor = ExecutePreprocessor(kernel_name='python%i' %
                                       version_info.major)
    preprocessor.preprocess(notebook, {'metadata': {'path': str(tmpdir)}})
