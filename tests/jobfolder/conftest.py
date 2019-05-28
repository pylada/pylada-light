from pytest import fixture


def Extract(outdir=None):
    from os.path import exists
    from os import getcwd
    from collections import namedtuple
    from pickle import load
    from pylada.misc import local_path

    if outdir == None:
        outdir = getcwd()

    Extract = namedtuple("Extract", ["success", "directory", "indiv", "functional"])
    if not exists(outdir):
        return Extract(False, outdir, None, functional)

    outdir = local_path(outdir)
    if not outdir.join("OUTCAR").check(file=True):
        return Extract(False, str(outdir), None, functional)
    indiv, value = load(outdir.join("OUTCAR").open("rb"))

    return Extract(True, outdir, indiv, functional_call)


def functional_call(indiv, outdir=None, value=False, **kwargs):
    from pylada.misc import local_path
    from pickle import dump

    outdir = local_path(outdir)
    outdir.ensure(dir=True)
    dump((indiv, value), outdir.join("OUTCAR").open("wb"))
    return Extract(str(outdir))


functional_call.Extract = Extract


@fixture(scope="session")
def functional():
    return functional_call
