""" Fake pwscf programs """


def data_path(*args):
    from py.path import local
    return local(__file__).dirpath().dirpath().join("data", *args)


def copyoutput(tmpdir, src, dest):
    """ Copies pre-determined output to given directory """
    from sys import executable
    from stat import S_IREAD, S_IWRITE, S_IEXEC
    result = tmpdir.join("copy.py") if tmpdir.check(dir=True) else tmpdir
    result.write(
        "#! %s\n" % executable
        + "from py.path import local\n"
        + "src = local('%s')\n" % src
        + "dest = local('%s')\n" % dest
        + "dest.ensure(dir=True)\n"
        + "for u in src.listdir(lambda x: x.basename != 'pwscf.in'):\n"
        + "  u.copy(dest.join(u.basename))\n"
    )
    result.chmod(S_IREAD | S_IWRITE | S_IEXEC)
    return result
