#!@PYTHON_EXECUTABLE@
def cmdl():
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description="Only usefull to debug the process module")
    parser.add_argument("--order", dest="order", type=int, default=1)
    parser.add_argument("--sleep", dest="sleep", type=float, default=0)
    parser.add_argument("--fail-mid-call",
                        dest="fail_mid_call", action='store_true')
    parser.add_argument("--fail-at-end",
                        dest="fail_at_end", action='store_true')
    return parser.parse_args()


def main():
    from platform import uname
    from sys import exit
    from time import sleep
    from numpy import array, pi
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = cmdl() if rank == 0 else None
    args = comm.bcast(args, root=0)

    h = 1.0 / float(max(1, args.order))
    result = 0e0
    for i in range(rank + 1, args.order + 1, size):
        x = h * (float(i) - 0.5)
        result += 4e0 / (1e0 + x * x)
        if args.sleep != 0:
            sleep(args.sleep)
    result = comm.reduce(result * h, op=MPI.SUM, root=0)

    if rank == 0:
        message = "pi to order %i is approximately %f, Error is %f" \
            "  -- slept %s seconds at each iteration -- "       \
            " mpi world size is %i " % (
                args.order, result, abs(result - pi),
                args.sleep, size
            )
        print(message)
        if args.fail_mid_call:
            exit(2)
        print("sysname: %s\nnodename: %s\nrelease: %s\n"
              "compilation: %s\nversion: %s\nmachine: %s\n" % uname())

    exit(2 if args.fail_at_end else 0)

if __name__ == '__main__':
    main()
