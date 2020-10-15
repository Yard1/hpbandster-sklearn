from time import sleep


class NameServerContext(object):
    def __init__(self, nameserver):
        self.nameserver = nameserver

    def __enter__(self):
        self.nameserver.start()

    def __exit__(self, type, value, traceback):
        self.nameserver.shutdown()
        sleep(0.2)


class OptimizerContext(object):
    def __init__(self, optimizer, *args, **kwargs):
        self.optimizer = optimizer
        self.run_args = args
        self.run_kwargs = kwargs

    def __enter__(self):
        return self.optimizer.run(*self.run_args, **self.run_kwargs)

    def __exit__(self, type, value, traceback):
        self.optimizer.shutdown(shutdown_workers=True)
        sleep(0.2)
