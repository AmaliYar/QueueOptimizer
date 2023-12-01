class ThreadProprs:
    def __init__(self, time: int, cores: int, ram: int):
        self.cores = cores
        self.ram = ram
        self.time = time


class Threads:
    threads_times: list = []
    threads_cores: list = []
    threads_memory: list = []
    total_time = 0

    def add_thread(self, thread: ThreadProprs):
        self.threads_times.append(thread.time)
        self.threads_cores.append(thread.cores)
        self.threads_memory.append(thread.ram)
        self.total_time += thread.time


class QueueProps:
    def __init__(self, time, cpu, ram):
        self.cpu = cpu
        self.ram = ram
        self.max_time = time


def pack_properties(threads: Threads, queue: QueueProps) -> dict:
    return {
        'v': list(zip(threads.threads_times,
                      threads.threads_cores,
                      threads.threads_memory)),
        'V': list(zip([queue.max_time],
                      [queue.cpu],
                      [queue.ram]))
    }
