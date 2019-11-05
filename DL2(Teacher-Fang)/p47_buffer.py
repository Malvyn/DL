import threading
import time


class Buffer:
    def __init__(self, capacity):
        self.buffer = [None] * capacity
        self.write_pointer = 0
        self.read_pointer = -1

        lock = threading.RLock()  # Reenterantable Lock
        self.has_data = threading.Condition(lock)
        self.has_buffer = threading.Condition(lock)

    def put(self, data):
        with self.has_buffer:   #  call self.has_buffer.__enter__()  ==> acquire()
            while True:
                next = (self.write_pointer + 1) % len(self.buffer)
                if next != self.read_pointer:
                    break
                self.has_buffer.wait()
            self.buffer[self.write_pointer] = data
            self.write_pointer = next
            self.has_data.notify_all()

    def get(self):
        with self.has_data:
            while True:
                next = (self.read_pointer + 1) % len(self.buffer)
                if next != self.write_pointer:
                    break
                self.has_data.wait()
            value = self.buffer[next]
            self.read_pointer = next
            self.has_buffer.notify_all()
            return value


class Samples:
    def __init__(self, capacity):
        self.buffer = Buffer(capacity)
        self.thread = threading.Thread(target=self._read, args=(), daemon=True)
        self.stopped = True

    def _read(self):
        num = 0
        while not self.stopped:
            self.buffer.put(num)
            print('---------- write %d' % num, flush=True)
            num += 1

    def start(self):
        """
        Start a thread that reading batch of samples from hard disk
        :return:
        """
        if not self.stopped:
            raise Exception('Thread is alread started, please start it just when it is stopped.')
        self.stopped = False
        self.thread.start()

    def stop(self):
        """
        Stop the thread
        :return:
        """
        self.stopped = True
        self.thread.join()  # wait until the thread is finished or stopped.

    def next_batch(self, batch_size):
        return self.buffer.get()


if __name__ == '__main__':
    samples = Samples(4)
    samples.start()  #  call read() asynchronizely

    for epoch in range(2000):
        data = samples.next_batch(123)
        print('%d: %s' % (epoch, data), flush=True)

    samples.stop()
