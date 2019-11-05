import threading


class H2O:
    def __init__(self):
        _lock = threading.Lock()
        self.lock = threading.Condition(_lock)
        self.hs = [None] * 2
        self.o = None
        self.h_count = 0

    def h_come(self, h):
        with self.lock:
            while self.hs[0] is not None and self.hs[1] is not None:
                self.lock.wait()
            if self.hs[0] is None:
                self.hs[0] = h
            else:
                self.hs[1] = h
            self.lock.notify_all()

            while self.o is None or self.hs[0] is None or self.hs[1] is None:
                self.lock.wait()

            o = self.o
            self.h_count += 1
            self.lock.notify_all()
        return o

    def o_come(self, o):
        with self.lock:
            while self.o is not None:
                self.lock.wait()

            self.o = o
            self.lock.notify_all()

            while self.hs[0] is None or self.hs[1] is None:
                self.lock.wait()

            while self.h_count < 2:
                self.lock.wait()
            hs = '%s_%s' % (self.hs[0], self.hs[1])
            self.hs[0] = None
            self.hs[1] = None
            self.o = None
            self.h_count = 0
            self.lock.notify_all()
        return hs


def generate(name, func):
    n = 0
    while True:
        nm = '%s_%d' % (name, n)
        n += 1
        other = func(nm)
        print('%s, %s' % (nm, other), flush=True)


if __name__ == '__main__':
    h2o = H2O()
    t_ha = threading.Thread(target=generate, args=['HA', h2o.h_come], daemon=True)
    t_hb = threading.Thread(target=generate, args=['HB', h2o.h_come], daemon=True)
    t_o = threading.Thread(target=generate, args=['O', h2o.o_come], daemon=True)

    t_ha.start()
    t_hb.start()
    t_o.start()

    import time
    time.sleep(1)

