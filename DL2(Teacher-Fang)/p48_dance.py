from p47_buffer import Buffer
import threading


class Dance:
    def __init__(self):
        _lock = threading.Lock()
        self.lock = threading.Condition(_lock)
        self.girl_name = None
        self.boy_name = None

    def girl_come(self, girl_name):
        with self.lock:
            while self.girl_name is not None:
                self.lock.wait()
            self.girl_name = girl_name

            while self.boy_name is None:
                self.lock.wait()

            boy_name = self.boy_name
            self.boy_name = None
            self.lock.notify_all()
        return boy_name

    def boy_come(self, boy_name):
        with self.lock:
            while self.boy_name is not None:
                self.lock.wait()
            self.boy_name = boy_name

            while self.girl_name is None:
                self.lock.wait()

            girl_name = self.girl_name
            self.girl_name = None
            self.lock.notify_all()
        return girl_name


def boys(dance: Dance):
    n = 0
    while True:
        boy_name = 'boy_%d' % n
        n += 1
        girl_name = dance.boy_come(boy_name)
        print('%s <==> %s' % (boy_name, girl_name), flush=True)


def girls(dance: Dance):
    n = 0
    while True:
        import time
        time.sleep(0.1)
        girl_name = 'girl_%d' % n
        n += 1
        boy_name = dance.girl_come(girl_name)
        print('%s <==> %s' % (girl_name, boy_name), flush=True)


if __name__ == '__main__':
    dance = Dance()
    t_boy = threading.Thread(target=boys, daemon=True, args=[dance])
    t_girl = threading.Thread(target=girls, daemon=True, args=[dance])

    t_girl.start()
    t_boy.start()

    import time
    time.sleep(2)
