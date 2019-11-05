from tornado import web, ioloop


class HelloHandler(web.RequestHandler):
    def __init__(self, application, request, name):
        super(HelloHandler, self).__init__(application, request)
        self.name = name

    def get(self, *args, **kwargs):
        name = self.get_argument('name', self.name)
        self.write('<span style="font-size: 56px">Hello %s!</span>' % name)


from sin import Sin

_sin_app = Sin()
# _sin_app.train()


class SinHandler(web.RequestHandler):
    def get(self, *args, **kwargs):
        x = self.get_argument('x')  # A exception will be raised if the x is not found
        x = float(x)
        result = _sin_app.predict(x)
        self.write('sin(%s) = %s' % (x, result))   # please use json other than a text in a real project.


if __name__ == '__main__':
    app = web.Application([
        ('/abc/(.*)', web.StaticFileHandler, {'path': './html/'}),
        ('/hello', HelloHandler, {'name': 'ZhangSan'}),
        ('/sin', SinHandler)
    ])
    app.listen(2345)
    current = ioloop.IOLoop.current()
    current.make_current()
    print('My web server is started')
    current.start()
    current.stop()
    print('My web server is stopped')
