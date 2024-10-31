class HelloDevice:
    def __init__(self):
        self._message = "Hello, World!"

    @property
    def message(self) -> str:
        return self._message

    @message.setter
    def message(self, value):
        self._message = value


devices = {'hello': HelloDevice()}
