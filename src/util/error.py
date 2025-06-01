class NotFoundError(Exception):
    reason: str

    def __init__(self, reason: str, *args):
        self.reason = reason
        super().__init__(*args)


class InvalidArgumentError(Exception):
    reason: str

    def __init__(self, reason: str, *args):
        self.reason = reason
        super().__init__(*args)
