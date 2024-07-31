class AutoDict(dict):
    def __missing__(self,
                    key
                    ):
        self[key] = type(self)()
        return self[key]
