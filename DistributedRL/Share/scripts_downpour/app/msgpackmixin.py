# -*- coding: utf-8 -*-

class MsgpackMixin:
    def to_msgpack(self, *args, **kwargs):
        return self.__dict__ #msgpack.dump(self.to_dict(*args, **kwargs))

    @classmethod
    def from_msgpack(cls, encoded):
        obj = cls()
        obj.__dict__ = {k.decode('utf-8'): v for k, v in encoded.items()}
        return obj