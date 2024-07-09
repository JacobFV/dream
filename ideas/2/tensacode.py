from __future__ import annotations
from typing import TypeVar, Generic, Type
import typing
import torch

EncT = TypeVar("EncT", bound=torch.Tensor)
ObjT = TypeVar("ObjT")


class Engine(Generic[EncT]):
    """TensaCode engine"""

    def __init__(self):
        super().__init__()

        # get the type of R
        if hasattr(self, "__orig_bases__"):
            param_type = self.__orig_bases__[0].__args__[0]
            self.enc_type = param_type
        else:
            # Fallback if no type is specified
            self.enc_type = torch.Tensor

    def encode(self, val: ObjT) -> EncT:
        """Encode a value of type T to a representation of type R"""
        return self.enc_type(val)  # type: ignore

    def decode(self, enc_val: EncT, obj_type: type[ObjT]) -> ObjT:
        """Decode a representation of type R to a value of type T"""
        return obj_type(enc_val)  # type: ignore
