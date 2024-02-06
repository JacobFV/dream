from __future__ import annotations
from collections import namedtuple

from typing import Any, Generic, Iterable, Iterator, Mapping, TypeVar

# Tree[T] = Tree[T] | List[T] | Dict[T] | Set[T] | T
# Tree[]
# T should store shape, type, and bits
# You only have to use Tree on the root node

"""
So the hcildren aren't the same as the values.
The leaves don't have children, but they have values.

"""

T = TypeVar('T')
class Tree(Generic[T], Iterable[T]):

  def __init__(self, *vals, **kwval):
    assert len(vals) == 0 or len(kwval) == 0, "Cannot mix positional and keyword arguments"

    if len(vals) == 1:
      self._val = vals[0]
    elif len(vals) > 1:
      self._val = vals
    elif len(kwval) > 0:
      self._val = kwval
    else:
      self._val = None

  @property
  def val(self):
    return self._val

  @property
  def val_as_list(self):
    v = self.val
    if isinstance(v, Tree):
      return v.flat_val
    elif isinstance(v, (list, tuple, set, frozenset, Iterable)):
      return list(v)
    elif isinstance(v, (dict, Mapping)):
      return list(v.values())
    elif isinstance(v, (namedtuple)):
      return list(v._asdict().values())
    elif isinstance(v, (object)):
      return list(v.__dict__.values())
    else:
      return [v]

  def flatten(self, depth=None):
    # depth=None means flatten all levels
    def convert_to_list(v):
      if isinstance(v, Tree):
        return v.flatten(depth)
      elif isinstance(v, (list, tuple, set, frozenset, Iterable)):
        return list(v)
      elif isinstance(v, (dict, Mapping)):
        return list(v.values())
      elif isinstance(v, (namedtuple)):
        return list(v._asdict().values())
      elif isinstance(v, (object)):
        return list(v.__dict__.values())
      else:
        return [v]
    list_tree = self.map(convert_to_list)

    pass

  def map(self, f, recursive=False, ordering='post', fn_args=(), fn_kwargs={}):
    pass

  def foreach(self, f, recursive=False, ordering='post', fn_args=(), fn_kwargs={}):
    pass

  def sum(self, weights: Tree[float], keepleaves=False) -> Tree[float]:
    return self.map(lambda x: x.sum(weights, keepleaves), recursive=True)

  def dot(self, other):
    ...

  def __iter__(self) -> Iterator[T]:
    # assume post-order traversal
    return iter(self.val)

  def __sum__(self):
    return sum(self.val_as_list)

  def __add__(self, other):
    return self.map(sum, recursive=True)

  def __sub__(self, other):


def treemap(f, ...):
  pass

def treereduce(f, ...):
  pass

