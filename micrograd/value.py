class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        # _children is a tuple when passed in, but maintained as a set (for efficiency?)
        self.data = data
        self.grad = 0 # by default gradient is 0 because we assume it does not affect loss function
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out