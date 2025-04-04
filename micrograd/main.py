from value import Value
from graphviz import Digraph

def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
    if n._op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._op, label = n._op)
      # and connect this node to it
      dot.edge(uid + n._op, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  return dot

def lol():
    # small amount h
    h = 0.001

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    # d = a * b + c; d.label = 'd' # creates d with -6, which is passed as child with the value of c to children
    e = a * b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'
    L1 = L.data

    # added to a
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    b.data += h
    c = Value(10.0, label='c')
    # d = a * b + c; d.label = 'd' # creates d with -6, which is passed as child with the value of c to children
    e = a * b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'
    L2 = L.data

    # change in L, normalize by H (rise / run)
    # deriv of L w/ respect to h
    print((L2 - L1) / h)

    L.grad = (L2 - L1) / h

def draw():
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    # d = a * b + c; d.label = 'd' # creates d with -6, which is passed as child with the value of c to children
    e = a * b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'; L.grad = 1.0
    
    f.grad = 4.0
    d.grad = -2.0
    c.grad = -2.0
    e.grad = -2.0
    b.grad = -4.0
    a.grad = 6.0

    # one step of an optimization, hyperparameter alpha
    alpha = 0.01
    a.data += alpha * a.grad
    b.data += alpha * b.grad
    c.data += alpha * c.grad
    f.data += alpha * f.grad

    e = a * b
    d = e + c
    L = d * f
    print(L.data)
    draw_dot(L).render()

if __name__ == "__main__":
    # manual backprop: how much does L change with a small change h?
    lol()
    draw()

    # back a step:
    # what is dL/dd? (propagating backwards)
    # dL / dd = f
    # L = d * f, so grad(d) = f, grad(f) = d

    # dL / dc = 
    # L = d * f = (c + e) * f
    # dL / dc = dL / dd * dd / dc = -2.0 * 1 = -2.0 = dL / de 
    # d = c + e, so dd/dc = 1 + 0, dd/de = 1.0

    # dL / db = dL / dd * dd / de * de / db
    # e = b * a, de / db = a, de / da = b
    # dL / db = -2.0 * 1.0 * 2.0 = -4.0
    # dL / da = -2.0 * 1.0 * -3.0 = 6.0