from torch_geometric.datasets import Planetoid

ds = Planetoid(root='/tmp/Cora', name='Cora')

data = ds[0]

