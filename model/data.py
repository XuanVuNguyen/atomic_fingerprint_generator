import csv
from collections import namedtuple
from torch.utils.data import DataLoader
from grover.data.moldataset import MoleculeDatapoint, MoleculeDataset
from grover.data.molgraph import MolCollator

def get_dataset(data_path: str):
    with open(data_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        lines = [line for line in csv_reader]
    mol_datapoints = [MoleculeDatapoint(line) for line in lines]
    mol_dataset = MoleculeDataset(mol_datapoints)
    return mol_dataset

def get_dataloader(dataset, batch_size: int=32):
    GraphArgs = namedtuple('GraphArgs',
                       ['bond_drop_rate',
                        'no_cache'])
    graph_args = GraphArgs(bond_drop_rate=0,
                           no_cache=True)
    mol_collator = MolCollator({}, graph_args)
    mol_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=mol_collator)
    return mol_loader