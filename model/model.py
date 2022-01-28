import torch
from torch import nn

from grover.model.models import GROVEREmbedding

class FpConfig:
    def __init__(self,
                 output_type = 'concat',
                 pretrained_model_path='models_pretrained/grover_base.pt',
                 device=None):
        self.output_type = output_type
        self.pretrained_model_path = pretrained_model_path
        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GroverAtomicFpGenerate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_type = config.output_type
        grover_base = torch.load(config.pretrained_model_path, map_location=config.device)
        grover_state_dict = grover_base['state_dict']
        grover_config = grover_base['args']
        setattr(grover_config, 'dropout', 0.1)
        setattr(grover_config, 'cuda', torch.cuda.is_available())
        self.grover = GROVEREmbedding(grover_config)
        # self.readout = AtomFpReadout(config)
        self.load_state_dict(grover_state_dict)
        self.eval()
    
    def forward(self, graph_batch: list):
        
        def _chop(embedding: torch.Tensor, atom_scope):
            atom_fp_per_mol = []
            for start_id, size in atom_scope:
                cur_embedding = embedding.narrow(0, start_id, size)
                atom_fp_per_mol.append(cur_embedding)

            return tuple(atom_fp_per_mol)
        
        _, _, _, _, _, atom_scope, _, _ = graph_batch
        embedding = self.grover(graph_batch)
        atom_from_atom = embedding['atom_from_atom']
        atom_from_bond = embedding['atom_from_bond']
        if self.output_type=='atom_from_atom':
            atom_embedding = atom_from_atom
        elif self.output_type=='atom_from_bond':
            atom_embedding = atom_from_bond
        elif self.output_type=='sum':
            atom_embedding = atom_from_atom + atom_from_bond
        elif self.output_type=='concat':
            atom_embedding = torch.cat([atom_from_atom, atom_from_bond], -1)
        else:
            raise ValueError(f'Unsupported output type: {self.output_type}.\n' + \
                             'Available output types: \'atom_from_atom\', \'atom_from_bond\', \'sum\', \'concat\'.')
        
        return _chop(atom_embedding, atom_scope)