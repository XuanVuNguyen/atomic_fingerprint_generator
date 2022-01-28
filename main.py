import argparse
import sys
import os
import logging
import numpy as np
from tqdm import tqdm
from torchinfo import summary
from model.model import GroverAtomicFpGenerate, FpConfig
from model.data import get_dataset, get_dataloader
from model.util import a2a_to_distance

def setup_parser():
    parser = argparse.ArgumentParser('Extract atom fingerprint')
    parser.add_argument('--output_type',
                        help='How to handle the atom features output by GROVEREmbedding',
                        choices=['concat', 'add', 'atom_from_atom', 'atom_from_bond'],
                        type=str,
                        default='concat')
    parser.add_argument('--capped_distance',
                        help='Entries of the relative distance matrices are capped by this value',
                        type=int,
                        default=30)
    parser.add_argument('--pretrained_model_path',
                        help='Path to a .pt file containing config and state dict of pretrained GROVER',
                        type=str,
                        default='models_pretrained/grover_base.pt')
    parser.add_argument('--device',
                        choices=['cpu', 'cuda', 'None'],
                        help='Device name. If None then use whatever device available',
                        type=str,
                        default='None')
    parser.add_argument('--data_path',
                        help='Path to a .csv file containing data',
                        type=str,
                        default='exampledata/pretrain/tryout.csv')
    parser.add_argument('--fingerprint_save_dir',
                        help='Directory to save the outputs',
                        type=str,
                        default='extracted_fingerprint/grover_base_tryout/')
    return parser

def parse_args(parser: argparse.ArgumentParser):
    args = parser.parse_args()
    for attr in args.__dict__:
        if getattr(args, attr) == 'None':
            setattr(args, attr, None)
    return args
                        
def main(args):
    logging.getLogger().setLevel(logging.INFO)
    
    config = FpConfig(output_type=args.output_type,
                      pretrained_model_path=args.pretrained_model_path,
                      device=args.device)
    
    model = GroverAtomicFpGenerate(config)
    logging.info(f'Model defined, pretrained state dict loaded from: \'{config.pretrained_model_path}\'')
    logging.info('Model summary:\n' + str(summary(model, verbose=0)))
        
    mol_dataset = get_dataset(args.data_path)
    mol_dataloader = get_dataloader(mol_dataset)
    logging.info(f'Data get from: \'{args.data_path}\'. Total: {len(mol_dataset)}')
    
    if not os.path.exists(args.fingerprint_save_dir):
        os.makedirs(args.fingerprint_save_dir)
        logging.info(f'Save dir initiated at: \'{args.fingerprint_save_dir}\'')
    else:
        logging.info(f'Save dir existed at: \'{args.fingerprint_save_dir}\'')
        print(f'Save dir existed at: \'{args.fingerprint_save_dir}\'')
        if not input('Overwrite? [y/n]: ') == 'y':
            sys.exit()
        
        
    smiles_spath = os.path.join(args.fingerprint_save_dir, 'smiles.txt')
    fp_spath = os.path.join(args.fingerprint_save_dir, 'atom_fp.npy')
    distances_spath = os.path.join(args.fingerprint_save_dir, 'distances.npy')
    
    logging.info('Inference starts')
    with open(smiles_spath, 'w') as smiles_file, \
        open(fp_spath, 'wb') as fp_file, \
        open(distances_spath, 'wb') as distances_file:

        for step, batch_item in enumerate(tqdm(mol_dataloader)):
            batch_smiles, batch_inputs, _, _, _ = batch_item
            _, _, _, _, _, atom_scope, _, batch_a2a = batch_inputs
            
            batch_outputs = model(batch_inputs)
            batch_distances = a2a_to_distance(batch_a2a, atom_scope, args.capped_distance)
            assert len(batch_smiles)==len(batch_outputs) and len(batch_smiles)==len(batch_distances)
            for smiles, outputs, distances in zip(batch_smiles, batch_outputs, batch_distances):
                smiles_file.write(f'{smiles}\n')
                np.save(fp_file, outputs.detach().numpy())
                np.save(distances_file, distances)
                
if __name__=='__main__':
    parser = setup_parser()
    args = parse_args(parser)
    main(args)