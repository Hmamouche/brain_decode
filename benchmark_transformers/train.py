import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import argparse

from src.Transformer import *
from src.Train_Function import train_model
from src.Test_Function import test_model
from src.Dataset import MyDataset
from src.Data_prepare import split_paths, read_split_append, read_statements, split_train_paths, split_val_paths, split_test_paths
from src.Tokenization import load_vocab_from_json, get_filler_tokens_id, add_filling_tokens_convert_to_tensor
from src.CNNEmbedding import CNNEmbedding
from src.Inference import inference

#from Metric import detokenize

models_dict = {
'Transformer':Transformer,
'CNNTransformer':CNNTransformer,
'DuplexTransformerConv':DuplexTransformerConv,
'BipartiteTransformerConv':BipartiteTransformerConv,
'DeconvBipartiteTransformerConv':DeconvBipartiteTransformerConv,
}
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_seeds", "-ns", help="Number of seeds to execute.", default = 1)
    parser.add_argument("--model_name", "-m", help="Name of the model to train.", choices = models_dict.keys())
    args = parser.parse_args()

    #name = 'BipartiteTransformerConv2 dff 512'
    name = args.model_name + ' dff 512'

    nb_seeds = args.nb_seeds
    
    lr = 0.0001
    d_model = 256
    d_ff = 512
    heads = 8
    N = 2
    epochs = 200
    batch_size = 64
    src_fmri_features = 274
    time_steps = 10
    wandb_log = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    column_name = 'Dynamic Token ids'
    vocab_path = 'speech_data/dynamic_statement.json'
    ######### Data loading ############

    fMRI_paths_train, fMRI_paths_val, fMRI_paths_test = split_paths('fMRI_data/sub')

    dialogues_paths_train = split_train_paths('speech_data/train/sub')
    #dialogues_paths_val = split_val_paths('speech_data/eval/sub')
    dialogues_paths_test = split_test_paths('speech_data/test/sub')
    dialogues_paths = dialogues_paths_train + dialogues_paths_test

    Tensor_list_train = read_split_append(fMRI_paths_train)
    #Tensor_list_val = read_split_append(fMRI_paths_val)
    Tensor_list_test = read_split_append(fMRI_paths_test)
    Tensor_list = Tensor_list_train + Tensor_list_test

    Statements_train = read_statements(dialogues_paths, column_name)
    #Statements_val = read_statements(dialogues_paths_val, column_name)
    Statements_test = read_statements(dialogues_paths_test, column_name)


    ############ Data Preparing and tokeization ##################

    vocab = load_vocab_from_json(vocab_path)
    flipped_vocab = {v: k for k, v in vocab.items()}


    vocab_len = len(vocab)
    padding_token_id, sos_token_id, eos_token_id = get_filler_tokens_id(vocab)

    token_ids_tensors_train = add_filling_tokens_convert_to_tensor(Statements_train, sos_token_id, eos_token_id)
    #token_ids_tensors_val = add_filling_tokens_convert_to_tensor(Statements_val, sos_token_id, eos_token_id)
    token_ids_tensors_test = add_filling_tokens_convert_to_tensor(Statements_test, sos_token_id, eos_token_id)

    max_seq_length_train = max([tensor.size(-1) for tensor in token_ids_tensors_train])
    #max_seq_length_val = max([tensor.size(-1) for tensor in token_ids_tensors_val])
    max_seq_length_test = max([tensor.size(-1) for tensor in token_ids_tensors_test])

    max_size = max(max_seq_length_train, max_seq_length_test)

    ############ Dataset innit #################

    Dataset = MyDataset(Tensor_list, token_ids_tensors_train, sos_token_id, eos_token_id, max_size)
    #Dataset_val = MyDataset(Tensor_list_val, token_ids_tensors_val, sos_token_id, eos_token_id, max_size)
    #Dataset_test = MyDataset(Tensor_list_test, token_ids_tensors_test, sos_token_id, eos_token_id, max_size)

    train_size = len(Dataset) - 250
    Dataset_train = Dataset.slice(0, train_size)

    txt = Dataset_train[0:1][1][0]

    #detokenize(txt[0], vocab)

    # val_size = len(Dataset_train) - 100
    # Dataset_val = Dataset_train.slice(val_size, train_size)
    Dataset_test = Dataset.slice(train_size, len(Dataset))

    ############### Model Innit ##############
    model_class = models_dict[args.model_name]
    model = model_class(time_steps, src_fmri_features, max_size, vocab_len, d_model, d_ff, N, heads, device).to(device)
    model = model.float()

    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    ########### Training ###################
    #model = torch.load('trained_models/%s.pt'%args.model_name)

    for seed in range (0 , args.nb_seeds, 1):
        torch.manual_seed(seed)
        train_model(name, model, Dataset_train, batch_size, optim, epochs, lr, N, sos_token_id, eos_token_id, padding_token_id,max_size, flipped_vocab, device, wandb_log)
    
        torch.save(model, 'trained_models/%s_%s.pt'%(args.model_name, seed))
        
        #test_model(name, model, Dataset_test, batch_size, lr, N, sos_token_id, eos_token_id, padding_token_id, max_size, flipped_vocab, device, wandb_log)
        #model, saving_file, vocab, dataset, sos_token_id, eos_token_id, padding_token_id, max_seq_len, device
        #saving_file = 'results/%s_%s.txt'%(args.model_name, seed)
        #inference(model, saving_file, vocab, Dataset_test, sos_token_id, eos_token_id, padding_token_id, 72, device)
