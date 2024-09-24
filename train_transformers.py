import torch
import argparse
from tokenizers import Tokenizer
from src.load_data import data_builder
from src.transformers_src.Transformer import *
from src.transformers_src.Train_Function import train_model
from src.transformers_src.Inference import inference


models_dict = {
'Transformer':Transformer,
'CNNTransformer':CNNTransformer,
'DuplexTransformerConv':DuplexTransformerConv,
'BipartiteTransformerConv':BipartiteTransformerConv,
'DeconvBipartiteTransformerConv':DeconvBipartiteTransformerConv,
}
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", help="Name of the model to train.", choices = models_dict.keys(), default = "DeconvBipartiteTransformerConv")
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--retrain", action='store_true')
    parser.add_argument("-seed", default = 3)
    args = parser.parse_args()

    name = args.model_name
    torch.manual_seed(args.seed)

    # TODO: Using a config file
    lr = 0.0001
    d_model = 256
    d_ff = 512
    heads = 8
    N = 2
    epochs = 200
    batch_size = 64
    src_fmri_features = 200
    time_steps = 10
    wandb_log = False
    max_size = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pad_token_id, sos_token_id, eos_token_id = 0, 1, 2

    tokenizer = Tokenizer.from_file("tools/tokenizer-trained.json")
    vocab_len = tokenizer.get_vocab_size()

    # TODO: increase batch size in testing
    if args.test:
        batch_size = 1

    ################ Datasets ##############
    data_loader = data_builder(batch_size=batch_size)

    ################ Model Init ##############
    model_class = models_dict[args.model_name]
    model = model_class(time_steps, src_fmri_features, max_size, vocab_len, d_model, d_ff, N, heads, device).to(device)
    model = model.float()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    ################ Model Training/Testing ##############
    if args.test:
        model.load_state_dict(torch.load('trained_models/%s.pt'%(args.model_name), weights_only=True))
        saving_file = 'results/%s.txt'%(args.model_name)
        inference(model, saving_file, tokenizer, vocab_len, data_loader["test"], sos_token_id, eos_token_id, pad_token_id, max_size, device)

    else:
        if args.retrain:
            model = torch.load('trained_models/%s.pt'%(args.model_name), map_location=torch.device(device))
        train_model(name, model, data_loader["train"], batch_size, optim, epochs, lr, N, sos_token_id, eos_token_id, pad_token_id,max_size, tokenizer, device, wandb_log)
        torch.save(model.state_dict(), 'trained_models/%s.pt'%(args.model_name))
