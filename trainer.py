import argparse
import torch
from src.load_data import data_builder
from src.models import *


def save_checkpoint(model, model_name, cur_epoch, is_best=False):

    param_grad_dic = {
        k: v.requires_grad for (k, v) in model.named_parameters()
    }
    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            # delete parameters that do not require gradient
            del state_dict[k]

    save_obj = {"model": state_dict,"epoch": cur_epoch}

    os.system ("rm trained_models/%s*"%model_name)
    save_to = "trained_models/%s_%s.pth"%(model_name, ("best" if is_best else str (cur_epoch)))
    print("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
    torch.save(save_obj, save_to)


def load_from_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    try:
        model.load_state_dict(checkpoint["model"])
    except RuntimeError as e:
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


def train (model, model_name, data_loader, epochs = 100, save_iters = 10, starting_epoch = 1):

	model.train()
	optim = Adam(model.parameters(), lr=0.0001)
     
	best_loss = 100000

	for epoch in range(starting_epoch, epochs + 1):
		print ('-------- Epoch: ', epoch)
		mean_loss = 0
        
		if epoch > (epochs - 20):
			save_iters = 2

		for sample in data_loader:
			loss = model(sample)
			mean_loss += loss
			optim.zero_grad()
			loss.backward()
			optim.step()

		print (mean_loss / len (data_loader))
		if epoch % save_iters == 0 and mean_loss < best_loss:
			best_loss = mean_loss
			save_checkpoint(model, model_name, epoch)

def test (model, data_loader, model_name):
    model.eval()
    f = open("results/%s.txt"%args.model_name, "w")

    for sample in data_loader:
        output_text = model.generate (sample)
        for predicted, target in zip (output_text, sample["text_output"]):
            f.write("The predicted Conversation :")
            f.write(predicted + "\n")
            f.write("The target Conversation :")
            f.write(target + "\n")

    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default = 32, type = int)
    parser.add_argument("-seed", default = 3)
    parser.add_argument("--model_name", "-m", help="Name of the model to train.", choices = ["MllmBrainToTextV0", "MllmBrainToTextV1", "MllmBrainToTextV2"])
    parser.add_argument('--test', action='store_true', help = "test the model")
    parser.add_argument('--retrain', action='store_true', help = "retrain from existing checkpoint")
    parser.add_argument("--starting_epoch", default = 1, type = int)
    parser.add_argument("--save_epochs", default = 5, type = int)
    parser.add_argument("--epochs", default = 300, type = int)
    parser.add_argument("--saved_checkpoint", type = str)

    args = parser.parse_args()

    models_dict = {
    'MllmBrainToTextV0':MllmBrainToTextV0,
    'MllmBrainToTextV1':MllmBrainToText,
    'MllmBrainToTextV2':MllmBrainToTextV2,
    }

    torch.manual_seed(args.seed)

    data_loader = data_builder(args.batch_size)
    llm = models_dict[args.model_name]()


    if args.test:
        llm = load_from_checkpoint(llm, args.saved_checkpoint)
        test (llm, data_loader["test"], args.model_name)
    else:
        if args.retrain:
            llm = load_from_checkpoint(llm, args.saved_checkpoint)
            
        train (llm, 
               args.model_name,
               data_loader["train"],
               epochs = args.epochs,
               save_iters = args.save_iters,
               starting_epoch = args.starting_epoch)
