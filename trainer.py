import argparse
import torch
from load_data import data_builder
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
    save_to = "trained_models/%s_{}.pth"%model_name.format("best" if is_best else cur_epoch)
    print("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
    torch.save(save_obj, save_to)


def load_checkpoint(model, checkpoint_path):
    #logging.info("Loading checkpoint from {}.".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    try:
        model.load_state_dict(checkpoint["model"])
    except RuntimeError as e:
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


def train (model, model_name, data_loader, epochs = 100, save_epochs = 10, starting_epoch = 1):

	model.train()
	optim = Adam(model.parameters(), lr=0.0001)

	for epoch in range(starting_epoch, epochs + 1):
		print ('Epoch %d'%epoch)
		mean_loss = 0

		for sample in data_loader:
			loss = model(sample)

			mean_loss += loss
			optim.zero_grad()
			loss.backward()
			optim.step()

		print (mean_loss / len (data_loader))
		if epoch % save_epochs == 0:
			print ('-------- Epoch: ', epoch)
			save_checkpoint(model, model_name, epoch)

def test (model, data_loader):

    model.eval()
    f = open("results/mllm_image_fmri_txt.txt", "a")

    for sample in data_loader:
        output_text = model.generate (sample)
        for predicted, target in zip (output_text, sample["text_output"]):
            f.write("The predicted Conversation :")
            f.write(predicted + "\n")
            f.write("The target Conversation :")
            f.write(target + "\n\n")

    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default = 32, type = int)
    parser.add_argument("--model_name", "-m", help="Name of the model to train.", choices = ["MllmBrainToText", "MllmBrainToTextV2"])
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument("--starting_epoch", default = 1, type = int)
    parser.add_argument("--save_epochs", default = 10, type = int)
    parser.add_argument("--epochs", default = 200, type = int)
    parser.add_argument("--saved_checkpoint", type = str, default = "trained_models/checkpoint_mllm_fmri_txt_200.pth")

    args = parser.parse_args()

    models_dict = {
    'MllmBrainToText':MllmBrainToText,
    'MllmBrainToTextV2':MllmBrainToTextV2,
    }

    data_loader = data_builder(args.batch_size)
    llm = models_dict[args.model_name]()

    if args.test:
        llm = load_checkpoint(llm, args.saved_checkpoint)
        test (llm, data_loader["test"])
    else:
        if args.retrain:
            llm = load_checkpoint(llm, args.saved_checkpoint)

        train (llm,
               args.model_name,
               data_loader["train"],
               epochs = args.epochs,
               save_epochs = args.save_epochs,
               starting_epoch = args.starting_epoch)
