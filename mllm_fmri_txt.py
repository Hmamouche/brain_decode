from transformers import AutoTokenizer
from packaging import version
import transformers
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM

from torch.optim import Adam
from benchmark_transformers.src.Transformer import DeconvBipartiteTransformerConv
from load_data import data_builder
import os


from PIL import Image
from CLIP import clip



class BrVicunaInstruct(nn.Module):

    def __init__(
        self,
        img_size=256,
        drop_path_rate=0,
        max_txt_len=128,
        max_output_txt_len=256,
    ):
        super().__init__()
        transformers_version = version.parse(transformers.__version__)

        model_name_or_path = "llm/vicuna-7b-v1.3"

        self.device = "cuda"

        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        self.frmi_encoder = torch.load('benchmark_transformers/trained_models/DeconvBipartiteTransformerConv_0.pt', map_location=torch.device(self.device)).encoder.to(self.device)


        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.llm_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                              device_map=self.device,
                                                              trust_remote_code=True,
                                                              torch_dtype=torch.bfloat16,
                                                              local_files_only=True)



        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        # self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False

        self.llm_proj = nn.Linear(256, 4096).to(self.device)


        self.vision_llm_proj = nn.Linear(512, 4096).to(self.device)

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len


    def maybe_autocast(self, dtype=torch.float16):
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()


    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len


    def forward (self, sample):

        output_text = sample["text_output"]
        # Input text
        input_text = ["En se basant sur ce contenu, réponds en Français à la phrase suivante '" + a  + "' : " for a in sample["text_input"]]

        # BOLD embeddings
        embeddings, masks = self.frmi_encoder (sample["bold_signal"].to(self.device))#.to(device))
        embeddings = embeddings[-1]
        inputs_llm_bold = self.llm_proj (embeddings)
        atts_llm_bold = torch.ones(inputs_llm_bold.size()[:-1], dtype=torch.long).to(self.device)


        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            input_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(self.device)

        self.llm_tokenizer.truncation_side = 'right'
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in output_text],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(self.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )


        # do not apply loss to the padding
        targets = llm_tokens['input_ids'].masked_fill(llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100)
        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        # do not apply loss to the image and bold
        empty_targets_bold = (torch.ones(atts_llm_bold.size(), dtype=torch.long).to(self.device).fill_(-100))
        #empty_targets_image = (torch.ones(atts_llm_image.size(), dtype=torch.long).to(self.device).fill_(-100))


        targets = torch.cat([empty_targets_bold, targets], dim=1)


        # Input embeddings
        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])


        inputs_embeds = torch.cat([inputs_llm_bold, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm_bold, llm_tokens['attention_mask']], dim=1)

        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        loss = outputs.loss

        return loss


    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=1,
        max_length=400,
        max_new_tokens = 100,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1):

        self.llm_tokenizer.padding_side = "left"

        image = samples["image"]
        bs = image.size(0)

        prompt = ["En se basant sur ce contenu, réponds en Français à la phrase suivante '" + a  + "' : " for a in samples["text_input"]]

        # Bold embedding
        bold_embeddings, _ = self.frmi_encoder (samples["bold_signal"].to(self.device))#.to(device))
        bold_embeddings = bold_embeddings[-1]
        inputs_llm_bold = self.llm_proj (bold_embeddings)
        atts_llm_bold = torch.ones(inputs_llm_bold.size()[:-1], dtype=torch.long).to(self.device)



        llm_tokens = self.llm_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt"
        ).to(self.device)

        with self.maybe_autocast():
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            #attention_mask = llm_tokens.attention_mask
            inputs_embeds = torch.cat([inputs_llm_bold, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm_bold, llm_tokens.attention_mask], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                # eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text



def save_checkpoint(model, cur_epoch, is_best=False):

    param_grad_dic = {
        k: v.requires_grad for (k, v) in model.named_parameters()
    }
    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            # delete parameters that do not require gradient
            del state_dict[k]

    save_obj = {"model": state_dict,"epoch": cur_epoch}

    os.system ("rm trained_models/checkpoint_mllm_fmri_txt*")
    save_to = "trained_models/checkpoint_mllm_fmri_txt_{}.pth".format("best" if is_best else cur_epoch)
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


def train (model, data_loader, epochs = 100, save_iters = 10, starting_epoch = 1):

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
		if epoch % save_iters == 0:
			print ('-------- Epoch: ', epoch)
			save_checkpoint(model, epoch)

def test (model, data_loader):

    model.eval()
    f = open("results/mllm_fmri_txt.txt", "a")


    for sample in data_loader:
        output_text = model.generate (sample)
        for predicted, target in zip (output_text, sample["text_output"]):
            f.write("The predicted Conversation :")
            f.write(predicted + "\n")
            f.write("The target Conversation :")
            f.write(target + "\n\n")

    f.close()

if __name__ == '__main__':
    batch_size = 32
    data_loader = data_builder(batch_size)
    llm = BrVicunaInstruct()
    #llm = load_checkpoint(llm, "checkpoint_vicuna_200.pth")
    train (llm, data_loader["train"], epochs = 200, save_iters = 10, starting_epoch=1)
    test (llm, data_loader["test"])
