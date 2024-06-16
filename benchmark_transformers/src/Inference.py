import torch
import torch.nn.functional as F
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import wandb
from src.LPTS import LPTS
from src.Metric import word_overlap_percentage, jaccard_similarity, detokenize, remove_word

def inference(model, saving_file, vocab, dataset, sos_token_id, eos_token_id, padding_token_id, max_seq_len, device):

#     if wandb_log:
#         wandb.init(
#             settings=wandb.Settings(start_method="thread"),
#             project="Final_Metrics",
#             name=f"{name}_bs{batch_size}_lr{lr}_N{N}",
#             config={
#                 "epochs": num_epochs,
#                 "batch_size": batch_size,
#             })
    total_loss = 0
    total_bleu = 0
    total_word_overlap_test = 0
    total_lpips_test = 0
    total_jaccard_test = 0
    criterion = nn.CrossEntropyLoss()
    nlp_lpips = LPTS()
    vocab = {v: k for k, v in vocab.items()}
    total_samples = len(dataset)
    for batch_start in range(0, total_samples, 1): #total_samples
        batch_end = min(batch_start + 1, total_samples)
        batch = dataset[batch_start:batch_end]

        src, trg = batch
        src = src.to(device)
        trg = trg.to(device)

        output_ids, _, output = generate_sentence_ids(model, src.float(), sos_token_id, eos_token_id, padding_token_id, max_seq_len, len(vocab), device)
        output = output.float()
        loss = criterion(output.view(-1, output.size(-1)), trg.view(-1))
        bleu_score, lpips_dist, overlap_score, jaccard_score = print_output_sentence(output_ids, trg, saving_file, vocab, padding_token_id, eos_token_id, nlp_lpips)
        total_loss += loss
        total_bleu += bleu_score
        total_lpips_test += lpips_dist.item()
        total_word_overlap_test += overlap_score
        total_jaccard_test += jaccard_score

    Loss = total_loss / total_samples
    bleu_score = total_bleu / total_samples
    word_overlap = total_word_overlap_test / total_samples
    lpips = total_lpips_test / total_samples
    jaccard = total_jaccard_test / total_samples

    print(f"Sequence to sequence: BLEU: {bleu_score:.4f} Word Overlap: {word_overlap:.4f}% Jaccard Similarity: {jaccard:.4f} Test NLPIPS: {lpips:.4f}")
#     print(counter)
#     if wandb_log:
#             metrics = {"Loss": Loss, "Bleu": bleu_score, "Word Overlap": word_overlap, "ENLS": lpips, "Jaccard": jaccard}
#             wandb.log(metrics)

def inference_without_sequence_to_sequence(model, saving_file, vocab, dataset, sos_token_id, eos_token_id, padding_token_id, max_seq_len, device, wandb_log, name, batch_size, lr, N, num_epochs):

    if wandb_log:
        wandb.init(
            settings=wandb.Settings(start_method="thread"),
            project="Final_Metrics",
            name=f"{name}_bs{batch_size}_lr{lr}_N{N}",
            config={
                "epochs": num_epochs,
                "batch_size": batch_size,
            })
    total_loss = 0
    total_bleu = 0
    total_word_overlap_test = 0
    total_lpips_test = 0
    total_jaccard_test = 0
    criterion = nn.CrossEntropyLoss()
    nlp_lpips = NLPLPIPS()
    vocab = {v: k for k, v in vocab.items()}
    total_samples = len(dataset)
    for batch_start in range(0, total_samples, 1):
        torch.cuda.empty_cache()
        batch_end = min(batch_start + 1, total_samples)
        batch = dataset[batch_start:batch_end]

        src, trg = batch
        src = src.to(device)
        trg = trg.to(device)
        output, softmax_output = model(src.float(), trg.float())
        output = output.float()
        loss = criterion(output.view(-1, output.size(-1)), trg.view(-1))
        bleu_score, lpips_dist, overlap_score, jaccard_score = print_output_sentence(softmax_output.argmax(dim=-1), trg, saving_file, vocab, padding_token_id, eos_token_id, nlp_lpips)

        total_loss += loss
        total_bleu += bleu_score
        total_lpips_test += lpips_dist.item()
        total_word_overlap_test += overlap_score
        total_jaccard_test += jaccard_score

    Loss = total_loss / total_samples
    bleu_score = total_bleu / total_samples
    word_overlap = total_word_overlap_test / total_samples
    lpips = total_lpips_test / total_samples
    jaccard = total_jaccard_test / total_samples

    print(f" Training Inference: BLEU: {bleu_score:.4f} Word Overlap: {word_overlap:.4f}% Jaccard Similarity: {jaccard:.4f} Test NLPIPS: {lpips:.4f}")
    if wandb_log:
            metrics = {"Loss": Loss, "Bleu": bleu_score, "Word Overlap": word_overlap, "ENLS": lpips, "Jaccard": jaccard}
            wandb.log(metrics)


def generate_sentence_ids(model, src, sos_token_id, eos_token_id, pad_token_id, max_length, vocab_len, device):
    model.eval()
    bs = src.size(0)

    # Preallocate memory
    sentences = torch.full((bs, max_length), pad_token_id, dtype=torch.float32, device=device)
    sentences[:, 0] = sos_token_id
    logits = torch.full((bs, max_length, vocab_len), 0, dtype=torch.float32, device=device)
    prob = torch.full((bs, max_length, vocab_len), 0, dtype=torch.float32, device=device)

    # Encoder once
    e_outputs, src_mask = model.encoder(src.float())
    with torch.no_grad():
        for t in range(1, max_length): #max_length
            current_tokens = sentences[:, :t]
            current_tokens_padded = torch.nn.functional.pad(current_tokens, (0, max_length - t), value=pad_token_id)
            d_output = model.decoder(current_tokens_padded.float(), e_outputs, src_mask)
            output = model.out(d_output)

            next_logit = output[:, t, :]
            logits[:, t, :] = next_logit

            softmax_output = F.softmax(output, dim=2)

            next_prob = softmax_output[:, t, :]
            prob[:, t, :] = next_prob

            next_tokens = next_prob.argmax(dim=-1)
            sentences[:, t] = next_tokens

    return sentences.to(device), prob.to(device), logits.to(device)

def print_output_sentence(output, trg_des, saving_file, vocabulary, padding_token_id, eos_token_id, nlp_lpips):
    smoothie = SmoothingFunction().method1
    trg_des = trg_des.flatten().tolist()
    trg_desc =[x for x in trg_des if x != padding_token_id]
    trg_desc = [x for x in trg_desc if x != eos_token_id]
    trg_desc = [x for x in trg_desc if x != 1]
    output_words = [vocabulary[token_id.item()] for token_id in output[0]]
    #print(output_words)
    output_words =[x for x in output_words if x != '[PAD]']
    output_words = [x for x in output_words if x != '[EOS]']
    output_words = [x for x in output_words if x != '[SOS]']
    unique_list = list(set(output_words))
    desc_words = [vocabulary[token_id] for token_id in trg_desc]

    output_sentence = ' '.join(output_words)
    output_sentence1 = ' '.join(unique_list)
    desc_sentence = ' '.join(desc_words)

    bleu_score = sentence_bleu([desc_sentence.split()], output_sentence.split(), smoothing_function=smoothie)
    lpips_dist = nlp_lpips.distance(desc_sentence, output_sentence)
    overlap_score = word_overlap_percentage(desc_sentence, output_sentence)
    jaccard_score = jaccard_similarity(desc_sentence, output_sentence)

    with open(saving_file, 'a') as f:
        f.write(f'The predicted Conversation : {output_sentence1}\nThe target Conversation : {desc_sentence}\n\n')
    return bleu_score, lpips_dist, overlap_score, jaccard_score
