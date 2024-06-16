import wandb
import torch
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from torch.distributions import MultivariateNormal
from src.Metric import word_overlap_percentage, jaccard_similarity, detokenize, remove_word
from src.Inference import generate_sentence_ids
from src.LPTS import LPTS

def train_model(name, model, train_dataset, batch_size, optimizer, num_epochs, lr, N, sos_token_id, eos_token_id, padding_token_id, max_seq_len, vocab, device, wandb_log):
    if wandb_log:
        wandb.init(
            settings=wandb.Settings(start_method="thread"),
            project="Final_Metrics",
            name=f"{name}_bs{batch_size}_lr{lr}_N{N}",
            config={
                "epochs": num_epochs,
                "batch_size": batch_size,
            })
    criterion = nn.CrossEntropyLoss()
    nlp_lpips = LPTS()
    train_num_samples = len(train_dataset)
    smoothie = SmoothingFunction().method1
    word_to_remove1 = '[PAD]'
    word_to_remove2 = '[EOS]'
    word_to_remove3 = '[SOS]'
    for epoch in range(num_epochs):
        model.train()

        total_loss_train = 0.0
        total_bleu_train = 0
        total_word_overlap_train = 0
        total_lpips_train = 0
        total_jaccard_train = 0
        total_samples_training = 0

        total_loss_val = 0.0
        total_bleu_val = 0
        total_word_overlap_val = 0
        total_lpips_val = 0
        total_jaccard_val = 0
        total_samples_val = 0

        # Iterate over batches
        for batch_start in range(0, train_num_samples, batch_size):
            batch_end = min(batch_start + batch_size, train_num_samples)
            batch = train_dataset[batch_start:batch_end]

            src, trg = batch

            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()  # Clear gradients

            # Forward pass
            output, softmax_output = model(src.float(), trg.float())
            # Compute loss
            loss = criterion(output.reshape(-1, output.size(-1)), trg.reshape(-1))

            # Backward pass
            loss.backward()

            # Update model parameters
            optimizer.step()

            total_loss_train += loss.item() * trg.size(0)  # Accumulate loss
            total_samples_training += trg.size(0)  # Accumulate number of samples

            # Add references and hypotheses for BLEU calculation

            for i in range(trg.size(0)):
                # Get the reference sentence
                ref = trg[i].tolist()
                ref_sentence = detokenize(ref, vocab)
                ref_sentence = remove_word(ref_sentence, word_to_remove1)
                ref_sentence = remove_word(ref_sentence, word_to_remove2)
                ref_sentence = remove_word(ref_sentence, word_to_remove3)


                # Get the hypothesis
                hyp = softmax_output.argmax(dim=-1)[i].tolist()
                hyp_sentence = detokenize(hyp, vocab)
                hyp_sentence = remove_word(hyp_sentence, word_to_remove1)
                hyp_sentence = remove_word(hyp_sentence, word_to_remove2)
                hyp_sentence = remove_word(hyp_sentence, word_to_remove3)

                # Calculate BLEU score for this sentence pair
                bleu_score = sentence_bleu([ref_sentence.split()], hyp_sentence.split(), smoothing_function=smoothie)
                total_bleu_train += bleu_score

                # Calculate word overlap
                overlap_score = word_overlap_percentage(ref_sentence, hyp_sentence)
                total_word_overlap_train += overlap_score

                # Calculate LPIPS distance (assuming you have the function nlp_lpips.distance)
                lpips_dist = nlp_lpips.distance(ref_sentence, hyp_sentence)
                total_lpips_train += lpips_dist.item()

                # Calculate Jaccard similarity
                jaccard_score = jaccard_similarity(ref_sentence, hyp_sentence)
                total_jaccard_train += jaccard_score

        epoch_loss = total_loss_train / total_samples_training
        epoch_bleu = total_bleu_train / total_samples_training
        epoch_word_overlap = total_word_overlap_train / total_samples_training
        epoch_lpips = total_lpips_train / total_samples_training
        epoch_jaccard = total_jaccard_train / total_samples_training

        if wandb_log:
            metrics = {"Train_loss": epoch_loss, "Train_Bleu": epoch_bleu, "Train_Word_Overlap": epoch_word_overlap, "Train NLPIPS": epoch_lpips, "Train Jaccard": epoch_jaccard}
            wandb.log(metrics)

        print(f"Epoch {epoch + 1}/{num_epochs} Loss: {epoch_loss:.4f} BLEU: {epoch_bleu:.4f} Word Overlap: {epoch_word_overlap:.4f}% Jaccard Similarity: {epoch_jaccard:.4f} Train LPIPS: {epoch_lpips:.4f}")

        ## Evaluation ####

#         for batch_start in range(0, len(val_dataset), 1):
#                     batch_end = min(batch_start + 1, len(val_dataset))
#                     batch = val_dataset[batch_start:batch_end]

#                     src, trg = batch
#                     src, trg = src.to(device), trg.to(device)

#                     sentences, softmax_output, output_ids = generate_sentence_ids(model, src.float(), sos_token_id, eos_token_id,
#                                                                                   padding_token_id, max_seq_len, len(vocab), device)
#                     output = output_ids.float()

#                     # Compute loss
#                     loss = criterion(output.view(-1, output.size(-1)), trg.view(-1))

#                     total_loss_val += loss.item()  # Accumulate loss
#                     total_samples_val += trg.size(0)  # Accumulate number of samples

#                     # Add references and hypotheses for BLEU calculation
#                     for i in range(trg.size(0)):
#                             # Get the reference sentence
#                             ref = trg[i].tolist()
#                             ref_sentence = detokenize(ref, vocab)
#                             ref_sentence = remove_word(ref_sentence, word_to_remove1)
#                             ref_sentence = remove_word(ref_sentence, word_to_remove2)
#                             ref_sentence = remove_word(ref_sentence, word_to_remove3)

#                             # Get the hypothesis
#                             #hyp = softmax_output.argmax(dim=-1)[0][i].tolist()
#                             hyp = sentences[i].tolist()
#                             hyp_sentence = detokenize(hyp, vocab)
#                             hyp_sentence = remove_word(hyp_sentence, word_to_remove1)
#                             hyp_sentence = remove_word(hyp_sentence, word_to_remove2)
#                             hyp_sentence = remove_word(hyp_sentence, word_to_remove3)

#                             # Calculate BLEU score for this sentence pair
#                             bleu_score = sentence_bleu([ref_sentence.split()], hyp_sentence.split(), smoothing_function=smoothie)
#                             total_bleu_val += bleu_score

#                             # Calculate word overlap
#                             overlap_score = word_overlap_percentage(ref_sentence, hyp_sentence)
#                             total_word_overlap_val += overlap_score

#                             # Calculate LPIPS distance (assuming you have the function nlp_lpips.distance)
#                             lpips_dist = nlp_lpips.distance(ref_sentence, hyp_sentence)
#                             total_lpips_val += lpips_dist.item()

#                             # Calculate Jaccard similarity
#                             jaccard_score = jaccard_similarity(ref_sentence, hyp_sentence)
#                             total_jaccard_val += jaccard_score

#         val_loss = total_loss_val / total_samples_val
#         val_bleu_score = total_bleu_val / total_samples_val
#         val_word_overlap = total_word_overlap_val / total_samples_val
#         val_lpips = total_lpips_val / total_samples_val
#         val_jaccard = total_jaccard_val / total_samples_val
#         if wandb_log:
#                     metrics = {"Val_Loss": val_loss, "Val_Bleu": val_bleu_score, "Val_Word_Overlap": val_word_overlap, "Val_NLPIPS": val_lpips, "Val_Jaccard": val_jaccard}
#                     wandb.log(metrics)

#         print(f"Val Loss: {val_loss:.4f} Val BLEU: {val_bleu_score:.4f} Val Word Overlap: {val_word_overlap:.4f}% Val Jaccard Similarity: {val_jaccard:.4f} Val NLPIPS: {val_lpips:.4f}")
    print("Training completed!")
