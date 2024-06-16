import torch
import torch.nn as nn
import wandb
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from src.Metric import word_overlap_percentage, jaccard_similarity, detokenize, remove_word
from src.Inference import generate_sentence_ids
from src.LPTS import LPTS


def test_model(name, model, dataset, batch_size, lr, N, sos_token_id, eos_token_id, padding_token_id,
                                           max_seq_len, vocab, device, wandb_log):
    if wandb_log:
        wandb.init(
                settings=wandb.Settings(start_method="thread"),
                project="Final_Metrics",
                name=f"{name}_bs{batch_size}_lr{lr}_N{N}",
                config={
                    "batch_size": batch_size,
            })
    model.eval()  # Set the model to evaluation mode

    criterion = nn.CrossEntropyLoss()
    nlp_lpips = LPTS()
    num_samples = len(dataset)
    word_to_remove1 = '[PAD]'
    word_to_remove2 = '[EOS]'
    word_to_remove3 = '[SOS]'
    total_loss = 0.0
    total_bleu = 0
    total_word_overlap_test = 0
    total_lpips_test = 0
    total_jaccard_test = 0
    total_samples = 0
    smoothie = SmoothingFunction().method1
    # Iterate over batches
    for batch_start in range(0, num_samples, 1):

                    batch_end = min(batch_start + 1, num_samples)
                    batch = dataset[batch_start:batch_end]

                    src, trg = batch
                    src, trg = src.to(device), trg.to(device)
                    #print(src)
                    sentences, softmax_output, output_ids = generate_sentence_ids(model, src.float(), sos_token_id, eos_token_id, padding_token_id, max_seq_len, len(vocab), device)
                    #print(sentences)
                    output = output_ids.float()

                    # Compute loss
                    loss = criterion(output.view(-1, output.size(-1)), trg.view(-1))

                    total_loss += loss.item()  # Accumulate loss
                    total_samples += trg.size(0)  # Accumulate number of samples

                    # Add references and hypotheses for BLEU calculation
                    for i in range(trg.size(0)):
                            # Get the reference sentence
                            ref = trg[i].tolist()
                            ref_sentence = detokenize(ref, vocab)
                            ref_sentence = remove_word(ref_sentence, word_to_remove1)
                            ref_sentence = remove_word(ref_sentence, word_to_remove2)
                            ref_sentence = remove_word(ref_sentence, word_to_remove3)

                            # Get the hypothesis

                            hyp = sentences[i].tolist()
                            hyp_sentence = detokenize(hyp, vocab)
                            hyp_sentence = remove_word(hyp_sentence, word_to_remove1)
                            hyp_sentence = remove_word(hyp_sentence, word_to_remove2)
                            hyp_sentence = remove_word(hyp_sentence, word_to_remove3)

                            # Calculate BLEU score for this sentence pair
                            bleu_score = sentence_bleu([ref_sentence.split()], hyp_sentence.split(), smoothing_function=smoothie)
                            print(bleu_score)
                            total_bleu += bleu_score

                            # Calculate word overlap
                            overlap_score = word_overlap_percentage(ref_sentence, hyp_sentence)
                            total_word_overlap_test += overlap_score

                            # Calculate LPIPS distance (assuming you have the function nlp_lpips.distance)
                            lpips_dist = nlp_lpips.distance(ref_sentence, hyp_sentence)
                            total_lpips_test += lpips_dist.item()

                            # Calculate Jaccard similarity
                            jaccard_score = jaccard_similarity(ref_sentence, hyp_sentence)
                            total_jaccard_test += jaccard_score

    avg_loss = total_loss / total_samples
    bleu_score = total_bleu / total_samples
    word_overlap = total_word_overlap_test / total_samples
    lpips = total_lpips_test / total_samples
    jaccard = total_jaccard_test / total_samples
    if wandb_log:
            metrics = {"Test_loss": avg_loss, "Test_Bleu": bleu_score, "Test_Word_Overlap": word_overlap, "Test NLPIPS": lpips, "Test Jaccard": jaccard}
            wandb.log(metrics)

    print(f"Loss: {avg_loss:.4f} BLEU: {bleu_score:.4f} Word Overlap: {word_overlap:.4f}% Jaccard Similarity: {jaccard:.4f} Test NLPIPS: {lpips:.4f}")

    #wandb.finish()

    return avg_loss, bleu_score
