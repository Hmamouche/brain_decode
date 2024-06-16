import glob
import torch
from transformers import FlaubertModel, FlaubertTokenizer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

class LPTS:
    def __init__(self):
        # Load Flaubert tokenizer and model
        self.tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_base_cased')
        self.model = FlaubertModel.from_pretrained('flaubert/flaubert_base_cased')
        self.model.eval()

        # If you have a GPU, move the model to GPU
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def encode(self, sentence):
        """Tokenizes and gets embeddings for the given sentence."""
        with torch.no_grad():
            inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
            if torch.cuda.is_available():
                for k, v in inputs.items():
                    inputs[k] = v.cuda()
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1)  # Average pooling over sequence dimension

    def distance(self, sentence1, sentence2):
        """Computes 'LPIPS' distance between two sentences."""
        embedding1 = self.encode(sentence1)
        embedding2 = self.encode(sentence2)
        dist = torch.nn.functional.cosine_similarity(embedding1, embedding2)
        return 1 - dist  # Convert similarity to distance

def remove_word(sentence, word_to_remove):
    words = sentence.split()
    words = [word for word in words if word != word_to_remove]
    return ' '.join(words)

def word_overlap_percentage(sentence_A, sentence_B):
    # Tokenize the sentences
    tokens_A = set(sentence_A.split())
    tokens_B = set(sentence_B.split())

    # Check the overlap
    overlap = tokens_A.intersection(tokens_B)

    # Calculate the percentage
    if len(tokens_A) == 0:
        return 0.0
    else:
        return len(overlap) / len(tokens_A) * 1

def gatherup(list):
  preds = []
  for i in range(50):
    pred_sentences = [' '.join(inner_list) for inner_list in list[i]]
    preds.append(pred_sentences)
  return preds

def jaccard_similarity(sentence_A, sentence_B):
    tokens_A = set(sentence_A)
    tokens_B = set(sentence_B)
    intersection = tokens_A.intersection(tokens_B)
    union = tokens_A.union(tokens_B)
    return len(intersection) / len(union) if union else 0.0


if __name__ == "__main__":
    
                
    filenames = sorted (glob.glob ("results/*.txt"))

    f = open("final_results.txt", "w")
    f.write ("Model;BLUE score;jaccard similarity;word overlap;LPIPS\n")

    
    for filename in filenames:

        predictions = []
        target = []
        # Open the file
        with open(filename, 'r') as file:
            # Loop through each line in the file
            for line in file:
                # Process each line
                if 'The predicted Conversation' in line:
                  predictions.append(line.split()[4:])
                else:
                  target.append(line.split()[4:])
    
        pred_conversations = [predictions[x:x+5] for x in range(0, len(predictions), 5)]
        true_conversations = [target[x:x+5] for x in range(0, len(target), 5)]
    
    
        preds = gatherup(pred_conversations)
        targs = gatherup(true_conversations)
    
    
        predic = []
        for pred in preds:
          for prediction in pred:
            predic.append(prediction)
    
        targets = []
        for pred in targs:
          targets.append(pred)
    
        full_preds = [' '.join(inner_list) for inner_list in preds]
        full_targs = [' '.join(inner_list) for inner_list in targs]
    
    
        context = ['batman', 'aubergine', 'citron', 'tortue', 'ninja', 'fraise', 'pomme' 'spiderman', 'poire',\
                    'framboise', 'fruit', 'batman', 'fatigué', 'super-héros', 'pourrie', 'boxeur', 'Batman',\
                    'pourrissent', 'abîmée', 'orange', 'Spiderman', 'moches', 'Furhat', 'rouges']
    
        total_count = 0
        for i, sentence in enumerate(predic):
            # Initialize a count for each sentence
            count = 0
            # Loop through each word to count
            for word in context:
                # Add to the count the occurrences of the word in the sentence
                count += sentence.lower().split().count(word.lower())
            total_count += count
    
        #print (total_count)
    
    
        total_bleu = 0
        total_word_overlap = 0
        total_lpips = 0
        total_jaccard = 0
        nlp_lpips = LPTS()
        smoothie = SmoothingFunction().method1
        for pred, targ in zip(full_preds, full_targs):
          bleu_score = sentence_bleu([targ], pred, smoothing_function=smoothie)
          total_bleu += bleu_score
          jaccard_score = jaccard_similarity(targ, pred)
          total_jaccard += jaccard_score
          lpips_dist = nlp_lpips.distance(targ, pred)
          total_lpips += lpips_dist
          overlap_score = word_overlap_percentage(targ, pred)
          total_word_overlap += overlap_score
        final_bleu = total_bleu / len(full_preds)
        final_jaccard = total_jaccard / len(full_preds)
        final_word_overlap = total_word_overlap / len(full_preds)
        final_lpips = total_lpips / len(full_preds)
    
        '''print (final_bleu)
        print (final_jaccard*100)
        print (final_word_overlap)
        print (final_lpips)'''
        
        f.write ("%s ; %s ; %s ; %s ; %s; %s\n"%(filename.split('.txt')[0], final_bleu, final_jaccard*100, final_word_overlap, final_lpips[0].cpu().detach().numpy(), total_count))
    f.close()
