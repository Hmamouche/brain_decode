from tokenizers import Tokenizer


if __name__ == '__main__':

    tokenizer = Tokenizer.from_file("./tokenizer-uloaded.json")
    vocab_len = tokenizer.get_vocab_size()

    input_string = "j'ai encore l'aubergine en forme de batman c'est la même c'était exactement la même c'est la même que tout à l'heure oui"
    print (input_string)
    output = tokenizer.encode(input_string, add_special_tokens=True)
    print(output)
    print (tokenizer.decode(output.ids, skip_special_tokens = True))
