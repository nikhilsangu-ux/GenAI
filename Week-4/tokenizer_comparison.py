
from transformers import AutoTokenizer

# 1. Define the three pretrained tokenizers
TOKENIZERS = {
    "BPE - GPT-2": "gpt2",
    "WordPiece - Multilingual BERT": "bert-base-multilingual-cased",
    "SentencePiece - XLM-R": "xlm-roberta-base"
}

# 2. Input sentences in English, Telugu and Hindi
TEXTS = {
    "English": "Artificial intelligence is changing the world.",
    "Telugu": "కృత్రిమ మేధస్సు ప్రపంచాన్ని మారుస్తోంది.",
    "Hindi": "कृत्रिम बुद्धिमत्ता दुनिया को बदल रही है।"
}


# 3. Load pretrained tokenizers
def load_tokenizers():
    loaded = {}

    for display_name, model_name in TOKENIZERS.items():
        print(f"\nLoading {display_name}...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        loaded[display_name] = tokenizer

    return loaded


# 4. Compare tokenizers for all three languages
def compare_tokenizers(loaded_tokenizers):

    for language, text in TEXTS.items():

        print("\n" + "=" * 75)
        print(f"LANGUAGE: {language}")
        print(f"INPUT TEXT: {text}")
        print("=" * 75)

        for tokenizer_name, tokenizer in loaded_tokenizers.items():

            # Convert input text into token IDs
            encoded = tokenizer(
                text,
                add_special_tokens=True,
                truncation=False
            )

            token_ids = encoded["input_ids"]

            # Convert token IDs into readable tokens
            tokens = tokenizer.convert_ids_to_tokens(token_ids)

            # Display results
            print(f"\n{tokenizer_name}")
            print("-" * 50)
            print("Tokens      :", tokens)
            print("Token IDs   :", token_ids)
            print("Token count :", len(token_ids))


# 5. Main function
def main():

    print("\nBPE, WordPiece AND SENTENCEPIECE COMPARISON")
    print("=" * 75)

    tokenizers = load_tokenizers()

    compare_tokenizers(tokenizers)

    print("\n" + "=" * 75)
    print("TOKENIZER COMPARISON COMPLETED SUCCESSFULLY!")
    print("=" * 75)


# 6. Program execution
if __name__ == "__main__":
    main()
