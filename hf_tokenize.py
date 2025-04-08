import argparse
import gzip
from transformers import AutoTokenizer

def open_input_file(input_file_path):
    """
    Opens a file normally or with gzip based on the file extension.
    """
    if input_file_path.endswith('.gz'):
        return gzip.open(input_file_path, 'rt', encoding='utf-8')
    else:
        return open(input_file_path, 'r', encoding='utf-8')

def hf_tokenize(input_file_path, output_file_path, tokenizer):
    """
    Tokenizes an input file line by line and writes the tokenized output to another file.
    """
    with open_input_file(input_file_path) as infile, \
         open(output_file_path, 'w', encoding='utf-8') as outfile:

        for line in infile:
            tokens = tokenizer.tokenize(line.strip())
            outfile.write(' '.join(tokens) + '\n')

    print(f"Tokenization complete. Output saved to {output_file_path}")

def hf_detokenize(input_file_path, output_file_path, tokenizer):
    """
    Detokenizes a file of space-separated tokens line by line.
    """
    with open_input_file(input_file_path) as infile, \
         open(output_file_path, 'w', encoding='utf-8') as outfile:

        for line in infile:
            tokens = line.strip().split()
            detok_line = tokenizer.convert_tokens_to_string(tokens).replace("\n","").replace("\t","")
            outfile.write(detok_line + '\n')

    #print(f"Detokenization complete. Output saved to {output_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Tokenize or detokenize a text file using a Hugging Face tokenizer.")
    parser.add_argument("--input_file", required=True, help="Path to the input text file (can be .gz).")
    parser.add_argument("--output_file", required=True, help="Path to save the output (always uncompressed).")
    parser.add_argument("--model_name", required=True, help="Name of the Hugging Face tokenizer model to use.")
    parser.add_argument("--mode", choices=["tok", "detok"], default="tok", help="Choose whether to tokenize (tok) or detokenize (detok).")

    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.mode == "tok":
        hf_tokenize(args.input_file, args.output_file, tokenizer)
    else:
        hf_detokenize(args.input_file, args.output_file, tokenizer)

if __name__ == "__main__":
    main()

