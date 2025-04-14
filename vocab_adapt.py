import numpy as np
import sentencepiece as spm
import yaml
from transformers import AutoTokenizer
import argparse
import os
import sys
import glob
import shutil
import re

#this maps tokens in one vocabulary to tokens in another, using multiple tokens
#if necessary
def vocab_intersection(lm_tokenizer, source_spm, target_spm, vocab2):
    # we need to use the sentencepiece vocabs directly, as  we can't get token ids without
    # prefix space from the HF tokenizer
    source_sp = spm.SentencePieceProcessor(model_file=source_spm)
    target_sp = spm.SentencePieceProcessor(model_file=target_spm)

    vocab1 = lm_tokenizer.get_vocab()
    
    vocab_map = {}
    for symbol in vocab1:
        symbol_index = vocab1[symbol]
        # this gets the text version of the symbol
        if symbol in ["<unk>","</s>","<pad>","<s>"]:
            if symbol in vocab2:
                vocab_map[symbol_index] = (symbol,symbol,[vocab2[symbol]],[symbol])
            else:
                vocab_map[symbol_index] = (symbol,symbol,[],[])
            continue
        detok_symbol = lm_tokenizer.decode(symbol_index)
 
        # now get that symbol from the spm file
        if detok_symbol.startswith(" "):
            target_sp.override_normalizer_spec(add_dummy_prefix=True)
            source_sp.override_normalizer_spec(add_dummy_prefix=True)
        else:
            target_sp.override_normalizer_spec(add_dummy_prefix=False)
            source_sp.override_normalizer_spec(add_dummy_prefix=False)

        source_sp_tokens = source_sp.encode(detok_symbol,out_type=str)
        target_sp_tokens = target_sp.encode(detok_symbol,out_type=str)

        if len(source_sp_tokens) > len(target_sp_tokens):
            tokens = target_sp_tokens
        else:
            tokens = source_sp_tokens
        common_tokens = [token for token in tokens if token in vocab2]
        indexes = [vocab2[token] for token in tokens if token in vocab2]
        vocab_map[symbol_index] = (symbol,detok_symbol,indexes,common_tokens)
        
    return vocab_map
    
def modify_yaml_file(file_path, out_file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    
    # Modify the required fields
    data['models'] = ['modified_model.npz']
    data['vocabs'] = ['modified_vocab.yml', 'modified_vocab.yml']
    
    # Write back the modified content
    with open(out_file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)

def update_vocab_embeddings(model_dir, vocab_map, unk_id, embedding_init="weighted_average"):
    # Load the npz file
    npz_file = glob.glob(os.path.join(model_dir, "*.npz"), recursive=False)[0]
    data = np.load(npz_file, allow_pickle=True)
    emb_matrix = data['Wemb']  # Extract embeddings
    logit_emb_matrix = data['decoder_ff_logit_out_b']  # Extract embeddings
    
    # Determine the shape of the new embeddings matrix
    vocab_size = len(vocab_map)
    embedding_dim = emb_matrix.shape[1]  # Assuming original embeddings have shape (N, 512)
    new_emb_matrix = np.zeros((vocab_size, embedding_dim),dtype=np.float32)
    new_logit_emb_matrix = np.zeros((1,vocab_size),dtype=np.float32)
    
    common_counter = 0
    split_ids = []
    
    for index, (symbol, detok_symbol, target_indexes, tokens) in vocab_map.items():
        if len(target_indexes) == 1:
            # Direct mapping
            new_emb_matrix[index] = emb_matrix[target_indexes[0]]
            new_logit_emb_matrix[0][index] = logit_emb_matrix[0][target_indexes[0]]
            common_counter += 1
        elif len(target_indexes) > 1:
            # Compute new embedding based on initialization strategy
            embeddings = emb_matrix[target_indexes]
            logit_embeddings = logit_emb_matrix[0][target_indexes]
            if embedding_init == "average":
                new_emb_matrix[index] = np.mean(embeddings, axis=0, dtype=np.float32)
                new_logit_emb_matrix[0][index] = np.mean(logit_embeddings, axis=0, dtype=np.float32)
            elif embedding_init == "weighted_average":
                weights = np.array([len(x) for x in tokens], dtype=np.float32)
                weights /= weights.sum()  # Normalize weights
                new_emb_matrix[index] = np.sum(embeddings * weights[:, None], axis=0)
                normalized_logit_embeddings = logit_embeddings * weights
                new_logit_emb_matrix[0][index] = np.sum(normalized_logit_embeddings, axis=0, dtype=np.float32)
            elif embedding_init == "zeros":
                new_emb_matrix[index] = np.zeros(embedding_dim)
                new_logit_emb_matrix[0][index] = np.float32(0)
            elif embedding_init == "random":
                new_emb_matrix[index] = np.random.rand(embedding_dim)
                new_logit_emb_matrix[0][index] = np.random.rand(1)
            split_ids.append(index)
        else:
            # If no matching indexes, use the unknown embedding
            new_emb_matrix[index] = emb_matrix[unk_id]
    
    print(f"{common_counter} common tokens, {len(split_ids)} tokens were split.")
    
    # Save the updated embeddings back to the .npz file
    np.savez(os.path.join(model_dir,"modified_model.npz"), **{key: data[key] for key in data.files if key != 'Wemb' and key != "decoder_ff_logit_out_b"}, Wemb=new_emb_matrix, decoder_ff_logit_out_b= new_logit_emb_matrix)
    return split_ids

def generate_train_config(npz_file_path, config_path):
    # Load the .npz file
    with np.load(npz_file_path, allow_pickle=True) as data:
        # Check if the specific file exists within the archive
        if 'special:model.yml' in data.files:
            yaml_content = yaml.safe_load(data['special:model.yml'].tobytes().decode('utf-8')[:-1])
        else:
            raise FileNotFoundError("'special:model.yml' not found in the .npz file")
    with open(config_path, 'w') as file:
        yaml_content['model'] = 'modified_model.npz'
        yaml_content['vocabs'] = ['modified_vocab.yml', 'modified_vocab.yml']
        yaml.dump(yaml_content, file, default_flow_style=False, sort_keys=False)

def generate_decoder_config(decoder_path, config_path):
    with open(decoder_path, 'r') as f:
        yaml_content = yaml.safe_load(f)
    with open(config_path, 'w') as file:
        yaml_content['models'] = ['modified_model.npz']
        yaml_content['vocabs'] = ['modified_vocab.yml', 'modified_vocab.yml']
        yaml.dump(yaml_content, file, default_flow_style=False, sort_keys=False)

def find_and_parse_vocab(local_model_dir):
    vocab_files = glob.glob(os.path.join(local_model_dir, "*.vocab.yml"), recursive=False)
    if not vocab_files:
        sys.exit("Error: vocab.yml file not found.")
    with open(vocab_files[0], "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def modify_special_model(npz_file_path, dim_vocab_1, dim_vocab_2,output_path):
    # Load the .npz file
    d = dict()
    with np.load(npz_file_path, allow_pickle=True) as data:
        for k in data:
            if k == "special:model.yml":
                info = data[k].tobytes().decode()
                print(info)
                replace_string = fr"\1 {dim_vocab_1}\3 {dim_vocab_2}"
                info = re.sub(r"(dim-vocabs:\n\s+-)\s+(\d+)(\n\s+-)\s+(\d+)",
                              replace_string,info, re.MULTILINE)
                d[k] = np.fromstring(info, dtype="int8")
                print(d[k])
            else:
                d[k] = data[k]
        np.savez(output_path, **d)

def main():
    parser = argparse.ArgumentParser(description="Adapt a Marian model to use the vocabulary of a HF transformers language model.")
    parser.add_argument("--local_model_dir", type=str, help="Path to the local model directory.")
    parser.add_argument("--hf_model_name", type=str, help="Name of the Hugging Face model.")
    parser.add_argument("--output_model_dir", type=str, help="Path to the output model directory.")
    parser.add_argument("--overwrite", action="store_true", help="Use this to overwrite output dir.")
    
    args = parser.parse_args()
    
    """
    if os.path.exists(args.output_model_dir) and not args.overwrite:
        print(f"Error: Output model directory '{args.output_model_dir}' already exists.", file=sys.stderr)
        sys.exit(1)
    else:
        os.makedirs(args.output_model_dir,exist_ok=True)
        shutil.copytree(args.local_model_dir, args.output_model_dir, dirs_exist_ok=True)
        print(f"Created Output Model Directory: {args.output_model_dir}")
    """

    lm_tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name)

    # Replace potential line breaks in the symbols, since those are not supported by Marian. This
    # should have no effect, since those symbols seem to be junk. I'm not sure if it's actually possible
    # to have line breaks in symbols (the bug was caused by yaml.dump adding line breaks), but keep this
    # in just in case
    
    lm_vocab = sorted(lm_tokenizer.get_vocab().items(), key=lambda item: item[1])
    lm_vocab_dict = {k.replace("\n",""): v for (k,v) in lm_vocab}
    print(f"Extracted {args.hf_model_name} vocab")
    vocab_yaml = yaml.dump(lm_vocab_dict, allow_unicode=True,sort_keys=False, width=10000000000000)

    # for some reason yaml dump just keep adding the line break to some entries, even with the width setting
    # fix manually
    
    with open(os.path.join(args.output_model_dir, "modified_vocab.yml"), 'w', encoding='utf-8') as vocab_yaml_file:
        fixed_vocab_yaml = ""
        partial_sentence = ""
        for line in vocab_yaml.split("\n"):
            if not re.match("^.*:\s+\d+$",line):
                # need to quote these partial sentences, otherwise downstream processing fails
                if not partial_sentence:
                    partial_sentence += '"'
                partial_sentence += line
            elif partial_sentence:
                vocab_yaml_file.write(partial_sentence + '"' + line + "\n")
                partial_sentence = ""
            else:
                vocab_yaml_file.write(line + "\n")


    marian_vocab = find_and_parse_vocab(args.local_model_dir)
    
    vocab_map = vocab_intersection(
        lm_tokenizer,
        os.path.join(args.output_model_dir,"source.spm"),
        os.path.join(args.output_model_dir,"target.spm"),
        marian_vocab)
    
    split_ids = update_vocab_embeddings(
        args.output_model_dir,
        vocab_map,
        marian_vocab["<unk>"],
        embedding_init="weighted_average")
    
    generate_train_config(
        os.path.join(args.output_model_dir,"modified_model.npz"),
        os.path.join(args.output_model_dir,"modified_train.yml"))
    

    generate_decoder_config(
        os.path.join(args.output_model_dir,"decoder.yml"),
        os.path.join(args.output_model_dir,"modified_decoder.yml"))
    
    #vocab_length = 131072
    vocab_length = len(lm_vocab)
    modify_special_model(os.path.join(args.output_model_dir,"modified_model.npz"),vocab_length,vocab_length,os.path.join(args.output_model_dir,"modified_model.npz"))
    
if __name__ == "__main__":
    main()
