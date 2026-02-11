#!/usr/bin/env python3
"""Извлечь токенизатор из .nemo файла и создать vocab.json."""
import tarfile, shutil, os, tempfile, json, sys

nemo = "models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo"
outdir = "models/parakeet-tdt-0.6b-v3"

with tarfile.open(nemo, "r:") as tar:
    names = tar.getnames()
    tok_files = [n for n in names if "tokenizer.model" in n]
    vocab_files = [n for n in names if "vocab.txt" in n or "tokenizer.vocab" in n]

    with tempfile.TemporaryDirectory() as td:
        for tf in tok_files + vocab_files:
            tar.extract(tf, path=td)
            src = os.path.join(td, tf)
            basename = os.path.basename(tf)
            # Убрать хеш-префикс
            if "_tokenizer." in basename:
                basename = "tokenizer." + basename.split("_tokenizer.")[1]
            elif "_vocab." in basename:
                basename = "vocab." + basename.split("_vocab.")[1]
            dst = os.path.join(outdir, basename)
            shutil.copy2(src, dst)
            print(f"Copied: {tf} -> {dst}")

# Создать vocab.json из tokenizer.model
tok_path = os.path.join(outdir, "tokenizer.model")
if os.path.exists(tok_path):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(tok_path)
    vocab = {}
    for i in range(sp.GetPieceSize()):
        piece = sp.IdToPiece(i)
        score = sp.GetScore(i)
        vocab[piece] = {"id": i, "score": score}
    vocab_path = os.path.join(outdir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=1)
    print(f"vocab.json: {sp.GetPieceSize()} tokens -> {vocab_path}")
else:
    print("tokenizer.model not found!")

print("Done")
