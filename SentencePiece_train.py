import sentencepiece as spm

vocabulary_size = 16 #単位はk
EN_TRAIN_FILE = "data/train.en"
KU_TRAIN_FILE = "data/train.ku"
EN_VALID_FILE = "data/valid.en"
KU_VALID_FILE = "data/valid.ku"
EN_TEST_FILE = "data/test.en"
KU_TEST_FILE = "data/test.ku"

EN_MODEL_PREFIX = f"data/spm_bpe_EN_{str(vocabulary_size)}k"
KU_MODEL_PREFIX = f"data/spm_bpe_KU_{str(vocabulary_size)}k"


def train_spm(input_path, model_prefix):
    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=str(model_prefix),
        vocab_size=vocabulary_size * 1000,
        model_type="bpe",
        character_coverage=0.9995,
    )


def load_sp(model_path):
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    return sp


def encode_file(sp, in_path, out_path):
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.rstrip("\n")
            # 空行も落とさない（train/validの行数一致が重要）
            pieces = sp.encode(line, out_type=str)
            fout.write(" ".join(pieces) + "\n")


# train data only で SentencePiece を学習
train_spm(EN_TRAIN_FILE, EN_MODEL_PREFIX)
train_spm(KU_TRAIN_FILE, KU_MODEL_PREFIX)

en_sp = load_sp(f"{EN_MODEL_PREFIX}.model")
ku_sp = load_sp(f"{KU_MODEL_PREFIX}.model")

# train
encode_file(en_sp, EN_TRAIN_FILE, f"data/train{str(vocabulary_size)}k.en")
encode_file(ku_sp, KU_TRAIN_FILE, f"data/train{str(vocabulary_size)}k.ku")

# valid
encode_file(en_sp, EN_VALID_FILE, f"data/valid{str(vocabulary_size)}k.en")
encode_file(ku_sp, KU_VALID_FILE, f"data/valid{str(vocabulary_size)}k.ku")

# test
encode_file(en_sp, EN_TEST_FILE, f"data/test{str(vocabulary_size)}k.en")
encode_file(ku_sp, KU_TEST_FILE, f"data/test{str(vocabulary_size)}k.ku")
