from vocab import EOT, char_vocab, clean, train_bpe

STORIES = [
    "Once upon a time, there was a little cat.\nThe cat was happy.",
    'Tom said, "I like the big red ball!"  He ran to the park.',
    "The little dog and the little cat played all day long.",
] * 20


def test_clean_drops_foreign_characters():
    assert clean("  café “hi”\n") == "caf hi"


def test_chars_round_trip():
    vocab = char_vocab()
    for story in STORIES[:3]:
        ids = next(vocab.encode_stories([story]))
        assert ids[-1] == EOT
        assert vocab.decode(ids) == story


def test_bpe_round_trip_and_reload(tmp_path):
    vocab = train_bpe(STORIES, 160)
    assert vocab.tokens[EOT] == "<|endoftext|>"
    path = tmp_path / "vocab.json"
    vocab.save(path)
    reloaded = type(vocab).load(path)
    for story in STORIES[:3]:
        ids = next(vocab.encode_stories([story]))
        assert vocab.decode(ids) == story
        assert next(reloaded.encode_stories([story])) == ids
        assert len(ids) < len(story)


def test_char_lengths_match_decoded_text():
    vocab = train_bpe(STORIES, 160)
    ids = next(vocab.encode_stories([STORIES[0]]))
    assert vocab.char_lengths()[ids].sum() == len(STORIES[0]) + 1
