import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    B = len(dataset_items)

    # --------------audio------------------
    audios = [x["audio"] for x in dataset_items]
    audio_len = torch.tensor([a.size(1) for a in audios], dtype=torch.long)
    max_audio_len = int(audio_len.max().item())

    batch_audio = torch.zeros(B, 1, max_audio_len, dtype=audios[0].dtype)
    for i, a in enumerate(audios):
        batch_audio[i, 0, : a.size(1)] = a[0]
    # -------------spectrogram------------------
    specs = [x["spectrogram"] for x in dataset_items]
    spec_len = torch.tensor([s.size(1) for s in specs], dtype=torch.long)
    max_spec_len = int(spec_len.max().item())

    n_mels = specs[0].size(0)
    batch_spec = torch.zeros(B, n_mels, max_spec_len, dtype=specs[0].dtype)
    for i, s in enumerate(specs):
        batch_spec[i, :, : s.size(1)] = s

    # -------------text------------------
    texts = [x["text"] for x in dataset_items]
    paths = [x["audio_path"] for x in dataset_items]

    encs = [x["text_encoded"] for x in dataset_items]
    enc_len = torch.tensor([t.size(1) for t in encs], dtype=torch.long)
    max_enc_len = int(enc_len.max().item())

    batch_text = torch.zeros(B, max_enc_len, dtype=torch.long)
    for i, t in enumerate(encs):
        batch_text[i, : t.size(1)] = t[0].long()

    return {
        "audio": batch_audio,
        "audio_length": audio_len,
        "spectrogram": batch_spec,
        "spectrogram_length": spec_len,
        "text_encoded": batch_text,
        "text_encoded_length": enc_len,
        "text": texts,
        "audio_path": paths,
    }
