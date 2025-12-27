import editdistance


def calc_cer(target_text, predicted_text) -> float:
    target_text = target_text or ""
    predicted_text = predicted_text or ""

    if target_text == "":
        return 0.0 if predicted_text == "" else 1.0

    dist = editdistance.eval(target_text, predicted_text)
    return dist / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    target_text = target_text or ""
    predicted_text = predicted_text or ""

    target_words = target_text.split()
    predicted_words = predicted_text.split()
    if len(target_words) == 0:
        return 0.0 if len(predicted_words) == 0 else 1.0

    dist = editdistance.eval(target_words, predicted_words)
    return dist / len(target_words)
