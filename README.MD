## A thin wrapper over sacrebleu for personal use.

```
from sacrebleu_wrapper import sacrebleu_wrapper

def translate(input: str) -> str:
    # stuff happens here
    return output

evaluator = sacrebleu_wrapper.Evaluator(translate_func=translate)

evaluator.translate(source=mya_sentences, target=rki_sentences)

print(evaluator.evaluate(scorer=evaluator.scorers['chrf']).score)

```