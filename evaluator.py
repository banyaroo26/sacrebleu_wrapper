import tqdm
import sacrebleu

class Evaluator():
    def __init__(self, translate_func=None, source=[], translations=[], references=[]):
        self.translate_func = translate_func
        self.source = source             # source sentences
        self.translations = translations # translated sentences
        self.references = references     # reference sentences

    def translate(self, source: list[str], target: list[str]) -> dict[list[str], list[str], list[str]]:
        self.source = source
        refs = []

        for src, ref in tqdm(zip(source, target), total=len(source)):
            self.translations.append(self.translate_func(src))
            refs.append(ref)
        self.references.append([refs]) # sacrebleu requires list of list of references

        return {
            'source': self.source,
            'translations': self.translations,
            'references': self.references
        }
    
    def evaluate(self, metric: str, **kwargs):
        metric = metric.lower()

        if metric == "bleu":
            scorer = sacrebleu.BLEU(**kwargs)
        elif metric == "chrf":
            scorer = sacrebleu.CHRF(**kwargs)
        elif metric == "ter":
            scorer = sacrebleu.TER(**kwargs)
        else:
            raise ValueError("invalid")

        return scorer.corpus_score(self.translations, self.references)