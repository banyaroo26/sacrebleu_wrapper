from tqdm import tqdm
import sacrebleu

class Evaluator():
    def __init__(self, translate_func=None, source=[], translations=[], references=[], scorers={}):
        self.translate_func = translate_func
        self.source = source             # source sentences
        self.translations = translations # translated sentences
        self.references = references     # reference sentences
        self.scorers = {
            'bleu': sacrebleu.BLEU(tokenize='intl'),
            'chrf': sacrebleu.CHRF(),
            'chrfpp': sacrebleu.CHRF(word_order=2),
            'ter': sacrebleu.TER()
        }        

    def translate(self, source: list[str], target: list[str]) -> dict[list[str], list[str], list[str]]:
        self.source = source
        refs = []

        for src, ref in tqdm.tqdm(zip(source, target), total=len(source)):
            self.translations.append(self.translate_func(src))
            refs.append(ref)
        self.references.append([refs]) # sacrebleu requires list of list of references

        return {
            'source': self.source,
            'translations': self.translations,
            'references': self.references
        }
    
    def evaluate(self, scorer):
        if scorer is None:
            raise ValueError("Scorer is not defined.")
        return scorer.corpus_score(self.translations, self.references)