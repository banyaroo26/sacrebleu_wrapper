from tqdm import tqdm
import sacrebleu

class Evaluator():
    def __init__(self, 
                translate_func=None, 
                scorers={
                        'bleu': sacrebleu.BLEU(tokenize='intl'),
                        'chrf': sacrebleu.CHRF(),
                        'chrfpp': sacrebleu.CHRF(word_order=2),
                        'ter': sacrebleu.TER()
                        }):
        self.translate_func = translate_func
        self.source = []             # source sentences
        self.translations = []       # machine-translated sentences
        self.references = []         # reference sentences
        self.scorers = scorers

    def translate(self, source: list[str], target: list[str], inverse: bool=False) -> dict[list[str], list[str], list[str]]:
        refs = []
        trans = []

        if len(source) != len(target):
            raise ValueError("Source and target lists must have the same length.")
        
        if inverse:
            source, target = target, source     # swap source and target langs if inverse is True

        self.source = source

        for src, ref in tqdm(zip(source, target), total=len(source)):
            trans.append(self.translate_func(src))
            refs.append(ref)
        self.references = [refs]    # sacrebleu requires list of list of references
        self.translations = trans

        return {
            'source': self.source,
            'translations': self.translations,
            'references': self.references
        }
    
    def evaluate(self, scorer_name: str):
        if scorer_name is None:
            raise ValueError("Scorer is not defined.")
        return self.scorers[scorer_name].corpus_score(self.translations, self.references)
    
    def output_results(self):
        for index, (translation, reference) in enumerate(zip(self.translations, self.references[0])):
            print(f"Source: {self.source[index]}")
            print(f"Translation: {translation}")
            print(f"Reference: {reference}\n")
            print('\n')