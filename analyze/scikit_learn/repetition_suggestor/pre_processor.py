from analyze.scikit_learn.common.pre_processor import ScikitLearnCommonSuggestorDataPreProcessor


class ScikitLearnRepetitionSuggestorDataPreProcessor(ScikitLearnCommonSuggestorDataPreProcessor):

    def get_features_list(self) -> list[str]:
        return ['exercicio', 'peso', 'serie']

    def get_target(self) -> str:
        return 'repeticoes'