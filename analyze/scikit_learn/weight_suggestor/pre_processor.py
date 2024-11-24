from analyze.scikit_learn.common.pre_processor import ScikitLearnCommonSuggestorDataPreProcessor


class ScikitLearnWeightSuggestorDataPreProcessor(ScikitLearnCommonSuggestorDataPreProcessor):

    def get_features_list(self) -> list[str]:
        return ['exercicio', 'repeticoes', 'serie']

    def get_target(self) -> str:
        return 'peso'