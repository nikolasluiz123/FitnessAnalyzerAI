from analyze.keras.common.pre_processor import KerasCommonSuggestorDataPreProcessor


class KerasWeightSuggestorDataPreProcessor(KerasCommonSuggestorDataPreProcessor):

    def get_features_list(self) -> list[str]:
        return ['exercicio', 'serie', 'repeticoes', 'dia_da_semana', 'dias_desde_inicio']

    def get_target(self) -> str:
        return 'peso'