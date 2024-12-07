from abc import abstractmethod, ABC


class CommonAgent(ABC):
    """
    Implementação utilizada como base para criação dos agentes de machine learning.
    """
    def __init__(self,
                 train_model: bool,
                 history_index: int | None,
                 force_execute_additional_validation: bool):
        """
        :param train_model: Flag que indica se o processo de treinamento deve ocorrer.
        :param history_index: Índice para busca do histórico.
        :param force_execute_additional_validation: Flag que indica se a validação adicional deve ser executada.
        """
        self.train_model = train_model
        self.history_index = history_index
        self._force_execute_additional_validation = force_execute_additional_validation
        self._data_pre_processor = None
        self._process_manager = None

        self._initialize_data_pre_processor()
        self._initialize_multi_process_manager()

    @abstractmethod
    def _initialize_data_pre_processor(self):
        """
        Função para inicializar a implementação de pré processamento dos dados
        """

    @abstractmethod
    def _initialize_multi_process_manager(self):
        """
        Função para inicializar a implementação que realiza a busca do melhor modelo para a sugestão de repetições.
        """

    def execute(self, dataset):
        """
        Função que deve ser utilizada para executar a sugestão. Essa implementação sempre realizará a chamada do treino
        do modelo, caso não seja necessário ele não realizará o processo.

        :param dataset: Dicionário com os dados que deseja realizar a sugestão.
        """
        if self.train_model:
            self._process_manager.process_pipelines()
            self._execute_additional_validation()

        return self._execute_prediction(dataset)

    @abstractmethod
    def _execute_additional_validation(self):
        """
        Realiza as validações adicionais para cada um dos modelos
        """

    @abstractmethod
    def _execute_prediction(self, dataset):
        """
        Retorna a previsão realizada com o melhor modelo encontrado

        :param dataset: Dados para prever
        """