## Introdução ao Analisador

Esse projeto tem como objetivo reunir diversas funcionalidades interessantes para o contexto do mundo
fitness, inicialmente focando no treinamento em academias e podendo ser expandido para outras áreas que
estão interligadas.

## Arquitetura do Projeto

O projeto utiliza como base para o processamento mais complexo a biblioteca [MLModelTuner](https://github.com/nikolasluiz123/MLModelTuner),
dessa forma, é possível realizar as implementações do que chamamos de Agentes que devem ter um único objetivo.

Um Agente deve obrigatoriamente implementar [CommonAgent](https://github.com/nikolasluiz123/FitnessAnalyzerAI/blob/master/analyze/common/common_agent.py#L4) para que possua uma estrutura básica e que siga
os padrões pré estabelecidos. A implementação do Agente deve conseguir lidar com o treinamento inicial do modelo e,
quando a base de dados se expandir ou mudar de alguma forma, deve ser possível realizar um retreinamento e
reavaliar qual será o melhor modelo para essa nova versão da base de dados.

## Utilidade do Projeto

No momento, para tornar a implementação um pouco mais simples, a base de dados utilizada pelos Agentes é
um arquivo CSV simples, mas, a ideia final é que os dados venham de um banco que é alimentado por uma aplicação
que oferece funcionalidades referentes ao treinamento e dieta, a qual ainda está em desenvolvimento e é
chamada [FitnessProApp](https://github.com/nikolasluiz123/FitnessProApp). Essa aplicação se comunica com um
serviço implementado em python, chamado [FitnessProService](https://github.com/nikolasluiz123/FitnessProService) e seria
nesse serviço onde FitnessAnalyzerAI seria utilizado para que pudessem ser realizadas as previsões ou classificações.

## Exemplos

Para que os Agentes implementados pudessem ser testados, foram criados alguns scripts simples como exemplos e
eles serão apresentados abaixo.

### Sugestões de Repetições e Peso usando Scikit-Learn

Uma primeira tentativa foi avaliar como os modelos de machine learning do scikit-learn se comportariam
com dados temporais, mesmo sabendo que não é exatamente o objetivo dos modelos dessa biblioteca.

A sugestão de repetições pode ser observada [aqui](https://github.com/nikolasluiz123/FitnessAnalyzerAI/blob/master/examples/scikit_learn/repetition_sugestor/example_repetitions_suggestor.py) e a sugestão de peso [aqui](https://github.com/nikolasluiz123/FitnessAnalyzerAI/blob/master/examples/scikit_learn/weight_suggestor/example_weight_suggestor.py). 

No [diretório de validações adicionais da sugestão de repetições](https://github.com/nikolasluiz123/FitnessAnalyzerAI/tree/master/examples/scikit_learn/repetition_sugestor/additional_validations) podem ser encontrados dados gráficos e relatórios com algumas métricas de regressão.

No [diretório de validações adicionais da sugestão de peso](https://github.com/nikolasluiz123/FitnessAnalyzerAI/tree/master/examples/scikit_learn/weight_suggestor/additional_validations) podem ser encontrados dados gráficos e relatórios com algumas métricas de regressão.

### Classificação de Tipos de Treinamento

Nesse exemplo foi utilizado o [seguinte dataset](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset). O objetivo foi realizar a sugestão de um tipo de treino baseado em alguns dos dados desse dataset que faziam sentido.

A ideia é passarmos os seguintes dados: Idade, Gênero, Peso, Altura, Duração do Treino, Frequência Semanal, Nível de Experiência e Percentual de Gordura. Com essas informações presentes no dataset utilizado seria possível classificar o Tipo de Treino.

A classificação pode ser observada [aqui](https://github.com/nikolasluiz123/FitnessAnalyzerAI/blob/master/examples/scikit_learn/classify_workout_type/example_classify_workout_type.py).

No [diretório de validações adicionais da classificação do tipo de treino](https://github.com/nikolasluiz123/FitnessAnalyzerAI/tree/master/examples/scikit_learn/classify_workout_type/additional_validations) podem ser encontradas as validações realizadas.

### Classificação de Frequência de Treinamento

Nesse exemplo foi utilizado o [seguinte dataset](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset). O objetivo foi realizar a sugestão de um tipo de treino baseado em alguns dos dados desse dataset que faziam sentido.

A ideia é passarmos os seguintes dados: Idade, Gênero, Peso, Altura, Duração do Treino, Tipo de Treino, Nível de Experiência e Percentual de Gordura. Com essas informações presentes no dataset utilizado seria possível classificar o Tipo de Treino.

A classificação pode ser observada [aqui](https://github.com/nikolasluiz123/FitnessAnalyzerAI/blob/master/examples/scikit_learn/classify_workout_frequency/example_classify_workout_frequency.py).

No [diretório de validações adicionais da classificação do tipo de treino](https://github.com/nikolasluiz123/FitnessAnalyzerAI/tree/master/examples/scikit_learn/classify_workout_frequency/additional_validations) podem ser encontradas as validações realizadas.

### Sugestões de Repetições e Peso usando Tensorflow e Keras

Como a implementação utilizando Scikit-Learn não chegou no ponto que gostaria foi preciso abordar o problema
com outro tipo de tecnologia, como as redes neurais. A ideia foi utilizar as redes LSTMs para que pudessem captar
a evolução dos dados de treino do indivíduo ao longo do tempo.

A sugestão de repetições pode ser observada [aqui](https://github.com/nikolasluiz123/FitnessAnalyzerAI/tree/master/examples/keras/repetition_sugestor/example_repetition_suggestor.py) e a sugestão de peso [aqui](https://github.com/nikolasluiz123/FitnessAnalyzerAI/tree/master/examples/keras/weight_suggestor/example_weight_suggestor.py). 

No [diretório de validações adicionais da sugestão de repetições](https://github.com/nikolasluiz123/FitnessAnalyzerAI/tree/master/examples/keras/repetition_sugestor/additional_validations) podem ser encontrados dados gráficos e relatórios com algumas métricas de regressão.

No [diretório de validações adicionais da sugestão de peso](https://github.com/nikolasluiz123/FitnessAnalyzerAI/tree/master/examples/keras/weight_suggestor/additional_validations) podem ser encontrados dados gráficos e relatórios com algumas métricas de regressão.

