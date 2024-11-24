## Introdução ao Analisador

Esse projeto tem como objetivo reunir diversas funcionalidades interessantes para o contexto do mundo
fitness, inicialmente focando no treinamento em academias e podendo ser expandido para outras áreas que
estão interligadas.

## Arquitetura do Projeto

O projeto utiliza como base para o processamento mais complexo a biblioteca [MLModelTuner](https://github.com/nikolasluiz123/MLModelTuner),
dessa forma, é possível realizar as implementações do que chamamos de Agentes que devem ter um único objetivo.

Um Agente deve obrigatoriamente implementar [CommonAgent]() para que possua uma estrutura básica e que siga
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

A sugestão de repetições pode ser observada [aqui]() e a sugestão de peso [aqui](). Para chegar a conclusão de que
os modelos não foram feitos para essa finalidade basta observar os arquivos de imagem dos gráficos gerados:
[RandomForestRegressor](), [DecisionTreeClassifier]() e [KNeighborsRegressor]().

Como um complemento para os gráficos também foi utilizado um relatório com métricas de regressão que foi salvo
como um arquivo CSV, os arquivos podem ser encontrados para cada modelo testado: [RandomForestRegressor](), [DecisionTreeClassifier]() e [KNeighborsRegressor]().

Para facilitar abaixo está descrita uma tabela contendo as informações de cada um dos modelos testados.

| Modelo                     | Mean Absolute Error | Mean Squared Error | Root Mean Squared Error | R² Score | Explained Variance |
|----------------------------|---------------------|--------------------|-------------------------|----------|--------------------|
| [RandomForestRegressor]()  |                     |                    |                         |          |                    |
| [DecisionTreeClassifier]() |                     |                    |                         |          |                    |
| [KNeighborsRegressor]()    |                     |                    |                         |          |                    |

**Mean Absolute Error:** Mede o erro médio absoluto entre os valores reais e os previstos. É a média dos valores absolutos das diferenças entre valores reais e previstos.

**Mean Squared Error:** Mede o erro médio quadrado entre os valores reais e os previstos, penalizando erros maiores de forma mais significativa.

**Root Mean Squared Error:** É a raiz quadrada do MSE, trazendo a métrica para a mesma escala dos valores reais.

**Mean Absolute Error:** Representa a proporção da variabilidade dos dados explicada pelo modelo.

### Sugestões de Repetições e Peso usando Tensorflow e Keras

Como a implementação utilizando Scikit-Learn não chegou no ponto que gostaria foi preciso abordar o problema
com outro tipo de tecnologia, como as redes neurais. A ideia foi utilizar as redes LSTMs para que pudessem captar
a evolução dos dados de treino do indivíduo ao longo do tempo.

Foram implementadas as mesmas sugestões, para repetições o exemplo encontra-se [aqui](), já para peso podemos
ver [aqui](). Agora, utilizando uma tecnologia mais apropriada para análises temporais, podemos ver graficamente
que as redes testadas se adaptaram até um certo ponto aos dados do treinamento, veja o resultado de cada versão aqui:
[V1](), [V2](), [V3](), [V4]().

Como um complemento para os gráficos também foi utilizado um relatório com métricas de regressão que foi salvo
como um arquivo CSV, os arquivos podem ser encontrados para cada modelo testado: [RandomForestRegressor](), [DecisionTreeClassifier]() e [KNeighborsRegressor]().

Para facilitar abaixo está descrita uma tabela contendo as informações de cada um dos modelos testados.

| Modelo | Mean Absolute Error | Mean Squared Error | Root Mean Squared Error | R² Score | Explained Variance |
|--------|---------------------|--------------------|-------------------------|----------|--------------------|
| [V1]() |                     |                    |                         |          |                    |
| [V2]() |                     |                    |                         |          |                    |
| [V3]() |                     |                    |                         |          |                    |
| [V4]() |                     |                    |                         |          |                    |

**Mean Absolute Error:** Mede o erro médio absoluto entre os valores reais e os previstos. É a média dos valores absolutos das diferenças entre valores reais e previstos.

**Mean Squared Error:** Mede o erro médio quadrado entre os valores reais e os previstos, penalizando erros maiores de forma mais significativa.

**Root Mean Squared Error:** É a raiz quadrada do MSE, trazendo a métrica para a mesma escala dos valores reais.

**Mean Absolute Error:** Representa a proporção da variabilidade dos dados explicada pelo modelo.

