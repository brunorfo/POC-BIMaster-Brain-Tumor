<!-- antes de enviar a versão final, solicitamos que todos os comentários, colocados para orientação ao aluno, sejam removidos do arquivo -->
# POC-BIMaster-Brain-Tumor

#### Aluno: [Bruno Rodrigues](https://github.com/brunorfo).
#### Orientadora: [Manoela Kohler](https://github.com/manoelakohler).

---

Trabalho apresentado ao curso [BI MASTER](https://ica.puc-rio.ai/bi-master) como pré-requisito para conclusão de curso e obtenção de crédito na disciplina "Projetos de Sistemas Inteligentes de Apoio à Decisão".

<!-- para os links a seguir, caso os arquivos estejam no mesmo repositório que este README, não há necessidade de incluir o link completo: basta incluir o nome do arquivo, com extensão, que o GitHub completa o link corretamente -->
- [Notebook kaggle](https://github.com/brunorfo/POC-BIMaster-Brain-Tumor/blob/main/notebook1a3375ef89_final.ipynb). <!-- caso não aplicável, remover esta linha -->

- [Competição kaggle](https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification)


---

### Resumo



O trabalho de conclusão de curso será a elaboração de uma rede neural para classificação da presença da sequência genética em tumores de cérebro conhecido como MGMT promoter methylation em imagens de ressonância magnética multiparamétricas. Esse problema foi proposto em uma competição hospedada na plataforma Kaggle e que tinha como patrocinadora a Radiological Society of North America (RSNA) que é uma organização sem fins lucrativos.

Atualmente para identificar se um cancer de cérebro tem o MGMT e definir o melhor tratamento, é preciso fazer a extração de uma pequena amostra e fazer a análise genética, o que pode levar várias semanas. Utilizando técnicas de inteligência artificial para determinar se um paciente tem ou não a presença do MGMT no tumor cerebral, os médicos podem tomar decisões de tratamento mais rápido que os métodos atuais e aumentar as chances de cura do paciente.



### 1. Introdução

O dataset é da competição da RSNA de Classificação de Tumor Cerebral hospedado no Kaggke é componsto por imagens de ressonância magnética multiparamétrica, contendo para cada paciente quatro tipos de imagens, sendo elas a Fluid Attenuated Inversion Recovery (FLAIR), T1-weighted pre-contrast (T1w), T1-weighted post-contrast (T1Gd) e T2-weighted (T2). Para simplificação do problema, o modelo proposto usou como referência cada imagem individual e a classificação de presença do MGMT ou não no paciente, transformando cada arquivo em uma entrada individual da rede neural com sua classificação atribuída.
 
A estrutura dos diretórios fornecidos pela RSNA é da seguinte forma:


### 2. Modelagem

Como se tratava de um desafio de classificação de imagens, optamos pela utilização de uma rede neural convolucional uma vez que esse tipo de rede é o que apresenta os melhores resultados ao se lidar com imagem. Num primeiro momento elaboramos uma rede simples com apenas quatro camadas de forma experimental, porém não obtivemos uma acurácia satisfatória e a rede não apresentava evolução no aprendizado. Foi utilizado então uma rede Xception pré-treinada fazendo assim um processo de transferência de aprendizado para o nosso problema. Ao treinar a nossa nova rede, passamos a observar a evolução no modelo proposto, conseguindo uma acurácia de aproximadamente setenta e nove porcento e sendo considerado satisfatória.



### 3. Resultados

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin pulvinar nisl vestibulum tortor fringilla, eget imperdiet neque condimentum. Proin vitae augue in nulla vehicula porttitor sit amet quis sapien. Nam rutrum mollis ligula, et semper justo maximus accumsan. Integer scelerisque egestas arcu, ac laoreet odio aliquet at. Sed sed bibendum dolor. Vestibulum commodo sodales erat, ut placerat nulla vulputate eu. In hac habitasse platea dictumst. Cras interdum bibendum sapien a vehicula.

Finalizado o treinamento, o modelo foi salvo em arquivo do formato ".h5" e posteriormente restaurado e utilizado para fazer as predições das imagens de teste, obtendo assim as probalidades de cada imagem serem do tipo com presença de MGMT ou não.


### 4. Conclusões



---

Matrícula: 192.671.026

Pontifícia Universidade Católica do Rio de Janeiro

Curso de Pós Graduação *Business Intelligence Master*
