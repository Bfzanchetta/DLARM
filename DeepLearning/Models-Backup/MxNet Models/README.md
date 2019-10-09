# Modelos em Teste #
**Data:** 15/07  
**Descrição:** Testando os modelos que serão utilizados nos experimentos.

# Tabela dos Parâmetros Finais #
| MODELO        | TESTADO       | FUNCIONANDO   |
| ------------- | ------------- | ------------- |
| `MX-GPU-L-ALEX-mIMGN10.py`| Sim | Sim |
| `MX-GPU-L-CNN-mIMGN10.py` | Sim | Sim |
| `MX-GPU-L-DNS-mIMGN10.py` | Sim | Sim(!) |
| `MX-GPU-L-GGL-mIMGN10.py` | Sim | Sim |
| `MX-GPU-L-RES-mIMGN10.py` | Sim | Sim |
| `MX-GPU-L-SQZ-mIMGN10.py` | Sim | Sim(!) |
| `MX-GPU-L-VGG-mIMGN10.py` | Sim | Sim |

Para compilar:  
-Posicione a pasta contendo os diretórios de CPU, GPU e README no mesmo diretório da pasta contendo o dataset.  
-`python MX-...-dataset.py`


Usar como base:
/osmr/imgclsmob/tree/master/gluon/gluoncv2/models
