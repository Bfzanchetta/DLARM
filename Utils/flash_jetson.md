# Steps to Flash NVIDIA Jetson TX2

## In English
To-do




## In Portuguese-BR

Ao iniciar a usar a placa, ela necessita de um Flash, ou seja, necessita ser reiniciada para as configurações de fábrica.
É verdade que ela já vem com um Sistema Operacional instalado, mas sempre que houver erros de instalação ou se desejar instalar um OS diferente, necessita-se realizar o processo de Flash (formatação da placa para configuração de fábrica).
Para realizar o Flash, deve-se entrar no site da NVIDIA e encontrar a versão de Ubuntu e JetPack se deseja instalar.
Para facilitar o processo, a NVIDIA já incorpora a instalação do Ubuntu à instalação do JetPack.

O Ubuntu normalmente usado na Jetson TX2 é o 14.04.
O JetPack é um pacote de bibliotecas e softwares de desenvolvimento para Jetson. O JetPack contém CUDA, OpenCV, CuDNN e várias outras bibliotecas que serão usadas na placa. 

##### Para a instalação do JetPack siga: 

Vá até a página da NVIDIA e baixe a versão de JetPack dejesada.
Deve-se manter atenção sobre as configuraçes de software que cada versão de JetPack propicia, especialmente em relação a CUDA, CuDNN e OpenCV, a fim de que não haja problemas de compatibilidade com os softwares e frameworks que serão instalados na placa.

Outro cuidado que se deve tomar é em relação ao sistema operacional do computador padro(host), pois as versões de instalação para Jetson TX1/TX2 não apresentam suporte a Ubuntu 18.04, mas funcionam para Ubuntu 14.04 e 16.04.

Às vezes o instalador também demonstra erros no instalador para Ubuntu 16.04, para isso é importante usar alguns comandos, que também servem a alguns instaladores em 18.04:

```
sudo wget -q -O /tmp/libpng12.deb http://mirrors.kernel.org/ubuntu/pool/main/libp/libpng/libpng12-0_1.2.54-1ubuntu1_amd64.deb \
&& sudo dpkg -i /tmp/libpng12.deb \
&& sudo rm /tmp/libpng12.deb  
```
Seguido de:

```
./JetPack-<version>.run --noexec #This will create a pre-installation folder
cd _installer #enter this folder
sudo gedit start_up.sh
  >>> remove code at end of file, from the last if to EOF
sudo ./Launcher
#Proceed to normal installation
```
Nesse instalador, o computador padrão(host) vai pré-instalar todos arquivos localmente, ao finalizar ele estará pronto para passar essa instalação à placa Jetson. Basta ligar as duas máquinas pela mini-USB (ponta USB normal no computador e mini-USB na Jetson).

Para confirmar a conexão, use o comando ```$lsusb``` para encontrar o hardware da NVIDIA Corp.
Além disso, sugere-se que tanto a placa quanto o computador padrão estejam ligados simultaneamente no mesmo switch por meio de cabos ethernet.

O procedimento leva alguns minutos(~20-35min) e termina pedindo que se feche a tela de instalação.

A placa está pronta para ser usada.  

##### Para a instalação apenas do SO sem JetPack siga: 

Primeiro, deve-se baixar os arquivos de instalação do Ubuntu em um computador padrão, que seja diferente da placa. O download deve consistir de dois arquivos: A pasta do Linux_for_Tegra e uma pasta de rootfs personalizada para tal instalação. Via terminal, o usuário deve extrair a pasta Linux_for_Tegra, para depois também extrair por terminal o conteúdo do .tar nomeado rootfs no diretório de mesmo nome de Linux_for_Tegra(Atenção, se não for realizado por terminal com sudo, a instalação dará errado).

Depois de ter realizado esse procedimento, esta é a pasta que carrega o Sistema Operacional que a placa vai conter. O Usuário deve conectar a placa TX2 ao computador padrão pela mini-USB.

Para verificar se a porta USB está sendo reconhecida, o usuário pode usar o comando shell $lsusb.
Caso não haja nenhum equipamento Nvidia Corp., pode significar que a mini-USB está com defeito, tente trocar o cabo.

O usuário deve colocar a placa em Power Recovery mode. Para fazer isso, basta ligar a placa com o botão POWER, depois pressionar e segurar o botão RST por alguns segundos. Logo, continue segurando o botão RST e pressione também o botão REC por dois segundos. Solte o botão REC sem soltar o RST, então segure o RST por mais dois segundos, para depois soltá-lo. 

Se sua placa estiver conectada a um monitor, ela permanecerá escura, ou seja, a placa está pronta para receber uma alteração. Repita o comando lsusb para validar que a placa está sendo reconhecida. 

Vá até a pasta Linux_for_Tegra e execute como superusuário o script apply_binnaries.sh, após a sua conclusão, execute com sudo o script com o comando $sudo ./flash.sh jetson-tx2 mmcblk0p1.

O Processo leva alguns minutos, que culmina com o reboot da placa Jetson. Está formatada a placa.


