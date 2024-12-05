# Demo Huemul Solutions IA DEVS 2024

Este repositorio contiene ejemplos prácticos sencillos que muestran la utilización de la librería **Enola AI**, una herramienta que permite la observabilidad de proyectos y desarrollos con modelos de inteligencia artificial generativa.

### Instalación de librerías
*Se recomienda realizar en una ambiente aislado (venv, container, etc) para evitar conflictos.*
El archivo requirements.txt contiene las librerías a instalar para poder ejecutar el código.
Se puede hacer con el comando ```pip install -r requirements.txt```.

### Variables de entorno
Se deben configurar las variables de entorno, lo cual se puede hacer usando un archivo .env, lo cual va a permitir usar Enola y el modelo LLM a elección, en este caso AzureOpenAI.

### Uso de los agentes
Existen dos archivos de python, agent.py y agent_enola.py, siendo la diferencia entre ellos que el segundo agrega el monitoreo usando Enola AI.
Mientras se está ejecutando el programa, si se quiere terminar la ejecución escribiendo "/exit".
Además, al ejecutar el segundo archivo (agent_enola.py) se tiene la opción extra de mandar una evaluación sobre la última respuesta generada por el agente al escribir "/eval" e ingresando los datos solicitados.