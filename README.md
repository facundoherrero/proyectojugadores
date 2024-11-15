# Global Redes Neuronales Profundas

Este proyecto se trata de un generador de imagenes que utiliza VAE para producir 5 imagenes tomando 2 fotos de jugadores al azar a partir de un filtro y realizando una interpolación.

## Estructura del Proyecto

- **data/**: Archivos de datos utilizados para entrenamiento y evaluación.  
- **dev/**: Notebooks y scripts para el desarrollo y experimentación.  
- **prod/**: Código preparado para el entorno de producción.

## Uso

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/facundoherrero/proyectojugadores.git
   ```
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecutar app de streamlit:
   ```bash
   streamlit run prod\app.py
   ```
