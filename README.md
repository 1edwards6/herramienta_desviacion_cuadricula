# Analizador de imágenes con cuadrícula 2×2

Aplicación en Streamlit para dividir una imagen en cuatro regiones ajustables y analizar sus estadísticas de intensidad.

## Funciones

- Carga de imágenes PNG, JPG/JPEG, BMP, WEBP y TIFF.
- Dos controles para mover los ejes vertical y horizontal; la cuadrícula no tiene que ser equitativa.
- Recalculo de las métricas al cambiar los controles.
- Desviación estándar y media de luminancia por región.
- Desviación estándar de los canales R, G y B por separado.
- Dimensiones, cantidad de píxeles, porcentaje de área y extremos de luminancia.
- Histogramas de luminancia opcionales.
- Descarga de métricas en CSV y de la imagen con cuadrícula en PNG.

## Ejecutar localmente

Requiere Python 3.10 o superior.

```bash
python -m venv .venv
```

Activar el entorno:

- Windows PowerShell: `\.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Instalar dependencias e iniciar Streamlit:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Se abrirá la aplicación en el navegador, normalmente en `http://localhost:8501`.

## Publicar usando GitHub y Streamlit Community Cloud

1. Crea un repositorio en GitHub, por ejemplo `analizador-cuadricula-2x2`.
2. Sube `app.py`, `requirements.txt` y `README.md` a la raíz del repositorio.
3. Entra en [Streamlit Community Cloud](https://share.streamlit.io/) y conecta tu cuenta de GitHub.
4. Crea una aplicación nueva, elige el repositorio y selecciona `app.py` como archivo principal.
5. Pulsa **Deploy**. Cuando termine, tendrás una URL pública para abrir la aplicación.

## Cómo se interpretan las métricas

La luminancia se aproxima mediante `Y = 0.2126 R + 0.7152 G + 0.0722 B`. Su desviación estándar es la desviación poblacional de todos los píxeles de una región. Un valor más alto significa que las intensidades están más dispersas, pero no implica automáticamente que la región se perciba mejor o tenga más detalle.

Las métricas se calculan sobre la imagen a resolución completa; la visualización del navegador puede mostrarse reducida para ajustarse a la pantalla.
