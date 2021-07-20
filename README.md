# YOLOv3-Streamlit

App implementada con Streamlit para la detección de objetos a través de modelo pre-entrenado YOLOv3.

**Acerca de Streamlit**

**Streamlit** es una biblioteca de python de código abierto que es útil para crear y compartir aplicaciones web de datos. Poco a poco está ganando mucho impulso en la comunidad de la ciencia de datos. Debido a la facilidad con la que se puede desarrollar una aplicación web de ciencia de datos, muchos desarrolladores la usan en su flujo de trabajo diario. El repositorio de GitHub streamlit tiene más de 14.1k estrellas y 1.2k bifurcaciones. Bajo el capó, utiliza React como un marco frontend para renderizar los datos en la pantalla. Por lo tanto, los desarrolladores de React pueden manipular fácilmente la interfaz de usuario con pocos cambios en el código.

#### IMPLEMENTACIÓN  

1. Inicialmente se realiza la instalación de la distribución Anaconda siguiendo la guía de [página oficial](https://docs.anaconda.com/anaconda/install/windows/). 
2. Creamos un nuevo entorno virtual (Python 3.8).

3. Abrimos una ventana *Open Terminal*, y realizamos la instalación de las siguientes dependencias requeridas:

- `pip install opencv-python`
- `pip install streamlit`
- `pip install -U scikit-learn`
- `pip install matplotlib`

4. Descargamos este repositorio.

5. Para realizar prueba de nuestra primera aplicación, abrimos una ventana *Open Terminal* de nuestro entorno virtual y nos ubicamos en la ruta de la carpeta descargada. Ejecutamos la aplicación `app.py` a través del siguiente comando:

   `streamlit run app.py`

   se abrirá una ventana en nuestro navegador, como se muestra a continuación:

  ![](https://github.com/carlosjulioph/YOLOv3-Streamlit/blob/main/Im%C3%A1genes_readme/1.png)

#### DETECCIÓN DE OBJETOS CON YOLOv3
Descargue el archivo de pesos YOLO v3 previamente entrenado desde este [enlace](https://drive.google.com/file/d/1RoP4rVJo8f2ERZNaTRF_A3hwOUHt3ODd/view?usp=sharing) y colócalo en el directorio descargado.

Ejecutamos la aplicación `app.py` a través del siguiente comando:

`streamlit run YOLOv3_app.py`

![](https://github.com/carlosjulioph/YOLOv3-Streamlit/blob/main/Im%C3%A1genes_readme/2.png)

####   
