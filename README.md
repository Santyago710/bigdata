# MyTwin · Sistema de recomendación con perfil virtual de usuario

Proyecto de **Big Data e Inteligencia de Negocios** (Universidad Distrital Francisco José de Caldas, Ingeniería de Sistemas).

MyTwin construye un **perfil virtual** de cada usuario (marcas, plataformas, rango de precio e intereses) a partir de sus reseñas e interacciones, y lo usa para recomendar productos. Incluye una **aplicación web** para ver cómo funciona el modelo con usuarios ya registrados y con usuarios nuevos.

**Datos:** [Amazon Reviews'23](https://amazon-reviews-2023.github.io/) (McAuley Lab, UCSD), categoría *Video Games*.

---

## Qué incluye la aplicación web

- **Usuarios existentes:** perfil virtual, últimas compras y 10 recomendaciones con su motivo.
- **Prueba de realidad:** muestra el siguiente producto que el usuario realmente reseñó (oculto para el modelo) y en qué posición lo habría recomendado.
- **Usuario nuevo (arranque en frío):** se crea el perfil con plataformas, marcas, palabras clave y presupuesto.
- **Eventos en vivo:** visto, guardado, descartado y comprado. El perfil se actualiza al instante y la lista se reordena.
- **Explicación:** cada tarjeta muestra cuánto aportó cada modelo (contenido, vecinos, SVD, segmento y popularidad).
- **Filtros** por plataforma y precio máximo.

---

## Estructura del repositorio

> Ajusta esta sección a cómo hayas organizado tu repo.

```
.
├── mytwin_app/
│   ├── server.py          # servidor web (solo librería estándar + engine.py)
│   ├── engine.py          # motor de recomendación
│   ├── static/
│   │   └── index.html     # interfaz web
│   └── requirements.txt   # dependencias
├── notebooks/             # notebooks de Colab (pasos 0 a 4)
└── README.md
```

La carpeta de datos **`app_data/` no se sube al repositorio** (pesa demasiado). Se genera con los notebooks, como se explica abajo.

---

## Requisitos

- **Python 3.9 o superior** (`python3 --version`).
- Dependencias de `mytwin_app/requirements.txt`:

```
numpy>=1.24
pandas>=2.0
scipy>=1.10
pyarrow>=14.0
```

No necesita GPU ni conexión a internet para correr la aplicación.

---

## Paso 1: generar los datos de la aplicación (`app_data`)

La app necesita una carpeta `app_data` con los modelos y datos ya procesados. Se genera en **Google Colab**:

1. Ejecuta los notebooks en orden hasta el paso 4 (`00`, `01`, `02`, `03a`, `04`). Producen las carpetas `data/` y `models/` en tu Google Drive (`MyDrive/MyTwin`).
2. Abre `04b_app_web_MyTwin.ipynb` y ejecuta **solo las celdas 1, 2 y 3** (montar Drive, preparar `app_data` y, opcionalmente, nombrar los segmentos).
3. En tu Drive quedará `MyTwin/app_data`. **Descárgala** a tu computador (Drive la entrega como `.zip`; descomprímela).

La carpeta debe contener estos archivos:

| Archivo | Obligatorio |
|---|---|
| `hybrid_config.json` | Sí |
| `interactions_light.parquet` | Sí |
| `items_catalog.parquet` | Sí |
| `users_index.parquet` | Sí |
| `user_segments.parquet` | Sí |
| `item_content_F.npz` | Sí |
| `knn_W.npz` | Sí |
| `svd_V.npy` | Sí |
| `svd_scale.npy` | Sí |
| `resultados_modelos.csv` | No (tabla de métricas en la pestaña «Cómo funciona») |
| `segment_names.json` | No (nombres de los segmentos) |

> **Importante:** al descomprimir, Drive a veces deja los archivos **dentro de una subcarpeta** (por ejemplo `app_data-2026XXXX.../app_data/`). La ruta que le pases al servidor debe ser la carpeta que contiene **directamente** estos archivos.

---

## Paso 2: instalar dependencias

Clona el repositorio y entra a la carpeta de la app:

```bash
git clone <URL_DE_TU_REPOSITORIO>
cd <NOMBRE_DEL_REPO>/mytwin_app
```

Instala las dependencias:

```bash
pip3 install -r requirements.txt
```

Si tu sistema responde `externally-managed-environment`, usa un entorno virtual:

```bash
python3 -m venv venv
source venv/bin/activate          # Windows (PowerShell): venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Paso 3: ejecutar la aplicación

El comando es el mismo en todos los sistemas. Lo único que cambia es **la ruta a `app_data`, que depende de dónde la hayas guardado**.

```bash
python3 server.py --data "<RUTA_A_APP_DATA>" --port 8000
```

Ejemplos de `<RUTA_A_APP_DATA>`:

| Sistema | Ejemplo |
|---|---|
| Linux / WSL | `/home/usuario/bigdata/app_data` |
| macOS | `/Users/usuario/bigdata/app_data` |
| Windows | `C:\Users\usuario\bigdata\app_data` |
| Ruta relativa | `../app_data` (si está junto a la carpeta `mytwin_app`) |

Si la ruta tiene espacios, déjala entre comillas.

Cuando aparezca un mensaje como:

```
MyTwin listo: 95,901 usuarios · 25,709 productos -> http://127.0.0.1:8000
```

abre **http://localhost:8000** en el navegador. Para detener el servidor, pulsa `Ctrl + C` en la terminal.

### Opciones

| Opción | Descripción | Valor por defecto |
|---|---|---|
| `--data` | Ruta a la carpeta `app_data` (obligatoria) | — |
| `--port` | Puerto del servidor | `8000` |
| `--host` | `127.0.0.1` solo tu equipo; `0.0.0.0` toda la red | `127.0.0.1` |

> La app **no tiene autenticación**. No la expongas con `--host 0.0.0.0` fuera de una red de confianza.

---

## Uso rápido de la web

1. **Usuarios existentes:** elige un usuario de la lista (o búscalo por ID) y observa su perfil y sus recomendaciones.
2. **Usuario nuevo:** define tus preferencias o pulsa «Rellenar con un ejemplo». También puedes buscar un producto que ya uses y agregarlo.
3. En cada tarjeta usa 👁 visto · ⭐ guardado · ✖ descartado · 🛒 comprado. Las flechas ▲▼ indican cómo cambió cada posición.
4. La pestaña **Cómo funciona** muestra los pesos del modelo híbrido, los pesos de los eventos y las métricas.

---

## Solución de problemas

| Problema | Causa y solución |
|---|---|
| `FileNotFoundError: ... hybrid_config.json` | La ruta de `--data` apunta a la carpeta equivocada. Revisa con `ls "<RUTA_A_APP_DATA>"`: debes ver directamente `hybrid_config.json`, `interactions_light.parquet`, etc. Si ves otra carpeta, entra a ella y usa esa ruta. |
| `FileNotFoundError` con otro nombre de archivo | Falta ese archivo en `app_data`. Vuelve a ejecutar la celda 2 de `04b_app_web_MyTwin.ipynb` y descarga de nuevo. |
| `ModuleNotFoundError: No module named 'pyarrow'` (o `pandas`, `scipy`, `numpy`) | Faltan dependencias. Repite `pip3 install -r requirements.txt` (dentro del entorno virtual si usas uno). |
| `externally-managed-environment` | Usa un entorno virtual (ver Paso 2). |
| `Address already in use` | El puerto está ocupado. Usa otro: `--port 8001` y abre `http://localhost:8001`. |
| Tarda en arrancar | Es normal: carga el catálogo, las matrices y todas las interacciones al inicio. Luego responde rápido. |
| Se queda sin memoria | Cierra otros programas. Se recomienda tener al menos 8 GB de RAM libres/disponibles. |
| En WSL no abre `localhost:8000` | Abre la URL desde el navegador de Windows. Si no funciona, ejecuta `wsl --shutdown` en PowerShell, vuelve a abrir la terminal e inténtalo de nuevo. |
| Archivos `*:Zone.Identifier` | Marcas de Windows al descargar archivos. No afectan nada; puedes borrarlos. |

---

## Ejecutar desde Google Colab (sin instalar nada)

Si prefieres no correr nada en local, abre `04b_app_web_MyTwin.ipynb` en Colab y ejecuta todas las celdas. La web se muestra dentro del propio notebook. Si el marco no carga, usa la línea `serve_kernel_port_as_window` que aparece comentada en la última celda de código.

---

## Pipeline del proyecto (notebooks en Colab)

| Notebook | Qué hace |
|---|---|
| `00_setup_entorno` | Descarga de datos y conversión a Parquet (capa *bronze*) con PySpark |
| `01_exploracion_calidad_silver` | Calidad de datos, limpieza y filtro k-core (capa *silver*) |
| `02_negocio_kpis_gold` | KPIs de negocio y tablas *gold* con split temporal |
| `03a_segmentacion_usuarios` | Segmentación de usuarios con K-Means |
| `03b_sentimientos_gpu` | Análisis de sentimientos de reseñas (GPU T4) |
| `04_recomendador` | Modelos de recomendación, evaluación y perfil dinámico |
| `04b_app_web` | Aplicación web del recomendador |

---

## Archivos que NO deben subirse al repositorio

Agrega un `.gitignore` con, al menos:

```
app_data*/
venv/
__pycache__/
*.parquet
*.npz
*.npy
*:Zone.Identifier
```

---

## Autores

- Dilan Stive Arboleda Zambrano · 20212020105
- David Santiago Buitrago · 20221020085

Universidad Distrital Francisco José de Caldas · Facultad de Ingeniería · Ingeniería de Sistemas · Bogotá, Colombia.

## Créditos de datos

Amazon Reviews'23, McAuley Lab, Universidad de California en San Diego (UCSD). Consulta las condiciones de uso en el [sitio oficial del dataset](https://amazon-reviews-2023.github.io/).
