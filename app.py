from __future__ import annotations

from io import BytesIO
import hashlib

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError


st.set_page_config(
    page_title="Analizador de cuadrícula 2×2",
    page_icon="▦",
    layout="wide",
)

st.title("Analizador de imágenes · Cuadrícula 2×2")
st.markdown(
    "Mueve los ejes vertical y horizontal para dividir la imagen en cuatro regiones "
    "de tamaños distintos y observa cómo cambian sus estadísticas."
)
st.caption(
    "La desviación estándar se calcula sobre la luminancia de cada píxel, "
    "en una escala aproximada de 0 a 255. También se muestran las desviaciones "
    "de los canales RGB."
)


@st.cache_data(show_spinner=False)
def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode an uploaded image and return an RGB array."""
    with Image.open(BytesIO(image_bytes)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        return np.asarray(image, dtype=np.uint8)


def get_luminance(rgb: np.ndarray) -> np.ndarray:
    """Approximate perceived luminance using the RGB components."""
    values = rgb.astype(np.float32)
    return (
        0.2126 * values[..., 0]
        + 0.7152 * values[..., 1]
        + 0.0722 * values[..., 2]
    )


def region_metrics(
    region: np.ndarray, name: str, image_height: int, image_width: int
) -> dict[str, float | int | str]:
    """Compute descriptive intensity statistics for an RGB region."""
    luminance = get_luminance(region)
    region_height, region_width = region.shape[:2]
    row: dict[str, float | int | str] = {
        "Región": name,
        "Ancho (px)": int(region_width),
        "Alto (px)": int(region_height),
        "Píxeles": int(region_height * region_width),
        "Área (%)": 100.0 * region_height * region_width / (image_height * image_width),
        "Media luminancia": float(np.mean(luminance)),
        "Desv. estándar luminancia": float(np.std(luminance)),
        "Mín. luminancia": float(np.min(luminance)),
        "Máx. luminancia": float(np.max(luminance)),
        "Desv. estándar R": float(np.std(region[..., 0].astype(np.float32))),
        "Desv. estándar G": float(np.std(region[..., 1].astype(np.float32))),
        "Desv. estándar B": float(np.std(region[..., 2].astype(np.float32))),
    }
    return row


def reset_axes() -> None:
    """Reset both sliders before their widgets are rendered on the next rerun."""
    st.session_state["x_percent"] = 50
    st.session_state["y_percent"] = 50


def add_grid_overlay(rgb: np.ndarray, x_cut: int, y_cut: int) -> Image.Image:
    """Draw visible, high-contrast grid axes over a copy of the image."""
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)
    width, height = image.size
    thickness = max(2, round(min(width, height) / 350))
    for offset in range(-thickness, thickness + 1):
        xx = max(0, min(width - 1, x_cut + offset))
        yy = max(0, min(height - 1, y_cut + offset))
        draw.line((xx, 0, xx, height - 1), fill=(20, 20, 20), width=1)
        draw.line((0, yy, width - 1, yy), fill=(20, 20, 20), width=1)
    draw.line((x_cut, 0, x_cut, height - 1), fill=(255, 70, 70), width=thickness)
    draw.line((0, y_cut, width - 1, y_cut), fill=(255, 70, 70), width=thickness)
    return image


def cut_positions(percentages: np.ndarray, size: int) -> np.ndarray:
    """Convert percentages into valid pixel cut positions."""
    cuts = np.rint(size * percentages / 100.0).astype(int)
    return np.clip(cuts, 1, size - 1)


def standard_deviation_from_sums(
    sums: np.ndarray, sums_squared: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    """Calculate population standard deviations from sums and squared sums."""
    means = sums / counts
    variances = sums_squared / counts - means**2
    # Floating-point arithmetic can produce tiny negative values for near-constant data.
    return np.sqrt(np.maximum(variances, 0.0))


def deviation_curve(
    luminance: np.ndarray,
    axis: str,
    region_number: int,
    x_cut: int,
    y_cut: int,
) -> pd.DataFrame:
    """Calculate the selected region's deviation across all slider positions.

    The other axis stays fixed. Column/row aggregates and cumulative sums avoid
    recalculating every complete region for each of the 99 possible positions.
    """
    height, width = luminance.shape
    percentages = np.arange(1, 100, dtype=int)

    if axis == "Eje vertical":
        # The horizontal axis is fixed; summarize each column in its top/bottom band.
        if region_number in (1, 2):
            band = luminance[:y_cut, :]
            band_height = y_cut
        else:
            band = luminance[y_cut:, :]
            band_height = height - y_cut

        column_sums = np.sum(band, axis=0, dtype=np.float64)
        column_squares = np.sum(band * band, axis=0, dtype=np.float64)
        prefix_sums = np.concatenate(([0.0], np.cumsum(column_sums, dtype=np.float64)))
        prefix_squares = np.concatenate(([0.0], np.cumsum(column_squares, dtype=np.float64)))

        cuts = cut_positions(percentages, width)
        if region_number in (1, 3):  # left-hand regions
            sums = prefix_sums[cuts]
            sums_squared = prefix_squares[cuts]
            region_widths = cuts
        else:  # right-hand regions
            sums = prefix_sums[-1] - prefix_sums[cuts]
            sums_squared = prefix_squares[-1] - prefix_squares[cuts]
            region_widths = width - cuts
        counts = region_widths.astype(np.float64) * band_height
        deviations = standard_deviation_from_sums(sums, sums_squared, counts)
        current_percent = int(st.session_state["x_percent"])
        axis_label = "Separación del eje vertical (%)"

    else:
        # The vertical axis is fixed; summarize each row in its left/right band.
        if region_number in (1, 3):
            band = luminance[:, :x_cut]
            band_width = x_cut
        else:
            band = luminance[:, x_cut:]
            band_width = width - x_cut

        row_sums = np.sum(band, axis=1, dtype=np.float64)
        row_squares = np.sum(band * band, axis=1, dtype=np.float64)
        prefix_sums = np.concatenate(([0.0], np.cumsum(row_sums, dtype=np.float64)))
        prefix_squares = np.concatenate(([0.0], np.cumsum(row_squares, dtype=np.float64)))

        cuts = cut_positions(percentages, height)
        if region_number in (1, 2):  # top regions
            sums = prefix_sums[cuts]
            sums_squared = prefix_squares[cuts]
            region_heights = cuts
        else:  # bottom regions
            sums = prefix_sums[-1] - prefix_sums[cuts]
            sums_squared = prefix_squares[-1] - prefix_squares[cuts]
            region_heights = height - cuts
        counts = region_heights.astype(np.float64) * band_width
        deviations = standard_deviation_from_sums(sums, sums_squared, counts)
        current_percent = int(st.session_state["y_percent"])
        axis_label = "Separación del eje horizontal (%)"

    curve = pd.DataFrame(
        {
            "Separación (%)": percentages,
            "Desviación estándar": deviations,
        }
    )
    curve["Posición actual"] = curve["Separación (%)"] == current_percent
    curve.attrs["axis_label"] = axis_label
    curve.attrs["current_percent"] = current_percent
    return curve


def strongest_derivative_by_region(
    luminance: np.ndarray,
    axis: str,
    x_cut: int,
    y_cut: int,
) -> tuple[int, dict[str, object]]:
    """Choose an axis position from the strongest derivative location in each region.

    For each of the four regions, locate the interval with the largest absolute
    finite-difference derivative. The four interval midpoints are then summarized:
    median if their population standard deviation is < 20 percentage points,
    otherwise mean. The returned position is rounded to the nearest slider value.
    """
    peak_rows: list[dict[str, float | int | str]] = []

    for region_number in range(1, 5):
        curve = deviation_curve(luminance, axis, region_number, x_cut, y_cut)
        curve = curve.sort_values("Separación (%)").reset_index(drop=True)
        positions = curve["Separación (%)"].to_numpy(dtype=float)
        deviations = curve["Desviación estándar"].to_numpy(dtype=float)

        slopes = np.diff(deviations) / np.diff(positions)
        interval_index = int(np.argmax(np.abs(slopes)))
        start = float(positions[interval_index])
        end = float(positions[interval_index + 1])
        peak_position = (start + end) / 2.0
        slope = float(slopes[interval_index])

        peak_rows.append(
            {
                "Eje analizado": axis,
                "Cuadrícula": region_number,
                "Intervalo (%)": f"{start:.0f}–{end:.0f}",
                "Posición del cambio (%)": peak_position,
                "Derivada aproximada": slope,
                "Valor absoluto de la derivada": abs(slope),
            }
        )

    peak_positions = np.array(
        [row["Posición del cambio (%)"] for row in peak_rows], dtype=float
    )
    spread = float(np.std(peak_positions))
    if spread < 20.0:
        method = "Mediana (desviación < 20)"
        summary_value = float(np.median(peak_positions))
    else:
        method = "Media (desviación ≥ 20)"
        summary_value = float(np.mean(peak_positions))

    recommended_position = int(np.clip(np.rint(summary_value), 1, 99))
    return recommended_position, {
        "eje": axis,
        "picos": peak_rows,
        "posiciones": peak_positions.tolist(),
        "desviacion_posiciones": spread,
        "metodo": method,
        "valor_resumen": summary_value,
        "posicion_recomendada": recommended_position,
    }


uploaded = st.file_uploader(
    "Carga una imagen",
    type=["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"],
    help="Los cálculos se realizan con la resolución completa de la imagen.",
)

if uploaded is None:
    st.info("Carga una imagen para comenzar. Después podrás mover los dos ejes de la cuadrícula 2×2.")
    st.markdown(
        "**Cómo funciona:** el eje vertical determina la separación izquierda/derecha y el horizontal "
        "determina la separación superior/inferior. Cada cambio recalcula las métricas y la curva seleccionada."
    )
    st.stop()

try:
    rgb = decode_image(uploaded.getvalue())
except (UnidentifiedImageError, OSError, ValueError) as exc:
    st.error(f"No fue posible leer la imagen. Prueba con otro archivo. Detalle: {exc}")
    st.stop()

image_height, image_width = rgb.shape[:2]
if image_height < 2 or image_width < 2:
    st.error("La imagen debe tener al menos 2 píxeles de ancho y 2 de alto para dividirse en cuatro regiones.")
    st.stop()

st.caption(f"Imagen: **{uploaded.name}** · Dimensiones: **{image_width} × {image_height} px**")

# Defaults are set before the sliders are created so the automatic button can
# update their values safely during this run.
st.session_state.setdefault("x_percent", 50)
st.session_state.setdefault("y_percent", 50)

# Avoid displaying a previous image's automatic-adjustment report after uploading another image.
image_token = hashlib.sha256(uploaded.getvalue()).hexdigest()
if st.session_state.get("auto_image_token") != image_token:
    st.session_state.pop("auto_axis_result", None)
    st.session_state["auto_image_token"] = image_token

button_col, explanation_col = st.columns([1, 2.4])
with button_col:
    if st.button(
        "Ajustar ejes automáticamente",
        type="primary",
        use_container_width=True,
        help="Busca en cada región el intervalo con mayor cambio absoluto y combina las cuatro posiciones detectadas.",
    ):
        auto_luminance = get_luminance(rgb)
        current_x = int(st.session_state.get("x_percent", 50))
        current_y = int(st.session_state.get("y_percent", 50))
        current_x_cut = int(np.clip(round(image_width * current_x / 100.0), 1, image_width - 1))
        current_y_cut = int(np.clip(round(image_height * current_y / 100.0), 1, image_height - 1))

        recommended_x, vertical_result = strongest_derivative_by_region(
            auto_luminance, "Eje vertical", current_x_cut, current_y_cut
        )
        recommended_y, horizontal_result = strongest_derivative_by_region(
            auto_luminance, "Eje horizontal", current_x_cut, current_y_cut
        )

        st.session_state["x_percent"] = recommended_x
        st.session_state["y_percent"] = recommended_y
        st.session_state["auto_axis_result"] = {
            "vertical": vertical_result,
            "horizontal": horizontal_result,
            "image_token": image_token,
        }
with explanation_col:
    st.caption(
        "Para cada una de las cuatro regiones, detecta el cambio más brusco (máximo |derivada|). "
        "Si la desviación estándar de las cuatro posiciones detectadas es menor que 20 puntos porcentuales, "
        "usa su mediana; de lo contrario, usa su media. Cada eje se calcula manteniendo fijo el otro "
        "en su posición anterior al ajuste."
    )

control_col1, control_col2, reset_col = st.columns([2, 2, 1])
with control_col1:
    x_percent = st.slider(
        "Eje vertical · separación izquierda/derecha (%)",
        min_value=1,
        max_value=99,
        step=1,
        help="Un porcentaje pequeño crea regiones izquierdas más estrechas; uno grande crea regiones derechas más estrechas.",
        key="x_percent",
    )
with control_col2:
    y_percent = st.slider(
        "Eje horizontal · separación arriba/abajo (%)",
        min_value=1,
        max_value=99,
        step=1,
        help="Un porcentaje pequeño crea una franja superior más baja; uno grande crea una franja inferior más baja.",
        key="y_percent",
    )
with reset_col:
    st.write(" ")
    st.button("Centrar ejes", use_container_width=True, on_click=reset_axes)

x_cut = int(np.clip(round(image_width * x_percent / 100.0), 1, image_width - 1))
y_cut = int(np.clip(round(image_height * y_percent / 100.0), 1, image_height - 1))

regions: list[tuple[str, np.ndarray]] = [
    ("Cuadrícula 1 · Superior izquierda", rgb[:y_cut, :x_cut]),
    ("Cuadrícula 2 · Superior derecha", rgb[:y_cut, x_cut:]),
    ("Cuadrícula 3 · Inferior izquierda", rgb[y_cut:, :x_cut]),
    ("Cuadrícula 4 · Inferior derecha", rgb[y_cut:, x_cut:]),
]

if st.session_state.get("auto_axis_result", {}).get("image_token") == image_token:
    automatic_result = st.session_state["auto_axis_result"]
    with st.expander("Detalle del ajuste automático", expanded=True):
        result_col1, result_col2 = st.columns(2)
        for column, key, title, current_value in [
            (result_col1, "vertical", "Eje vertical", x_percent),
            (result_col2, "horizontal", "Eje horizontal", y_percent),
        ]:
            details = automatic_result[key]
            with column:
                st.markdown(f"**{title}: {current_value} %**")
                st.caption(
                    f"Posiciones de los cuatro picos: "
                    f"{', '.join(f'{value:.1f} %' for value in details['posiciones'])}. "
                    f"Desviación estándar de las posiciones: {details['desviacion_posiciones']:.2f}. "
                    f"Criterio aplicado: {details['metodo']}; valor antes de redondear: "
                    f"{details['valor_resumen']:.2f} %."
                )
        all_peak_rows = (
            automatic_result["vertical"]["picos"]
            + automatic_result["horizontal"]["picos"]
        )
        peak_table = pd.DataFrame(all_peak_rows)
        peak_table["Posición del cambio (%)"] = peak_table["Posición del cambio (%)"].map(
            lambda value: f"{value:.1f}"
        )
        for numeric_col in ["Derivada aproximada", "Valor absoluto de la derivada"]:
            peak_table[numeric_col] = peak_table[numeric_col].map(lambda value: f"{value:.4f}")
        st.dataframe(peak_table, hide_index=True, use_container_width=True)

metrics = pd.DataFrame(
    [
        region_metrics(region, name, image_height, image_width)
        for name, region in regions
    ]
)

preview_col, info_col = st.columns([1.45, 1])
with preview_col:
    st.subheader("Imagen y ejes")
    overlay = add_grid_overlay(rgb, x_cut, y_cut)
    st.image(overlay, use_container_width=True)
    st.caption(
        f"Eje vertical: x = {x_cut} px ({x_percent} %) · "
        f"Eje horizontal: y = {y_cut} px ({y_percent} %). Las líneas rojas marcan los cortes."
    )
with info_col:
    st.subheader("Desviación estándar por región")
    st.markdown("La desviación estándar mide cuánto varían los niveles de intensidad dentro de cada región.")
    for _, row in metrics.iterrows():
        with st.container(border=True):
            st.markdown(f"**{row['Región']}**")
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Desv. luminancia", f"{row['Desv. estándar luminancia']:.2f}")
            metric_col2.metric("Media luminancia", f"{row['Media luminancia']:.2f}")
            st.caption(
                f"Tamaño: {int(row['Ancho (px)'])} × {int(row['Alto (px)'])} px · "
                f"Área: {row['Área (%)']:.1f} % · "
                f"Desv. RGB: {row['Desv. estándar R']:.2f} / "
                f"{row['Desv. estándar G']:.2f} / {row['Desv. estándar B']:.2f}"
            )

st.subheader("Curva de desviación estándar")
st.write(
    "Elige una región y un eje para ver cómo variaría su desviación estándar al mover ese eje. "
    "El otro eje se mantiene fijo en la posición indicada por su deslizador."
)
chart_control_col1, chart_control_col2 = st.columns(2)
with chart_control_col1:
    chart_axis = st.selectbox(
        "Variable del eje horizontal de la gráfica",
        ["Eje vertical", "Eje horizontal"],
        index=0,
    )
with chart_control_col2:
    chart_region_label = st.selectbox(
        "Región que se desea analizar",
        [name for name, _ in regions],
        index=0,
    )
chart_region_number = [name for name, _ in regions].index(chart_region_label) + 1

# The cumulative row/column sums make the 99-point curve much faster to calculate
# than slicing and rescanning the full image for every possible slider position.
luminance = get_luminance(rgb)
curve = deviation_curve(luminance, chart_axis, chart_region_number, x_cut, y_cut)
current_percent = curve.attrs["current_percent"]
axis_label = curve.attrs["axis_label"]
current_deviation = float(
    curve.loc[curve["Separación (%)"] == current_percent, "Desviación estándar"].iloc[0]
)

# Aproximación numérica de la derivada: diferencia entre posiciones consecutivas
# del deslizador, expresada en unidades de desviación estándar por punto porcentual.
curve_sorted = curve.sort_values("Separación (%)").reset_index(drop=True)
slider_positions = curve_sorted["Separación (%)"].to_numpy(dtype=float)
deviation_values = curve_sorted["Desviación estándar"].to_numpy(dtype=float)
derivative_values = np.diff(deviation_values) / np.diff(slider_positions)
interval_starts = slider_positions[:-1]
interval_ends = slider_positions[1:]
interval_midpoints = (interval_starts + interval_ends) / 2.0
midpoint_deviations = (deviation_values[:-1] + deviation_values[1:]) / 2.0

max_positive_index = int(np.argmax(derivative_values))
max_negative_index = int(np.argmin(derivative_values))
steepest_index = int(np.argmax(np.abs(derivative_values)))

def describe_derivative(index: int) -> str:
    return (
        f"{interval_starts[index]:.0f}–{interval_ends[index]:.0f} % "
        f"(pendiente {derivative_values[index]:+.4f} por punto porcentual)"
    )

st.markdown("**Derivada aproximada de la curva**")
derivative_col1, derivative_col2, derivative_col3 = st.columns(3)
with derivative_col1:
    st.metric(
        "Mayor derivada positiva",
        f"{derivative_values[max_positive_index]:+.4f}",
        help="Mayor aumento de la desviación estándar entre dos posiciones consecutivas del deslizador.",
    )
    st.caption(f"Intervalo: {describe_derivative(max_positive_index)}")
with derivative_col2:
    st.metric(
        "Cambio más brusco |derivada|",
        f"{abs(derivative_values[steepest_index]):.4f}",
        help="Mayor pendiente en valor absoluto; detecta tanto aumentos como descensos pronunciados.",
    )
    st.caption(f"Intervalo: {describe_derivative(steepest_index)}")
with derivative_col3:
    st.metric(
        "Mayor derivada negativa",
        f"{derivative_values[max_negative_index]:+.4f}",
        help="Descenso más pronunciado de la desviación estándar.",
    )
    st.caption(f"Intervalo: {describe_derivative(max_negative_index)}")

steepest_point_data = pd.DataFrame({
    "Separación (%)": [interval_midpoints[steepest_index]],
    "Desviación estándar": [midpoint_deviations[steepest_index]],
    "Pendiente aproximada": [derivative_values[steepest_index]],
})
line = alt.Chart(curve).mark_line().encode(
    x=alt.X(
        "Separación (%):Q",
        title=axis_label,
        scale=alt.Scale(domain=[1, 99]),
    ),
    y=alt.Y("Desviación estándar:Q", title="Desviación estándar de luminancia"),
    tooltip=[
        alt.Tooltip("Separación (%):Q", title="Separación (%)", format=".0f"),
        alt.Tooltip("Desviación estándar:Q", title="Desviación estándar", format=".3f"),
    ],
)
current_point = alt.Chart(curve[curve["Posición actual"]]).mark_point(
    color="#dc2626", filled=True, size=90
).encode(
    x="Separación (%):Q",
    y="Desviación estándar:Q",
    tooltip=[
        alt.Tooltip("Separación (%):Q", title="Posición actual (%)", format=".0f"),
        alt.Tooltip("Desviación estándar:Q", title="Desviación actual", format=".3f"),
    ],
)
current_rule = alt.Chart(pd.DataFrame({"Separación (%)": [current_percent]})).mark_rule(
    color="#dc2626", strokeDash=[5, 4], opacity=0.8
).encode(x="Separación (%):Q")
steepest_rule = alt.Chart(steepest_point_data).mark_rule(
    color="#f59e0b", strokeDash=[4, 3], opacity=0.95
).encode(
    x="Separación (%):Q",
    tooltip=[
        alt.Tooltip("Separación (%):Q", title="Centro del intervalo (%)", format=".1f"),
        alt.Tooltip("Pendiente aproximada:Q", title="Derivada aproximada", format="+.4f"),
    ],
)
steepest_point = alt.Chart(steepest_point_data).mark_point(
    color="#f59e0b", filled=True, size=110, shape="diamond"
).encode(
    x="Separación (%):Q",
    y="Desviación estándar:Q",
    tooltip=[
        alt.Tooltip("Separación (%):Q", title="Centro del intervalo (%)", format=".1f"),
        alt.Tooltip("Desviación estándar:Q", title="Desviación aproximada", format=".3f"),
        alt.Tooltip("Pendiente aproximada:Q", title="Derivada aproximada", format="+.4f"),
    ],
)
chart = alt.layer(
    line, current_rule, current_point, steepest_rule, steepest_point
).properties(height=240).interactive()
st.altair_chart(chart, use_container_width=True)
st.caption(
    f"Posición actual: {current_percent} % · Desviación estándar: {current_deviation:.3f}. "
    "El punto rojo es la posición actual; el rombo naranja marca el intervalo con el cambio más brusco."
)
st.info(
    "La derivada se aproxima con diferencias finitas entre posiciones consecutivas del deslizador. "
    "Como la imagen tiene píxeles discretos y los cortes se redondean a píxeles, la curva también es discreta; "
    "por eso el resultado identifica el intervalo de cambio más fuerte, no un punto exacto de una función continua."
)

st.subheader("Tabla comparativa")
columns_to_show = [
    "Región", "Ancho (px)", "Alto (px)", "Píxeles", "Área (%)",
    "Media luminancia", "Desv. estándar luminancia", "Mín. luminancia", "Máx. luminancia",
    "Desv. estándar R", "Desv. estándar G", "Desv. estándar B",
]
formatted = metrics[columns_to_show].copy()
formatted["Área (%)"] = formatted["Área (%)"].map(lambda value: f"{value:.2f}")
for column in [
    "Media luminancia", "Desv. estándar luminancia", "Mín. luminancia", "Máx. luminancia",
    "Desv. estándar R", "Desv. estándar G", "Desv. estándar B",
]:
    formatted[column] = formatted[column].map(lambda value: f"{value:.3f}")
st.dataframe(formatted, hide_index=True, use_container_width=True)

show_histograms = st.checkbox("Mostrar histogramas de luminancia por región", value=False)
if show_histograms:
    histogram_data: dict[str, np.ndarray] = {}
    for name, region in regions:
        region_luminance = np.clip(np.rint(get_luminance(region)), 0, 255).astype(np.uint8)
        histogram_data[name] = np.bincount(region_luminance.ravel(), minlength=256)
    hist_df = pd.DataFrame(histogram_data, index=np.arange(256))
    hist_df.index.name = "Luminancia"
    st.line_chart(hist_df, x_label="Nivel de luminancia (0–255)", y_label="Cantidad de píxeles")

st.subheader("Descargas")
csv_bytes = metrics.to_csv(index=False, float_format="%.6f").encode("utf-8-sig")
image_buffer = BytesIO()
overlay.save(image_buffer, format="PNG")
download_col1, download_col2 = st.columns(2)
with download_col1:
    st.download_button(
        "Descargar métricas (CSV)",
        data=csv_bytes,
        file_name="metricas_cuadricula_2x2.csv",
        mime="text/csv",
        use_container_width=True,
    )
with download_col2:
    st.download_button(
        "Descargar imagen con cuadrícula (PNG)",
        data=image_buffer.getvalue(),
        file_name="imagen_cuadricula_2x2.png",
        mime="image/png",
        use_container_width=True,
    )

with st.expander("Definición de las métricas"):
    st.latex(r"Y = 0.2126R + 0.7152G + 0.0722B")
    st.latex(r"\sigma_Y = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(Y_i-\bar{Y})^2}")
    st.markdown(
        "La desviación estándar se calcula como desviación **poblacional** (divisor `N`), "
        "porque se consideran todos los píxeles de la región analizada. Una desviación mayor "
        "indica más dispersión de intensidades, pero no garantiza por sí sola una mejor calidad visual. "
        "Las desviaciones RGB se calculan por separado para cada canal."
    )
