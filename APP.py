import streamlit as st
import pandas as pd
from io import BytesIO
import altair as alt

# =========================
# Configuración y estilos
# =========================
st.set_page_config(page_title="🚀  Evaluador de Polizas de Seguro :: [Roadvisors](https://roadvisors.com.mx)", layout="wide")

st.markdown("""
    <style>
    h1 { color: #FF6600; text-align: center; }
    h2 { color: #333333; text-align: center; }
    .stButton>button {
        background-color: #FF6600;
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
    }
    .stDownloadButton>button {
        background-color: #003366;
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
    }
    .logo-container img {
        max-height: 120px;
        margin-bottom: 10px;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# Banner corporativo
# =========================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Asegúrate de tener el archivo del logo en la ruta indicada
    st.markdown("<h2>🚀 -> Evaluador de Polizas de Seguros ", unsafe_allow_html=True)
    st.markdown("<h4> Registra los datos de las Pólizas en el panel de la izquierda</h2>", unsafe_allow_html=True)
    st.caption(" Desarrollo [Roadvisors](https://roadvisors.com.mx)", unsafe_allow_html=True)

# =========================
# Conceptos clave
# =========================
st.header("🧾 Conceptos clave de seguros médicos")
st.markdown("""
### 🧾 Deducible
- Es la cantidad fija de dinero que el asegurado debe pagar de su bolsillo antes de que la aseguradora empiece a cubrir los gastos.
- Funciona como una barrera inicial: hasta que no se alcanza ese monto, la aseguradora no interviene.
- **Ejemplo:**
  Si tu póliza tiene un deducible de $10,000 y sufres un accidente con gastos médicos de $50,000, tú pagas los primeros $10,000 y la aseguradora cubre el resto.

### 🤝 Coaseguro
- Es el porcentaje de los gastos cubiertos que el asegurado debe pagar junto con la aseguradora, después de haber cubierto el deducible.
- Representa una participación compartida en el costo del siniestro.
- **Ejemplo:**
  Si tu póliza establece un coaseguro del 20% y los gastos médicos (ya descontando el deducible) son $40,000, tú pagas $8,000 y la aseguradora cubre $32,000.
""")

# =========================
# Entradas en barra lateral
# =========================
num_policies = st.sidebar.number_input("Número de pólizas a comparar", min_value=1, max_value=5, value=2)

policies = []
for i in range(num_policies):
    st.sidebar.subheader(f"Poliza{i+1}")
    nombre = f"Poliza{i+1}"
    cobertura = st.sidebar.number_input(f"Cobertura {nombre} (MXN o UMAM)", value=85000000.0, step=100000.0, key=f"cobertura{i}")
    deducible = st.sidebar.number_input(f"Deducible {nombre} (MXN o UMAM)", value=34395.0, step=100.0, key=f"deducible{i}")
    coaseguro = st.sidebar.number_input(f"Coaseguro {nombre} (%)", value=10.0, step=1.0, key=f"coaseguro{i}")
    tope = st.sidebar.number_input(f"Tope de coaseguro {nombre} (MXN o UMAM)", value=50000.0, step=1000.0, key=f"tope{i}")
    policies.append({"nombre": nombre, "cobertura": cobertura, "deducible": deducible, "coaseguro": coaseguro, "tope": tope})

# =========================
# Escenarios base
# =========================
escenarios = list(range(50000, 5000001, 50000)) + [50000000, 100000000, 200000000]

# =========================
# Cálculo (controlado con botón)
# =========================
if st.button("Calcular"):
    comparativas = {}
    for pol in policies:
        resultados = []
        for gasto in escenarios:
            pago_coaseguro = max(0, (gasto - pol["deducible"]) * (pol["coaseguro"] / 100))
            pago_coaseguro = min(pago_coaseguro, pol["tope"])
            pago_asegurado = min(gasto, pol["deducible"] + pago_coaseguro)
            pago_aseguradora = gasto - pago_asegurado
            resultados.append([gasto, pago_asegurado, pago_aseguradora])

        df = pd.DataFrame(
            resultados,
            columns=["Gasto Total", f"{pol['nombre']} - Pago Asegurado", f"{pol['nombre']} - Pago Aseguradora"]
        )
        comparativas[pol["nombre"]] = df

    # Unir resultados en un solo DataFrame
    df_final = comparativas[policies[0]["nombre"]]
    for pol in policies[1:]:
        df_final = df_final.merge(comparativas[pol["nombre"]], on="Gasto Total")

    # Tipos numéricos seguros
    df_final["Gasto Total"] = pd.to_numeric(df_final["Gasto Total"], errors="coerce")
    for col in df_final.columns:
        if col != "Gasto Total":
            df_final[col] = pd.to_numeric(df_final[col], errors="coerce")

    # Persistir resultados
    st.session_state["df_final"] = df_final

# =========================
# Visualización si hay resultados
# =========================
if "df_final" in st.session_state:
    df_final = st.session_state["df_final"]

    # Slider de rango que aplica a tabla y gráfica
    min_val = int(df_final["Gasto Total"].min())
    max_val = int(df_final["Gasto Total"].max())
    rango = st.slider(
        "Selecciona rango de gastos médicos a visualizar",
        min_val, max_val, (min_val, min(1000000, max_val)), step=50000
    )

    # Filtrado por rango
    df_filtrado = df_final[(df_final["Gasto Total"] >= rango[0]) & (df_final["Gasto Total"] <= rango[1])].copy()

    # Tabla filtrada
    st.header("📋 Comparativa de escenarios")
    st.dataframe(df_filtrado)

    # Comentario interpretativo
    st.header("📝 Comentario")
    st.markdown("""
    - En **gastos pequeños**, el deducible es el factor más importante.
    - En **gastos medianos**, el coaseguro empieza a pesar más.
    - En **gastos altos**, el **tope de coaseguro** define la conveniencia de cada póliza.
    """)

    # Combo para elegir qué graficar (coincidencia exacta del sufijo)
    opcion = st.selectbox("Selecciona qué graficar", ["Pago Asegurado", "Pago Aseguradora"])
    sufijo = " - Pago Asegurado" if opcion == "Pago Asegurado" else " - Pago Aseguradora"

    # Preparar datos para líneas (una serie por póliza)
    cols_grafica = [col for col in df_filtrado.columns if col.endswith(sufijo)]
    long_df = df_filtrado.melt(
        id_vars=["Gasto Total"],
        value_vars=cols_grafica,
        var_name="Serie",
        value_name="Monto"
    )
    long_df["Poliza"] = long_df["Serie"].str.split(" - ").str[0]

    # Paleta fija de colores por póliza
    palette = {
        "Poliza1": "#FF6600",  # naranja corporativo
        "Poliza2": "#003366",  # azul profundo
        "Poliza3": "#339933",  # verde
        "Poliza4": "#CC0000",  # rojo
        "Poliza5": "#990099"   # morado
    }

    # Gráfica de líneas con Altair y colores fijos
    st.header("📈 Visualización comparativa (líneas)")
    line_chart = alt.Chart(long_df).mark_line(point=True).encode(
        x=alt.X("Gasto Total:Q", title="Gasto Total"),
        y=alt.Y("Monto:Q", title=opcion),
        color=alt.Color("Poliza:N", title="Póliza",
                        scale=alt.Scale(domain=list(palette.keys()), range=list(palette.values()))),
        tooltip=["Gasto Total", "Poliza", "Monto"]
    ).properties(height=400)

    st.altair_chart(line_chart, use_container_width=True)

    # =========================
    # Exportar a Excel (con logo en el reporte)
    # =========================
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        # Escribimos la tabla empezando en la fila 7 para dejar espacio al encabezado y logo
        df_final.to_excel(writer, sheet_name="Comparativa", index=False, startrow=6)
        workbook  = writer.book
        worksheet = writer.sheets["Comparativa"]

        # Formatos
        title_format = workbook.add_format({
            "bold": True, "font_color": "#FF6600", "font_size": 16
        })
        subtitle_format = workbook.add_format({
            "bold": True, "font_color": "#333333", "font_size": 12
        })
        note_format = workbook.add_format({
            "font_color": "#333333", "font_size": 10
        })

        # Encabezado
        worksheet.write(0, 3, "Roadvisors", title_format)  # D1
        worksheet.write(1, 3, "Consultoría técnica y de Soporte a PYMES", subtitle_format)

        # Opcional: ancho de columnas
        worksheet.set_column(0, 0, 16)  # Columna A (Gasto Total)
        worksheet.set_column(1, len(df_final.columns)-1, 20)

        # Insertar logo (usa tu archivo local)
        # Ajusta 'logo_roadvisors.png' a la ruta real si está en otra carpeta
        try:
            worksheet.insert_image("A1", "logo_roadvisors.JPG", {
                "x_scale": 0.5,  # escala horizontal
                "y_scale": 0.5   # escala vertical
            })
        except Exception as e:
            # Nota si el logo no se pudo insertar
            worksheet.write(3, 0, f"Nota: No se pudo insertar el logo ({e}).", note_format)

    st.download_button(
        label="📥 Descargar en Excel (con logo)",
        data=buffer.getvalue(),
        file_name="comparativa_polizas.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("Configura las pólizas y presiona 'Calcular' para generar la comparativa.")
