import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Messwerte
# Temperatur in °C
T_C = np.array([80.9, 60.1, 37.7, 30.3, 22.8, -1.5, -0.7])

# Widerstandswerte in kΩ
R = np.array([1.01, 2.02, 4.62, 7.38, 10.28, 31.0, 30.3])

# Temperatur von °C in K umrechnen
T_K = T_C + 273.15

# x-Werte für Arrhenius-Auftragung: 1/T
x = 1 / T_K

# y-Werte logarithmieren: ln(R)
y = np.log(R)

# Lineare Ausgleichsgerade
m, b = np.polyfit(x, y, 1)

# Achsenabschnitt als t bezeichnen
t = b

# Gleichung der optimalen Gerade für die Legende
if t >= 0:
    gleichung_opt = f"y = {m:.1f}x + {t:.3f}"
else:
    gleichung_opt = f"y = {m:.1f}x - {abs(t):.3f}"

# Wert bei 25 °C berechnen
T_25 = 25 + 273.15
x_25 = 1 / T_25
ln_R_25 = m * x_25 + b
R_25 = np.exp(ln_R_25)

# Wertebereich für Geraden
x_fit = np.linspace(min(x), max(x), 100)
y_fit = m * x_fit + b

# Fehlerstreifen nach AMW-Idee: Trapez
x_links = min(x)
x_rechts = max(x)

# Halbe Breite des Fehlerstreifens links und rechts
w_links = 0.04
w_rechts = 0.10

# Optimale Gerade an den Randpunkten
y_links = m * x_links + b
y_rechts = m * x_rechts + b

# Ober- und Unterkante des Fehlerstreifens an den Rändern
y_links_oben = y_links + w_links
y_links_unten = y_links - w_links

y_rechts_oben = y_rechts + w_rechts
y_rechts_unten = y_rechts - w_rechts

# Fehlerstreifen über den gesamten x-Bereich
w_fit = w_links + (w_rechts - w_links) * (x_fit - x_links) / (x_rechts - x_links)

y_streifen_oben = y_fit + w_fit
y_streifen_unten = y_fit - w_fit

# Grenzgeraden = Diagonalen des Fehlerstreifens
m_diag1 = (y_rechts_unten - y_links_oben) / (x_rechts - x_links)
b_diag1 = y_links_oben - m_diag1 * x_links

m_diag2 = (y_rechts_oben - y_links_unten) / (x_rechts - x_links)
b_diag2 = y_links_unten - m_diag2 * x_links

# Schwerpunkt = Schnittpunkt der Diagonalen
x_s = (b_diag2 - b_diag1) / (m_diag1 - m_diag2)
y_s = m_diag1 * x_s + b_diag1

# kleinere und größere Steigung bestimmen
if m_diag1 < m_diag2:
    m_min = m_diag1
    b_min = b_diag1
    m_max = m_diag2
    b_max = b_diag2
else:
    m_min = m_diag2
    b_min = b_diag2
    m_max = m_diag1
    b_max = b_diag1

# Unsicherheit der Steigung
delta_m = (m_max - m_min) / 2

# y-Werte der Grenzgeraden
y_min = m_min * x_fit + b_min
y_max = m_max * x_fit + b_max

# Kontrolle: Wie viele Punkte im Fehlerstreifen liegen
w_punkte = w_links + (w_rechts - w_links) * (x - x_links) / (x_rechts - x_links)
y_fit_punkte = m * x + b

innerhalb = np.abs(y - y_fit_punkte) <= w_punkte
punkte_innerhalb = np.sum(innerhalb)
punkte_ausserhalb = len(x) - punkte_innerhalb

# Graph erstellen
fig, ax = plt.subplots(figsize=(8, 5))

# Messpunkte
ax.scatter(
    x, y,
    color="black",
    marker="x",
    s=45,
    label="Messwerte"
)

# Fehlerstreifen
ax.fill_between(
    x_fit,
    y_streifen_unten,
    y_streifen_oben,
    color="gray",
    alpha=0.25,
    label="Fehlerstreifen"
)

# Optimale Gerade
ax.plot(
    x_fit,
    y_fit,
    color="red",
    linewidth=1,
    label=f"Optimale Gerade: {gleichung_opt}"
)

# Grenzgeraden
ax.plot(
    x_fit,
    y_min,
    color="blue",
    linestyle="-",
    linewidth=1,
    label=f"Grenzgerade: m_min = {m_min:.1f}"
)

ax.plot(
    x_fit,
    y_max,
    color="green",
    linestyle="-",
    linewidth=1,
    label=f"Grenzgerade: m_max = {m_max:.1f}"
)

# Schwerpunkt einzeichnen
ax.scatter(
    x_s, y_s,
    color="purple",
    marker="o",
    s=25,
    label="Schwerpunkt"
)

ax.annotate(
    "Schwerpunkt",
    xy=(x_s, y_s),
    xytext=(8, 8),
    textcoords="offset points",
    fontsize=9,
    color="purple"
)

# Punkt bei 25 °C einzeichnen
ax.scatter(
    x_25,
    ln_R_25,
    color="green",
    marker="o",
    s=35,
    label="Wert bei 25 °C"
)

ax.annotate(
    "25 °C",
    xy=(x_25, ln_R_25),
    xytext=(8, -12),
    textcoords="offset points",
    fontsize=9,
    color="green"
)

# Beschriftung
ax.set_title("Arrhenius-Auftragung des Widerstandes gegen die Temperatur", fontsize=14)
ax.set_xlabel(r"1/T [1/K]", fontsize=12)
ax.set_ylabel(r"$\ln(R(T)) \ [\ln(\mathrm{k}\Omega)]$", fontsize=12)

# Achsengrenzen
ax.set_xlim(min(x)-0.0000, max(x) + 0.00001)
ax.set_ylim(min(y)-0.1, max(y) + 0.2)

# Skalierung
ax.xaxis.set_major_locator(MultipleLocator(0.0001))
ax.yaxis.set_major_locator(MultipleLocator(0.5))

ax.tick_params(axis="both", labelsize=11)
ax.grid(True, linestyle="", linewidth=0.7, alpha=0.6)
ax.legend(fontsize=9, loc="best", frameon=True)

plt.show()

print("Wert bei 25 °C:")
print(f"T = {T_25:.2f} K")
print(f"1/T = {x_25:.6f} 1/K")
print(f"ln(R(25 °C)) = {ln_R_25:.4f}")
print(f"R(25 °C) = {R_25:.4f} kΩ")
