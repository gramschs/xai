---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: python311
  language: python
  name: python3
---

# Erklärbarkeit von KI-Modellen

Mit Künstlicher Intelligenz (KI) beschäftigen sich Informatikerinnen und
Informatiker bereits seit 1955. Die Mehrheit der Bevölkerung in Deutschland hat
jedoch erst mit der Veröffentlichung von ChatGPT im November 2022 erstmals
bewusst ein KI-System genutzt. Seitdem findet eine überfällige Diskussion statt,
wie wir als Gesellschaft zukünftig mit KI-Anwendungen umgehen sollen. Eine
besondere Herausforderung in diesem Kontext ist die **Erklärbarkeit von
KI-Modellen**, also die Nachvollziehbarkeit, wie KI-Systeme ihre Prognosen und
Entscheidungen treffen. Die Herausforderung besteht besonders bei komplexen
Modellen wie tiefen neuronalen Netzen, deren Struktur und Entscheidungslogik für
Menschen oft schwer nachvollziehbar sind.

## Lernziele

```{admonition} Lernziele
:class: goals
* Sie verstehen, warum **Erklärbarkeit von KI-Modellen** gesellschaftlich
  relevant ist.
* Sie kennen die Begriffe **White-Box-Modell** und **Black-Box-Modell**.
* Sie können die Funktionsweise des Erklärungsmodells **LIME** (Local
  Interpretable Model-Agnostic Explanations) beschreiben und an einem einfachen
  Beispiel anwenden.
* Sie können Erklärungswerkzeuge unterscheiden nach
  1. **Komplexität** (intrinsische Modelle oder Post-hoc-Methoden),
  2. **Umfang** (global oder lokal) und
  3. **Modellabhängigkeit** (modellspezifisch oder modellagnostisch).
```

## Warum brauchen wir erklärbare KI-Modelle?

Neuronale Netze, insbesondere mehrschichtige neuronale Netze im Deep Learning,
haben in den letzten Jahren zu einem enormen Anstieg bei der Anwendung von
KI-Systemen geführt. Laut einer [Studie von
NextMSC](https://www.nextmsc.com/report/artificial-intelligence-market) wird
sich der weltweite Umsatz von KI (einschließlich Anwendungen, Infrastruktur
sowie IT- und Unternehmensdienstleistungen) bis zum Jahr 2030 verzwanzigfachen.
Dennoch zögern deutsche Unternehmen noch, KI-Systeme umfassend einzusetzen, wie
der Bericht [Statista → KI Perspektive der deutschen
Wirtschaft](https://de.statista.com/themen/9400/ki-in-der-deutschen-wirtschaft/#topicOverview)
zeigt. Viele Unternehmen zweifeln an der Qualität der verfügbaren Daten für das
Training der KI-Systeme und stufen die mangelnde Transparenz der
KI-Entscheidungen als Risiko ein. Ohne Vertrauen in die Vorhersagefähigkeiten
der KI bleiben sie daher vorsichtig bei der Einführung solcher Systeme.

Selbst für Fachleute bleiben manche KI-Modelle wie beispielsweise die tiefen
neuronalen Netze intransparent. Hier setzen erklärbare KI-Modelle an: Sie sollen
das Vertrauen in die Entscheidungen der Systeme stärken und nachvollziehbare
Einblicke in die Entscheidungslogik der KI ermöglichen.

## Anwendungsbeispiel: das Schuheinlagen-Orakel

Anhand eines fiktiven Anwendungsbeispiels aus der Produktion erarbeiten wir uns
das Thema Erklärbarkeit von KI-Modellen. Angenommen, eine Firma stellt
personalisierte Schuheinlagen her. Dabei sollen die Einlagen sowohl an die
Geometrie des Fußes angepasst werden als auch an die Druckbelastungen des Fußes
beim Gehen. Sowohl Geometrie als auch Druckbelastungen werden digital erfasst,
und basierend auf den Messungen wird die Schuheinlage in einzelne
Belastungszonen eingeteilt und in 3D gedruckt. Zwar ist die Firma in diesem
Beispiel fiktiv, ein Prototyp dieses Prozesses existiert jedoch bereits (siehe
{cite}`voelz:2023`) und könnte so tatsächlich in naher Zukunft von einem
Start-Up umgesetzt werden.

Beim 3D-Druck gibt es Zielkonflikte. Einerseits soll möglichst wenig Material
eingesetzt werden, andererseits muss die gedruckte Struktur dennoch die
Belastung aushalten, die für diese Belastungszone vorgesehen ist. Zur
Unterstützung der Entwicklung der personalisierten Schuheinlage gibt es ein
KI-Modell, das die maximale Kraft (in Newton) prognostiziert, mit der eine
3D-gedruckte Gitterstruktur belastet werden kann. Aus diesen 3D-Bauteilen werden
dann die Schuheinlagen zusammengesetzt.

Spaßeshalber nennen wir dieses KI-Modell **Schuheinlagen-Orakel**, denn leider
liegt es nur binär vor. Daher müssen wir auch das Modul `dill` benutzen, um es
zu laden.

```{code-cell}
import dill
with open('schuheinlagen_orakel.dill', 'rb') as f:
    ki_modell = dill.load(f)
```

Als nächstes benutzen wir die eingebaute Hilfe des KI-Systems, um mehr über das
Schuheinlagen-Orakel zu erfahren.

```{code-cell}
help(ki_modell)
```

Die Hilfe gibt Auskunft darüber, wie das Schuheinlagen-Orakel verwendet werden
kann. Das KI-Modell liefert Prognosen zur maximalen Kraft, die ein Bauteil
aushalten kann, basierend auf den Eingabemerkmalen Zellenform, Zellengröße und
Füllgrad. Laut Dokumentation ist die Zellenform entweder `1` für eine X-Zelle
oder `2` für einen Gyroiden. Was sich hinter diesen Fachbegriffen verbirgt,
können nur die Ingenieure beantworten, die diesen Produktionsprozess entwickelt
haben (siehe Abbildung {ref}`zellenform`).

```{figure} pics/zellenform.png
:alt: Zellenform der Bauteile
:align: center
:name: zellenform
Zellenform der Bauteile: links X-Zelle (codiert als 1) und rechts Gyroid (codiert als 2); 
Quelle: Tim Schwitzner {cite}`schwitzner:2024`.
```

Darüber hinaus muss die Zellengröße zwischen $\pu{2 mm}$ und $\pu{10 mm}$
liegen. Der Füllgrad liegt zwischen $\pu{20 \%}$ und $\pu{45 \%}$, wobei dieser
Wert als Zahl im Intervall $[0,1]$ angegeben wird.

Um mit dem KI-Modell vertraut zu werden, lassen wir eine Prognose erstellen. Wir
nutzen das Modul Pandas zur Verwaltung der Daten und importieren es daher in
einem ersten Schritt mit seiner üblichen Abkürzung `pd`. Als nächstes definieren
wir ein Bauteil. Dazu verwenden wir einen Pandas-DataFrame als Datenstruktur,
der durch ein Dictionary initialisiert wird. Die Schlüssel des Dictionaries sind
die Merkmale `'Zellenform'`, `'Zellengroesse'` und `'Fuellgrad'`. Die Werte sind
Listen mit den entsprechenden Eigenschaften der 3D-gedruckten Bauteile. Auch
wenn nur ein Bauteil betrachtet wird, ist eine Liste aufgrund der Syntax
erforderlich. Zuletzt lassen wir uns die initialisierte Datenstruktur mit der
Pandas-Methode `.head()` anzeigen.

```{code-cell}
import pandas as pd 

bauteil = pd.DataFrame({
    'Zellenform': [1],
    'Zellengroesse': [3.0],
    'Fuellgrad': [0.3]
})

bauteil.head()
```

Nun können wir die `predict()`-Methode nutzen, um die maximale Kraft
prognostizieren zu lassen. Wir speichern das Ergebnis in der Variablen
`maximale_kraft`.

```{code-cell}
maximale_kraft = ki_modell.predict(bauteil)
print(maximale_kraft)
```

Das Bauteil hält eine maximale Kraft von $\pu{42.3 N}$ aus.

Obwohl das Schuheinlagen-Orakel eine Prognose liefert, bleibt das System für uns
eine **Black Box**. Die innere Struktur des KI-Modells ist nicht transparent;
wir wissen nicht, *wie* das KI-Modell zu seiner Prognose gekommen ist.

Ist ein solches Szenario realistisch? Tatsächlich kommen solche Szenarien
häufiger vor als erwünscht. Verlassen beispielsweise Wissensträger ein
Unternehmen, ist es oft nicht möglich, ihre Expertise ausreichend für die
nachfolgenden Mitarbeiterinnen und Mitarbeiter zu dokumentieren. Aber auch die
zunehmend *komplexeren KI-Modelle* sind für uns Menschen undurchsichtig. Auch
können die Eingabedaten für ein KI-Modell so stark weiterverarbeitet worden
sein, dass auch die Eingabedaten selbst nicht mehr für uns Menschen
nachvollziehbar sind. Ist ein KI-Modell undurchsichtig oder sind die
Eingangsdaten nicht nachvollziehbar, wird das KI-Modell als Black-Box-Modell
bezeichnet.

```{admonition} Was ist ... ein Black-Box-Modell?
:class: note
Ein KI-Modell wird als Black-Box-Modell bezeichnet, wenn es keine transparente
Entscheidungslogik oder nachvollziehbare Eingangsdaten besitzt.
```

Das Gegenteil eines Black-Box-Modells ist das sogenannte **White-Box-Modell**.
Es ist im Kontext der erklärbaren KI folgendermaßen definiert.

```{admonition} Was ist ... ein White-Box-Modell?
:class: note
Ein KI-Modell wird als White-Box-Modell bezeichnet, wenn seine
Entscheidungslogik transparent ist und es nachvollziehbare Eingangsdaten
besitzt.
```

## Das LIME-Modell

Kehren wir zurück zu dem fiktiven Anwendungsbeispiel und stellen uns vor, die
Entwickler des Schuheinlagen-Orakels haben das Unternehmen verlassen. Die
Konstruktionsabteilung möchte nun verstehen, warum für ein 3D-gedrucktes Bauteil
mit der Zellenform X-Zelle, einer Zellengröße von $\pu{3 mm}$ und einem Füllgrad
von $\pu{30 %}$ eine maximale Kraft von $\pu{42.3 N}$ prognostiziert wird. Um
ein KI-Modell erklärbar zu machen, gibt es verschiedene Ansätze. Ein häufig
verwendetes Verfahren ist **LIME**. LIME ist ein Akronym und steht für

**L**ocal **I**nterpretable **M**odel-agnostic **E**xplanations.

*Lokal* bedeutet, dass eine Erklärung für ein einzelnes Beispiel gesucht wird.
Gleichzeitig soll das LIME-Modell *interpretierbar* sein, also ein
White-Box-Modell darstellen. *Modellagnostisch* bedeutet, dass die LIME-Methode
unabhängig von der Struktur des zugrunde liegenden KI-Modells funktioniert und
für verschiedene KI-Modelle eingesetzt werden kann.

```{admonition} Wie funktioniert die LIME-Methode?
:class: notes
1. *Variation der Daten*: Für ein ausgewähltes Beispiel (hier das Bauteil)
   erzeugen wir abgewandelte Varianten der Eingabedaten mit kleinen Änderungen
   im Vergleich zum Referenzbeispiel.
2. *Berechnung der Prognosen*: Für jede dieser leicht abgeänderten Eingabedaten
   berechnen wir mit dem ursprünglichen KI-Modell eine Prognose.
3. *Gewichtung der Eingabedaten*: Die abgeänderten Eingabedaten werden gewichtet.
   Je ähnlicher eine Datenpunkt zum ausgewählten Beispiel ist, desto höher ist das
   Gewicht.
4. *Training eines Ersatzmodells*: Wir trainieren ein einfaches, gut
   interpretierbares Ersatzmodell (z.B. ein linares Regressionsmodell) auf den
   gewichteten, leicht abgeänderten Eingabedaten. Die Prognosen des
   ursprünglichen Modells sind dabei die Ausgabedaten.
5. *Erklärung der Prognose*: Da das Ersatzmodell aus Schritt 4 ein
   White-Box-Modell ist, können wir es nun benutzen, um zu erklären, welche
   Merkmale besonders die Prognose beeinflussen.
```

Obwohl für das LIME-Modell, das erstmals 2016 vorgestellt wurde
{cite}`ribeiro:2016`, ein Python-Modul namens
[lime](https://github.com/marcotcr/lime) existiert, erstellen wir das
LIME-Modell hier Schritt für Schritt von Grund auf, um die Funktionsweise besser
zu verstehen.

### Schritt 1: Variation der Daten

Als erstes ändern wir die Merkmale des ausgewählten Beispiels leicht ab. Da es
nur zwei mögliche Zellenformen gibt, können wir diese nicht "leicht" variieren.
Daher belassen wir es bei der Zellenform `X-Zelle` und generieren eine Liste mit
`N` Einsen. Etwas einfacher wird es, wenn wir dazu die Funktion `np.ones()` des
Moduls `NumPy` nutzen, das wir mit der üblichen Abkürzung `np` importieren.

```{code-cell}
import numpy as np

N = 100
variation_zellenform = np.ones(N)
```

Anschließend möchten wir die Zellengröße und den Füllgrad leicht variieren.
Dafür verwenden wir Zufallszahlen. Aus didaktischen Gründen fixieren wir den
Seed der Zufallszahlen auf 42 mit `np.random.seed(42)`. Dann ziehen wir mit der
Funktion `np.random.normal()` normalverteilte Zufallszahlen (mit Mittelwert 0
und Standardabweichung 0.1) und addieren diese Zufallszahlen zur Zellengröße 3
und zum Füllgrad 0.3:

```{code-cell}
np.random.seed(42) 

variation_zellengroesse = 3.0 + np.random.normal(0, 0.1, N)
variation_fuellgrad = 0.3 + np.random.normal(0, 0.1, N)
```

Auf diese Weise erhalten wir die variierten Eingabedaten, die wir anschließend
in einem Pandas-DataFrame zusammenfassen:

```{code-cell}
eingabedaten = pd.DataFrame({
    'Zellenform': variation_zellenform,
    'Zellengroesse': variation_zellengroesse,
    'Fuellgrad': variation_fuellgrad
}) 
```

### Schritt 2: Berechnung der Prognosen

Die Prognosen des ursprünglichen KI-Modells lassen sich einfach mit der
`predict()`-Methode berechnen.

```{code-cell}
ausgabedaten = ki_modell.predict(eingabedaten)
```

### Schritt 3: Gewichtung der Eingabedaten

Für das LIME-Verfahren ist es wichtig, dass die variierte Eingabedaten gewichtet
werden. Je ähnlicher ein Datenpunkt zum ausgewählten Beispiel ist, desto mehr
Gewicht soll dieser Datenpunkt beim Training des Ersatzmodells sein. In diesem
Beispiel verwenden wir den euklidischen Abstand, um die Ähnlichkeit zu
berechnen.

```{code-cell}
abstaende = ((eingabedaten['Zellengroesse'] - 3.0)**2 + (eingabedaten['Fuellgrad'] - 0.3)**2)**0.5
```

Das Gewicht soll 1 sein, wenn der Abstand zum ausgewählten Beispiel 0 ist. Ein
schneller Check der gestörten Eingabedaten zeigt, dass der maximale Abstand der
Eingabedaten zum Referenzbeispiel kleiner als 0.5 ist. Für einen Abstand von 0.5
fordern wir ein Gewicht von 0. Dazwischen sollen die Gewichte linear abfallen.
Der folgende Code erfüllt diese Forderungen:

```{code-cell}
gewichte = -2 * abstaende + 1
```

### Schritt 4: Training eines Ersatzmodells

Häufig wird ein lineares Regressionsmodell oder ein Entscheidungsbaum verwendet,
um ein lokal interpretierbares Modell zu erzeugen, dass das ursprüngliche
KI-Modell erklärt. Wir werfen mit Hilfe eines Streudiagramms (Scatterplot) einen
kurzen Blick auf die variierten Eingabedaten und die vom KI-Modell
prognostizierten Ausgabedaten (maximale Kräfte). Wir importieren das Modul
`Plotly Express` als `px`. Auf der x-Achse tragen wir die Zellengrößen ein und
auf der y-Achse die Füllgrade der variierten Bauteile. Durch die Farbe
kennzeichen wir die prognostizierten maximalen Kräfte.

```{code-cell}
import plotly.express as px 

fig = px.scatter( eingabedaten, x='Zellengroesse', y='Fuellgrad', color=ausgabedaten,
    title='Gestörte Daten um ausgewähltes Beispiel (3, 0.3)', 
    labels={'color': 'maximale Kraft [N]'}
)
fig.update_xaxes(range=[2.7, 3.3])
fig.update_yaxes(range=[0.0, 0.6])

fig.show()
```

Scheinbar ist vor allem der Füllgrad entscheidend für die Prognose der maximalen
Kraft. Wir entscheiden uns für ein lineares Regressionsmodell, das wir zunächst
aus dem Modul Scikit-Learn importieren. Dann trainieren wir das lineare
Regressionsmodell mit den Eingabe- und Ausgabedaten (wir verzichten auf eine
Skalierung der Daten, da diese in derselben Größenordnung liegen und lassen auch
den üblichen Split in Trainings- und Testdaten weg). Dafür geben wir über das
optionale Argument `sample_weights` noch die Gewichte an.

```{code-cell}
from sklearn.linear_model import LinearRegression

modell = LinearRegression()
modell.fit(eingabedaten, ausgabedaten, sample_weight=gewichte)
```

Mithilfe der `score()`-Methode lassen wir die Qualität des Modells bestimmen:

```{code-cell}
score = modell.score(eingabedaten, ausgabedaten)
print(score)
```

Ein Score von 1 wäre perfekt, 0.87 ist sehr gut. Wir haben daher ein
interpretierbares Ersatzmodell gefunden, das hilft, das ausgewählte Beispiel zu
interpretieren.

### Schritt 5: Erklärung der Prognose

Lineare Regressionsmodelle sind leicht verständlich und gut interpretierbar.
Wenn $x_0$ für die Zellenform steht, $x_1$ für die Zellengröße und $x_2$ für den
Füllgrad, dann wird die maximale Kraft $y$ bei einem linearen Regressionsmodell
als

$$y = w_0\cdot x_0 + w_1\cdot x_1 + w_2\cdot x_2 + b$$

berechnet. $w_0, w_1$ und $w_2$ sind die Koeffizienten des linearen Modells und
$b$ die Steigung. Wir lassen uns nun die Koeffizienten des linearen
Regressionsmodells ausgeben, das als Ersatzmodell trainiert wurde. Diese werden
von Scikit-Learn im trainierten Modell im Attribut `coef_` gespeichert.

```{code-cell}
print(modell.coef_)
```

Die Zellenform spielt keine Rolle, der Koeffizient ist 0. Die Zellengröße wird
mit ungefähr 3.8 gewichtet, der Füllgrad mit ungefähr 474.3. Der Koeffizient für
den Füllgrad ist um eingies größer als der Koeffizient für die Zellengröße. Dies
ist auch dann noch der Fall, wenn wir berücksichtigen, dass die Werte der
Zellengröße 10-mal so groß sind wie der Füllgrad (wir haben die Daten nicht
skaliert).

Dazu kommt noch die Steigung, die im Attribut `intercept_` gespeichert ist.

```{code-cell}
print(modell.intercept_)
```

Insgesamt lautet das lineare Regressionsmodell also

$$y = 3.8\cdot x_1 + 474.3\cdot x_2 - 97.4,$$

wobei $x_1$ die Zellengröße in Millimetern und $x_2$ der Füllgrad ist.

## Kategorien der erklärbaren KI-Modelle

TODO

* Komplexität (intrinsische Modelle oder Post-hoc-Methoden),
* Umfang (global oder lokal) und
* Modellabhängigkeit (modellspezifisch oder modellagnostisch)

## Zusammenfassung und Ausblick

TODO
