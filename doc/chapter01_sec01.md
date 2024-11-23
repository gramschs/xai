---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
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

```{code-cell} ipython3
import dill
with open('schuheinlagen_orakel.dill', 'rb') as f:
    ki_modell = dill.load(f)
```

Als nächstes benutzen wir die eingebaute Hilfe des KI-Systems, um mehr über das
Schuheinlagen-Orakel zu erfahren.

```{code-cell} ipython3
help(ki_modell)
```

Die Hilfe gibt Auskunft darüber, wie das Schuheinlagen-Orakel verwendet werden
kann. Das KI-Modell liefert Prognosen zur maximalen Kraft, die ein Bauteil
aushalten kann, basierend auf den Eingabemerkmalen Zellenform, Zellengröße und
Füllgrad. Die Messung der maximalen Kraft wird dabei durch die Be- und
Entlastung des Bauteils bestimmt.

```{figure} ./pics/messung.mp4
:alt: Video der Be- und Entlastungsmessung
:align: center
:name: messung
:width: 100%
Video der Be- und Entlastungsmessung eines Bauteils
(Quelle: Tim Schwitzner {cite}`schwitzner:2024`).
```

Laut Dokumentation ist die Zellenform entweder `1` für eine X-Zelle
oder `2` für einen Gyroiden. Was sich hinter diesen Fachbegriffen verbirgt,
können nur die Ingenieure beantworten, die diesen Produktionsprozess entwickelt
haben (siehe Abbildung 1).

```{figure} pics/zellenform.png
:alt: Zellenform der Bauteile
:align: center
:name: zellenform
Zellenform der Bauteile: links X-Zelle codiert als 1 und rechts Gyroid codiert als 2  
(Quelle: Tim Schwitzner {cite}`schwitzner:2024`).
```

Darüber hinaus muss die Zellengröße zwischen $\pu{2 mm}$ und $\pu{10 mm}$
liegen. Der Füllgrad liegt zwischen $\pu{20 \%}$ und $\pu{45 \%}$, wobei dieser
Prozentwert als Fließkommazahl (Float) im Intervall $[0.2,0.45]$ angegeben wird.

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

```{code-cell} ipython3
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

```{code-cell} ipython3
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
Entscheidungslogik besitzt oder die Eingangsdaten nicht nachvollziehbar sind.
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
von $\pu{30 \%}$ eine maximale Kraft von $\pu{42.3 N}$ prognostiziert wird. Um
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
1. *Variation der Daten*: Für ein ausgewähltes Beispiel, die sogenannte
   Referenz, erzeugen wir abgewandelte Varianten der Eingabedaten mit kleinen
   Änderungen im Vergleich zum Referenzbeispiel.
2. *Berechnung der Prognosen*: Für jede dieser leicht abgeänderten Eingabedaten
   berechnen wir mit dem ursprünglichen KI-Modell eine Prognose.
3. *Gewichtung der Eingabedaten*: Die abgeänderten Eingabedaten werden
   gewichtet. Je ähnlicher eine Datenpunkt zur Referenz ist, desto höher ist das
   Gewicht.
4. *Training eines Ersatzmodells*: Wir trainieren ein einfaches, gut
   interpretierbares Ersatzmodell (z.B. ein lineares Regressionsmodell oder einen
   Entscheidungsbaum) auf den gewichteten, leicht abgeänderten Eingabedaten. Die
   Prognosen des ursprünglichen Modells sind dabei die Ausgabedaten.
5. *Erklärung der Prognose*: Da das Ersatzmodell aus Schritt 4 ein
   White-Box-Modell ist, können wir es nun benutzen, um das Black-Box-KI-Modell
   lokal zu erklären.
```

Obwohl für das LIME-Modell, das erstmals 2016 vorgestellt wurde
{cite}`ribeiro:2016`, ein Python-Modul namens
[lime](https://github.com/marcotcr/lime) existiert, erstellen wir das
LIME-Modell hier Schritt für Schritt von Grund auf, um die Funktionsweise besser
zu verstehen.

### Schritt 1: Variation der Daten

Als erstes ändern wir die Merkmale des ausgewählten Referenzbeispiels leicht ab.
Da es nur zwei mögliche Zellenformen gibt, können wir diese nicht "leicht"
variieren. Daher belassen wir es bei der Zellenform `X-Zelle` und generieren
eine Liste mit `N` Einsen. Etwas einfacher wird es, wenn wir dazu die Funktion
`np.ones()` des Moduls `NumPy` nutzen, das wir mit der üblichen Abkürzung `np`
importieren.

```{code-cell} ipython3
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

```{code-cell} ipython3
np.random.seed(42) 

variation_zellengroesse = 3.0 + np.random.normal(0, 0.5, N)
variation_fuellgrad = 0.3 + np.random.normal(0, 0.05, N)
```

Wir überprüfen visuell mit einem Streudiagramm (Scatterplot), ob die variierten
Eingabedaten im zulässigen Bereich liegen. Die Zellengröße muss ja zwischen 2
und 10 mm liegen, der Füllgrad im Intervall [0.2,0.45]. Dazu importieren wir das
Modul `Plotly Express` als `px`. Der Scatterplot wird durch die Funktion
`scatter()` erzeugt. Auf der x-Achse tragen wir mit `x=variation_zellengroesse`
die Zellengrößen ein und auf der y-Achse mit `y=variation_fuellgrad` die
Füllgrade der variierten Bauteile. Zusätzlich setzen wir mit dem optionalen
Argument `title=` noch einen Titel. Zuletzt ergänzen wir noch das
Referenzbeispiel durch einen zweiten Scatterplot mit `add_scatter()`. Insgesamt
fixieren wir den Ausschnitt für die x-Achse auf [1.5, 4.5] und für die y-Achse
auf [0.15, 0.45] mit `fig.update_layout(xaxis_range=[1.5, 4.5],
yaxis_range=[0.15,0.45])`, damit das Referenzbeispiel im Zentrum des Diagramms
liegt.

```{code-cell} ipython3
import plotly.express as px 

fig = px.scatter(x=variation_zellengroesse, y=variation_fuellgrad,
    title='Variierte Eingabedaten um ausgewähltes Referenzbeispiel (3, 0.3)'
)
fig.add_scatter(x=[3.0], y=[0.3], name='Referenz')
fig.update_layout(xaxis_range=[1.5, 4.5], yaxis_range=[0.15,0.45])

fig.show()
```

Auf diese Weise erhalten wir die variierten Eingabedaten, die wir anschließend
in einem Pandas-DataFrame zusammenfassen:

```{code-cell} ipython3
eingabedaten = pd.DataFrame({
    'Zellenform': variation_zellenform,
    'Zellengroesse': variation_zellengroesse,
    'Fuellgrad': variation_fuellgrad
}) 
```

### Schritt 2: Berechnung der Prognosen

Die Prognosen des ursprünglichen KI-Modells lassen sich einfach mit der
`predict()`-Methode berechnen.

```{code-cell} ipython3
ausgabedaten = ki_modell.predict(eingabedaten)
```

### Schritt 3: Gewichtung der Eingabedaten

Für das LIME-Verfahren ist es wichtig, dass die variierte Eingabedaten gewichtet
werden. Je ähnlicher ein Datenpunkt zum ausgewählten Beispiel ist, desto mehr
Gewicht soll dieser Datenpunkt beim Training des Ersatzmodells haben. In diesem
Beispiel verwenden wir den euklidischen Abstand, um die Ähnlichkeit der
Eingabedaten zur Referenz zu berechnen.

```{figure} pics/variierte_eingabedaten_annotated.svg
:alt: Euklidischer Abstand zur Referenz
:align: center
:name: variierte_eingabedaten_annotate
Der euklidische Abstand $r$ zur Referenz kann mit dem Satz des Pythagoras 
als $r=\sqrt{(\Delta x)^2 + (\Delta y)^2}$ berechnet werden.
(Quelle: eigene Darstellung)
```

Damit ergibt sich der folgende Python-Code zur Berechnung der Abstände.

```{code-cell} ipython3
abstaende = ((eingabedaten['Zellengroesse'] - 3.0)**2 + (eingabedaten['Fuellgrad'] - 0.3)**2)**0.5
```

Wie die Gewichte nun basierend auf der Ähnlichkeit der Eingabedaten zur Referenz
gewählt werden, wird in der Forschung intensiv diskutiert. Häufig werden
exponentielle Gewichte gewählt. Wir nehmen hier Gewichte, die linear vom Abstand
der Eingabedaten zur Referenz abhängen. Das Gewicht soll 1 sein, wenn der
Abstand zum ausgewählten Beispiel 0 ist. Ein schneller Check der gestörten
Eingabedaten zeigt, dass der maximale Abstand der Eingabedaten zum
Referenzbeispiel kleiner als 1.5 ist. Für einen Abstand von 1.5 fordern wir ein
Gewicht von 0. Dazwischen sollen die Gewichte linear abfallen.

```{figure} pics/gewichtsfunktionen.svg
:alt: Gewichtsfunktionen
:align: center
:name: gewichtsfunktionen

Mögliche Gewichtsfunktionen: links eine lineare Gewichtsfunktion, die so
parametriert wurde, dass ein Abstand $r=0$ zu einem Gewicht von Eins führt und
ab $r=1.5$ Null ist. Die rechte exponentielle Gewichtsfunktion ist ähnlich zur
linearen Gewichtsfunktion, bietet aber den zusätzlichen Vorteil, differenzierbar
zu sein.
(Quelle: eigene Darstellung)
```

Mit dem folgenden Code implementieren wir die lineare Gewichtsfunktion.

```{code-cell} ipython3
gewichte = -2/3 * abstaende + 1
```

### Schritt 4: Training eines Ersatzmodells

Häufig wird ein lineares Regressionsmodell oder ein Entscheidungsbaum verwendet,
um ein lokal interpretierbares Modell zu erzeugen, dass das ursprüngliche
KI-Modell erklärt. Wir werfen mit Hilfe eines Streudiagramms (Scatterplot) einen
kurzen Blick auf die variierten Eingabedaten und die vom KI-Modell
prognostizierten Ausgabedaten (maximale Kräfte). Auf der x-Achse tragen wir die
Zellengrößen ein und auf der y-Achse die Füllgrade der variierten Bauteile.
Durch die Farbe kennzeichen wir die prognostizierten maximalen Kräfte.

```{code-cell} ipython3
fig = px.scatter( eingabedaten, x='Zellengroesse', y='Fuellgrad', color=ausgabedaten,
    title='Variierte Eingabedaten und dazugehörige Prognosen', 
    labels={'color': 'maximale Kraft [N]'}
)
fig.update_layout(xaxis_range=[1.5, 4.5], yaxis_range=[0.15,0.45])

fig.show()
```

Scheinbar ist vor allem der Füllgrad entscheidend für die Prognose der maximalen
Kraft. Wir visualisieren daher die Prognosen der maximalen Kräfte abhängig vom
Füllgrad.

```{code-cell} ipython3
import plotly.express as px 

fig = px.scatter(eingabedaten, x='Fuellgrad', y=ausgabedaten,
    title='Prognostizierte maximale Kraft abhängig vom Füllgrad', 
    labels={'y': 'maximale Kraft[N]'}
)

fig.show()
```

Wir wählen als White-Box-Modell ein lineares Regressionsmodell, das lokal eine
gut interpretierbare Erklärung liefern soll. Dabei beschränken wir uns zunächst
auf den Füllgrad als Ursache $x$ und die maximale Kraft (in Newton) als Wirkung
$y$. Wir suchen also Parameter $w$ (Steigung) und $b$ (y-Achsenabschnitt), so
dass die lineare Funktion

$$y = w\cdot x + b$$

möglichst gut die Punkte trifft. Wie gut die maximalen Kräfte basierend auf dem
Füllgrad durch die Gerade angenähert werden, bewertet das sogenannte
R²-Bestimmtheitsmaß (siehe [Wikipedia →
Bestimmtheitsmaß](https://de.wikipedia.org/wiki/Bestimmtheitsmaß)).
Normalerweise liegt das R²-Bestimmtheitsmaß zwischen 0 und 1, wobei ein Wert von
Eins perfekt wäre, aber es kann auch negativ werden.

```{admonition} Interaktive Bestimmung des linearen Regressionsmodells
:class: miniexercise

Probieren Sie aus, für welche Steigung $w$ und für welchen y-Achsenabschnitt $b$
die lineare Regressionsgerade am besten die Datenpunkte annähert. Das
R²-Bestimmtheitsmaß wird dabei im Titel angezeigt und sollte möglichst nahe 1 sein.

<iframe src="https://gramschs.github.io/xai/_static/extra/linear_regression.html"
width=100% height="600" frameborder="0" scrolling="yes"></iframe>
```

Lineare Regressionsmodelle sind nicht darauf beschränkt, nur *ein* Merkmal als
Ursache zu betrachten. Wir können auch ein sogenanntes multiples lineares
Regressionsmodell benutzen, bei dem die drei Merkmale Zellenform $x_0$,
Zellengröße $x_1$ und Füllgrad $x_2$ linear kombiniert werden, um die maximale
Kraft $y$ zu prognostizieren:

$$y = w_0\cdot x_0 + w_1\cdot x_1 + w_2\cdot x_2 + b.$$

Bei drei Merkmalen haben wir nicht nur die Steigung (für den Füllgrad), sondern
auch die Steigungen für die Zellenform und die Zellengröße. Üblicherweise werden
diese Koeffizienten Gewichte (englisch weight) genannt und mit $w_0$, $w_1$ und
$w_2$ abgekürzt.

Die Bestimmung der bestmöglichen Gewichte $w_0$, $w_1$, $w_2$ und $b$ überlassen
wir diesmal dem Modul Scikit-Learn.
[Scikit-Learn](https://scikit-learn.org/stable/index.html) ist eine bekanntesten
Bibliotheken für das maschinelle Lernen und beinhaltet auch lineare
Regressionsmodelle. Die linearen Regressionsmodelle sind dabei in einem
Untermodul namens `sklearn.linear_model` gesammelt. Daraus importieren wir das
lineare Regressionsmodell `LinearRegression` und instanziieren es als `modell`.
Dann trainieren wir das lineare Regressionsmodell mit den Eingabe- und
Ausgabedaten und benutzen dafür die die `fit()`-Methode. Wir verzichten auf eine
Skalierung der Daten, da diese in derselben Größenordnung liegen und lassen auch
den üblichen Split in Trainings- und Testdaten weg. Stattdessen übergeben wir
zusätzlich über das optionale Argument `sample_weights` noch die Gewichte, so
dass die Ähnlichkeit eines Datenpunktes zur Referenz bei der Bestimmung der
Gewichte berücksichtigt wird.

```{code-cell} ipython3
from sklearn.linear_model import LinearRegression

modell = LinearRegression()
modell.fit(eingabedaten, ausgabedaten, sample_weight=gewichte)
```

Mithilfe der `score()`-Methode lassen wir die Qualität des Modells bestimmen:

```{code-cell} ipython3
score = modell.score(eingabedaten, ausgabedaten)
print(score)
```

Ein Score von 1 wäre perfekt, ungefähr 0.89 ist sehr gut. Wir haben daher ein
White-Box-Ersatzmodell gefunden, das hilft, das ausgewählte Beispiel zu
interpretieren.

### Schritt 5: Erklärung der Prognose

Als nächstes lassen wir uns die Gewichte $w_0, w_1$ und $w_2$ und den
y-Achsenabschnitt $b$ des linearen Regressionsmodells

$$y = w_0\cdot x_0 + w_1\cdot x_1 + w_2\cdot x_2 + b$$

ausgeben. Diese werden von Scikit-Learn im trainierten Modell im Attribut
`coef_` gespeichert.

```{code-cell} ipython3
print(modell.coef_)
```

Dazu kommt noch der y-Achsenabschnitt $b$, der im Attribut `intercept_` gespeichert ist.

```{code-cell} ipython3
print(modell.intercept_)
```

Insgesamt lautet das lineare Regressionsmodell also

$$y = 0\cdot x_0 -2.46\cdot x_1 +  635.33 \cdot x_2 - 134.79,$$

wobei $y$ die maximale Kraft in Newton bezeichnet, $x_0$ die Zellenform, $x_1$
die Zellengröße und $x_2$ den Füllgrad. Das Gewicht für die Zellenform ist $0$,
sie spielt bei unsererm Erklärmodell keine Rolle. Wir haben die Zellenform auch
nicht variiert. Die Zellengröße hat einen leicht negativen Effekt, denn
$w_1\approx -2.46$. Der deutlich wichtigere Effekt ist jedoch der Füllgrad
($w_2\approx 635.33$). Selbst wenn wir berücksichtigen, dass die Füllgrade aus
dem Intervall $[0.2, 0.45]$ Faktor 10 kleiner sind als die Zellengrößen aus dem
Intervall $[2, 8]$, ist $w_2$ um einiges gewichtiger als $w_1$ und hat einen
positiven Effekt. Je höher der Füllgrad ist, desto höher ist die prognostizierte
maximale Kraft. Damit können wir Ingenieurinnen und Ingenieuren Hinweise geben,
wie die Bauteile für die Schuheinlage konstruiert werden sollten.

## Kategorien der erklärbaren KI-Modelle

Bei dem obigen Beispiel des Schuheinlagen-Orakels haben wir die LIME-Methode
benutzt, um das KI-Modell zu erklären. LIME steht dabei für »Local Interpretable
Model-agnostic Explanations«. Allein diese Begrifflichkeiten deuten schon an,
dass es viele verschiedene Möglichkeiten gibt, KI-Modelle zu erklären.
Beispielsweise werden Erklärkonzepte nach ihrem **Umfang** unterschieden. **Lokale
Modelle** erklären, wie die Entscheidungslogik des KI-Modells für ein einzelnes
Beispiel zustandekommt und was für Datenpunkte in der unmittelbaren
Nachbarschaft prognostiziert werden würde. Dem gegenüber stehen **globale
Modelle**, die einen Einblick in die Gesamtstruktur und Funktionsweise eines
KI-Modells geben.

Ein weiteres Unterscheidungsmerkmal von erklärbaren KI-Modellen ist die
**Modellabhängigkeit**. Funktioniert die Methode für jedes KI-Modell, ist das
erklärbare KI-Modell als unabhängig vom Originalmodell, so nennt man die Methode
**modellagnostisch**. Das Gegenteil von modellagnostisch ist
**modellspezifisch**. Bei modellspezifischen Methoden ist die Erklärmethode auf
ein bestimmtes KI-Modell zugeschnitten. Ein typischer Vertreter dieser Kategorie
ist die Analyse der sogenannten Feature Importance bei Random Forests.

Es gibt noch einige weitere Unterscheidungsmerkmale. In diesem Kapitel gehen wir
noch auf die **Komplexität** ein, bei der zwischen **intrinsischen** Modellen
und **Post-hoc-Methoden** unterschieden wird. Intrinsische Modelle sind von sich
aus interpretierbar  wie biespielsweise die lineare Regression oder
Entscheidungsbäume. Dahingegen werden Post-hoc-Modelle nachträglich auf
KI-Modelle angewendet, so wie wir im obigen Beispiel die Post-hoc-Methode LIME
eingesetzt haben, um die Funktionsweise des Schuheinlagen-Orakels im Nachhinein
zu erklären. Eine weitere sehr bekannte Post-hoc-Methode ist das **SHAP**-Verfahren.

## Zusammenfassung und Ausblick

In diesem Kapitel haben wir die Erklärbarkeit von KI-Modellen untersucht. Nach
einer Einführung in die Relevanz des Themas haben wir das populäre
Post-hoc-Verfahren LIME kennengelernt, das universell einsetzbar ist und die
Entscheidungslogik für einzelne Referenzbeispiele lokal interpretieren kann.
Außerdem haben wir wichtige Kategorien der Erklärbarkeit von KI-Modellen
betrachtet. Im nächsten Kapitel würden wir uns mit dem SHAP-Verfahren
beschäftigen.
