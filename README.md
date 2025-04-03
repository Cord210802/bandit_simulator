clave unica 205597 

# Problema de Multi-Bandas (Multi-Armed Bandit): Teoría e Implementación

La tarea se entrega por discord antes del miercoles de la siguiente clase. Incluye llenar cuidadosamente en latex todos los snippets mencionados aqui, mas el codigo ya sea con link a colab o al repositorio. No olviden poner su clave unica. La idea es que investiguen, entiendan y proponga una git@github.com:Cord210802/bandit_simulator.gitsolucion al problema. Utilicen chatgpt y los tutoriales de la tarea (cursor especialmente) para hacer codigo y entender el problema.  

**Nota**  git@github.com:Cord210802/bandit_simulator.git
No pueden utilizar machine learning salvo regresion lineal si asi lo desean (no arboles, deep learning, etc..). 

La proxima clase vamos a continuar con un ejercicio parecido, pero usando cadenas de markov. Vamos a modificar el bandit para que sea mas interesante ante cadenas de markov.  

**Examen**  
El lunes hay examen sobre estos ejercicios a papel y lapiz, la calificacion sera el $min\{examen, ejercicios\}$, si $|examen - ejercicios|<1$ entonces sera el $maximo$. 


## 1. Introducción a los Problemas de Multi-Bandas

### 1.1 Definición y Enunciado del Problema

El problema de Multi-Bandas (MAB, por sus siglas en inglés) es un problema clásico en teoría de la decisión y aprendizaje por refuerzo. Su nombre surge del escenario de un jugador que enfrenta múltiples máquinas tragamonedas (a veces llamadas "bandidos de un solo brazo"), cada una con diferentes probabilidades de recompensa desconocidas. El jugador debe decidir qué máquinas jugar, en qué orden y cuántas veces, para maximizar su recompensa total.

En este modelo:
- Existen $K$ brazos (o acciones) diferentes.
- Cada brazo, cuando se jala, otorga una recompensa extraída de una distribución de probabilidad específica de ese brazo.
- Las distribuciones de recompensa son inicialmente desconocidas para el tomador de decisiones.
- El objetivo es maximizar la recompensa acumulada a lo largo de una serie de jugadas.

El problema captura la disyuntiva fundamental entre **exploración** (probar diferentes brazos para reunir información sobre sus distribuciones de recompensa) y **explotación** (elegir el brazo que actualmente parece ser el mejor).

### 1.2 Dilema de Exploración vs. Explotación

Este dilema está en el corazón del problema de multi-bandas:

- **Exploración**: Seleccionar brazos para aprender más sobre sus distribuciones de recompensa, potencialmente sacrificando recompensas inmediatas.
- **Explotación**: Seleccionar el brazo que actualmente parece ofrecer la mayor recompensa esperada en función de la información reunida hasta el momento.

Equilibrar estos dos aspectos es crucial. Demasiada exploración desperdicia recursos en brazos subóptimos. Demasiada explotación puede impedir descubrir un brazo mejor.

### 1.3 Formulación Matemática General

Formalicemos el problema estándar de bandas estocásticas:

- Sea $K$ el número de brazos.
- Para cada brazo $i \in \{1, 2, \ldots, K\}$, existe una distribución de probabilidad desconocida $\mathcal{D}_i$ con media $\mu_i$.
- En cada paso de tiempo $t \in \{1, 2, \ldots, T\}$:
  - El agente selecciona un brazo $a_t \in \{1, 2, \ldots, K\}$.
  - El agente recibe una recompensa $r_t \sim \mathcal{D}_{a_t}$.
- El objetivo es maximizar la recompensa acumulada $\sum_{t=1}^{T} r_t$.

Alternativamente, el problema puede enmarcarse en términos de minimizar **el arrepentimiento**. El arrepentimiento se define como la diferencia entre la recompensa obtenida al seleccionar siempre el brazo óptimo y la recompensa realmente obtenida por el agente:

$\text{Regret}(T) = T \cdot \max_{i} \mu_i - \mathbb{E}\left[\sum_{t=1}^{T} r_t\right]$

## 2. Escenarios de Información en Nuestro Entorno de Bandas

En nuestro entorno de multi-bandas, exploramos tres escenarios de información distintos, cada uno proporcionando al agente diferentes niveles de conocimiento:

### 2.1 Escenario de Información Completa

En este escenario, el agente observa:
- El número de turno actual.
- El número total de turnos T.
- La probabilidad de recompensa para el brazo 1 (p1) y brazo 2 (p2).
- El historial completo de acciones y recompensas pasadas.

Este es el escenario más informativo, ya que el agente conoce la probabilidad de uno de los brazos directamente y puede inferir la del otro con base en las recompensas observadas.

### 2.2 Escenario de Información Parcial

En este escenario, el agente observa:
- El número de turno actual.
- El número total de turnos T.
- La probabilidad de recompensa para el brazo 1 (p1).
- El historial de acciones y recompensas pasadas.

El agente conoce la probabilidad de un brazo pero debe aprender la del otro a través de la experimentación.

### 2.3 Escenario de Solo Recompensa

En este escenario, el agente observa:
- El número de turno actual.
- El historial de acciones y recompensas pasadas.

Este es el escenario más desafiante porque:
1. El agente no conoce la probabilidad de ninguno de los dos brazos.
2. El agente no conoce el número total de turnos T.

El agente debe aprender las probabilidades de ambos brazos mediante la experimentación y no puede optimizar su estrategia en función de la duración conocida del juego.

## 3. Entornos de Bandas en Nuestro Playground

Nuestro entorno implementa cuatro tipos diferentes de entornos de multi-bandas, cada uno con características distintas que afectan cómo cambian las probabilidades de los brazos a lo largo del tiempo.

### 3.1 Entorno de Banda Fija

#### Descripción
En el entorno de Banda Fija, cada brazo tiene una probabilidad constante de recompensa durante todo el juego. Estas probabilidades se asignan aleatoriamente al inicio de cada juego (uniforme entre 0.01 y 0.99) y permanecen sin cambios.

#### Formulación Matemática
- Dos brazos: $a \in \{0, 1\}$
- Probabilidades fijas: $p_1, p_2 \in [0.01, 0.99]$
- En el turno $t$, al seleccionar el brazo $a$:
  - Se recibe recompensa $r_t = 1$ con probabilidad $p_{a+1}$
  - Se recibe recompensa $r_t = 0$ con probabilidad $1 - p_{a+1}$

#### Decisión (T Fijo)

### **EJERCICIO**  
**RESPUESTA**  
Definir el problema de decisión para la Banda Fija con horizonte de tiempo conocido T = 100. ¿Cuál es la función objetivo? ¿Cuáles son las restricciones? ¿Cuál es la política óptima?
---
### Mi respuesta:
### 3.1 Entorno de Banda Fija

#### Problema de Decisión para $T = 100$

En el entorno de Banda Fija, las probabilidades de recompensa de cada brazo ($p_1$ y $p_2$) son constantes pero desconocidas para el agente (con excepción del escenario completo). A continuación, se describen los elementos del problema según cada uno de los **tres escenarios de información**.

Además, se presenta una estrategia común para tomar decisiones en este tipo de entornos: **UCB1 (Upper Confidence Bound)**.

---

### ¿Qué es UCB1 y por qué lo usamos?

**UCB1 (Upper Confidence Bound 1)** es una estrategia de decisión para problemas de multi-bandas que busca equilibrar automáticamente la **exploración** y la **explotación**. La idea es seleccionar, en cada paso, el brazo cuya recompensa esperada es **potencialmente** la mayor, considerando tanto su media observada como una medida de incertidumbre.

En cada turno $t$, el algoritmo selecciona el brazo $i$ que maximiza:

$$
\hat{\mu}_i + \sqrt{\frac{2 \log t}{n_i}}
$$

Donde:
- $\hat{\mu}_i$ es la media empírica de recompensas del brazo $i$
- $n_i$ es el número de veces que se ha jugado el brazo $i$

El segundo término actúa como un "bonus" que favorece los brazos menos explorados. Este enfoque ha demostrado ser eficiente en muchos escenarios porque garantiza un **arrepentimiento logarítmico** en el número de turnos: esto significa que, a largo plazo, el algoritmo se acerca al rendimiento óptimo.

---

## Escenarios de Información

### **Escenario 1: Información Completa**

#### Estados

El agente conoce:
- El turno actual $t$
- El total de turnos $T = 100$
- Las probabilidades reales de recompensa $p_1$, $p_2$
- El historial completo $(a_1, r_1), \ldots, (a_{t-1}, r_{t-1})$

Estado:

$$
s_t = (t, p_1, p_2)
$$

#### Acciones

$a_t \in \{0, 1\}$

#### Función Objetivo

$$
\max_{\pi} \mathbb{E}_\pi\left[\sum_{t=1}^{100} r_t \right]
$$

#### Política Óptima

Como las probabilidades son conocidas, simplemente se explota el mejor brazo:

$$
\pi^*(s_t) = 
\begin{cases}
0 & \text{si } p_1 \geq p_2 \\
1 & \text{si } p_2 > p_1
\end{cases}
$$

---

### **Escenario 2: Información Parcial**

#### Estados

El agente conoce:
- El turno actual $t$
- El total de turnos $T = 100$
- La probabilidad $p_1$ (conocida)
- El historial de acciones y recompensas pasadas

Estado:

$$
s_t = (t, p_1, n_1, s_1)
$$

#### Acciones

$a_t \in \{0, 1\}$

#### Función Objetivo

$$
\max_{\pi} \mathbb{E}_\pi\left[\sum_{t=1}^{100} r_t \right]
$$

#### Política Óptima

Ya que $p_1$ es conocida, se puede comparar contra una **estimación confiable** de $p_2$. Para lograr esto, se utiliza una variante del algoritmo **UCB1**, que aplica la cota de confianza solo al brazo desconocido.

Política:

$$
\pi^*(s_t) = 
\begin{cases}
0 & \text{si } p_1 \geq \hat{p}_2 + \sqrt{\frac{2 \log t}{n_1}} \\
1 & \text{en otro caso}
\end{cases}
$$

También puede utilizarse **Thompson Sampling** sobre el brazo desconocido como alternativa bayesiana.

---

### **Escenario 3: Solo Recompensa**

#### Estados

El agente conoce:
- El turno actual $t$
- Solo el historial de acciones y recompensas pasadas

Estado:

$$
s_t = (t, n_0, s_0, n_1, s_1)
$$

#### Acciones

$a_t \in \{0, 1\}$

#### Función Objetivo

$$
\max_{\pi} \mathbb{E}_\pi\left[\sum_{t=1}^{100} r_t \right]
$$

#### Política Óptima

En este escenario no se conoce ninguna probabilidad ni el valor de $T$ al inicio. Por tanto, se requiere una política que **aprenda ambas probabilidades** desde cero mientras maximiza la recompensa.

Aquí es donde **UCB1** brilla: selecciona el brazo que maximiza:

$$
\hat{\mu}_i + \sqrt{\frac{2 \log t}{n_i}}
$$

Esto asegura que:
- Al inicio, se exploran ambos brazos lo suficiente.
- Conforme avanza el tiempo, se explota el mejor.

Alternativamente, **Thompson Sampling** también puede ser usado, actualizando para cada brazo su distribución posterior:

$$
\text{Beta}(s_i + 1, n_i - s_i + 1)
$$

Y seleccionando el brazo con el mayor valor muestreado.


---

#### Decisión (T Aleatorio)

### **EJERCICIO**  
**RESPUESTA**  
Definir el problema de decisión para la Banda Fija con horizonte de tiempo desconocido T ~ Uniform(1, 300). ¿Cómo afecta el horizonte de tiempo aleatorio la estrategia óptima?
Claro, aquí tienes la **respuesta explicada y redactada** para el ejercicio:  

---
### Mi respuesta:
### 3.2 Entorno de Banda Fija con Horizonte de Tiempo Aleatorio

#### Problema de Decisión para $T \sim \text{Uniform}(1, 300)$

En este escenario, el número total de turnos $T$ no es fijo, sino que es una variable aleatoria extraída de una distribución uniforme discreta entre $1$ y $300$. Esto implica que el agente no sabe con certeza cuántas jugadas podrá realizar, lo que afecta su forma de balancear exploración y explotación.

---

### ¿Cómo afecta el horizonte aleatorio a la estrategia?

Cuando $T$ es **desconocido**, el agente debe **ser más conservador con la exploración**, especialmente en las primeras rondas. Explorar demasiado al inicio puede ser costoso si el juego termina pronto, ya que **no habrá tiempo para aprovechar lo aprendido**. Por tanto, las estrategias óptimas tienden a:
- Explorar, pero con menor agresividad.
- Comenzar la explotación más temprano.
- Usar variantes de UCB o Thompson Sampling adaptadas a horizontes inciertos.

---

## Escenarios de Información

### **Escenario 1: Información Completa**

#### Supuestos

El agente conoce:
- El turno actual $t$
- La probabilidad de recompensa real de ambos brazos: $p_1$, $p_2$
- El valor actual de $T$ (aunque fue generado aleatoriamente)

#### Estado

$$
s_t = (t, p_1, p_2, T)
$$

#### Acción

$a_t \in \{0, 1\}$

#### Política Óptima

Si el agente conoce el valor de $T$ y las probabilidades reales $p_1$, $p_2$, la estrategia es idéntica al caso con horizonte fijo: simplemente explota el mejor brazo durante todo el juego.

$$
\pi^*(s_t) = \arg\max_{a \in \{0, 1\}} p_{a+1}
$$

---

### **Escenario 2: Información Parcial**

#### Supuestos

El agente conoce:
- El turno actual $t$
- La probabilidad de recompensa del brazo 0: $p_1$
- El historial de recompensas y acciones anteriores
- No conoce el valor de $T$

#### Estado

$$
s_t = (t, p_1, n_1, s_1)
$$

#### Acción

$a_t \in \{0, 1\}$

#### Política Óptima

Debido a la incertidumbre sobre la duración del juego, el agente debe:
- Considerar el valor esperado de $T$, que es $E[T] = 150.5$
- Explorar el brazo desconocido (brazo 1) solo lo suficiente para reducir incertidumbre

Una estrategia efectiva aquí es usar una variante de **UCB1**, pero con una penalización más rápida al término de exploración:

$$
a_t = 
\begin{cases}
0 & \text{si } p_1 \geq \hat{p}_2 + \sqrt{\frac{2 \log t}{n_1}} \\
1 & \text{otro caso}
\end{cases}
$$

Esto permite aprovechar el conocimiento sobre el brazo 0 sin arriesgarse a explorar demasiado un brazo incierto.

También puede utilizarse **Thompson Sampling restringido** al brazo desconocido, para ajustar la exploración de forma bayesiana.

---

### **Escenario 3: Solo Recompensa**

#### Supuestos

El agente solo conoce:
- El turno actual $t$
- El historial de acciones y recompensas pasadas
- No conoce ni $p_1$, ni $p_2$, ni el valor de $T$

#### Estado

$$
s_t = (t, n_0, s_0, n_1, s_1)
$$

#### Acción

$a_t \in \{0, 1\}$

#### Política Óptima

En este escenario el agente está completamente en la oscuridad, por lo que debe:
- Asumir una duración esperada de $E[T] = 150.5$
- Controlar cuidadosamente la exploración para evitar malgastar turnos si el juego termina pronto

Una forma de hacerlo es usando una versión ajustada de **UCB1** con un término de exploración más agresivo:

$$
a_t = \arg\max_{i \in \{0,1\}} \left( \hat{\mu}_i + \sqrt{\frac{1.5 \log t}{n_i}} \right)
$$

Donde el factor $1.5$ (en vez del típico $2$) puede ayudar a explorar menos agresivamente ante horizontes inciertos.

Alternativamente, **Thompson Sampling** sigue siendo una excelente opción:
- Se modela cada brazo con una distribución Beta: $\text{Beta}(s_i + 1, n_i - s_i + 1)$
- Se selecciona el brazo que produce el mayor valor muestreado

Este método ajusta naturalmente la cantidad de exploración en función de la información acumulada, sin necesidad de saber cuánto durará el juego.

---

### 3.2 Entorno de Banda Periódica

#### Descripción
En el entorno de Banda Periódica, la probabilidad de recompensa de cada brazo cambia cada k turnos (por defecto, k=10). En cada punto de cambio, se asignan nuevas probabilidades aleatorias (uniforme entre 0.01 y 0.99) a ambos brazos.

#### Formulación Matemática
- Dos brazos: $a \in \{0, 1\}$
- En el turno $t$, las probabilidades son:
  - $p_1(t) = p_1^{\lfloor t/k \rfloor}$, donde $p_1^j \sim \text{Uniform}(0.01, 0.99)$
  - $p_2(t) = p_2^{\lfloor t/k \rfloor}$, donde $p_2^j \sim \text{Uniform}(0.01, 0.99)$
- El superíndice $j = \lfloor t/k \rfloor$ indica el número de "período".
- En cada punto de cambio (cuando $t$ es divisible por $k$), se asignan nuevos valores aleatorios.

#### Decisión (T Fijo)

### **EJERCICIO**  
**RESPUESTA**  
Definir el problema de decisión para la Banda Periódica con horizonte de tiempo conocido T = 100 y período k = 10. ¿Cómo abordarías la búsqueda de una estrategia óptima? ¿Qué información adicional sería valiosa rastrear?

---
### Mi respuesta:
### 3.3 Entorno de Banda Periódica (Con Cambios Aleatorios)

#### Descripción

En este entorno, las probabilidades de recompensa de los brazos **cambian cada $k$ turnos**. En cada nuevo período $j = \left\lfloor \frac{t}{k} \right\rfloor$, se generan nuevas probabilidades independientes para ambos brazos:

- $p_1^{(j)} \sim \text{Uniform}(0.01, 0.99)$  
- $p_2^{(j)} \sim \text{Uniform}(0.01, 0.99)$

Estas probabilidades se mantienen constantes dentro del período $j$, pero cambian abruptamente al inicio del siguiente.  
El horizonte total de tiempo es $T = 100$ y el período de cambio es $k = 10$, por lo tanto, habrá $10$ períodos distintos.

---

## Implicaciones

Este es un entorno **no estacionario con cambios abruptos**, y por tanto:
- Las recompensas pasadas **no sirven directamente para predecir las futuras**.
- Las estrategias que acumulan promedios históricos (como UCB1 puro) **pueden fallar**.
- Es necesario usar estrategias que se **adapten rápidamente** a cambios.

---

## Escenarios de Información

### **Escenario 1: Información Completa**

#### Supuestos

El agente conoce:
- El turno actual $t$
- El total de turnos $T = 100$
- El valor de $k = 10$
- Las probabilidades reales $p_1^{(j)}$, $p_2^{(j)}$ del período actual $j = \left\lfloor \frac{t}{k} \right\rfloor$

#### Estado

$$
s_t = (t, j, p_1^{(j)}, p_2^{(j)})
$$

#### Acción

$a_t \in \{0, 1\}$

#### Política Óptima

Como el agente conoce las probabilidades en cada período, simplemente debe explotar el mejor brazo **durante los $k$ turnos del período**:

$$
\pi^*(s_t) = 
\begin{cases}
0 & \text{si } p_1^{(j)} \geq p_2^{(j)} \\
1 & \text{si } p_2^{(j)} > p_1^{(j)}
\end{cases}
$$

---

### **Escenario 2: Información Parcial**

#### Supuestos

El agente conoce:
- El turno actual $t$
- El total de turnos $T = 100$
- El valor de $k = 10$
- Solo la probabilidad del brazo 1: $p_1^{(j)}$
- No conoce $p_2^{(j)}$, ni sus valores futuros

#### Estado

$$
s_t = (t, j, p_1^{(j)}, n_2^{(j)}, s_2^{(j)})
$$

Donde:
- $n_2^{(j)}$: veces que se ha jugado el brazo 2 en el período $j$
- $s_2^{(j)}$: recompensas acumuladas del brazo 2 en el período $j$

#### Acción

$a_t \in \{0, 1\}$

#### Política Óptima

El agente debe tomar una decisión **por período**. Como $p_1^{(j)}$ es conocido, se puede explorar el brazo 2 dentro del mismo período para estimar si vale la pena explotarlo. 

Una estrategia adecuada es **UCB adaptado por período**:

$$
\pi^*(s_t) = 
\begin{cases}
0 & \text{si } p_1^{(j)} \geq \hat{p}_2^{(j)} + \sqrt{\frac{2 \log(k')}{n_2^{(j)}}} \\
1 & \text{en otro caso}
\end{cases}
$$

Donde $k'$ es el número de turnos transcurridos **dentro del período** $j$.

Se puede usar también **Thompson Sampling por período** para el brazo desconocido.

---

### **Escenario 3: Solo Recompensa**

#### Supuestos

El agente conoce:
- El turno actual $t$
- El valor de $k = 10$
- Solo el historial de acciones y recompensas pasadas

No conoce:
- Las probabilidades actuales ni pasadas
- El valor de $p_1^{(j)}$, $p_2^{(j)}$

#### Estado

$$
s_t = (t, j = \left\lfloor \frac{t}{k} \right\rfloor, n_0^{(j)}, s_0^{(j)}, n_1^{(j)}, s_1^{(j)})
$$

#### Acción

$a_t \in \{0, 1\}$

#### Política Óptima

Dado que los cambios son abruptos y las probabilidades antiguas no son útiles para el siguiente período, se recomienda usar **estrategias reactivas**, como:

- **Reiniciar las estadísticas al comenzar cada nuevo período $j$**
- Aplicar **UCB1 por período**, reiniciando los contadores al inicio de cada $k$:

$$
a_t = \arg\max_{i \in \{0, 1\}} \left( \hat{\mu}_i^{(j)} + \sqrt{\frac{2 \log(k')}{n_i^{(j)}}} \right)
$$

También se puede usar **Thompson Sampling reinicializado cada $k$ turnos**, usando:

$$
\text{Beta}(s_i^{(j)} + 1, n_i^{(j)} - s_i^{(j)} + 1)
$$

---

## Información adicional valiosa a rastrear

Para adaptarse al entorno, el agente debería llevar:

- El índice del período actual: $j = \left\lfloor \frac{t}{k} \right\rfloor$
- Para cada brazo $i$:
  - $n_i^{(j)}$: número de veces jugado en el período actual
  - $s_i^{(j)}$: recompensas acumuladas en ese período
- Estadísticas **locales por período**, ya que las globales pierden valor por los cambios abruptos

---
#### Decisión (T Aleatorio)

### **EJERCICIO**  
**RESPUESTA**  
Definir el problema de decisión para la Banda Periódica con horizonte de tiempo desconocido T ~ Uniform(1, 300) y período k = 10. ¿Cómo interactúa la aleatoriedad en T con la naturaleza periódica del entorno?
Problema de Decisión para la Banda Periódica (T ~ Uniform(1, 300), k=10)
---
### Mi respuesta:
### 3.4 Entorno de Banda Periódica con Horizonte Aleatorio

#### Descripción

En este entorno:
- Hay **dos brazos**: $a \in \{0, 1\}$
- Las probabilidades de recompensa **cambian cada $k = 10$ turnos**:
  - En cada período $j = \left\lfloor \frac{t}{k} \right\rfloor$:
    - $p_1^{(j)} \sim \text{Uniform}(0.01, 0.99)$
    - $p_2^{(j)} \sim \text{Uniform}(0.01, 0.99)$
- El número total de turnos $T$ **no se conoce**, y es extraído de una distribución uniforme:
  - $T \sim \text{Uniform}(1, 300)$

---

### ¿Cómo interactúa la aleatoriedad en $T$ con la naturaleza periódica?

Esta combinación implica que:

- No se puede saber cuántos **períodos completos** habrá (en promedio, $15$).
- Explorar al inicio de un nuevo período es **riesgoso**, ya que el juego puede terminar **antes de que se aproveche** lo aprendido.
- Se vuelve crucial **aprender rápido** dentro de cada período y adaptarse **de forma local**, sin asumir que la información persistirá o será reutilizable.

---

## Escenarios de Información

### **Escenario 1: Información Completa**

#### Supuestos

El agente conoce:
- El turno actual $t$
- El valor de $k = 10$
- Las probabilidades reales $p_1^{(j)}$, $p_2^{(j)}$ en el período actual $j$
- No conoce el valor exacto de $T$, pero sabe que $T \sim \text{Uniform}(1, 300)$

#### Estado

$$
s_t = (t, j = \left\lfloor \frac{t}{k} \right\rfloor, p_1^{(j)}, p_2^{(j)})
$$

#### Acción

$a_t \in \{0, 1\}$

#### Política Óptima

A pesar de que $T$ es aleatorio, si se conocen las probabilidades del período actual, el agente simplemente debe explotar el mejor brazo:

$$
\pi^*(s_t) = \arg\max \{p_1^{(j)}, p_2^{(j)}\}
$$

No es necesario explorar, porque la información del otro brazo ya está disponible.

---

### **Escenario 2: Información Parcial**

#### Supuestos

El agente conoce:
- El turno actual $t$
- El valor de $k = 10$
- Solo $p_1^{(j)}$ del período actual
- No conoce $p_2^{(j)}$ ni el valor exacto de $T$

#### Estado

$$
s_t = (t, j, p_1^{(j)}, n_2^{(j)}, s_2^{(j)})
$$

#### Acción

$a_t \in \{0, 1\}$

#### Política Óptima

La política debe:
- Considerar que $E[T] = 150.5$
- Aprovechar que cada período solo dura $10$ turnos
- Limitar la exploración al inicio del período

Se recomienda una política tipo **UCB con reinicio por período**, pero con **mayor peso a la explotación**:

$$
\pi^*(s_t) = 
\begin{cases}
0 & \text{si } p_1^{(j)} \geq \hat{p}_2^{(j)} + \sqrt{\frac{1.5 \log(k')}{n_2^{(j)}}} \\
1 & \text{en otro caso}
\end{cases}
$$

También puede usarse **Thompson Sampling por período** sobre el brazo desconocido, reinicializando los parámetros al comienzo de cada $k$.

---

### **Escenario 3: Solo Recompensa**

#### Supuestos

El agente conoce:
- El turno actual $t$
- El valor de $k = 10$
- Solo el historial de acciones y recompensas
- No conoce $T$, $p_1^{(j)}$, ni $p_2^{(j)}$

#### Estado

$$
s_t = (t, j = \left\lfloor \frac{t}{k} \right\rfloor, n_0^{(j)}, s_0^{(j)}, n_1^{(j)}, s_1^{(j)})
$$

#### Acción

$a_t \in \{0, 1\}$

#### Política Óptima

Se deben **reiniciar las estadísticas cada nuevo período** y actuar de manera **rápida**:

- Usar **UCB1 por período**, con penalización más fuerte a la exploración para evitar desperdicio si $T$ termina antes:

$$
a_t = \arg\max_{i \in \{0, 1\}} \left( \hat{\mu}_i^{(j)} + \sqrt{\frac{1.5 \log(k')}{n_i^{(j)}}} \right)
$$

- Alternativamente, usar **Thompson Sampling con reinicio**:

  - En cada nuevo $j$, iniciar:
  
    $$
    \text{Beta}(s_i^{(j)} + 1, n_i^{(j)} - s_i^{(j)} + 1)
    $$

  - Seleccionar el brazo con mayor valor muestreado

---

## Información adicional valiosa a rastrear

Para cada período $j$:
- Número de jugadas por brazo: $n_i^{(j)}$
- Recompensas acumuladas: $s_i^{(j)}$
- Tiempo dentro del período: $k' = t \bmod k$

Esto permite al agente:
- Estimar rápidamente las probabilidades dentro del período
- Decidir cuándo dejar de explorar
- Reiniciar correctamente las estadísticas cada vez que $j$ cambia

---
### 3.3 Entorno de Banda Dinámica

#### Descripción
En el entorno de Banda Dinámica, las probabilidades de recompensa para ambos brazos cambian en cada turno. Cada turno se asignan probabilidades aleatorias completamente nuevas (uniforme entre 0.01 y 0.99) a ambos brazos.

#### Formulación Matemática
- Dos brazos: $a \in \{0, 1\}$
- En el turno $t$, las probabilidades son:
  - $p_1(t) \sim \text{Uniform}(0.01, 0.99)$
  - $p_2(t) \sim \text{Uniform}(0.01, 0.99)$
- Se generan nuevos valores aleatorios en cada turno.

#### Decisión (T Fijo)

### **EJERCICIO**  
**RESPUESTA**  
Definir el problema de decisión para la Banda Dinámica con horizonte de tiempo conocido T = 100. ¿Hay una forma significativa de aprender de observaciones pasadas en este entorno? ¿Cuál sería la estrategia óptima?
¡Claro! Aquí tienes la **respuesta completa en formato markdown (`.md`)**, siguiendo tu estilo preferido, para el caso del **Entorno de Banda Dinámica** con **horizonte fijo $T = 100$**, y considerando los tres escenarios de información.

---
### Mi respuesta:
### 3.5 Entorno de Banda Dinámica con Horizonte Fijo

#### Descripción

En este entorno, las probabilidades de recompensa de ambos brazos **cambian completamente en cada turno**. Es decir, para cada $t \in \{1, 2, \ldots, T\}$:

- $p_1(t) \sim \text{Uniform}(0.01, 0.99)$  
- $p_2(t) \sim \text{Uniform}(0.01, 0.99)$

Estas probabilidades son **generadas de forma independiente** en cada turno, por lo que no existe ningún patrón temporal ni continuidad entre turnos.

---

## Implicaciones del entorno dinámico

Este entorno es **totalmente no estacionario**, con cambios **abruptos e impredecibles en cada instante**. Como consecuencia:

- **No es posible aprender** patrones a largo plazo, porque las probabilidades no se mantienen.
- Las observaciones pasadas **no son informativas** para predecir recompensas futuras.
- Las estrategias tradicionales basadas en promedios acumulados (como UCB1 o Thompson Sampling) **no funcionan bien** aquí.

En cambio, la mejor estrategia será **mixta o aleatoria**, y deberá tratar de **adaptarse en tiempo real**, si es posible.

---

## Escenarios de Información

### **Escenario 1: Información Completa**

#### Supuestos

El agente conoce:
- El turno actual $t$
- El valor de $T = 100$
- Las probabilidades reales $p_1(t)$ y $p_2(t)$ en cada turno

#### Estado

$$
s_t = (t, p_1(t), p_2(t))
$$

#### Acción

$a_t \in \{0, 1\}$

#### Política Óptima

Como se conocen las probabilidades actuales, la política óptima es simplemente **explotar el mejor brazo del turno actual**:

$$
\pi^*(s_t) = 
\begin{cases}
0 & \text{si } p_1(t) \geq p_2(t) \\
1 & \text{si } p_2(t) > p_1(t)
\end{cases}
$$

**No es necesario explorar**, ya que el agente conoce toda la información relevante en cada instante.

---

### **Escenario 2: Información Parcial**

#### Supuestos

El agente conoce:
- El turno actual $t$
- El valor de $T = 100$
- Solo la probabilidad actual $p_1(t)$
- No conoce $p_2(t)$
- No tiene información sobre turnos futuros

#### Estado

$$
s_t = (t, p_1(t))
$$

#### Acción

$a_t \in \{0, 1\}$

#### Política Óptima

Dado que $p_2(t)$ cambia en cada turno y no puede predecirse, no tiene sentido intentar aprenderla.

La mejor estrategia posible en este caso es una **regla de comparación directa**:

- Si $p_1(t)$ es alto (por ejemplo, $p_1(t) > 0.6$), explotar el brazo 0.
- Si $p_1(t)$ es bajo, entonces vale la pena **arriesgarse** a probar el brazo 1.

Una posible política es:

$$
\pi^*(s_t) = 
\begin{cases}
0 & \text{si } p_1(t) \geq \theta \\
1 & \text{si } p_1(t) < \theta
\end{cases}
$$

Donde $\theta$ es un umbral heurístico (por ejemplo, 0.5 o el valor esperado de una uniforme: 0.5).

Esta estrategia **no garantiza óptimo**, pero **maximiza la recompensa esperada** ante incertidumbre total sobre el otro brazo.

---

### **Escenario 3: Solo Recompensa**

#### Supuestos

El agente conoce:
- El turno actual $t$
- El historial de acciones y recompensas pasadas
- No conoce $p_1(t)$ ni $p_2(t)$
- No conoce los valores futuros

#### Estado

$$
s_t = (t, a_1, r_1, \ldots, a_{t-1}, r_{t-1})
$$

#### Acción

$a_t \in \{0, 1\}$

#### Política Óptima

Como las probabilidades **cambian aleatoriamente en cada turno**, el historial **no tiene valor predictivo**. El agente no puede aprender nada útil de las recompensas anteriores.

La mejor estrategia aquí es simplemente **elegir brazos al azar**, por ejemplo:

- Escoger cada brazo con probabilidad $0.5$  
- O bien, usar una política estocástica más sofisticada (como softmax), aunque sus ventajas son limitadas en este caso

Por lo tanto, la política óptima es simplemente:

$$
\pi^*(s_t) = \text{Random}(0, 1)
$$

Esto maximiza la recompensa esperada si ambos brazos son independientes y aleatorios.

---

## Conclusión: ¿Se puede aprender en este entorno?

No. En el **entorno de Banda Dinámica**, **no existe memoria útil** de turnos anteriores. Toda la información relevante está en el presente.

La estrategia óptima depende exclusivamente de:
- La información disponible **en el turno actual**
- La **ausencia total de correlación temporal** en las probabilidades

---

#### Decisión (T Aleatorio)

### **EJERCICIO**  
**RESPUESTA**  
Definir el problema de decisión para la Banda Dinámica con horizonte de tiempo desconocido T ~ Uniform(1, 300). ¿Cambia significativamente el enfoque óptimo en este entorno altamente dinámico si el horizonte de tiempo es desconocido?
---
### Mi respuesta:
### 3.6 Entorno de Banda Dinámica con Horizonte Aleatorio

#### Descripción

En este entorno:
- Hay **dos brazos**: $a \in \{0, 1\}$
- En cada turno $t$, se generan **nuevas probabilidades independientes**:

  $$
  p_1(t), p_2(t) \sim \text{Uniform}(0.01, 0.99)
  $$

- Las probabilidades cambian **completamente en cada turno**.
- El horizonte total de tiempo $T$ es **desconocido** y extraído aleatoriamente de una distribución uniforme:

  $$
  T \sim \text{Uniform}(1, 300)
  $$

---

## Implicaciones

Este entorno es **altamente dinámico y no estacionario**, con la dificultad adicional de que **no se sabe cuándo termina** el proceso. Las probabilidades pasadas **no se repiten ni influyen en el futuro**, y el agente **no puede hacer planes a largo plazo**.

---

## Escenarios de Información

### **Escenario 1: Información Completa**

#### Supuestos

El agente conoce:
- El turno actual $t$
- Las probabilidades actuales $p_1(t)$ y $p_2(t)$
- No conoce el valor de $T$

#### Estado

$$
s_t = (t, p_1(t), p_2(t))
$$

#### Acción

$a_t \in \{0, 1\}$

#### Política Óptima

La información pasada y futura no influye, así que el agente solo debe maximizar la recompensa del turno actual:

$$
\pi^*(s_t) = 
\begin{cases}
0 & \text{si } p_1(t) \geq p_2(t) \\
1 & \text{si } p_2(t) > p_1(t)
\end{cases}
$$

El hecho de que $T$ sea aleatorio **no cambia la estrategia óptima**, porque no hay nada que aprender ni planificar.

---

### **Escenario 2: Información Parcial**

#### Supuestos

El agente conoce:
- El turno actual $t$
- Solo $p_1(t)$
- No conoce $p_2(t)$ ni $T$

#### Estado

$$
s_t = (t, p_1(t))
$$

#### Acción

$a_t \in \{0, 1\}$

#### Política Óptima

Como $p_2(t)$ es desconocido y cambia constantemente, **no se puede aprender sobre él**, ni siquiera dentro del mismo juego.

La mejor estrategia es usar una regla basada en umbral:

$$
\pi^*(s_t) = 
\begin{cases}
0 & \text{si } p_1(t) \geq \theta \\
1 & \text{si } p_1(t) < \theta
\end{cases}
$$

Donde $\theta$ es un valor heurístico (como $0.5$ o el valor esperado de una uniforme).  
Tampoco en este caso el valor desconocido de $T$ afecta la política, ya que no existe oportunidad de aprendizaje secuencial.

---

### **Escenario 3: Solo Recompensa**

#### Supuestos

El agente conoce:
- El turno actual $t$
- Solo el historial de acciones y recompensas pasadas
- No conoce $p_1(t)$, $p_2(t)$ ni $T$

#### Estado

$$
s_t = (t, a_1, r_1, \ldots, a_{t-1}, r_{t-1})
$$

#### Acción

$a_t \in \{0, 1\}$

#### Política Óptima

En este escenario, no hay forma de inferir nada útil:
- Las probabilidades cambian cada turno y no siguen un patrón
- El horizonte es desconocido
- El historial no tiene valor predictivo

La política óptima es simplemente **aleatoria**:

$$
\pi^*(s_t) = \text{Random}(0, 1)
$$

Al igual que en los otros casos, **no cambia la estrategia** por el hecho de que $T$ sea aleatorio.

---

## ¿Cambia significativamente la estrategia si $T$ es desconocido?

**No.** En este entorno altamente dinámico:

- Las probabilidades cambian cada turno de manera aleatoria
- Las observaciones pasadas no aportan nada para el futuro
- No existe valor en planificar para el largo plazo

Por tanto, **la estrategia óptima depende únicamente del turno actual**, y conocer o no el valor de $T$ **no afecta el enfoque**.
---
### 3.4 Entorno de Banda Totalmente Aleatorio

#### Descripción
En el entorno de Banda Totalmente Aleatorio, las probabilidades de los brazos se inicializan de forma aleatoria y luego cambian aleatoriamente con una pequeña probabilidad (5%) en cada turno. Esto crea un entorno donde los cambios son impredecibles pero ocurren con menos frecuencia que en el entorno Dinámico.

#### Formulación Matemática
- Dos brazos: $a \in \{0, 1\}$
- Probabilidades iniciales: $p_1(0), p_2(0) \sim \text{Uniform}(0.01, 0.99)$
- En el turno $t > 0$, con probabilidad 0.05:
  - $p_1(t) \sim \text{Uniform}(0.01, 0.99)$
  - $p_2(t) \sim \text{Uniform}(0.01, 0.99)$
- De lo contrario (con probabilidad 0.95):
  - $p_1(t) = p_1(t-1)$
  - $p_2(t) = p_2(t-1)$

#### Decisión (T Fijo)

### **EJERCICIO**  
**RESPUESTA**  
Definir el problema de decisión para la Banda Totalmente Aleatoria con horizonte de tiempo conocido T = 100. ¿Cómo equilibrarías la exploración y explotación sabiendo que las probabilidades de los brazos podrían cambiar repentinamente?
---
### Mi respuesta:
### 3.7 Entorno de Banda Totalmente Aleatorio con Horizonte Fijo

#### Descripción

En este entorno:
- Hay **dos brazos**: $a \in \{0, 1\}$
- Las **probabilidades iniciales** son aleatorias:

  $$
  p_1(0),\ p_2(0) \sim \text{Uniform}(0.01, 0.99)
  $$

- En cada turno $t > 0$:
  - Con probabilidad $0.05$, se **reemplazan las probabilidades** por nuevos valores aleatorios:

    $$
    p_1(t),\ p_2(t) \sim \text{Uniform}(0.01, 0.99)
    $$

  - Con probabilidad $0.95$, **se mantienen** las probabilidades del turno anterior:

    $$
    p_1(t) = p_1(t-1),\quad p_2(t) = p_2(t-1)
    $$

Este entorno es un punto intermedio entre uno **estacionario** y uno **altamente dinámico**. Los cambios son **poco frecuentes pero impredecibles**.

---

### ¿Cómo equilibrar exploración y explotación?

Dado que los cambios ocurren **ocasionalmente**, la estrategia óptima debe:

- **Explorar continuamente**, pero de manera **suave**, para poder detectar cambios cuando ocurran.
- **Desconfiar ligeramente** de datos antiguos, ya que podrían haber quedado obsoletos tras un cambio.

Esto se puede lograr usando versiones **modificadas de UCB o Thompson Sampling** que incorporen **descuento temporal** o **ventanas móviles**.

---

## Escenarios de Información

### **Escenario 1: Información Completa**

#### Supuestos

El agente conoce:
- El turno actual $t$
- El valor de $T = 100$
- Las probabilidades actuales $p_1(t)$, $p_2(t)$

#### Estado

$$
s_t = (t, p_1(t), p_2(t))
$$

#### Acción

$a_t \in \{0, 1\}$

#### Política Óptima

Como las probabilidades actuales son conocidas, el agente simplemente debe **explotar el mejor brazo**:

$$
\pi^*(s_t) = \arg\max \{p_1(t), p_2(t)\}
$$

No es necesario explorar porque no hay incertidumbre en el estado actual.

---

### **Escenario 2: Información Parcial**

#### Supuestos

El agente conoce:
- El turno actual $t$
- El valor de $T = 100$
- Solo la probabilidad actual $p_1(t)$
- No conoce $p_2(t)$ ni si hubo un cambio

#### Estado

$$
s_t = (t, p_1(t), n_2, s_2)
$$

#### Acción

$a_t \in \{0, 1\}$

#### Política Óptima

El agente debe:
- Aprovechar que conoce $p_1(t)$
- Explorar el brazo 2 para detectar si **p_2(t)** es superior, sobre todo si $p_1(t)$ es bajo
- Reiniciar parcialmente las estadísticas si sospecha un cambio

Una posible estrategia es:

$$
\pi^*(s_t) = 
\begin{cases}
0 & \text{si } p_1(t) \geq \hat{p}_2 + \sqrt{\frac{2 \log t}{n_2}} \\
1 & \text{en otro caso}
\end{cases}
$$

Además, se recomienda:
- Reiniciar $n_2$, $s_2$ cada cierto número de turnos, o
- Aplicar **decaimiento exponencial** a las estadísticas anteriores para adaptarse a cambios

---

### **Escenario 3: Solo Recompensa**

#### Supuestos

El agente conoce:
- El turno actual $t$
- El historial de acciones y recompensas pasadas
- No conoce $p_1(t)$, $p_2(t)$ ni cuándo ocurren los cambios

#### Estado

$$
s_t = (t, a_1, r_1, \ldots, a_{t-1}, r_{t-1})
$$

#### Acción

$a_t \in \{0, 1\}$

#### Política Óptima

Dado que los cambios son poco frecuentes pero posibles, el agente debe:

- Explorar constantemente para detectar si una probabilidad ha cambiado
- Desconfiar de medias históricas muy antiguas

Una política adecuada es **UCB1 con ventana móvil o decaimiento**:

$$
a_t = \arg\max_{i \in \{0,1\}} \left( \hat{\mu}_i^{(w)} + \sqrt{\frac{2 \log t}{n_i^{(w)}}} \right)
$$

Donde:
- $\hat{\mu}_i^{(w)}$ es la media de las últimas $w$ observaciones del brazo $i$
- $n_i^{(w)}$ es la cantidad de observaciones en esa ventana

También puede usarse **Thompson Sampling con olvido**, es decir, actualizando distribuciones Beta pero ponderando más las observaciones recientes.

---

### Información adicional valiosa a rastrear

Para cada brazo $i$:
- Ventanas móviles de observaciones recientes $(r_{t-w}, \ldots, r_{t})$
- Estimaciones suavizadas (media móvil o exponencial)
- Conteo desde el último cambio percibido (si el agente quiere detectar rupturas)

---

#### Decisión (T Aleatorio)

### **EJERCICIO**  
**RESPUESTA**  
Definir el problema de decisión para la Banda Totalmente Aleatoria con horizonte de tiempo desconocido T ~ Uniform(1, 300). ¿Cómo interactúan las dos formas de aleatoriedad (en las probabilidades de los brazos y en el horizonte de tiempo)?
---
### Mi respuesta:
### 3.8 Entorno de Banda Totalmente Aleatoria con Horizonte Aleatorio

#### Descripción

En este entorno:
- Hay **dos brazos**: $a \in \{0, 1\}$
- Las **probabilidades iniciales** se sortean aleatoriamente:

  $$
  p_1(0),\ p_2(0) \sim \text{Uniform}(0.01, 0.99)
  $$

- En cada turno $t > 0$:
  - Con probabilidad $0.05$ (5%), **ambas probabilidades cambian**:

    $$
    p_1(t),\ p_2(t) \sim \text{Uniform}(0.01, 0.99)
    $$

  - Con probabilidad $0.95$ (95%), **se mantienen**:

    $$
    p_1(t) = p_1(t-1),\quad p_2(t) = p_2(t-1)
    $$

- El número total de turnos $T$ es **desconocido** y se distribuye uniformemente:

  $$
  T \sim \text{Uniform}(1, 300)
  $$

---

## ¿Cómo interactúan estas dos fuentes de aleatoriedad?

Hay **dos niveles de incertidumbre**:

1. **Aleatoriedad en las probabilidades**: implica que el entorno puede cambiar en cualquier turno (aunque rara vez).
2. **Aleatoriedad en el horizonte de tiempo**: el agente no sabe cuánto durará el juego.

La combinación de ambas hace que:
- **Exploraciones muy largas sean arriesgadas**, porque el juego podría terminar antes de que se aproveche lo aprendido.
- Las **observaciones viejas puedan volverse obsoletas** si ocurre un cambio en las probabilidades.
- Se requiera una estrategia que combine **exploración constante pero acotada**, y **desconfianza moderada en datos históricos**.

---

## Escenarios de Información

### **Escenario 1: Información Completa**

#### Supuestos

El agente conoce:
- El turno actual $t$
- Las probabilidades actuales $p_1(t)$, $p_2(t)$
- No conoce $T$

#### Estado

$$
s_t = (t, p_1(t), p_2(t))
$$

#### Acción

$a_t \in \{0, 1\}$

#### Política Óptima

Como se conocen las probabilidades actuales, no importa el valor de $T$ ni los cambios anteriores. La mejor decisión es:

$$
\pi^*(s_t) = \arg\max \{p_1(t), p_2(t)\}
$$

Ni el cambio de probabilidades ni el valor aleatorio de $T$ afecta esta política cuando se conoce el estado actual.

---

### **Escenario 2: Información Parcial**

#### Supuestos

El agente conoce:
- El turno actual $t$
- Solo la probabilidad $p_1(t)$
- No conoce $p_2(t)$ ni $T$
- No sabe cuándo ocurren cambios

#### Estado

$$
s_t = (t, p_1(t), n_2, s_2)
$$

#### Acción

$a_t \in \{0, 1\}$

#### Política Óptima

Aquí el agente debe:
- Tomar decisiones basadas en **comparar $p_1(t)$ con la estimación actual de $p_2$**
- Tener en cuenta que la información sobre $p_2$ **puede quedar obsoleta** si hubo un cambio reciente
- No puede planear a largo plazo, ya que **$T$ es incierto**

Se recomienda:
- Aplicar **UCB con ventana o con decaimiento exponencial** para el brazo desconocido
- Tomar decisiones por:

$$
\pi^*(s_t) = 
\begin{cases}
0 & \text{si } p_1(t) \geq \hat{p}_2 + \sqrt{\frac{1.5 \log t}{n_2}} \\
1 & \text{en otro caso}
\end{cases}
$$

Reiniciar la estadística de $p_2$ si se detecta que su rendimiento cambió repentinamente.

---

### **Escenario 3: Solo Recompensa**

#### Supuestos

El agente conoce:
- El turno actual $t$
- Solo el historial de acciones y recompensas
- No conoce $p_1(t)$, $p_2(t)$ ni $T$

#### Estado

$$
s_t = (t, a_1, r_1, \ldots, a_{t-1}, r_{t-1})
$$

#### Acción

$a_t \in \{0, 1\}$

#### Política Óptima

La estrategia debe:
- Estar preparada para **cambios impredecibles**, aunque poco frecuentes
- Tener en cuenta que **el tiempo es limitado y desconocido**
- Usar aprendizaje que valore **más las observaciones recientes**

Opciones recomendadas:
- **UCB con ventana móvil**:

  $$
  a_t = \arg\max_{i \in \{0, 1\}} \left( \hat{\mu}_i^{(w)} + \sqrt{\frac{2 \log t}{n_i^{(w)}}} \right)
  $$

- **Thompson Sampling con decaimiento** (ponderando más las recompensas recientes)

Esto permite:
- Reaccionar rápidamente a cambios de probabilidad
- No sobre-explorar por si $T$ es bajo

---

## Conclusión

La interacción entre:
- **Aleatoriedad en las probabilidades** (cambios súbitos pero poco frecuentes)
- **Horizonte aleatorio** (duración impredecible del juego)

hace necesario un enfoque que:
- **Explore de forma constante pero moderada**
- **Aprenda de lo reciente**
- **Sea conservador en la planificación**

En resumen: **sí cambia la estrategia óptima**, pero no de forma radical. Más bien, **afina el balance entre explorar y explotar**.

---
## 4. Implementación de Agentes

En nuestro entorno, implementarás tres tipos de agentes correspondientes a los tres escenarios de información descritos anteriormente. Esto es lo que cada agente debe manejar:

### 4.1 Agente de Información Completa

**Entrada:**
```python
env_info = {
    'current_turn': int,        # Número de turno actual
    'total_turns': int,         # Número total de turnos en el juego
    'p1': float,                # Probabilidad de recompensa del brazo 1
    'history': {
        'actions': [int, ...],   # Acciones pasadas (0 para brazo 1, 1 para brazo 2)
        'rewards': [float, ...], # Recompensas pasadas
        'p1': [float, ...],      # Historial de probabilidades del brazo 1
        'p2': [float, ...]       # Historial de probabilidades del brazo 2 (solo para evaluación)
    }
}
```

**Salida:**
```python
action = 0 or 1  # 0 para el brazo 1, 1 para el brazo 2
```

### 4.2 Agente de Información Parcial

**Entrada:**
```python
env_info = {
    'current_turn': int,        # Número de turno actual
    'total_turns': int,         # Número total de turnos en el juego
    'p1': float,                # Probabilidad de recompensa del brazo 1
    'history': {
        'actions': [int, ...],   # Acciones pasadas (0 para brazo 1, 1 para brazo 2)
        'rewards': [float, ...]  # Recompensas pasadas
    }
}
```

**Salida:**
```python
action = 0 or 1  # 0 para el brazo 1, 1 para el brazo 2
```

### 4.3 Agente de Solo Recompensa

**Entrada:**
```python
env_info = {
    'current_turn': int,        # Número de turno actual
    'history': {
        'actions': [int, ...],   # Acciones pasadas (0 para brazo 1, 1 para brazo 2)
        'rewards': [float, ...]  # Recompensas pasadas
    }
}
```

**Salida:**
```python
action = 0 or 1  # 0 para el brazo 1, 1 para el brazo 2
```

## 5. Métricas de Rendimiento

El entorno evalúa el rendimiento de los agentes usando varias métricas clave:

### 5.1 Recompensa Promedio

Esta es la recompensa media obtenida por turno, calculada como:

$\text{Recompensa Promedio} = \frac{1}{T} \sum_{t=1}^{T} r_t$

Esta métrica mide directamente qué tan bien el agente está maximizando su función objetivo. Valores más altos indican un mejor rendimiento.

### 5.2 Porcentaje de Acciones Óptimas

Esta métrica mide el porcentaje de veces que el agente seleccionó el brazo con la mayor probabilidad de recompensa:

$\text{Acciones Óptimas (\%)} = \frac{100}{T} \sum_{t=1}^{T} \mathbf{1}\{a_t = \arg\max_i p_i(t)\}$

Donde $\mathbf{1}$ es la función indicadora que vale 1 cuando la condición es verdadera y 0 en caso contrario.

Esta métrica muestra con qué frecuencia el agente elige el mejor brazo, independientemente de la recompensa real recibida. Valores más altos indican una mejor selección de brazos.

### 5.3 Arrepentimiento (Regret)

El arrepentimiento mide la diferencia entre la recompensa esperada de elegir siempre el brazo óptimo y la recompensa esperada de las elecciones del agente:

$\text{Regret} = \sum_{t=1}^{T} \max_i p_i(t) - \sum_{t=1}^{T} p_{a_t+1}(t)$

Valores más bajos de arrepentimiento indican un mejor rendimiento.

### 5.4 Distribución de Recompensas

El entorno visualiza la distribución de recompensas en diferentes entornos usando diagramas de caja (boxplots) y diagramas de violín (violin plots). Estas visualizaciones ayudan a entender:
- La mediana del rendimiento
- La variabilidad en el rendimiento
- La presencia de valores atípicos
- La forma general de la distribución de recompensas

## 6. Pautas de Estrategia

### 6.1 Enfoques Generales

Aquí hay algunos enfoques generales a considerar para la implementación de tus agentes:

1. **Selección Aleatoria**: Elegir brazos aleatoriamente (enfoque de referencia).
2. **Greedy (Codicioso)**: Elegir siempre el brazo con la recompensa estimada más alta.
3. **ε-Greedy**: Casi siempre elegir el mejor brazo, pero explorar ocasionalmente.
4. **UCB (Upper Confidence Bound)**: Elegir brazos basados en estimaciones optimistas de su valor.
5. **Thompson Sampling**: Elegir brazos basados en emparejar probabilidades con distribuciones a posteriori.
6. **Enfoques Bayesianos**: Mantener distribuciones de probabilidad sobre los valores de los brazos.

### 6.2 Consideraciones Específicas del Entorno

#### Banda Fija
- Enfocarse en identificar rápidamente el mejor brazo.
- La exploración se vuelve menos valiosa conforme avanza el juego.
- Con T conocido, se puede planificar un programa decreciente de exploración.

#### Banda Periódica
- Detectar la estructura periódica (k=10).
- Restablecer estimaciones al comienzo de cada período.
- Asignar más exploración al inicio de cada período.

#### Banda Dinámica
- Las observaciones recientes valen más que las antiguas.
- Considerar el uso de una ventana deslizante de observaciones.
- Podría necesitar alta capacidad de respuesta a los cambios.

#### Banda Totalmente Aleatoria
- Estar alerta a cambios repentinos en los patrones de recompensa.
- Equilibrar la persistencia (usar historial) con la adaptabilidad.
- Considerar métodos de detección de cambios.

### 6.3 Consideraciones Específicas de la Información

#### Agente de Información Completa
- Aprovechar el valor conocido p1.
- Enfocarse en estimar p2 con eficiencia.
- Ajustar la estrategia dinámicamente con base en los valores relativos.

#### Agente de Información Parcial
- Similar a información completa, pero más limitado.
- Podría requerir más exploración en ciertos entornos.

#### Agente de Solo Recompensa
- Debe estimar las probabilidades de ambos brazos.
- Necesita lidiar con el horizonte de tiempo desconocido.
- Considerar estrategias adaptativas en el tiempo.

## 7. Conclusión

El problema de Multi-Bandas ofrece un marco fundamental para estudiar la toma de decisiones secuenciales bajo incertidumbre. Los entornos y escenarios de información en este playground brindan un conjunto rico de desafíos que resaltan diferentes aspectos del dilema exploración-explotación.

Al implementar agentes para estos escenarios, obtendrás experiencia práctica con conceptos clave en aprendizaje por refuerzo y teoría de la decisión, y desarrollarás intuición para equilibrar la recolección de información con la maximización de recompensas en diversos contextos.

Mientras trabajas en tus implementaciones, considera cómo se extenderían tus estrategias a:
- Bandas con más de dos brazos.
- Espacios de acción continuos.
- Distribuciones de recompensa no estacionarias con diferentes patrones.
- Bandas contextuales donde se dispone de información adicional.

