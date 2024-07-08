
# Resultados:

Como resultados tenemos 3 predictores independientes entre si que pueden complementarse para tomar decisiones sobre el problema original, todos ellos han sido testados contra conjunto de imágenes para rastrear estos,
todos los conjuntos utilizados se trata de tener la mayoría de las imagenes desde la perspectiva de un paso de peatones, todos han dado una precisión superior al 90% lo cual dado que se utiliza un modelo el cual está diseñado para dispositivos de bajos recursos computacionales, además del escaso tiempo para poder profundizar en los detalles de cada predictor. 

Después de la investigación y experimentación llevada a cabo sobre este punto en particular del problema de cruce de calles para personas con dificultades visuales concluimos que es probable mediante recursos a la mano de un ciudadano medio poder tener mayor autonomía mediante el uso de soluciones de visión computacional, hemos podido afirmar que aunque YOLOV8n tiene buenos resultados para generalizar con relativamente pequeños volumenes de datos, puede existir complicaciones, en caso de tener una mala selección del dataset, así como la aplicación como K-fold para poder sortear las dificultades de la poca disponibilidad de datos no es necesario altos números, con k=3 y k=5 ha arrojado muy buenos resultados, dado los escasos recursos, puede ser util cambiar la perspectiva hacia que predecir como fue en el caso de los cruces de peatones los cuales haciendo que detectara solo porciones de este arrojaba mayor probabilidad así como existía un descenso del nivel de Falsos Positivos los cuales es nuestro objetivo disminuir tanto como nos sea posible.

# Aplicación Final

Para poder dar una aproximación a una solución del subproblema hemos tomado los tres modelos y creado una serie de reglas sencillas donde establece que se tiene que cumplir al mismo tiempo.

Se aconseja cruzar, extremando las precauciones en caso de :
-  Existencia del cruce peatonal,luz verde y la no existencia de  ningun 
     auto sobre el cuadro delimitador del cruce de peatones. 

Se aconseja repetir la captura en caso de (Si en varios intentos no se a  logrado respuesta contundente, pedir ayuda) :
- Existencia de cruce peatonal, luz amarilla o roja o existencia de un vehiculo de grandes dimensiones en el cruce.

- No existencia del cruce de peatones o luces de peatones 



## Discusión de los resultados:

En nuestras pruebas sintécticas basándonos en un pequeño dataset creado para ello, el cual tiene True o False para si se puede realizar o no el cruce. 

################M Mete aca la matriz de confución

El resultado de analizar unas 64 imagenes distintas nos ha llevado a la conclusión que:
- Hemos logrado el objetivo de que nuestros predictores funcionen como un todo, minimizando los casos de los Falsos Positivos, en nuestro experimento no encontramos ninguno, ni en pruebas dirigidas donde entre el equipo conociendo las deficiencias de cada predictor tratamos de buscar escenarios complejos para ello y aun ha sido resistente.

- Este sistema es muy primitivo aun y necesita muchos hiperparámetros a ajustar para poder ser una aplicación usable dado que aun mantenemos una tasa de 48% respecto a la predicción de no cruzar cuando se podía.


## Conclusiones :











