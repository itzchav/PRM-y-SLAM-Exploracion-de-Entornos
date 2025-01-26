# PRM-SLAM
Navegación Autónoma en Línea: Integración de PRM y SLAM para la Exploración de Entornos Desconocidos

Este trabajo presenta una integración de los algoritmos SLAM y PRM para la navegación en entornos interiores y de grandes dimensiones. El sistema opera en dos modos principales: *Modo meta definida* y *Modo exploración*.

## Contenido Digital

- Vídeos
  - Modo meta definida
     https://www.youtube.com/watch?v=03RZZ6ntFaE
    
      - **0:00** Presentación de PRM-SLAM: Modo Meta definida, mostrando el funcionamiento del algoritmo.
      - **2:33** Navegación hacia un mismo punto objetivo considerando diferentes sistemas de coordenadas globales.
      - **12:34** Navegación con diferentes puntos de inicio y final hacia la misma meta.
      - **15:55** Navegación asistida, empleando el mapa previamente generado para asignar nuevas metas.
      - **16:50** Implementación del método en el Turtlebot 2 real.
  -  Modo exploración
     https://www.youtube.com/watch?v=GfzVuH_KnKs
-  Diagramas
-  Pseudocódigos



### Entorno División de Estudios de Posgrado de la UTM


Si se desea consultar información del [Entorno](https://github.com/itzchav/Entorno-Division-Estudios-Posgrado/tree/main)

La ejecución del entorno con Gmapping se realiza con los siguientes comandos:

```shell
cd ~/catkin_ws
source ./devel/setup.bash
export TURTLEBOT3_MODEL=burger
roslaunch turtlebot3_gazebo plano_completo_gmapping_noetic.launch 
```

# Ejecución

## Online

###  PRM-SLAM: Modo meta definida


Para ejecutar PRM-SLAM: Modo meta definida:
```shell
cd ~/prm_slam_ws/
source devel/setup.bash 
rosrun prm_slam prm_slam.py
```
###  PRM-SLAM: Modo meta exploración

Para ejecutar PRM-SLAM: Modo exploración:
```shell
cd ~/prm_slam_ws/
source devel/setup.bash 
rosrun prm_slam prm_slam_explore_2..py
```


## Offline
Para la ejecución del modo offline:
Teleoperación del robot:
```shell
cd ~/catkin_ws
source ./devel/setup.bash
export TURTLEBOT3_MODEL=burger
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
```

Comandos para la obtención del mapa como imagen:
```shell
cd ~/catkin_ws
source ./devel/setup.bash
rosrun map_server map_saver -f map
```

Comandos para obtener del mapa como imagen binarizada:
```shell
cd prm_slam_ws/codigos_offline 
python3 rviz_binarizado.py
```

Comandos para obtener la trayectoria con PRM:
```shell
cd prm_slam_ws/codigos_offline 
python3 Prm_code.py
```


Comandos para recorrer la trayectoria:
```shell
cd ~/prm_slam_ws/
source devel/setup.bash 
rosrun prm_slam prm_slam_offline.py 
```
<p align="center">
    <img width=60% src="https://github.com/itzchav/PRM-y-SLAM-Exploracion-de-Entornos/blob/main/offline.png">
</p>

