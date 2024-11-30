# Prm-slam
Navegación Autónoma en Línea: Integración de PRM y SLAM para la Exploración de Entornos Desconocidos

Este trabajo presenta una integración de los algoritmos SLAM y PRM para la navegación en entornos interiores y de grandes dimensiones. El sistema opera en dos modos principales: *Modo meta definida y *Modo exploración*.

# PRM_SLAM_Control_cinemático

Si se desea consultar información del [Entorno](https://github.com/itzchav/Entorno-Division-Estudios-Posgrado/tree/main)

La ejecución del entorno con Gmapping se realiza con los siguientes comandos:

```shell
cd ~/catkin_ws
source ./devel/setup.bash
export TURTLEBOT3_MODEL=burger
roslaunch turtlebot3_gazebo plano_completo_gmapping_noetic.launch 
```

## Online



## Diagrama PRM-SLAM: Modo meta definida
<p align="center">
    <img width=50% src="https://github.com/itzchav/PRM-y-SLAM-Exploracion-de-Entornos/blob/main/Diagrama_meta_definida.png">
</p>

## Diagrama Navegación
El diagrama para navegar una vez que se establece una ruta
<p align="center">
    <img width=50% src="https://github.com/itzchav/PRM-y-SLAM-Exploracion-de-Entornos/blob/main/Diagrama_navegacion.png">
</p>

### Pruebas PRM-SLAM: Modo meta definida
<p align="center">
    <img width=60% src="https://github.com/itzchav/PRM-y-SLAM-Exploracion-de-Entornos/blob/main/Pruebas.png">
</p>

## Diagrama PRM-SLAM: Modo exploración
<p align="center">
    <img width=50% src="https://github.com/itzchav/PRM-y-SLAM-Exploracion-de-Entornos/blob/main/Diagrama_exploracion.png">
</p>

### Pruebas PRM-SLAM: Modo exploración
<p align="center">
    <img width=60% src="https://github.com/itzchav/PRM-y-SLAM-Exploracion-de-Entornos/blob/main/Pruebas_explore.png">
</p>

Para ejecutar PRM-SLAM: Modo meta definida:
```shell
cd ~/prm_slam_ws/
source devel/setup.bash 
rosrun prm_slam prm_slam.py
```

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

Comando para la obtención del mapa como imagen:
```shell
cd ~/catkin_ws
source ./devel/setup.bash
rosrun map_server map_saver -f map
```

Código para obtener del mapa como imagen binarizada:
```shell
cd prm_slam_ws/codigos_offline 
python3 rviz_binarizado.py
```

Código para obtener la trayectoria con PRM:
```shell
cd prm_slam_ws/codigos_offline 
python3 Prm_code.py
```


Código para recorrer la trayectoria:
```shell
cd ~/prm_slam_ws/
source devel/setup.bash 
rosrun prm_slam prm_slam_offline.py 
```

://github.com/itzchav/PRM-y-SLAM-Exploracion-de-Entornos/blob/main/Pruebas_explore.png">
</p>
