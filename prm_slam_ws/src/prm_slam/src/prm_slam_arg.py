#!/usr/bin/python3

import subprocess
import sys
from math import sin, cos, pi, sqrt
import math
import rospy
import cv2
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import networkx as nx

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import tf
from nav_msgs.msg import OccupancyGrid

from scipy.spatial import KDTree
from heapq import heappush, heappop
import time

import random
import sys, select, os
if os.name == 'nt':
  import msvcrt
else:
  import tty, termios


from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


V_max = 0.22; W_max = 2.84; 
num=1147



def publish_markers(self):
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.punto_desado_xh
        marker.pose.position.y = self.punto_desado_yh
        marker.pose.position.z = 0
        
        markerArray.markers.append(marker)
        self.publisher.publish(markerArray)


def verificar_region_blanca(imagen_binaria, x_pixeles, y_pixeles):
    if 0 <= x_pixeles < imagen_binaria.shape[1] and 0 <= y_pixeles < imagen_binaria.shape[0]:
        if imagen_binaria[int(y_pixeles), int(x_pixeles)] == 255:
            imagen_binaria[int(y_pixeles), int(x_pixeles)] == 0
            return True  # El punto esta en una region blanca
    return False  # El punto esta fuera de la imagen o en una region negra

def metros_a_pixeles(self, x, y):  
    pixel_x=self.pixel_origen_x+round(x/self.resx)
    pixel_y=self.pixel_origen_y-round(y/self.resy)
    return pixel_x, pixel_y


def pixeles_a_metros(self, pixel_x, pixel_y):
    y=(-self.origen_y+self.y_max+self.origen_global_y)
    metros_x =  self.origen_global_x + (pixel_x) * self.resx 
    metros_y =  y-(pixel_y) * self.resy     
    return metros_x, metros_y


def generar_puntos_aleatorios_radio(self, imagen_binaria, imagen_color):
    pixeles_aleatorios = []
    self.distancia_minima = 40
    self.distancia_min_puntos_ruta = 15
    pcm = None
    distancia_pcm = None
    
    self.puntos_guardados_pixeles= [metros_a_pixeles(self, px, py) for px, py in self.centros]
    self.puntos_ruta_pixeles= [metros_a_pixeles(self, px, py) for px, py in self.ruta]

    for (px, py) in self.puntos_guardados_pixeles:
        imagen_color[py, px] = [255, 105, 180]

    for _ in range(self.nodos):
        while True:
            rand_x = random.randint(0, imagen_binaria.shape[1] - 1)
            rand_y = random.randint(0, imagen_binaria.shape[0] - 1)
            # Verifica si el pixel en las coordenadas es blanco
            if imagen_binaria[rand_y, rand_x] == 255:
                # Calcula la distancia mínima con todos los puntos guardados
                distancias = [distancia_entre_puntos((rand_x, rand_y), (y,x)) for y,x in self.puntos_guardados_pixeles]
                distancia_2= [distancia_entre_puntos((rand_x, rand_y), (y,x)) for y,x in self.puntos_ruta_pixeles]
                # Verifica si la distancia mínima es al menos 40 píxeles de cada punto guardado
                if all(distancia >= self.distancia_minima for distancia in distancias) and all(distancia >= self.distancia_min_puntos_ruta for distancia in distancia_2):
                    # El punto es válido
                    imagen_color[rand_y, rand_x] = [0, 100, 255]  # Amarillo
                    pixeles_aleatorios.append((rand_x, rand_y))
                    x_metros, y_metros = pixeles_a_metros(self, rand_x, rand_y)
                    distancia = distancia_entre_puntos((x_metros, y_metros), (self.meta_x, self.meta_y))
                    
                    if pcm is None or distancia < distancia_pcm:
                        pcm = (rand_x, rand_y)
                        distancia_pcm = distancia
                    break

    if pcm is not None:
        imagen_color[pcm[1], pcm[0]] = [0, 255, 0]  # Verde

    return pcm


def generar_puntos_aleatorios_al_no_encotrar_ruta(self, imagen_binaria, imagen_color):
    pixeles_aleatorios = []
    distancia_pcm_na = None
    self.distancia_minima = 60
    self.distancia_min_puntos_ruta = 15
    pcm_na = None#Punto cercano a la meta no alcanzable

    self.puntos_guardados_pixeles= [metros_a_pixeles(self, px, py) for px, py in self.centros]
    self.puntos_ruta_pixeles= [metros_a_pixeles(self, px, py) for px, py in self.ruta]



    for (px, py) in self.puntos_guardados_pixeles:
        imagen_color[py, px] = [255, 105, 180]
    for _ in range(self.nodos):
        while True:
            rand_x = random.randint(0, imagen_binaria.shape[1] - 1)
            rand_y = random.randint(0, imagen_binaria.shape[0] - 1)
            # Verifica si el pixel en las coordenadas es blanco
            if imagen_binaria[rand_y, rand_x] == 255:
                # Calcula la distancia mínima con todos los puntos guardados
                distancias_meta = [distancia_entre_puntos((rand_x, rand_y), (y,x)) for y,x in self.meta]
                distancias = [distancia_entre_puntos((rand_x, rand_y), (y,x)) for y,x in self.puntos_guardados_pixeles]
                distancia_2= [distancia_entre_puntos((rand_x, rand_y), (y,x)) for y,x in self.puntos_ruta_pixeles]
                # Verifica si la distancia mínima es al menos 40 píxeles de cada punto guardado
                if all(distancia >= self.distancia_minima for distancia in distancias) and all(distancia >= self.distancia_min_puntos_ruta for distancia in distancia_2) and all(distancia >= self.distancia_minima for distancia in distancias_meta):
                
                
                    imagen_color[rand_y, rand_x] = [0, 100, 255]  # Amarillo
                    pixeles_aleatorios.append((rand_x, rand_y))
                    x_metros, y_metros = pixeles_a_metros(self, rand_x, rand_y)
                    x_meta, y_meta= pixeles_a_metros(self, self.meta[0][0], self.meta[0][1])
                    distancia = distancia_entre_puntos((x_metros, y_metros), (x_meta,y_meta))
                    
                    if pcm_na is None or distancia < distancia_pcm_na:
                        pcm_na = (rand_x, rand_y)
                        distancia_pcm_na = distancia
                    break

    if pcm_na is not None:
        imagen_color[pcm_na[1], pcm_na[0]] = [0, 255, 0]  # Verde

    return pcm_na



    



def distancia_entre_puntos(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def hay_obstaculo_entre_puntos(punto1, punto2, imagen_binaria):
    # Verificar si hay obstáculos entre dos puntos utilizando la línea entre ellos
    linea = np.linspace(punto1, punto2, num=10).astype(int)
    for punto in linea:
        if imagen_binaria[punto[0], punto[1]] == 0:
            return True
    return False

def generar_muestras_validas(imagen_binaria, numero_de_muestras):
    # Obtener las coordenadas de los puntos blancos en la imagen binaria
    puntos_validos = np.column_stack(np.where(imagen_binaria == 255))  # Puntos blancos
    
    # Seleccionar aleatoriamente un número específico de muestras
    indices_muestras = np.random.choice(len(puntos_validos), numero_de_muestras, replace=False)
    muestras = puntos_validos[indices_muestras]
    
    # Convertir la imagen binaria a una imagen en color (BGR)
    imagen_color = cv2.cvtColor(imagen_binaria, cv2.COLOR_GRAY2BGR)
    
    # Marcar los puntos de muestra en amarillo (255, 255, 0) en la imagen en color
    for (y, x) in muestras:
        imagen_color[y, x] = [0, 255, 255]  # Color amarillo en formato BGR
        
    return muestras


def shortest(graph, start_pixel, goal_pixel):
    start_node = tuple(start_pixel)
    goal_node = tuple(goal_pixel)
    
    distancia = {node: float('inf') for node in graph.nodes}
    parent = {node: None for node in graph.nodes}
    
    distancia[start_node] = 0
    
    queue = [(0, start_node)]
    
    while queue:
        current_distance, current_node = min(queue, key=lambda x: x[0])
        queue.remove((current_distance, current_node))
        
        for neighbor in graph.neighbors(current_node):
            if distancia[current_node] + 1 < distancia[neighbor]:
                distancia[neighbor] = distancia[current_node] + 1
                parent[neighbor] = current_node
                queue.append((distancia[neighbor], neighbor))
    
    if distancia[goal_node] == float('inf'):
        return float('inf'), None
    camino = obten_camino(parent, goal_node)

    return camino

def obten_camino(parent, final):
    camino = []
    while final:
        camino.append(final)
        final = parent[final]
    return camino[::-1]

def get_occupancy(image, pos):
    # Verificar si el píxel está dentro de los límites de la imagen
    if 0 <= pos[1] < image.shape[0] and 0 <= pos[0] < image.shape[1]:
        # Obtener el valor del píxel en la posición dada
        pixel_value = image[pos[1], pos[0]]
        # Devuelve 0 si el píxel es blanco (espacio libre), 1 si no lo es.
        return 0 if np.all(pixel_value == 255) else 1
    return 1


def dibujar_grafo(imagen_color, grafo, puntos):
    # Copiar la imagen original para dibujar sobre ella
    imagen_dibujo = imagen_color.copy()
    # Dibujar nodos
    for punto in puntos:
        cv2.circle(imagen_dibujo, (int(punto[1]), int(punto[0])), 5, (0, 255, 255), -1)  # Amarillo
    # Dibujar aristas
    for (p1, p2, data) in grafo.edges(data=True):
        cv2.line(imagen_dibujo, (int(p1[1]), int(p1[0])), (int(p2[1]), int(p2[0])), (255, 0, 0), 1)  # Rojo
    return imagen_dibujo

def encontrar_pixel_blanco_mas_cercano(image, x, y):
    # Verificar si el píxel está en una zona blanca
    if image[x,y] == 255:
        return x, y#, True    
    radio = 2
    while True:
        for i in range(-radio, radio + 1):
            for j in range(-radio, radio + 1):
                x_vecino = x + i
                y_vecino = y + j
                if 0 <= x_vecino < image.shape[0] and 0 <= y_vecino < image.shape[1]:
                    if image[x_vecino, y_vecino] == 255:
                        # Mostrar el punto blanco encontrado en la imagen
                        image[x_vecino,y_vecino] = 180                       
                        return x_vecino, y_vecino#, True   
        radio += 1
        # Si no se encontró un píxel blanco en el radio actual, continuar con un mayor radio de búsqueda.
        if radio > max(image.shape):
            break
    # Si no se encontró ningún píxel blanco cercano, devolver el punto original y bandera falsa.
    return x, y#, False

def prm(self, imagen, imagen_reducida, im_bin,  imagen_color, numero_de_muestras, distancia_max, punto_inicio, punto_destino,carpeta):
    path = None
    self.meta=[]
    print("generanding")
    self.Velocidad_Lineal=0
    self.Velocidad_Angular=0
    self.velocidad.linear.x = self.Velocidad_Lineal 
    self.velocidad.angular.z = self.Velocidad_Angular
    self.pub.publish(self.velocidad)

    
    img_color = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    img_color[punto_inicio] = [255, 0, 0]  # Blue
    img_color[punto_destino] = [0, 0, 255]  # Red
    ruta_completa = os.path.join(carpeta, 'inicio_fin.jpg')
    cv2.imwrite(ruta_completa, img_color)
    

    
    # Bucle para intentar encontrar una ruta
    while path is None:
        print("generating while")
         # Verificar y ajustar el punto de inicio
        x_start, y_start = encontrar_pixel_blanco_mas_cercano(imagen_reducida, punto_inicio[0], punto_inicio[1])
        punto_inicio = (x_start,y_start)

        x_end, y_end = encontrar_pixel_blanco_mas_cercano(imagen_reducida, punto_destino[0], punto_destino[1])
        punto_destino = (x_end,y_end)
        self.pcm=punto_destino
        self.pcm_x, self.pcm_y = pixeles_a_metros(self, self.pcm[1], self.pcm[0])

        img_color[punto_inicio] = [0, 255, 0]  # Green
        img_color[punto_destino] = [255, 255, 0]  # Cian

        print(punto_inicio, punto_destino)
        print("inicio, fin")
        

        grafo = nx.Graph()
        muestras_validas = generar_muestras_validas(imagen, numero_de_muestras)
        img_color[punto_inicio] = [255, 255, 0]  # Amarillo
        img_color[punto_destino] = [0, 255, 255]  # Cian

        muestras = np.vstack([muestras_validas, punto_inicio, punto_destino])  # Agregar puntos de inicio y destino
        kdtree = KDTree(muestras)
        for i, punto in enumerate(muestras):
            # Encontrar vecinos dentro de la distancia máxima
            distancias, vecinos_indices = kdtree.query(punto, k=25, distance_upper_bound=distancia_max)
            for vecino_indice, distancia in zip(vecinos_indices, distancias):
                if distancia < distancia_max and not hay_obstaculo_entre_puntos(punto, muestras[vecino_indice], im_bin):
                    grafo.add_edge(tuple(punto), tuple(muestras[vecino_indice]), weight=distancia)

        # Dibuja la ruta en la imagen original.
        img_color = cv2.cvtColor(im_bin, cv2.COLOR_GRAY2BGR)
        imagen_con_ruta = img_color.copy()  # Copia la imagen original.
        imagen_con_ruta_2 = img_color.copy()  # Copia la imagen original.
        """
        im=dibujar_grafo(imagen_color, grafo, muestras)
        nombre_imagen = "grafo" + str(self.num_new_img)
        self.num_new_img += 1
        nombre_imagen_con_extension = nombre_imagen + ".jpg"
        ruta_completa = os.path.join(carpeta, nombre_imagen_con_extension)
            

        cv2.imwrite(ruta_completa, im)
        """

        try:
            if nx.has_path(grafo, tuple(punto_inicio), tuple(punto_destino)):
                ruta_optima = nx.shortest_path(grafo, source=tuple(punto_inicio), target=tuple(punto_destino), weight='weight')
                print(len(ruta_optima))

                punto_inicio = pixeles_a_metros(self, punto_inicio[0], punto_inicio[1])
                punto_destino = pixeles_a_metros(self, punto_destino[0], punto_destino[1])
                distancia_total = sqrt((punto_inicio[0] - punto_destino[0])**2 + (punto_inicio[1] - punto_destino[1])**2)

                path = ruta_optima
                for i in range(len(path) - 1):
                    point1 = (path[i][1], path[i][0])  # Intercambiar coordenadas xy a yx
                    point2 = (path[i + 1][1], path[i + 1][0])  # Intercambiar coordenadas xy a yx
                    cv2.line(imagen_con_ruta, point1, point2, (0, 0, 255), 1)  # Dibuja la línea roja entre nodos de la ruta.

                # Dibuja los puntos aleatorios en la imagen
                if self.centros is not None:
                    for punto_guardado in self.centros:
                        xs, ys = metros_a_pixeles(self, punto_guardado[0], punto_guardado[1])
                        x_pixel = xs
                        y_pixel = ys

                        if self.ruta is not None:
                            for punto_guardado1 in self.ruta:
                                xs, ys = metros_a_pixeles(self, punto_guardado1[0], punto_guardado1[1])
                                x_pixel_r = xs
                                y_pixel_r = ys
                                imagen_color[y_pixel_r, x_pixel_r] = [0, 150, 0]  # Verde

                        imagen_con_ruta[y_pixel, x_pixel] = [255, 0, 255]  # Magenta
                
                nombre_imagen = "imagen_con_ruta" + str(self.numero_de_imagen)
                nombre_imagen_con_extension = nombre_imagen + ".png"
                ruta_completa = os.path.join(carpeta, nombre_imagen_con_extension)
                cv2.imwrite(ruta_completa, imagen_con_ruta)  # Guarda la imagen con la ruta y los puntos aleatorios dibujados.
                ruta_completa = os.path.join(carpeta, "imagen_ruta.png")
                cv2.imwrite(ruta_completa, imagen_con_ruta)
                print(path)
                self.contador=self.contador+1
                return ruta_optima
            path = None

            #self.meta_anterior = punto_destino
            punto_destino = (y_end,x_end)
            self.meta.append(punto_destino)

            punto_destino = generar_puntos_aleatorios_al_no_encotrar_ruta(self, imagen, imagen_color)
            self.pcm = punto_destino
            self.pcm_x, self.pcm_y = pixeles_a_metros(self, self.pcm[0], self.pcm[1])
            
            imagen_color[punto_destino[1], punto_destino[0]] = [150, 255, 0]  # Marca el nuevo destino en la imagen
            y=punto_destino[1]
            x=punto_destino[0]
            punto_destino=[y,x]

            nombre_imagen = "new_meta" + str(self.num_new_img)
            self.num_new_img += 1
            nombre_imagen_con_extension = nombre_imagen + ".jpg"
            ruta_completa = os.path.join(carpeta, nombre_imagen_con_extension)
            

            cv2.imwrite(ruta_completa, imagen_color)

        
        except Exception as e:
            print(f"Error al encontrar la ruta: {e}")
            path = None
            punto_destino = (y_end,x_end)
            self.meta.append(punto_destino)

            punto_destino = generar_puntos_aleatorios_al_no_encotrar_ruta(self, imagen, imagen_color)
            self.pcm = punto_destino
            self.pcm_x, self.pcm_y = pixeles_a_metros(self, self.pcm[0], self.pcm[1])
            self.pcm_en_metros = pixeles_a_metros(self, self.pcm[0], self.pcm[1])
            
            imagen_color[punto_destino[1], punto_destino[0]] = [150, 255, 0]  # Marca el nuevo destino en la imagen
            y=punto_destino[1]
            x=punto_destino[0]
            punto_destino=[y,x]

            nombre_imagen = "new_meta_2_" + str(self.num_new_img)
            self.num_new_img += 1
            nombre_imagen_con_extension = nombre_imagen + ".jpg"
            ruta_completa = os.path.join(carpeta, nombre_imagen_con_extension)
            

            cv2.imwrite(ruta_completa, imagen_color)


           

def control_velocidad(self):
    k=1; 
    self.h=0.1    
    tetha=self.yaw

    self.xh = self.x+self.h*cos(tetha)
    self.yh = self.y+self.h*sin(tetha)

    ex = self.xh-self.punto_desado_xh;  ey = self.yh-self.punto_desado_yh
    Ux = -k*ex;  Uy =-k*ey
    self.Velocidad_Lineal= Ux*cos(tetha)+Uy*sin(tetha)
    self.Velocidad_Angular=-Ux*sin(tetha)/self.h+Uy*cos(tetha)/self.h
    
    #Evitar la saturacion en las velocidades 
    if (abs(self.Velocidad_Lineal)>V_max):
        self.Velocidad_Lineal = V_max*abs(self.Velocidad_Lineal)/self.Velocidad_Lineal
    if (abs(self.Velocidad_Angular)>W_max):
        self.Velocidad_Angular = W_max*abs(self.Velocidad_Angular)/self.Velocidad_Angular


class Nodo(object):
    def __init__(self):
        self.loop_rate = rospy.Rate(30)

        #Variables
        self.x = None
        self.velocidad=Twist()
        self.Velocidad_Lineal=0
        self.Velocidad_Angular=0
        self.punto_desado_xh=None
        self.pcm_x=0; self.pcm_y=0
        self.xh=0
        self.yh=0
        self.nodos=30
        self.nodos=100
        self.contador=0
        self.limite=20
        
        self.path=None
        self.centros = []  # Vector para almacenar los puntos generados
        self.ruta = []
        self.contador=0
        self.bandera_espacios=False
        self.limite=40
        self.lectura=0
        self.pixeles_anterior=0
        self.map_file = "map" 
        self.numero_de_imagen=0
        self.num_new_img=0
        self.fin=False
        self.meta=[]


        # Subscribirse
        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)        
        odom_sub =rospy.Subscriber('/odom',Odometry, self.Callback)
        self.ph_pub =rospy.Publisher('/punto_h',Odometry, queue_size=10)
        self.odom_msg = Odometry()
        self.pub = rospy.Publisher('/cmd_vel',Twist,queue_size=10)
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        


    def Callback(self,msg):
        #Posicion
        self.x=msg.pose.pose.position.x
        self.y=msg.pose.pose.position.y
        print("X={0} Y={1}".format(self.x,self.y))
        print("Xh={0} Yh={1}".format(self.xh,self.yh))
        print("Xhd={0} Yhd={1}".format(self.pcm_x,self.pcm_y))
        #Inicializa velocidad
        self.velocidad=Twist()
        #Orientacion
        quater = msg.pose.pose.orientation
        quater_list = [quater.x, quater.y, quater.z, quater.w]
        (roll, pitch, self.yaw) = euler_from_quaternion(quater_list)
        #Funcion de control
        control_velocidad(self)
        publish_markers(self)
    
    def map_callback(self, msg):
        # Establecer un flag para indicar que estamos procesando el mapa
        self.processing_map = True

        try:
            # Obtener la resolución, origen y dimensiones del mapa
            resolution = msg.info.resolution
            origen_x = msg.info.origin.position.x
            self.origen_y = msg.info.origin.position.y
            self.width = msg.info.width
            self.height = msg.info.height

            # Dimensiones del mapa en metros
            self.x_min = origen_x
            self.x_max  = origen_x + (self.width * resolution)
            self.y_min = self.origen_y
            self.y_max = self.origen_y + (self.height * resolution)

            # Crear un listener de tf
            listener = tf.TransformListener()

            # Esperar hasta que la transformación de /map a /base_link esté disponible
            listener.waitForTransform('/map', '/base_link', rospy.Time(0), rospy.Duration(3.0))
            
            # Obtener la transformación de /map a /base_link
            (trans, rot) = listener.lookupTransform('/map', '/base_link', rospy.Time(0))
            
            # La posición del robot en el marco de referencia /map
            self.robot_x = trans[0]
            self.robot_y = trans[1]
            
            
            if self.lectura == 0:
                self.punto_desado_xh = self.robot_x
                self.punto_desado_yh = self.robot_y

                self.pcm_x = self.robot_x
                self.pcm_y = self.robot_y

                self.inicio_x = self.robot_x
                self.inicio_y = self.robot_y

                self.pcm_en_metros=[self.robot_x,self.robot_y]
                self.path_metros=[(self.robot_x,self.robot_y),(self.robot_x,self.robot_y)]
                

            # Esperar hasta que la transformación de /map a /odom esté disponible
            listener.waitForTransform('/map', '/odom', rospy.Time(0), rospy.Duration(1.0))
            
            # Obtener la transformación de /map a /odom
            (trans, rot) = listener.lookupTransform('/odom', '/map', rospy.Time(0))
            
            # Calcular el Local Origin en el marco global /odom
            self.origen_global_x = origen_x + trans[0]
            self.origen_global_y = self.origen_y + trans[1]

            self.resx = (abs(self.x_min) + abs(self.x_max )) / self.width
            self.resy = (abs(self.y_min) + abs(self.y_max)) / self.height


            self.pixel_origen_x=round(abs(self.origen_global_x/self.resx))
            self.pixel_origen_y=self.height-round(abs(self.origen_global_y/self.resy))
            
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr("Error de transformación: %s", str(e))
        
        finally:
            # Marcar que hemos terminado de procesar el mapa
            self.processing_map = False


    def start(self,arg1,arg2,arg3):
        #self.meta_x=16.5#31#8
        #self.meta_y=17.5#-6.5#-11
        self.meta_x=float(arg1)
        self.meta_y=float(arg2)
        random.seed(20)
        rospy.loginfo("Comenzando codigo")
        numero_de_muestras=2000+100*int(arg3)
        distancia_k_vecinos=10
        self.nodos=100*int(arg3)
        

        nombre_carpeta = "Prueba" + '_'+str(arg3)+'_'+str(arg1)+'_'+str(arg2)
        # Obtén la carpeta personal del usuario
        carpeta_raiz = os.path.expanduser("~")

        # Define la ruta a la carpeta "pruebas"
        carpeta_pruebas = os.path.join(carpeta_raiz, 'Pruebas')
        carpeta1 = os.path.join(carpeta_pruebas, 'Prueba_metas4')
        carpeta = os.path.join(carpeta1, nombre_carpeta)

        
        if not os.path.exists(carpeta):
            os.makedirs(carpeta)
        
        while not rospy.is_shutdown() and self.fin is False:
            if self.x is not None and self.punto_desado_xh is not None:
                
                #Asigna la velocidad y el angulo del robot
                self.velocidad.linear.x = self.Velocidad_Lineal 
                self.velocidad.angular.z = self.Velocidad_Angular
                self.pub.publish(self.velocidad)
                #Publicación de ph en RVIZ
                self.odom_msg.pose.pose.position.x = self.xh
                self.odom_msg.pose.pose.position.y = self.yh
                self.ph_pub.publish(self.odom_msg)


                distancia_meta = sqrt((self.x - self.meta_x)**2 + (self.y - self.meta_y)**2)
                distancia_pcm = sqrt((self.x - self.pcm_x)**2 + (self.y - self.pcm_y)**2)
                if distancia_meta < 0.12:
                    print("METAAAAAAAAAAAAAAAAAAAAAAAA")
                    cmd = ["rosrun", "map_server", "map_saver", "-f", ruta_completa]
                    proc = subprocess.Popen(cmd)
                    proc.wait()
                    
                    # Carga la imagen del mapa
                    imagen_original = cv2.imread(ruta_completa+ '.pgm', cv2.IMREAD_GRAYSCALE)
                    umbral = 240  
                    _, imagen_binaria = cv2.threshold(imagen_original, umbral, 255, cv2.THRESH_BINARY)
                    
                    # Guardar la imagen binarizada
                    nombre_imagen = "imagen_binarizada_final" + str(self.numero_de_imagen)
                    nombre_imagen_con_extension = nombre_imagen + ".png"
                    ruta_completa = os.path.join(carpeta, nombre_imagen_con_extension)

                    cv2.imwrite(ruta_completa, imagen_binaria)
                    self.Velocidad_Lineal=0
                    self.Velocidad_Angular=0
                    self.velocidad.linear.x = self.Velocidad_Lineal 
                    self.velocidad.angular.z = self.Velocidad_Angular
                    self.pub.publish(self.velocidad)
                    self.fin=True

                # Distancia meta o
                if distancia_meta < 0.65:
                    self.punto_desado_xh=self.meta_x
                    self.punto_desado_yh=self.meta_y
                # Distancia de tolerancia al punto mas cercano a la meta 
                elif distancia_pcm < 0.2:
                    
                 #Obtener imagen del mapa
                    print("PCM")
                    
                    numero_de_muestras=numero_de_muestras+50
                    #distancia_k_vecinos=distancia_k_vecinos+1
                    ruta_completa = os.path.join(carpeta, self.map_file)
                
                    cmd = ["rosrun", "map_server", "map_saver", "-f", ruta_completa]
                    proc = subprocess.Popen(cmd)
                    proc.wait()
                    
                    # Carga la imagen del mapa
                    imagen_original = cv2.imread(ruta_completa+ '.pgm', cv2.IMREAD_GRAYSCALE)
                    umbral = 240  
                    _, imagen_binaria = cv2.threshold(imagen_original, umbral, 255, cv2.THRESH_BINARY)
                    
                    # Guardar la imagen_original binarizada
                    nombre_imagen = "imagen_binarizada" + str(self.numero_de_imagen)
                    nombre_imagen_con_extension = nombre_imagen + ".png"
                    ruta_completa = os.path.join(carpeta, nombre_imagen_con_extension)

                    cv2.imwrite(ruta_completa, imagen_binaria)                       

                    self.lectura=1
                    self.nodos=self.nodos+5
                    self.nodos=self.nodos+100
                    if self.contador > self.limite:
                        self.fin=True
                    
                    print(self.contador)
                    print("contador")

                    kernel = np.ones((9,9),np.uint8)
                    img_bin = cv2.erode(imagen_binaria,kernel,iterations = 1)

                    kernel = np.ones((10,10),np.uint8)
                    img = cv2.erode(imagen_binaria,kernel,iterations = 1)

                    kernel = np.ones((11,11),np.uint8)
                    imagen_reducida = cv2.erode(imagen_binaria,kernel,iterations = 1)

                    imagen_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    imagen_color2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    imagen_origen = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    sp,j=metros_a_pixeles(self,self.robot_x,self.robot_y)

                    imagen_origen[self.pixel_origen_y,self.pixel_origen_x] = [0, 0, 255]  # Morado

                    ruta_completa = os.path.join(carpeta, "inicio.jpg")
                    cv2.imwrite(ruta_completa, imagen_origen)
                                        
                    xd,yd=metros_a_pixeles(self,self.meta_x,self.meta_y)
                    print(xd,yd)

                    if verificar_region_blanca(img,xd,yd):
                        
                       # try:
                            # Si esta en un area blanca, simplemente devuelve el punto deseado  
                            self.pcm_x=self.meta_x
                            self.pcm_y=self.meta_y
                            #xs,ys=metros_a_pixeles(self, self.x, self.y)
                            xs,ys=metros_a_pixeles(self, self.x, self.y)
                            start=(ys,xs)
                            print(xs,ys)
                            xg,yg=metros_a_pixeles(self, self.meta_x, self.meta_y)
                            
                            goal=(yg,xg)

                            imagen_origen[start] = [0, 0, 255]  # Morado
                            imagen_origen[goal] = [255,0,0]

                            self.numero_de_imagen = self.numero_de_imagen + 1
                            self.centros.append(self.pcm_en_metros)
                            self.ruta.extend(self.path_metros)

                            nombre_imagen = "imagen_origen" + str(self.numero_de_imagen)
                            nombre_imagen_con_extension = nombre_imagen + ".png"
                            ruta_completa = os.path.join(carpeta, nombre_imagen_con_extension)
                            cv2.imwrite(ruta_completa, imagen_origen)

                            self.path=prm(self, img,imagen_reducida, img_bin, imagen_color, numero_de_muestras, distancia_k_vecinos, start, goal,carpeta)
                            self.path_metros= [pixeles_a_metros(self, px, py) for py, px in self.path]
                            print(self.path)
                            bandera=0
                        #except:
                         #   print("No hubo ruta")
                        
                    else:
                        #try:
                            xs,ys=metros_a_pixeles(self, self.punto_desado_xh, self.punto_desado_yh)
                            start=(ys,xs)
                            print("No blanco")
                            self.meta=[]
                            self.centros.append(self.pcm_en_metros)
                            self.ruta.extend(self.path_metros)


                            imagen_origen[ys,xs] = [0, 100, 255]  # Morado

                            self.pcm = generar_puntos_aleatorios_radio(self, imagen_reducida, imagen_color)
                            
                            self.pcm_x, self.pcm_y = pixeles_a_metros(self, self.pcm[0], self.pcm[1])
                            
                            goal=(self.pcm[1], self.pcm[0])


                            imagen_origen[start] = [0, 0, 255]  # Morado
                            imagen_origen[goal] = [255,0,0]

                            self.numero_de_imagen = self.numero_de_imagen + 1
                            
                            self.path=prm(self, img,imagen_reducida, img_bin, imagen_color, numero_de_muestras, distancia_k_vecinos, start, goal,carpeta)
                            nombre_imagen = "imagen_" + str(self.numero_de_imagen)
                            nombre_imagen_con_extension = nombre_imagen + ".png"
                            ruta_completa = os.path.join(carpeta, nombre_imagen_con_extension)
                            
                            cv2.imwrite(ruta_completa, imagen_color)
                            self.pcm_en_metros = pixeles_a_metros(self, self.pcm[1], self.pcm[0])
                            print("METROOOOOOOOOOOOOOOS")
                            print(self.pcm_en_metros)
                            bandera=0
                            self.path_metros= [pixeles_a_metros(self, px, py) for py, px in self.path]
                            
                            print(self.path_metros)
                        #except:
                         #   print("No hubo ruta")
                        
                    
                    nombre_imagen = "imagen_" + str(self.numero_de_imagen)
                    nombre_imagen_con_extension = nombre_imagen + ".png"
                    ruta_completa = os.path.join(carpeta, nombre_imagen_con_extension)
                        
                    cv2.imwrite(ruta_completa, imagen_color)

                    
                    
                distancia_control = sqrt((self.x - self.punto_desado_xh)**2 + (self.y - self.punto_desado_yh)**2)

                # Distancia de tolerancia al punto deseado 
                if distancia_control < 0.13:
                    sp,j=metros_a_pixeles(self,-1,2)
                    rp,t=pixeles_a_metros(self,sp,j)

                    if bandera == (len(self.path_metros)-1):
                        self.Velocidad_Lineal=0
                        self.Velocidad_Angular=0
                        self.velocidad.linear.x = self.Velocidad_Lineal 
                        self.velocidad.angular.z = self.Velocidad_Angular
                        self.pub.publish(self.velocidad)
                        print("pausaaaaaaaaaaa")
                        #time.sleep(5)

                    if bandera < len(self.path_metros):
                        self.punto_desado_xh= self.path_metros[bandera][0]
                        self.punto_desado_yh = self.path_metros[bandera][1]
                        
                        bandera=bandera+1
                        print("posicion xd")
                        print(self.punto_desado_xh, self.punto_desado_yh)
                    
                    
                control_velocidad(self)
                
            self.loop_rate.sleep()


if __name__ == '__main__':
    rospy.init_node("PRM_SLAM", anonymous=True)


    if os.name != 'nt':
        settings = termios.tcgetattr(sys.stdin)

    try:
        my_node = Nodo()
        if len(sys.argv) < 3:
            print("usage: my_node.py arg1=x arg2=y arg3=num_prueba")
        else:
            #my_node(sys.argv[1], sys.argv[2])
            my_node.start(sys.argv[1], sys.argv[2], sys.argv[3])
    except rospy.ROSInterruptException:
        pass
    
    
