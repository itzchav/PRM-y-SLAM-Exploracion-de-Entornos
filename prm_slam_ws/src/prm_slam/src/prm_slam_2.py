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
from sensor_msgs.msg import LaserScan
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
        marker.pose.position.x = self.Xhd
        marker.pose.position.y = self.Yhd
        marker.pose.position.z = 0
        
        markerArray.markers.append(marker)
        self.publisher.publish(markerArray)


def verificar_region_blanca(imagen_gris, x_pixeles, y_pixeles):

    if 0 <= x_pixeles < imagen_gris.shape[1] and 0 <= y_pixeles < imagen_gris.shape[0]:
        if imagen_gris[int(y_pixeles), int(x_pixeles)] == 255:
            
            return True  # El punto esta en una region blanca
    return False  # El punto esta fuera de la imagen o en una region negra

def metros_a_pixeles(self, x, y):
    pix_origen_x=round(abs(self.origin_global_x/self.resx))#width-round(abs(origin_x/resx))
    pix_origen_y=self.height-round(abs(self.origin_global_y/self.resy))#height-round(abs(origin_y/resy))
    
    print("metros a pixels")
    print(pix_origen_x,self.pix_origen_y)
    print(round(abs(y/self.resy)))

    pixel_x=pix_origen_x-round(abs(x/self.resx))
    pixel_y=self.pix_origen_y-round(abs(y/self.resy))

    
    return pixel_x, pixel_y


def pixeles_a_metros(self, pixel_x, pixel_y):
    # Calcula las coordenadas en metros a partir de las coordenadas en pixeles
    pix_origen_x=round(abs(self.origin_global_x/self.resx))#width-round(abs(origin_x/resx))
    pix_origen_y=round(abs(self.origin_global_y/self.resy))#height-round(abs(origin_y/resy))

    

    print(self.origin_global_x,self.origin_global_y)
    #print("pixel")
    print(pixel_x,pixel_y)
    #print((pixel_y) * self.resy )
    y=(-self.origin_y+self.y_max+self.origin_global_y)
    #print(y)

    metros_x =  self.origin_global_x + (pixel_x) * self.resx 
    metros_y =  y-(pixel_y) * self.resy 
    
    
    
    print("Coordenadas en metros del punto:", metros_x, metros_y)
    
    return metros_x, metros_y

def generar_puntos_aleatorios_radio(self, imagen_gris, imagen_color):
    pixeles_aleatorios = []
    self.distancia_minima = 30

    closest_pixel = None
    closest_distance = None

    if self.contador > self.limite:
        self.puntos_guardados.append(self.nuevo_pd)
        print("puntos_guardados--")
        print(self.contador)
        print(self.puntos_guardados)

    for _ in range(self.nodos):
        while True:
            rand_x = random.randint(0, imagen_gris.shape[1] - 1)
            rand_y = random.randint(0, imagen_gris.shape[0] - 1)

            # Verifica si el pixel en las coordenadas es blanco
            if imagen_gris[rand_y, rand_x] == 255:
                # Calcula la distancia mínima con todos los puntos guardados
                distancias = [distancia_entre_puntos((rand_x, rand_y), punto_guardado) for punto_guardado in self.puntos_guardados]

                # Verifica si la distancia mínima es al menos 40 píxeles de cada punto guardado
                if all(distancia >= self.distancia_minima for distancia in distancias):
                    # El punto es válido
                    imagen_color[rand_y, rand_x] = [0, 255, 255]  # Amarillo
                    pixeles_aleatorios.append((rand_x, rand_y))

                    x_metros, y_metros = pixeles_a_metros(self, rand_x, rand_y)
                    distancia = distancia_entre_puntos((x_metros, y_metros), (self.pd_x, self.pd_y))

                    if closest_pixel is None or distancia < closest_distance:
                        closest_pixel = (rand_x, rand_y)
                        closest_distance = distancia

                    break

    if closest_pixel is not None:
        imagen_color[closest_pixel[1], closest_pixel[0]] = [0, 160, 160]  # Gris

    return closest_pixel





def generar_puntos_aleatorios_2(self,imagen_gris, imagen_color):
    pixeles_aleatorios = []
    closest_pixel = None
    closest_distance = None
    
    for _ in range(self.nodos):
        while True:
            # Genera coordenadas aleatorias dentro de la imagen

            rand_x = random.randint(0, imagen_gris.shape[1] - 1)
            rand_y = random.randint(0, imagen_gris.shape[0] - 1)

            # Verifica si el pixel en las coordenadas es blanco (puedes ajustar el criterio de blanco segun tu necesidad)
            if imagen_gris[rand_y, rand_x] == 255:
                # Pone un punto morado en las coordenadas generadas
                imagen_color[rand_y, rand_x] = [255, 0, 255]  # Morado
                pixeles_aleatorios.append((rand_x, rand_y))
                
                # Convierte las coordenadas del pixel al sistema de metros
                x_metros, y_metros = pixeles_a_metros(self,rand_x, rand_y)

                # Calcula la distancia entre el punto generado y el objetivo (3, 5)
                distancia = distancia_entre_puntos((x_metros, y_metros), (self.pd_x,self.pd_y))
                
                # Actualiza el pixel mas cercano
                if closest_pixel is None or distancia < closest_distance:
                    closest_pixel = (rand_x, rand_y)
                    closest_distance = distancia

                break

    # Cambia el color del pixel mas cercano a verde
    if closest_pixel is not None:
        imagen_color[closest_pixel[1], closest_pixel[0]] = [0, 255, 0]  # Verde

    return  closest_pixel

def distancia_entre_puntos(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def hay_obstaculo_entre_puntos(punto1, punto2, imagen_binaria):
    # Verificar si hay obstáculos entre dos puntos utilizando la línea entre ellos
    linea = np.linspace(punto1, punto2, num=10).astype(int)
    for punto in linea:
        if imagen_binaria[punto[0], punto[1]] == 0:
            return True
    return False

def generar_muestras_validas(imagen_binaria, num_muestras):
    puntos_validos = np.column_stack(np.where(imagen_binaria == 255))  # Puntos no blancos
    indices_muestras = np.random.choice(len(puntos_validos), num_muestras, replace=False)
    return puntos_validos[indices_muestras]


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
    
    distancia_inicio_a_fin = distancia[goal_node]
    camino = obten_camino(parent, goal_node)

    return camino

def obten_camino(parent, final):
    camino = []
    while final:
        camino.append(final)
        final = parent[final]
    return camino[::-1]

	
	
def prm(self,image, image_color, num_muestras, distancia_max, punto_inicio, punto_destino):
    path=None
    
    points = []  # Lista para almacenar los puntos aleatorios
 
# Close all windows
    cv2.destroyAllWindows()
    imagen_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    imagen_color[punto_inicio] = [255, 0, 0]  # Morado
    imagen_color[punto_destino] = [0, 0, 255]  # Morado
    while path is None:
        print(punto_inicio, punto_destino)
        ##Codigo PRM
        cv2.imwrite('inicio_fin.jpg', imagen_color)
        
        
        grafo = nx.Graph()
        muestras_validas = generar_muestras_validas(image, num_muestras)

        muestras = np.vstack([muestras_validas, punto_inicio, punto_destino])  # Agregar puntos de inicio y destino
        kdtree = KDTree(muestras)
        for i, punto in enumerate(muestras):
        # Encontrar vecinos dentro de la distancia máxima
            distancias, vecinos_indices = kdtree.query(punto, k=10, distance_upper_bound=distancia_max)
            for vecino_indice, distancia in zip(vecinos_indices, distancias):
            		if distancia < distancia_max and not hay_obstaculo_entre_puntos(punto, muestras[vecino_indice], image):
                		grafo.add_edge(tuple(punto), tuple(muestras[vecino_indice]), weight=distancia)
        # Dibuja la ruta en la imagen original.
        image_with_path = image_color.copy()  # Copia la imagen original.
        
        if nx.has_path(grafo, punto_inicio, punto_destino):
            ruta_optima = shortest(grafo, tuple(punto_inicio), tuple(punto_destino))
            print(len(ruta_optima))
            pi=pixeles_a_metros(self,punto_inicio[0],punto_inicio[1])
            pf=pixeles_a_metros(self,punto_destino[0],punto_destino[1])
            distancia_total = sqrt((pi[0] - pf[0])**2 + (pi[1] - pf[1])**2)
            print(pi, pf, distancia_total)
            pixeles_blancos = np.sum(image == 255)
            print(pixeles_blancos)
            porcentaje_diferencia = ((pixeles_blancos - self.pixeles_anterior) / self.pixeles_anterior) * 100
            print(porcentaje_diferencia)
            if porcentaje_diferencia<10:
                self.contador=self.contador+1
                print(self.contador)
            if self.contador>self.limite:
                self.bandera_espacios=True
                print(self.contador)
            path=ruta_optima
            for i in range(len(path) - 1):
                point1 = (path[i][1], path[i][0])  # Intercambiar coordenadas xy a yx
                point2 = (path[i + 1][1], path[i + 1][0])  # Intercambiar coordenadas xy a yx
                cv2.line(image_with_path, point1, point2, (0, 0, 255), 1)  # Dibuja la linea roja entre nodos de la ruta.
            # Dibuja los puntos aleatorios en la imagen
            if self.puntos_guardados is not None:
                for punto_guardado in self.puntos_guardados:
                    x_pixel = punto_guardado[0]
                    y_pixel = punto_guardado[1]
                    image_with_path[y_pixel, x_pixel] = [0, 0, 255]  # Azul

            nombre_imagen = "imagen_con_ruta" + str(self.num_img)
            nombre_imagen_con_extension = nombre_imagen + ".png"

            cv2.imwrite(nombre_imagen_con_extension, image_with_path)  # Guarda la imagen con la ruta y los puntos aleatorios dibujados.
            cv2.imwrite("imagen_ruta.png", image_with_path)
            print(path)
            self.pixeles_anterior=pixeles_blancos
            return ruta_optima
    

def control_velocidad(self):
    k=1; 
    self.h=0.1    
    tetha=self.yaw

    self.xh = self.x+self.h*cos(tetha)
    self.yh = self.y+self.h*sin(tetha)

    
   
    ex = self.xh-self.Xhd;  ey = self.yh-self.Yhd
    Ux = -k*ex;  Uy =-k*ey
    self.V= Ux*cos(tetha)+Uy*sin(tetha)
    self.W=-Ux*sin(tetha)/self.h+Uy*cos(tetha)/self.h
    
    #Evitar la saturacion en las velocidades 
    if (abs(self.V)>V_max):
        self.V = V_max*abs(self.V)/self.V
        #print("Saturacion en V\t")
    if (abs(self.W)>W_max):
        self.W = W_max*abs(self.W)/self.W
        #print("Saturacion en W\t")


class Nodo(object):
    def __init__(self):
        self.loop_rate = rospy.Rate(30)

        #Variables
        self.x = None
        self.velocidad=Twist()
        self.V=0
        self.W=0
        self.Xhd=None
        self.Xhd1=0; self.Yhd1=0
        self.xh=0
        self.yh=0
        #self.pix_origen_x=0
        #self.pix_origen_y=0
        self.nodos=0
        
        self.path=None
        self.puntos_guardados = []  # Vector para almacenar los puntos generados
        self.contador=0
        self.bandera_espacios=False
        self.limite=2
        self.lectura=0


        # Subscribirse
        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)        
        odom_sub =rospy.Subscriber('/odom',Odometry, self.Callback)
        self.ph_pub =rospy.Publisher('/punto_h',Odometry, queue_size=10)
        self.odom_msg = Odometry()
        self.pub = rospy.Publisher('/cmd_vel',Twist,queue_size=10)
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        
        
        #Inicializar Tiempo
        self.t0 = rospy.Time.now().to_sec()
        self.map_file = "map" 
        self.i=0
        self.pixeles_anterior=0
        self.xs=np.zeros(num)
        self.ys=np.zeros(num)
        self.num_img=0
        
        


    def Callback(self,msg):
        #Posicion
        self.x=msg.pose.pose.position.x
        self.y=msg.pose.pose.position.y
        #print("X={0} Y={1}".format(self.x,self.y))
        #print("Xh={0} Yh={1}".format(self.Xhd,self.Yhd))
        #Inicializa velocidad
        self.velocidad=Twist()
        #Orientacion
        quater = msg.pose.pose.orientation
        quater_list = [quater.x, quater.y, quater.z, quater.w]
        (roll, pitch, self.yaw) = euler_from_quaternion(quater_list)
        #Tiempo
        self.time = rospy.Time.now().to_sec()-self.t0
        #Funcion de control
        control_velocidad(self)
        publish_markers(self)
    
    def map_callback(self, msg):
        # Establecer un flag para indicar que estamos procesando el mapa
        self.processing_map = True

        try:
            # Obtener la resolución, origen y dimensiones del mapa
            resolution = msg.info.resolution
            origin_x = msg.info.origin.position.x
            self.origin_y = msg.info.origin.position.y
            self.width = msg.info.width
            self.height = msg.info.height

            # Dimensiones del mapa en metros
            self.x_min = origin_x
            self.x_max  = origin_x + (self.width * resolution)
            self.y_min = self.origin_y
            self.y_max = self.origin_y + (self.height * resolution)

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
                rospy.loginfo("Ya asigné")
                self.Xhd = self.robot_x
                self.Yhd = self.robot_y

                self.Xhd1 = self.robot_x
                self.Yhd1 = self.robot_y
            
            rospy.loginfo("Dimensiones del mapa en metros:")
            rospy.loginfo(f"x_min: {self.x_min:.2f} m")
            rospy.loginfo(f"x_max: {self.x_max :.2f} m")
            rospy.loginfo(f"y_min: {self.y_min:.2f} m")
            rospy.loginfo(f"y_max: {self.y_max:.2f} m")

            rospy.loginfo("Dimensiones del mapa en píxeles:")
            rospy.loginfo(f"Ancho: {self.width} píxeles")
            rospy.loginfo(f"Alto: {self.height} píxeles")

            # Esperar hasta que la transformación de /map a /odom esté disponible
            listener.waitForTransform('/map', '/odom', rospy.Time(0), rospy.Duration(1.0))
            
            # Obtener la transformación de /map a /odom
            (trans, rot) = listener.lookupTransform('/odom', '/map', rospy.Time(0))
            
            # Calcular el Local Origin en el marco global /odom
            self.origin_global_x = origin_x + trans[0]
            self.origin_global_y = self.origin_y + trans[1]

            rospy.loginfo("Local Origin in World (odom frame):")
            rospy.loginfo(f"x: {self.origin_global_x:.2f} m")
            rospy.loginfo(f"y: {self.origin_global_y:.2f} m")
            self.resx = (abs(self.x_min) + abs(self.x_max )) / self.width
            self.resy = (abs(self.y_min) + abs(self.y_max)) / self.height
            print(self.height)


            self.pix_origen_x=round(abs(self.origin_global_x/self.resx))#width-round(abs(origin_x/resx))
            self.pix_origen_y=self.height-round(abs(self.origin_global_y/self.resy))#height-round(abs(origin_y/resy))
    
            print(self.pix_origen_x, self.pix_origen_y)
        
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr("Error de transformación: %s", str(e))
        
        finally:
            # Marcar que hemos terminado de procesar el mapa
            self.processing_map = False


    

    def start(self):
        self.pd_x=-9
        self.pd_y=21
        random.seed(20)
        rospy.loginfo("Comenzando codigo")
        
        

        while not rospy.is_shutdown():
            if self.x is not None and self.Xhd is not None:
                


                #Asigna la velocidad y el angulo del robot
                self.velocidad.linear.x = self.V 
                self.velocidad.angular.z = self.W
                self.pub.publish(self.velocidad)
                #Publicación de ph en RVIZ
                self.odom_msg.pose.pose.position.x = self.xh
                self.odom_msg.pose.pose.position.y = self.yh
                self.ph_pub.publish(self.odom_msg)


                distancia_meta = sqrt((self.x - self.pd_x)**2 + (self.y - self.pd_y)**2)
                distancia_pcm = sqrt((self.x - self.Xhd1)**2 + (self.y - self.Yhd1)**2)
                
                
                # Distancia meta 
                if distancia_meta < 0.65:
                    self.Xhd=self.pd_x
                    self.Yhd=self.pd_y

                # Distancia de tolerancia al punto mas cercano a la meta 
                elif distancia_pcm < 0.25:
                 #Obtener imagen del mapa
                    cmd = ["rosrun", "map_server", "map_saver", "-f", self.map_file]
                    proc = subprocess.Popen(cmd)
                    proc.wait()
                    
                    # Carga la imagen del mapa
                    imagen = cv2.imread(self.map_file + '.pgm', cv2.IMREAD_GRAYSCALE)
                    umbral = 240  
                    _, imagen_binaria = cv2.threshold(imagen, umbral, 255, cv2.THRESH_BINARY)
                    
                    # Guardar la imagen binarizada
                    nombre_imagen = "imagen_binarizada" + str(self.num_img)
                    nombre_imagen_con_extension = nombre_imagen + ".png"
                    cv2.imwrite(nombre_imagen_con_extension, imagen_binaria)                       

                    self.lectura=1
                    print("resolucion")
                    self.nodos=self.nodos+5
                    


                    kernel = np.ones((11,11),np.uint8)
                    img = cv2.erode(imagen_binaria,kernel,iterations = 1)

                    imagen_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    imagen_origen = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    #print(self.pix_origen_y,self.pix_origen_x)
                    print("s,j")

                    sp,j=metros_a_pixeles(self,self.robot_x,self.robot_y)

                    sp,j=metros_a_pixeles(self,-1,2)
                    print("0,0")
                    imagen_origen[self.pix_origen_y,self.pix_origen_x] = [255, 0, 0]  # Morado
                    cv2.imwrite('inicio.jpg', imagen_origen)
                    cv2.imshow('imagen_2',imagen_origen)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    print(sp,j)
                    print("-1,2")
                    imagen_origen[j,sp] = [255, 0, 0]  # Morado
                    rp,t=pixeles_a_metros(self,sp,j)
                    print("r,t")
                    print(rp,t)



                    cv2.imwrite('inicio.jpg', imagen_origen)
                    cv2.imshow('imagen_2',imagen_origen)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    
                    
                    xd,yd=metros_a_pixeles(self,self.pd_x,self.pd_y)

                    s,r = pixeles_a_metros(self, xd, yd)

                    if verificar_region_blanca(img,xd,yd):
                        print('punto d')
                    # Si esta en un area blanca, simplemente devuelve el punto deseado  
                        self.Xhd1=self.pd_x
                        self.Yhd1=self.pd_y
                        xs,ys=metros_a_pixeles(self, self.x, self.y)
                        start=(ys,xs)
                        print("Metros a pixeles")
                        print(xs,ys)
                        xg,yg=metros_a_pixeles(self, self.pd_x, self.pd_y)
                        num_samples=2000
                        k_neighbors=150
                        goal=(yg,xg)
                        self.num_img = self.num_img + 1

                        self.path=prm(self, img, imagen_color, num_samples, k_neighbors, start, goal)
                        print(self.path)
                        bandera=0
                        
                    else:
                        xs,ys=metros_a_pixeles(self, self.x, self.y)
                        start=(ys,xs)
                        print("Metros a pixeles")
                        print(xs,ys)
                        if self.bandera_espacios is not True:
                            self.nuevo_pd = generar_puntos_aleatorios_2(self, img, imagen_color)
                        else:
                            self.nuevo_pd = generar_puntos_aleatorios_radio(self, img, imagen_color)
                        #print(self.nuevo_pd[1], self.nuevo_pd[0])
                        print("prm_2")
                        self.Xhd1, self.Yhd1 = pixeles_a_metros(self, self.nuevo_pd[0], self.nuevo_pd[1])
                        num_samples=2000
                        k_neighbors=150
                        goal=(self.nuevo_pd[1], self.nuevo_pd[0])
                        self.num_img = self.num_img + 1
                        
                        self.path=prm(self, img, imagen_color, num_samples, k_neighbors, start, goal)
                        bandera=0
                        self.path_metros= [pixeles_a_metros(self, py, px) for px, py in self.path]
                        print(self.path_metros)
                    
                    
                    nombre_imagen = "imagen_" + str(self.num_img)
                    nombre_imagen_con_extension = nombre_imagen + ".png"

                    nombre_imagen_slam = "imagen_slam" + str(self.num_img)
                    nombre_imagen_con_extension_slam = nombre_imagen_slam + ".png"
                        
                    cv2.imwrite(nombre_imagen_con_extension_slam, imagen_color)
                    
                    #cv2.imwrite(nombre_imagen_con_extension, img_slam)
                distancia_control = sqrt((self.x - self.Xhd)**2 + (self.y - self.Yhd)**2)

                # Distancia de tolerancia al punto deseado 
                if distancia_control < 0.2:
                    sp,j=metros_a_pixeles(self,-1,2)
                    rp,t=pixeles_a_metros(self,sp,j)
                    print("r,t")
                    print(rp,t)
                    print(bandera)
                    print(len(self.path_metros))
                    
                    if bandera < len(self.path_metros):
                        #self.Xhd, self.Yhd = pixeles_a_metros(self,self.path[bandera][1],self.path[bandera][0])
                        self.Xhd, self.path_metros[bandera][0]
                        self.Yhd = self.path_metros[bandera][1]
                        bandera=bandera+1
                        print(self.Xhd, self.Yhd)                    
                    
                control_velocidad(self)
                    
                #print('***********************************************')

                
            self.loop_rate.sleep()


if __name__ == '__main__':
    rospy.init_node("PRM_SLAM", anonymous=True)


    if os.name != 'nt':
        settings = termios.tcgetattr(sys.stdin)

    try:
        my_node = Nodo()
        my_node.start()
    except rospy.ROSInterruptException:
        pass
    
    
