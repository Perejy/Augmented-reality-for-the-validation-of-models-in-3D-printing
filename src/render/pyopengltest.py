from OpenGL.GL import *
from OpenGL.GLU import * 
from OpenGL.GLUT import * 
import sys

def draw(): 
    glutWireTeapot(0.5) 
    glFlush()

def main():
    glutInit(sys.argv) 
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB) 
    glutCreateWindow("My First OGL Program") 
    glutDisplayFunc(draw) 
    glutMainLoop()