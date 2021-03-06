from sympy import *
from sympy.parsing.latex import parse_latex
def pre(expr):
    print(expr)
#    print(expr.func)
    for arg in expr.args:
        pre(arg)

#Solución a derivada de y=uv+wz



salida = open("/tmp/salida.txt","w")
init_printing()
x = var('x')
u = var('u')
v = var('v')
w = var('w')
z = var('z')
u = parse_latex(r"1/x")
v = parse_latex(r"\sin(6x)")
w = parse_latex(r"x^2")
z = parse_latex(r"\cos(8x)")
du = diff(u)
dv = diff(v)
dw = diff(w)
dz = diff(z)
xp = u*v+w*z
print(xp)
pre(xp)
salida.write("$$y=%s$$<br/>\n" % latex(u*v+w*z))
salida.write("$$u=%s$$<br/>\n" % latex(u))
salida.write("$$v=%s$$<br/>\n" % latex(v))
salida.write("$$w=%s$$<br/>\n" % latex(w))
salida.write("$$z=%s$$<br/>\n" % latex(z))
#derivadas
salida.write("$$u'=%s$$<br/>\n" % latex(du))
salida.write("$$v'=%s$$<br/>\n" % latex(dv))
salida.write("$$w'=%s$$<br/>\n" % latex(dw))
salida.write("$$z'=%s$$<br/>\n" % latex(dz))

result = du*v+dv*u+dw*z+dz*w
salida.write("$$y'=(u\'v+v\'u)+(w\'z+z\'w)=[(%s) (%s) +(%s) (%s)]+[(%s) (%s)]+[(%s)(%s)]$$<br/>\n"% ( latex(du), latex(v),latex(dv), latex(u), latex(dw), latex(z),latex(dz), latex(w) ) )
salida.write("$$y'=%s$$<br/>\n" % latex(result))
print(result)
salida.close()