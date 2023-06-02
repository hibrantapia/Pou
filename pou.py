import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0.0, 1.0, 1000)

p0 = np.array([0.0, 0.0])
p1 = np.array([0.5, 1.0])
p2 = np.array([1.0, 0.0])

x = (1 - t)**2 * p0[0] + 2*(1 - t)*t * p1[0] + t**2 * p2[0]
y = (1 - t)**2 * p0[1] + 2*(1 - t)*t * p1[1] + t**2 * p2[1]

plt.figure(figsize = (6, 4))
plt.plot(x, y)

plt.plot([p0[0], p1[0]], [p0[1], p1[1]], 'r--')  
plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r--')  
plt.plot([p2[0], p0[0]], [p2[1], p0[1]], 'r--')  

plt.title("Bezier Curve with Convex Hull")
plt.xlabel("x")
plt.ylabel("y")

plt.show()

#--------------------------------------------------------------

t = np.linspace(0.0, 1.0, 100)


# CUERPO

p0_1 = np.array([0.0, 0.0])
p1_1 = np.array([0.5, 1.5])
p2_1 = np.array([1.0, 0.0])

p0_2 = np.array([1.0, 0.0])
p1_2 = np.array([1.5, -1.5])
p2_2 = np.array([0.5, -1.5])

p0_3 = np.array([0.5, -1.5])
p1_3 = np.array([-0.5, -1.5])
p2_3 = np.array([0.0, 0.0])

x1 = (1 - t)**2 * p0_1[0] + 2*(1 - t)*t * p1_1[0] + t**2 * p2_1[0]
y1 = (1 - t)**2 * p0_1[1] + 2*(1 - t)*t * p1_1[1] + t**2 * p2_1[1]

x2 = (1 - t)**2 * p0_2[0] + 2*(1 - t)*t * p1_2[0] + t**2 * p2_2[0]
y2 = (1 - t)**2 * p0_2[1] + 2*(1 - t)*t * p1_2[1] + t**2 * p2_2[1]

x3 = (1 - t)**2 * p0_3[0] + 2*(1 - t)*t * p1_3[0] + t**2 * p2_3[0]
y3 = (1 - t)**2 * p0_3[1] + 2*(1 - t)*t * p1_3[1] + t**2 * p2_3[1]


# SONRISA

p0_smile = np.array([0.20, -0.10])
p1_smile = np.array([0.15, -0.35])
p2_smile = np.array([0.30, -0.30])

x_smile = (1 - t)**2 * p0_smile[0] + 2*(1 - t)*t * p1_smile[0] + t**2 * p2_smile[0]
y_smile = (1 - t)**2 * p0_smile[1] + 2*(1 - t)*t * p1_smile[1] + t**2 * p2_smile[1]


# OJOS

# Izquierdo
p0_eye_left = np.array([0.3, 0.2])
p1_eye_left = np.array([0.35, 0.3])
p2_eye_left = np.array([0.4, 0.2])

x_eye_left = (1 - t)**2 * p0_eye_left[0] + 2*(1 - t)*t * p1_eye_left[0] + t**2 * p2_eye_left[0]
y_eye_left = (1 - t)**2 * p0_eye_left[1] + 2*(1 - t)*t * p1_eye_left[1] + t**2 * p2_eye_left[1]

# Derecho
p0_eye_right = np.array([0.6, 0.2])
p1_eye_right = np.array([0.65, 0.3])
p2_eye_right = np.array([0.7, 0.2])

x_eye_right = (1 - t)**2 * p0_eye_right[0] + 2*(1 - t)*t * p1_eye_right[0] + t**2 * p2_eye_right[0]
y_eye_right = (1 - t)**2 * p0_eye_right[1] + 2*(1 - t)*t * p1_eye_right[1] + t**2 * p2_eye_right[1]


plt.figure(figsize = (8, 6))

plt.plot(x1, y1, '#161513', linewidth = 2.5)
plt.plot(x2, y2, '#161513', linewidth = 2.5)
plt.plot(x3, y3, '#161513', linewidth = 2.5)

plt.plot(x_smile, y_smile, '#161513', linewidth = 2.5)  

plt.plot(x_eye_left, y_eye_left, '#161513', linewidth = 2.5)
plt.plot(x_eye_right, y_eye_right, '#161513', linewidth = 2.5)

plt.fill(np.concatenate([x1, x2[::-1], x3[::-1]]), np.concatenate([y1, y2[::-1], y3[::-1]]), '#d3a15b', edgecolor = '#161513')
plt.fill([p2_1[0], p2_2[0], p2_3[0]], [p2_1[1], p2_2[1], p2_3[1]], '#d3a15b', edgecolor = '#d3a15b')

plt.title("Hibran's and Elias's Pou")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(-0.5,1.5)

plt.show()

#--------------------------------------------------------------

# CUERPO

plt.plot([p0_1[0], p1_1[0]], [p0_1[1], p1_1[1]], 'r--')  
plt.plot([p1_1[0], p2_1[0]], [p1_1[1], p2_1[1]], 'r--')  
plt.plot([p2_1[0], p0_1[0]], [p2_1[1], p0_1[1]], 'r--')  

plt.plot([p0_2[0], p1_2[0]], [p0_2[1], p1_2[1]], 'r--')  
plt.plot([p1_2[0], p2_2[0]], [p1_2[1], p2_2[1]], 'r--') 
plt.plot([p2_2[0], p0_2[0]], [p2_2[1], p0_2[1]], 'r--') 

plt.plot([p0_3[0], p1_3[0]], [p0_3[1], p1_3[1]], 'r--')  
plt.plot([p1_3[0], p2_3[0]], [p1_3[1], p2_3[1]], 'r--') 
plt.plot([p2_3[0], p0_3[0]], [p2_3[1], p0_3[1]], 'r--') 

# SONRISA

plt.plot([p0_smile[0], p1_smile[0]], [p0_smile[1], p1_smile[1]], 'r--')  
plt.plot([p1_smile[0], p2_smile[0]], [p1_smile[1], p2_smile[1]], 'r--') 
plt.plot([p2_smile[0], p0_smile[0]], [p2_smile[1], p0_smile[1]], 'r--') 

# OJOS

plt.plot([p0_eye_left[0], p1_eye_left[0]], [p0_eye_left[1], p1_eye_left[1]], 'r--')  
plt.plot([p1_eye_left[0], p2_eye_left[0]], [p1_eye_left[1], p2_eye_left[1]], 'r--') 
plt.plot([p2_eye_left[0], p0_eye_left[0]], [p2_eye_left[1], p0_eye_left[1]], 'r--') 

plt.plot([p0_eye_right[0], p1_eye_right[0]], [p0_eye_right[1], p1_eye_right[1]], 'r--')  
plt.plot([p1_eye_right[0], p2_eye_right[0]], [p1_eye_right[1], p2_eye_right[1]], 'r--') 
plt.plot([p2_eye_right[0], p0_eye_right[0]], [p2_eye_right[1], p0_eye_right[1]], 'r--') 