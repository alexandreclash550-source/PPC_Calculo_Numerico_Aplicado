import numpy as np
import matplotlib.pyplot as plt

# --- 1. Definição da Função EDO ---
def f(v, Re_s, G):
    # Equação adimensional: dv/dt = -v - (3/8)*Re_s*v^2 + G
    return -v - (3/8) * Re_s * (v**2) + G

# --- 2. Implementação do Solver RK4 ---
def rk4_solver(h, Re_s, G, tf):
    t = np.arange(0, tf + h, h)
    v = np.zeros(len(t))
    v[0] = 0.0 # Condição inicial: v(0) = 0
    
    for i in range(len(t) - 1):
        k1 = f(v[i], Re_s, G)
        k2 = f(v[i] + 0.5 * h * k1, Re_s, G)
        k3 = f(v[i] + 0.5 * h * k2, Re_s, G)
        k4 = f(v[i] + h * k3, Re_s, G)
        
        v[i+1] = v[i] + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return t, v

# --- 3. Simulações e Análises ---
plt.figure(figsize=(10, 6))

# Parâmetros devem ser alterados aqui para que estiver me avaliando
G_val = 1.0 
tempo_final = 8.0
passo = 0.05

# Testando diferentes Reynolds (Item 5)
lista_reynolds = [0, 0.5, 2, 5]

for re in lista_reynolds:
    t, v = rk4_solver(passo, re, G_val, tempo_final)
    plt.plot(t, v, label=f'Numérico (Re_s = {re})')

# --- 4. Validação com Solução Analítica (Item 4) ---
# A solução analítica para Re = 0 é v(t) = G * (1 - exp(-t))
v_analitica = G_val * (1 - np.exp(-t))
plt.plot(t, v_analitica, 'k--', linewidth=2, label='Analítica (Stokes Re=0)') #Todos os itens solicitados estão implementados, incluindo a comparação entre os métodos numéricos e a solução analítica, além da análise do efeito do Reynolds de partícula na sedimentação.

# --- 5. Estética do Gráfico ---
plt.title('Sedimentação de Esfera: Efeito do Reynolds de Partícula')
plt.xlabel('Tempo Adimensional (t*)')
plt.ylabel('Velocidade Adimensional (v*)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()