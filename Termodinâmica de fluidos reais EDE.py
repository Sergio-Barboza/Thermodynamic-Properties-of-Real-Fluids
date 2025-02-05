import numpy as np

# Função para calcular os parâmetros da equação de estado pela equação de Peng-Robinson
def calcular_ab(T, Tc, Pc, w, R):
    Tr = T / Tc
    m = 0.37464 + 1.54226 * w - 0.26992 * w**2
    alfa_T = (1 + m * (1 - np.sqrt(Tr)))**2
    a = 0.45724 * R**2 * Tc**2 / Pc * alfa_T
    b = 0.07780 * R * Tc / Pc
    return a, b

# Função para resolver a equação cúbica analiticamente e encontrar Z
def resolver_equacao_cubica_analitica(A, B):
    # Coeficientes da equação cúbica Z^3 + c2*Z^2 + c1*Z + c0 = 0
    c2 = -(1 - B)
    c1 = A - 2 * B - 3 * B**2
    c0 = -(A * B - B**2 - B**3)

    # Coeficientes para solução analítica
    Q = (3*c1 - c2**2) / 9
    R = (9*c2*c1 - 27*c0 - 2*c2**3) / 54
    D = Q**3 + R**2  # Discriminante

    if D >= 0:  # Uma raiz real e duas complexas
        S = np.cbrt(R + np.sqrt(D))
        T = np.cbrt(R - np.sqrt(D))
        Z1 = -c2/3 + (S + T)
        Z2 = -c2/3 - (S + T)/2
        Z3 = Z2  # Raízes complexas conjugadas, considere apenas a parte real
    else:  # Três raízes reais
        theta = np.arccos(R / np.sqrt(-Q**3))
        Z1 = 2 * np.sqrt(-Q) * np.cos(theta/3) - c2/3
        Z2 = 2 * np.sqrt(-Q) * np.cos((theta + 2*np.pi)/3) - c2/3
        Z3 = 2 * np.sqrt(-Q) * np.cos((theta + 4*np.pi)/3) - c2/3

    return sorted([Z1, Z2, Z3])

# Função para calcular o coeficiente de fugacidade
def coeficiente_fugacidade(Z, A, B):
    try:
        return np.exp((Z - 1) - np.log(Z - B) + (A / (2 * np.sqrt(2) * B)) * np.log((Z + (1 - np.sqrt(2)) * B) / (Z + (1 + np.sqrt(2)) * B)))
    except ValueError as e:
        print(f"Erro no cálculo da fugacidade: {e}")
        return np.nan

# Dados para o Cloreto de Hidrogênio:
Tc = 324.65  # K
Pc = 8.310  # MPa
w = 0.13154  # Fator acêntrico
R = 0.08314  # L·bar/(K·mol)

# Lista de temperaturas para as quais queremos calcular a pressão de saturação
temperaturas = [160, 180, 200, 220, 240, 260, 280, 300, 320]

# Loop sobre as temperaturas para calcular a pressão de saturação para cada uma
for T in temperaturas:
    P = 0.14  # Pressão inicial suposta em MPa

    for _ in range(100):  # Máximo de iterações
        a, b = calcular_ab(T, Tc, Pc, w, R)
        A = a * P / (R**2 * T**2)
        B = b * P / (R * T)

        Z = resolver_equacao_cubica_analitica(A, B)

        if np.any(np.isnan(Z)):
            print(f"Falha ao calcular Z para T={T} K. Pulando esta temperatura.")
            break

        Z_L = Z[0]  # Assumindo Z_L como menor raiz
        Z_V = Z[-1]  # Assumindo Z_V como maior raiz

        F_L = coeficiente_fugacidade(Z_L, A, B)
        F_V = coeficiente_fugacidade(Z_V, A, B)

        if np.isnan(F_L) or np.isnan(F_V):
            print(f"Falha ao calcular coeficiente de fugacidade para T={T} K. Pulando esta temperatura.")
            break

        # Verificar a condição de convergência
        if abs(F_V / F_L - 1) < 0.0001:
            break

        # Atualizar a pressão
        P = P * (F_L / F_V)

    P_bar = P * 10  # Convertendo MPa para bar
    print("Temperatura:", T, "K")
    print("Pressão de saturação:", round(P_bar, 6), "bar")
    print("-" * 30)


