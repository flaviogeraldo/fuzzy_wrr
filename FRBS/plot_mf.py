import matplotlib.pyplot as plt
import skfuzzy as fuzz


def plot_membership_functions(var, title):
    """
    Função para plotar as funções de pertinência de uma variável fuzzy.
    
    Args:
        var: A variável fuzzy (ex.: BufferOccupancy1 ou BufferOccupancy2).
        title: O título do gráfico.
    """
    for label in var.terms:
        mf_params = var.terms[label].mf  # Pega os parâmetros da função de pertinência
        x = var.universe  # Universo da variável
        y = fuzz.gaussmf(x, mf_params[1], mf_params[2])  # Função de pertinência ajustada
        plt.plot(x, y, label=f"{label} (mean={mf_params[1]:.2f}, sigma={mf_params[2]:.2f})")
    
    plt.title(title)
    plt.xlabel('Input value')
    plt.ylabel('Membership degree')
    plt.legend()
    plt.grid(True)
    plt.show()

def show_adjusted_membership_functions(BufferOccupancy1, BufferOccupancy2):
    """
    Exibe e plota as funções de pertinência ajustadas.
    
    Args:
        BufferOccupancy1: Variável fuzzy 1.
        BufferOccupancy2: Variável fuzzy 2.
    """
    print("Funções de pertinência ajustadas para BufferOccupancy1:")
    for label in BufferOccupancy1.terms:
        print(f"Termo: {label}, Parâmetros: {BufferOccupancy1.terms[label].mf}")

    print("\nFunções de pertinência ajustadas para BufferOccupancy2:")
    for label in BufferOccupancy2.terms:
        print(f"Termo: {label}, Parâmetros: {BufferOccupancy2.terms[label].mf}")

    # Plotar as funções de pertinência ajustadas
    plot_membership_functions(BufferOccupancy1, "BufferOccupancy1 Ajustado")
    plot_membership_functions(BufferOccupancy2, "BufferOccupancy2 Ajustado")
