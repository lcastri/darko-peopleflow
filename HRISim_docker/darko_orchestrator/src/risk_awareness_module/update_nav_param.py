#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, skewnorm

#%%
# percorso_file_csv = 'output_nav_times_new.csv'
percorso_file_csv = 'times_1501_alpha01.csv'

df = pd.read_csv(percorso_file_csv)

# # Creare nuove colonne con start minimo ed end massimo
# df['min_start'] = df[['start', 'end']].min(axis=1)
# df['max_end'] = df[['start', 'end']].max(axis=1)

# # Raggruppare i dati per le nuove colonne min_start e max_end
# grouped_data = df.groupby(['min_start', 'max_end'])['actual_time'].apply(list).reset_index()

# print(grouped_data)
from scipy.stats import gaussian_kde
# Calcolo della KDE
data = df['actual_time']
kde = gaussian_kde(data, bw_method=0.5)

# Creazione dei punti per l'asse x
x = np.linspace(data.min(), data.max(), 500)

# Calcolo dei valori della KDE
kde_values = kde.evaluate(x)

# Calcolo di media, varianza e percentili
mean = data.mean()
variance = data.var()
percentile_25 = np.percentile(data, 25)
percentile_50 = np.percentile(data, 50)
percentile_75 = np.percentile(data, 75)
percentile_80 = np.percentile(data, 80)
percentile_90 = np.percentile(data, 90)

# Aggiunta dell'istogramma
plt.hist(data, bins=30, density=True, alpha=0.5, label='Istogramma')

# Aggiunta di un riquadro con statistiche
textstr = '\n'.join((
    f"Media: {mean:.2f}",
    f"Varianza: {variance:.2f}",
    f"25° percentile: {percentile_25:.2f}",
    f"50° percentile: {percentile_50:.2f}",
    f"75° percentile: {percentile_75:.2f}",
    f"80° percentile: {percentile_80:.2f}",
    f"90° percentile: {percentile_90:.2f}"
))
plt.gca().text(0.95, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle="round", alpha=0.5))

# Creazione del grafico
plt.plot(x, kde_values)
plt.title('Kernel Density Estimate of Actual Times')
plt.xlabel('Actual Time')
plt.ylabel('Density')
plt.show()






# values_for_4_10 = grouped_data.loc[(grouped_data['min_start'] == 10) & (grouped_data['max_end'] == 14), 'actual_time'].values[0]


# # Supponiamo che values_for_4_4 sia la tua lista di valori
# values_for_4_4 = np.array(values_for_4_10)

# # Creazione dell'istogramma

# # Creazione della distribuzione normale approssimata
# mu, std = norm.fit(values_for_4_4)
# mu0, std0 = norm.fit((values_for_4_4-mu)/mu)
# plt.hist((values_for_4_4-mu)/mu, bins=20, density=True, alpha=0.6, color='g', label='Dati')
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p = norm.pdf(x, mu0, std0)
# plt.plot(x, p, 'k', linewidth=2, label='Distribuzione normale approssimata')

# # Aggiunta di legende e titolo
# plt.title('Confronto dati con distribuzione normale approssimata')
# plt.legend()

# # Visualizzazione del grafico
# plt.show()

# %%

# CREAZIONE PLOT E SOTTOPLOT
# Calcolare il numero totale di sottoplot
num_totale_sottoplot = len(grouped_data)
# Impostare il numero di colonne per la griglia
num_colonne = 3  # Modifica il numero di colonne a tuo piacimento
# Calcolare il numero di righe necessarie
num_righe = int(np.ceil(num_totale_sottoplot / num_colonne))
# Creare una griglia di sottoplot
fig, axs = plt.subplots(num_righe, num_colonne, figsize=(15, 5 * num_righe))

# FITTING DEI DATI E CREAZIONE DEL SOTTOPLOT PER OGNI COPPIA DI NODI 
mu0_list= []
std0_list= []
normalized_times_list = [0]
# Iterare su ogni coppia di min_start e max_end
for indice, (index, row) in enumerate(grouped_data.iterrows()):
    actual_time_list = row['actual_time']
    print(len(actual_time_list))
    # Fai il fitting per ogni coppia di nodi e genera mu e std
    mu, std = norm.fit(actual_time_list)
    
    # Normalizza i dati e ri-esegui il fitting
    normalized_list = (actual_time_list - mu) / mu
    mu0, std0 = norm.fit(normalized_list)
    
    # Salva i dati
    normalized_times_list.extend(normalized_list)
    mu0_list.append(mu0)
    std0_list.append(std0)

    # Posizionare il plot nella griglia di sottoplot
    riga = indice // num_colonne
    colonna = indice % num_colonne

    # Plot dell'istogramma e della distribuzione normale
    axs[riga, colonna].hist(normalized_list, bins=20, density=True, alpha=0.6, color='g', label='Dati')
    xmin, xmax = axs[riga, colonna].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu0, std0)
    axs[riga, colonna].plot(x, p, 'k', linewidth=2, label='Distr. normale appross.')

    # Aggiungere legenda e titolo al sottoplot
    axs[riga, colonna].set_title(f'Confronto dati con distr. normale appross.[{row["min_start"]},{row["max_end"]}]')
    axs[riga, colonna].legend(bbox_to_anchor=(0.45, 1), loc='upper left')

    # Aggiungere un box con i valori di mu, std, mu0, std0
    box_text = f'mu: {mu:.2f}\nstd: {std:.2f}\nmu0: {mu0:.2f}\nstd0: {std0:.2f}'
    axs[riga, colonna].text(0.95, 0.80, box_text, transform=axs[riga, colonna].transAxes, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Aggiungere titolo al maxi-plot
fig.suptitle('Confronto dati con distribuzione normale approssimata')
# Regolare automaticamente la posizione della griglia
fig.tight_layout(rect=[0, 0, 1, 0.96])
# Visualizzare il maxi-plot
plt.show()



# PLOT DI CONFRONTO TRA TUTTE LE CURVE FITTATE
xmin, xmax = -1, 1
x = np.linspace(xmin, xmax, 100)

# Iterare su ogni coppia di mu0 e std0
for mu0, std0 in zip(mu0_list, std0_list):
    # Calcolare la PDF della distribuzione normale approssimata
    p = norm.pdf(x, mu0, std0)

    # Plottare la distribuzione normale approssimata
    plt.plot(x, p, label=f'std0={std0:.3f}')

# Aggiungere legende e titolo
plt.title('Distribuzioni normali approssimate con diverse mu0 e std0')
plt.legend(loc='upper right')
plt.show()


# FITTING CONSIDERANDO TUTTI I DATI INSIEME E PLOT DELLA RISULTANTE DISTRIB. NORMALE APPROSS. 

normalized_times_list.remove(0)
mu0, std0 = norm.fit(normalized_times_list)
a, loc, scale = skewnorm.fit(normalized_times_list)
plt.hist(normalized_times_list, bins=20, density=True, alpha=0.6, color='g', label='Dati')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = skewnorm.pdf(x, a, loc, scale)
# p = norm.pdf(x, mu0, std0)
plt.plot(x, p, 'k', linewidth=2, label='Distr. normale appross.')

# Aggiunta di legende e titolo
plt.title('Confronto dati con distribuzione normale approssimata')
plt.legend(bbox_to_anchor=(0.55, 1), loc='upper left')

# Aggiungere un box con i valori di mu, std, mu0, std0
box_text = f'mu0: {mu0:.2f}\nstd0: {std0:.2f}'
plt.text(0.95, 0.80, box_text, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Calcola i percentili (ad esempio, 25%, 50%, 75%)
# percentile_25 = norm.ppf(0.25, mu0, std0)
# percentile_50 = norm.ppf(0.50, mu0, std0)  # Mediana
# percentile_75 = norm.ppf(0.75, mu0, std0)

percentile_25 = skewnorm.ppf(0.10, a, loc, scale)
percentile_35 = skewnorm.ppf(0.35, a, loc, scale)  # Mediana
percentile_50 = skewnorm.ppf(0.50, a, loc, scale)  # Mediana
percentile_75 = skewnorm.ppf(0.90, a, loc, scale)

# Aggiungi linee verticali per i percentili
plt.axvline(percentile_25, color='r', linestyle='--', label='25% percentile')
plt.axvline(percentile_35, color='r', linestyle='--', label='35% percentile')
plt.axvline(percentile_50, color='b', linestyle='--', label='50% percentile (Mediana)')
plt.axvline(percentile_75, color='m', linestyle='--', label='75% percentile')

# Aggiungi un box con i valori dei percentili
box_text = f'25% percentile: {percentile_25:.2f}\n35% percentile: {percentile_35:.2f}\n50% percentile (Mediana): {percentile_50:.2f}\n75% percentile: {percentile_75:.2f}'
plt.text(0.95, 0.65, box_text, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Visualizzazione del grafico
plt.show()

# %%

# percorso_file_csv = 'output_nav_times_v_agg.csv'
percorso_file_csv = 'times_1501_alpha01.csv'
df = pd.read_csv(percorso_file_csv)

act_time = df["actual_time"][200:300]
est_time = df["estimated_time"][200:300]

diff = act_time/est_time

print(diff)


df['error'] = act_time - est_time

# Crea un grafico a dispersione dell'errore di previsione
plt.figure(figsize=(10, 6))
# plt.scatter(range(len(df)), df['error'], color='red')
plt.plot(range(len(df)), df['error'].values, color='blue', marker='o', linestyle='-')
plt.title('Errore di Previsione')
plt.xlabel('Iterazione')
plt.ylabel('Errore (Actual Time - Estimated Time)')
plt.show()
#%%

import matplotlib.pyplot as plt


# # Creazione di un grafico di dispersione
# plt.scatter(range(len(diff)), diff, label='Dati')

# # Aggiunta di legenda e titolo
# plt.title('Grafico di Dispersione')
# plt.xlabel('Indice')
# plt.ylabel('Valori')
# plt.legend()

xmin, xmax = -5, 8
x = np.linspace(xmin, xmax, 100)

mu, std = norm.fit(diff)
p = norm.pdf(x, mu, std)

# plt.plot((x-mu)/mu, p, 'k', linewidth=2, label='Distr. normale appross.')
plt.plot(x, p, 'k', linewidth=2, label='Distr. normale appross.')

box_text = f'mu: {mu:.2f}\nstd: {std:.2f}'
plt.text(0.95, 0.65, box_text, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


# Aggiunta di legende e titolo
plt.title('Confronto dati con distribuzione normale approssimata')
plt.legend(bbox_to_anchor=(0.55, 1), loc='upper left')


# Visualizzazione del grafico
plt.show()


#%%

# est_time = df["estimated_time"]
est_time_adj = est_time*mu

diff2 = act_time/est_time_adj

mu2, std2 = norm.fit(diff2)
diff_norm = (diff2-mu2)/mu2
xmin, xmax = -5, 8
x = np.linspace(xmin, xmax, 100)

mu0, std0 = norm.fit(diff_norm)
p = norm.pdf(x, mu0, std0)

plt.plot(x, p, 'k', linewidth=2, label='Distr. normale appross.')

# Aggiunta di legende e titolo
plt.title('Confronto dati con distribuzione normale approssimata')
plt.legend(bbox_to_anchor=(0.55, 1), loc='upper left')

box_text = f'mu: {mu2:.2f}\nstd: {std2:.2f}\nmu0: {mu0:.2f}\nstd0: {std0:.2f}'
plt.text(0.95, 0.65, box_text, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Visualizzazione del grafico
plt.show()

# %%
