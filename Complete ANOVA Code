import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

def run_bartletts_test(list1, list2):
    stat, p = stats.bartlett(list1, list2)
    return p

def run_levenes_test(list1, list2):
    stat, p = stats.levene(list1, list2)
    return p

def run_shapiro_wilk_test(data):
    stat, p = stats.shapiro(data)
    return p

def run_anova(list1, list2):
    stat, p = stats.f_oneway(list1, list2)
    return p

def run_mann_whitney_u_test(list1, list2):
    stat, p = stats.mannwhitneyu(list1, list2)
    return p

def plot_histogram_with_fit(data, title='Histogram with Fit'):
    mu, std = stats.norm.fit(data)
    plt.hist(data, bins=25, alpha=0.6, color='g', density=True)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.title(f'{title}: mu = {mu:.2f}, std = {std:.2f}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.savefig(f'{title.replace(" ", "_").lower()}.png')
    plt.show()

def plot_qq(data, title='Normal Q-Q Plot'):
    plt.figure()
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(title)
    plt.savefig(f'{title.replace(" ", "_").lower()}.png')
    plt.show()

def plot_run_chart(data, title='Run Chart'):
    plt.plot(data, linestyle='-', marker='o')
    plt.axhline(np.mean(data), color='red', linestyle='dashed')
    plt.title(title)
    plt.xlabel('Observation')
    plt.ylabel('Value')
    plt.savefig(f'{title.replace(" ", "_").lower()}.png')
    plt.show()

def plot_capability_analysis(data, spec_limits, title='Capability Analysis'):
    Cp = (spec_limits[1] - spec_limits[0]) / (6 * np.std(data))
    Cpu = (spec_limits[1] - np.mean(data)) / (3 * np.std(data))
    Cpl = (np.mean(data) - spec_limits[0]) / (3 * np.std(data))
    Cpk = min(Cpu, Cpl)
    plt.hist(data, bins=25, alpha=0.6, color='g', density=True)
    plt.axvline(x=spec_limits[0], color='r', linestyle='--')
    plt.axvline(x=spec_limits[1], color='r', linestyle='--')
    plt.title(f'{title}\nCp = {Cp:.2f}, Cpk = {Cpk:.2f}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.savefig(f'{title.replace(" ", "_").lower()}.png')
    plt.show()

# Example usage:
list1 = [1.0, 2.0, 3.0, 4.0, 5.0, 4.76, 3.40, 3.98, 5.24, 4.87, 2.02, 3.95, 2.85, 2.90, 3.41]
list2 = [14.0, 6.0, 7.0, 8.0, 9.0, 14.82, 14.71, 14.48, 15.73, 15.62, 9.71, 1, 10.29, 11.42, 10.11]
spec_limits = [0.5, 5.5]  # Replace with your specification limits

# Run statistical tests
bartlett_p = run_bartletts_test(list1, list2)
levene_p = run_levenes_test(list1, list2)
shapiro_p1 = run_shapiro_wilk_test(list1)
shapiro_p2 = run_shapiro_wilk_test(list2)
anova_p = run_anova(list1, list2) if shapiro_p1 > 0.05 and shapiro_p2 > 0.05 else None
mannwhitney_p = run_mann_whitney_u_test(list1, list2) if anova_p is None else None

# Output the results
print(f"Bartlett's test p-value: {bartlett_p:.4f}")
print(f"Levene's test p-value: {levene_p:.4f}")
print(f"Shapiro-Wilk test p-value for list 1: {shapiro_p1:.4f}")
print(f"Shapiro-Wilk test p-value for list 2: {shapiro_p2:.4f}")
if anova_p is not None:
    print(f"ANOVA p-value: {anova_p:.4f}")
elif mannwhitney_p is not None:
    print(f"Mann-Whitney U test p-value: {mannwhitney_p:.4f}")

# Plotting for List 1
plot_histogram_with_fit(list1, 'Histogram with Fit for List 1')
plot_histogram_with_fit(list2, 'Histogram with Fit for List 2')

plot_run_chart(list1, 'Run Chart for List 1')
plot_run_chart(list2, 'Run Chart for List 2')

plot_capability_analysis(list1, spec_limits, 'Capability Analysis for List 1')
plot_capability_analysis(list2, spec_limits, 'Capability Analysis for List 2')

plot_qq(list1, 'Q-Q Plot for List 1')
plot_qq(list2, 'Q-Q Plot for List 2')


# Determine the result based on p-values
result = "There is a difference" if (anova_p is not None and anova_p <= 0.05) else \
         "There is a difference (Mann-Whitney U test)" if (mannwhitney_p is not None and mannwhitney_p <= 0.05) else \
         "There is no difference"
print(result)
