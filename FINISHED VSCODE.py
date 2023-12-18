from matplotlib import table
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency
import seaborn as sns
from prettytable import PrettyTable
import subprocess
import statsmodels.api as sm

subprocess.check_call(["pip", "install", "prettytable"])

# Read the data from the CSV file CHANGE THIS TO YOUR FILE PATH!!
data = pd.read_csv(r'c:\Users\sm16\OneDrive - HydraForce Inc\Documents\CRAP FOLDER\2.5278616 - EHPR98-G35 - G3 Correlation INC VS LTD.csv', skiprows=1)

# Get the column names
columns = data.columns

#------------------------------------------------------------------------------------

# Bartlett's Test for Equal Variances (Initial step in the ANOVA process) Very sensitive test
def run_bartletts_test(listINC, listLTD):
    stat, p = stats.bartlett(listINC, listLTD)
    return p

def run_kruskal_wallis_test(listINC, listLTD):
    # Run the Kruskal-Wallis test
    kruskal_statistic, kruskal_p = stats.kruskal(listINC, listLTD)
    return kruskal_statistic, kruskal_p

def run_moods_median_test(listINC, listLTD):
    # Run Mood's Median test
    moods_median_statistic, moods_median_p, mood_table, _ = stats.median_test(listINC, listLTD)
    return moods_median_statistic, moods_median_p, mood_table

def run_chi_square_test(listINC, listLTD):
    # Create a contingency table
    contingency_table = pd.crosstab(listINC, listLTD)
    # Run the Chi-Square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return chi2, p

# Levene's Test for Equal Variances (Alternative if data fails Bartlett's test) Less sensitive test
def run_levenes_test(listINC, listLTD):
    stat, p = stats.levene(listINC, listLTD)
    return p

def run_runs_test(data):
    """
    Perform a simple Wald-Wolfowitz runs test on the data.
    
    :param data: 1D list or array of numbers
    :return: Z-score, p-value
    """
    n1 = np.sum(data < np.median(data))
    n2 = np.sum(data >= np.median(data))
    runs = np.sum(np.diff(data < np.median(data)) != 0) + 1
    expected_runs = 1 + 2 * n1 * n2 / (n1 + n2)
    std_dev_runs = np.sqrt(2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2)**2 * (n1 + n2 - 1)))
    
    # Avoid division by zero
    if std_dev_runs == 0:
        return np.nan, np.nan
    
    z_score = (runs - expected_runs) / std_dev_runs
    p_value = 2 * stats.norm.sf(np.abs(z_score))
    
    return z_score, p_value

# Shapiro-Wilk Test for Normality (Check if data is normally distributed)
def run_shapiro_wilk_test(data):
    # Check if the range of the data is zero
    if max(data) - min(data) == 0:
        print("Skipping Shapiro-Wilk test because data range is zero")
        return None
    else:
        # Run the Shapiro-Wilk test
        statistic, p_value = stats.shapiro(data)
        return p_value

# ANOVA F-Test (Used when data passes normality and equal variance tests)
def run_anova(listINC, listLTD):
    stat, p = stats.f_oneway(listINC, listLTD)
    return p

def run_mann_whitney_u_test(listINC, listLTD):
    # Note: 'exact' method may take a long time for large samples
    stat, p = stats.mannwhitneyu(listINC, listLTD, alternative='two-sided', method='exact')
    return p

def run_paired_t_test(listINC, listLTD):
    if len(listINC) != len(listLTD):
        raise ValueError("Lists are of unequal length")
    stat, p = stats.ttest_rel(listINC, listLTD)
    return stat, p

# Define the generate_report function
def generate_report(listINC, listLTD, test_results):
    table_r = PrettyTable()
    table_r.field_names = ["Test", "Statistic", "p-value", "Result"]

    for test_name, values in test_results.items():
        if values is not None:
            statistic, p_value = values
            result = "Significant" if p_value < 0.05 else "Not significant"
            table_r.add_row([test_name, f"{statistic:.4f}", f"{p_value:.4f}", result])
        else:
            table_r.add_row([test_name, "N/A", "N/A", "Not Applicable"])

    return table_r

def detect_outliers_iqr(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    return any((data < lower_bound) | (data > upper_bound))

# Plot Histogram with Fit (Part of investigating data variances)
def plot_histogram_with_fit(data, title='Histogram with Fit'):
    mu, std = stats.norm.fit(data)
    sns.histplot(data, kde=True, color='g')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.title(f'{title}: mu = {mu:.2f}, std = {std:.2f}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.savefig(f'{title.replace(" ", "_").lower()}.png')
    plt.show()

# Plot Q-Q Plot (Part of testing for normality in the ANOVA process)
def plot_qq(data, title='Normal Q-Q Plot'):
    plt.figure()
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(title)
    plt.savefig(f'{title.replace(" ", "_").lower()}.png')
    plt.show()

# Plot Run Chart (Part of investigating data independence)
def plot_run_chart(data, title='Run Chart'):
    plt.plot(data, linestyle='-', marker='o')
    plt.axhline(np.mean(data), color='red', linestyle='dashed')
    plt.title(title)
    plt.xlabel('Observation')
    plt.ylabel('Value')
    plt.savefig(f'{title.replace(" ", "_").lower()}.png')
    plt.show()

# Plot Capability Analysis (Part of Quality Control before ANOVA)
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


def plot_boxplot(listINC, listLTD):
    # Create a box plot
    plt.figure(figsize=(12, 6))
    plt.boxplot([listINC, listLTD])
    plt.title('Box Plot')
    plt.show()

def plot_scatter(listINC, listLTD, window_size=3):
    # Create a scatter plot
    plt.scatter(range(len(listINC)), listINC, label='listINC')
    plt.scatter(range(len(listLTD)), listLTD, label='listLTD')
    
    # Calculate moving averages
    moving_avg_listINC = np.convolve(listINC, np.ones(window_size)/window_size, mode='valid')
    moving_avg_listLTD = np.convolve(listLTD, np.ones(window_size)/window_size, mode='valid')
    
    # Plot moving average lines
    plt.plot(range(window_size-1, len(listINC)), moving_avg_listINC, color='red', label='Moving Average (listINC)')
    plt.plot(range(window_size-1, len(listLTD)), moving_avg_listLTD, color='blue', label='Moving Average (listLTD)')
    
    plt.title('Scatter Plot with Moving Average')
    plt.legend()
    plt.show()


def run_paired_t_test(listINC, listLTD):
    if len(listINC) != len(listLTD):
        raise ValueError("Lists are of unequal length")
    stat, p = stats.ttest_rel(listINC, listLTD)
    return stat, p

# Function to interpret p-value
def interpret_p_value(p):
    if p < 0.05:
        return "Significant"
    elif p >= 0.05:
        return "Not significant"
    else:
        return "N/A"

# Automated Analysis Function
def automated_analysis(test_data):
    analysis_statements = []

    # Interpret Chi-Square Test
    chi_square_interpretation = "significant association" if test_data[0][3] == "Significant" else "no significant association"
    analysis_statements.append(f"Chi-Square test indicates {chi_square_interpretation} between the two categorical variables.")

    # Interpret Variance Test (Levene's)
    if test_data[1][3] == "Significant":
        analysis_statements.append("Levene's test suggests that the variances are not equal.")
    else:
        analysis_statements.append("Levene's test indicates equal variances across the groups.")

    # Interpret Normality Tests
    normality_listINC = "normal distribution" if test_data[2][3] == "Significant" else "non-normal distribution"
    normality_listLTD = "normal distribution" if test_data[3][3] == "Significant" else "non-normal distribution"
    analysis_statements.append(f"listINC shows a {normality_listINC}. listLTD shows a {normality_listLTD}.")

    # Interpret ANOVA or Mann-Whitney U Test
    if test_data[4][3] != "N/A":
        anova_interpretation = "significant differences" if test_data[4][3] == "Significant" else "no significant differences"
        analysis_statements.append(f"ANOVA test indicates {anova_interpretation} between group means.")
    elif test_data[5][3] != "N/A":
        mann_whitney_interpretation = "significant differences" if test_data[5][3] == "Significant" else "no significant differences"
        analysis_statements.append(f"Mann-Whitney U test indicates {mann_whitney_interpretation} between distributions of two groups.")

    # Interpret Paired t-test
    if test_data[6][3] == "Significant":
        analysis_statements.append("Paired t-test shows significant differences between the paired samples.")
    elif test_data[6][3] == "Not significant":
        analysis_statements.append("Paired t-test indicates no significant differences between the paired samples.")
    elif test_data[6][3] == "Error":
        analysis_statements.append("Paired t-test could not be conducted due to data issues.")

    # Interpret Kruskal-Wallis Test
    kruskal_interpretation = "significant differences" if test_data[-2][3] == "Significant" else "no significant differences"
    analysis_statements.append(f"Kruskal-Wallis test indicates {kruskal_interpretation} between group distributions.")

    # Interpret Mood's Median Test
    moods_median_interpretation = "significant differences" if test_data[-1][3] == "Significant" else "no significant differences"
    analysis_statements.append(f"Mood's Median test indicates {moods_median_interpretation} in medians of the two groups.")

    return " ".join(analysis_statements)

def create_summary_statement(test_data):
    significant_count = sum(1 for _, _, _, interpretation in test_data if interpretation == "Significant")
    total_tests = len(test_data)
    normal_count = sum(1 for test, _, _, interpretation in test_data if "Normality" in test and interpretation == "Significant")

    if significant_count == 0:
        return "The tests show no significant differences or associations, indicating consistent and expected results across the data."
    elif significant_count == total_tests:
        return "The tests indicate significant differences or associations in all aspects, suggesting considerable variability in the data."
    elif normal_count > 0 and significant_count < total_tests:
        return "The tests reveal a mix of normality in some aspects of the data and significant differences in others, indicating variability and some consistency."
    else:
        return "The tests show that some aspects of the data are normal and consistent, while others differ significantly from what was expected."

#------------------------------------------------------------------------------------

for i in range(0, len(columns), 2):
    try:
        listINC = data[columns[i]]
        listLTD = data[columns[i+1]]

        # What columns we are now testing
        print(f"Now Analysing {columns[i]} and {columns[i+1]}:\n")
    
        # Run statistical tests
        chi2, p = run_chi_square_test(listINC, listLTD)
        levene_p = run_levenes_test(listINC, listLTD)
        shapiro_p1 = run_shapiro_wilk_test(listINC)
        shapiro_p2 = run_shapiro_wilk_test(listLTD)
        anova_p = run_anova(listINC, listLTD) if shapiro_p1 > 0.05 and shapiro_p2 > 0.05 else None
        mannwhitney_p = run_mann_whitney_u_test(listINC, listLTD) if anova_p is None else None
        kruskal_statistic, kruskal_p = run_kruskal_wallis_test(listINC, listLTD)
        moods_median_statistic, moods_median_p, _ = run_moods_median_test(listINC, listLTD)
        
        # Run the Runs Test for both datasets
        z_stat_inc, p_value_inc = run_runs_test(listINC)
        z_stat_ltd, p_value_ltd = run_runs_test(listLTD)
        
        # Run the paired t-test
        try:
            paired_t_stat, paired_t_p = run_paired_t_test(listINC, listLTD)
            paired_t_result = (f"{paired_t_stat:.5f}", f"{paired_t_p:.5f}")
        except ValueError as e:
            paired_t_result = ("Error", f"{e}")
            
            # Prepare your test data
        test_data = [
            ["Runs Test INC", f"{z_stat_inc:.5f}", f"{p_value_inc:.5f}", interpret_p_value(p_value_inc)],
            ["Runs Test LTD", f"{z_stat_ltd:.5f}", f"{p_value_ltd:.5f}", interpret_p_value(p_value_ltd)],
            ["Chi-Square (Independence)", f"{chi2:.5f}", f"{p:.5f}", interpret_p_value(p)],
            ["Levene's (Variances)", "N/A", f"{levene_p:.5f}", interpret_p_value(levene_p)],
            ["Shapiro-Wilk (Normality, listINC)", "N/A", f"{shapiro_p1:.5f}", interpret_p_value(shapiro_p1)],
            ["Shapiro-Wilk (Normality, listLTD)", "N/A", f"{shapiro_p2:.5f}", interpret_p_value(shapiro_p2)],
            ["ANOVA (Means)", "N/A", f"{anova_p:.5f}" if anova_p is not None else "Not applicable", interpret_p_value(anova_p) if anova_p is not None else "N/A"],
            ["Mann-Whitney U (Distributions)", "N/A", f"{mannwhitney_p:.5f}" if mannwhitney_p is not None else "Not applicable", interpret_p_value(mannwhitney_p) if mannwhitney_p is not None else "N/A"],
            ["Paired t-test (Parametric)", paired_t_result[0], paired_t_result[1], interpret_p_value(float(paired_t_result[1])) if paired_t_result[1] != "Error" else "Error"],
            ["Kruskal-Wallis Test (Non-Parametric)", f"{kruskal_statistic:.5f}", f"{kruskal_p:.5f}", interpret_p_value(kruskal_p)],
            ["Mood's Median Test (Non-Parametric)", f"{moods_median_statistic:.5f}", f"{moods_median_p:.5f}", interpret_p_value(moods_median_p)],
        ]

        
        # Create a PrettyTable object
        table = PrettyTable()
        table.field_names = ["Test", "Statistic", "p-value", "Interpretation"]

        # Add rows to the table
        for row in test_data:
            table.add_row(row)

        # Print the table
        print(table)

        # plot data from previous functions made
        plot_boxplot(listINC, listLTD)
        plot_scatter(listINC, listLTD)

        # Plot Histogram with Fit (Part of investigating data variances)
        plot_histogram_with_fit(listINC, f'Histogram with Fit for {columns[i]}')
        plot_histogram_with_fit(listLTD, f'Histogram with Fit for {columns[i+1]}')


        # Plot Q-Q Plot (Part of testing for normality in the ANOVA process)
        plot_qq(listINC, f'Q-Q Plot for {columns[i]}')
        plot_qq(listLTD, f'Q-Q Plot for {columns[i+1]}')

        # Plot Run Chart (Part of investigating data independence)
        plot_run_chart(listINC, f'Run Chart for {columns[i]}')
        plot_run_chart(listLTD, f'Run Chart for {columns[i+1]}')

        spec_limits = [0.5, 5.5]  # Replace with your specification limits
        # REMOVE HASH TO RUN THE GRAPH/REMEMBER TO CHANGE THE SPEC LIMITS!
        #plot_capability_analysis(listINC, spec_limits, f'Capability Analysis for {columns[i]}')
        #plot_capability_analysis(listLTD, spec_limits, f'Capability Analysis for {columns[i+1]}')

        # Run preliminary tests
        p_chi2, p_levene, p_shapiro1, p_shapiro2 = run_chi_square_test(listINC, listLTD)[1], run_levenes_test(listINC, listLTD), run_shapiro_wilk_test(listINC), run_shapiro_wilk_test(listLTD)

        # Check for independence and equal variances
        # Run preliminary tests
        p_chi2, p_levene, p_shapiro1, p_shapiro2 = run_chi_square_test(listINC, listLTD)[1], run_levenes_test(listINC, listLTD), run_shapiro_wilk_test(listINC), run_shapiro_wilk_test(listLTD)

        print("\nAlgorithmic Analysis:")

        # First, check for randomness in the data using the Runs Test
        if p_value_inc > 0.05 and p_value_ltd > 0.05:
            print(f"Data sequences are random (Runs Test p-values: INC = {p_value_inc:.5f}, LTD = {p_value_ltd:.5f}). Proceeding with variance test.")

            # Check for equal variances using Levene's test
            levene_p = run_levenes_test(listINC, listLTD)
            if levene_p <= 0.05:
                print(f"Variances are equal (Levene's test p-value: {levene_p:.5f}). Proceeding with normality test.")

                # Check normality of data
                shapiro_p1 = run_shapiro_wilk_test(listINC)
                shapiro_p2 = run_shapiro_wilk_test(listLTD)
                if shapiro_p1 <= 0.05 and shapiro_p2 <= 0.05:
                    print(f"Data is normally distributed (Shapiro-Wilk p-values: INC = {shapiro_p1:.5f}, LTD = {shapiro_p2:.5f}). Running ANOVA test.")
                    p_anova = run_anova(listINC, listLTD)
                    print(f"ANOVA test completed. There is a {'difference' if p_anova <= 0.05 else 'no difference'} (ANOVA p-value: {p_anova:.5f}).")
                else:
                    print(f"Data is not normally distributed (Shapiro-Wilk p-values: INC = {shapiro_p1:.5f}, LTD = {shapiro_p2:.5f}). Running Kruskal-Wallis test.")
                    kruskal_statistic, kruskal_p = run_kruskal_wallis_test(listINC, listLTD)
                    print(f"Kruskal-Wallis test completed. There is a {'difference' if kruskal_p <= 0.05 else 'no difference'} (Kruskal-Wallis p-value: {kruskal_p:.5f}).")
            
            else:
                print(f"Variances are not equal (Levene's test p-value: {levene_p:.5f}). Testing for normality.")
                shapiro_p1 = run_shapiro_wilk_test(listINC)
                shapiro_p2 = run_shapiro_wilk_test(listLTD)
                if shapiro_p1 > 0.05 and shapiro_p2 > 0.05:
                    print(f"Data is normally distributed (Shapiro-Wilk p-values: INC = {shapiro_p1:.5f}, LTD = {shapiro_p2:.5f}). Running Welch's t-test.")
                    t_stat, p_value = run_paired_t_test(listINC, listLTD)
                    print(f"Welch's t-test result: t-statistic = {t_stat:.5f}, p-value = {p_value:.5f}")
                    # Add a statement to interpret the result of the Welch's t-test
                    if p_value <= 0.05:
                        print("There is a statistically significant difference between the datasets.")
                    else:
                        print("There is no statistically significant difference between the datasets.")
                else:
                    print(f"Data is not normally distributed (Shapiro-Wilk p-values: INC = {shapiro_p1:.5f}, LTD = {shapiro_p2:.5f}). Checking for outliers.")
                    if detect_outliers_iqr(listINC) or detect_outliers_iqr(listLTD):
                        moods_median_statistic, moods_median_p, _ = run_moods_median_test(listINC, listLTD)
                        print(f"Mood's Median Test completed. There is a {'difference' if moods_median_p <= 0.05 else 'no difference'} (Mood's Median p-value: {moods_median_p:.5f}).")
                    else:
                        kruskal_statistic, kruskal_p = run_kruskal_wallis_test(listINC, listLTD)
                        print(f"Kruskal-Wallis test completed. There is a {'difference' if kruskal_p <= 0.05 else 'no difference'} (Kruskal-Wallis p-value: {kruskal_p:.5f}).")


        else:
            print(f"Data sequences are not random (Runs Test p-values: INC = {p_value_inc:.5f}, LTD = {p_value_ltd:.5f}). This might indicate a trend or structure in the data.")



        print(automated_analysis(test_data))

        # Generate and print summary statement with column names
        print("\nSummary Statement for Columns: " + columns[i] + " and " + columns[i+1] + ": ")
        print(create_summary_statement(test_data))

        # Generate and print automated analysis
        print("\nAutomated Analysis:")
        print(automated_analysis(test_data))


        print("\n----------------------------------------------------------------------------------------------------------------\n")


    except Exception as e:
        print(f"An error occurred while processing columns {columns[i]} and {columns[i+1]}: {e}\n")
