# Programming Final Project: Analyzing the Relationship between Health Expenditure and Malnutrition Death Rate
# By: Elena Comellas

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.iolib.summary2 import summary_col


def load_and_merge_data(death_path: str, health_path: str, gdp_path: str) -> pd.DataFrame:
    """
    Loads and merges death rate, healthcare expenditure, and GDP datasets.

    Args:
        death_path (str): File path for malnutrition death rate dataset.
        health_path (str): File path for health expenditure dataset.
        gdp_path (str): File path for GDP per capita dataset.

    Returns:
        pd.DataFrame: Cleaned and merged dataset for the year 2021.
    """
    death = pd.read_csv(death_path)[['Entity', 'Year', 'Death rate from protein-energy malnutrition among both sexes']]
    death.columns = ['Country', 'Year', 'Malnutrition_Death_Rate']

    health = pd.read_csv(health_path)[[
        'Entity', 'Year', 'Current health expenditure (CHE) as percentage of gross domestic product (GDP) (%)']]
    health.columns = ['Country', 'Year', 'Health_Expenditure_GDP']

    gdp = pd.read_csv(gdp_path)[['Entity', 'Year', 'GDP per capita, PPP (constant 2021 international $)']]
    gdp.columns = ['Country', 'Year', 'GDP_per_capita']

    merged = pd.merge(pd.merge(death, health, on=['Country', 'Year']), gdp, on=['Country', 'Year'])
    return merged[merged['Year'] == 2021].dropna()


def assign_gdp_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns GDP per capita tertile groups to countries.

    Args:
        df (pd.DataFrame): Merged dataset.

    Returns:
        pd.DataFrame: Dataset with new GDP group column.
    """
    df['GDP_Group'] = pd.qcut(df['GDP_per_capita'], 3, labels=['Low-GDP', 'Medium-GDP', 'High-GDP'])
    return df


def summarize_statistics(df: pd.DataFrame) -> None:
    """
    Prints summary statistics overall and by GDP group.

    Args:
        df (pd.DataFrame): Dataset with GDP group labels.
    """
    print("Overall Summary Statistics:")
    print(df[['Malnutrition_Death_Rate', 'Health_Expenditure_GDP', 'GDP_per_capita']].describe())

    for group in df['GDP_Group'].unique():
        print(f"\nSummary for {group}:")
        print(df[df['GDP_Group'] == group][['Malnutrition_Death_Rate', 'Health_Expenditure_GDP']].describe())


def plot_distributions(df: pd.DataFrame) -> None:
    """
    Plots histograms and boxplots of variables.

    Args:
        df (pd.DataFrame): Dataset to visualize.
    """
    # Histograms
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df['Malnutrition_Death_Rate'], bins=20, color='skyblue')
    plt.title("Histogram: Malnutrition Death Rate")

    plt.subplot(1, 2, 2)
    sns.histplot(df['Health_Expenditure_GDP'], bins=20, color='lightgreen')
    plt.title("Histogram: Health Expenditure (% of GDP)")
    plt.tight_layout()
    plt.show()

    # Boxplots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df['Malnutrition_Death_Rate'], color='skyblue')
    plt.title("Boxplot: Malnutrition Death Rate")

    plt.subplot(1, 2, 2)
    sns.boxplot(y=df['Health_Expenditure_GDP'], color='lightgreen')
    plt.title("Boxplot: Health Expenditure (% of GDP)")
    plt.tight_layout()
    plt.show()


def plot_scatter(df: pd.DataFrame) -> None:
    """
    Plots a scatterplot with regression lines by GDP group.

    Args:
        df (pd.DataFrame): Dataset with GDP groups.
    """
    sns.lmplot(data=df, x='Health_Expenditure_GDP', y='Malnutrition_Death_Rate',
               hue='GDP_Group', height=5, aspect=1.5)
    plt.title('Scatter: Health Expenditure vs Malnutrition Death Rate (2021)')
    plt.show()


def run_regressions(df: pd.DataFrame) -> dict:
    """
    Runs OLS regressions by GDP group.

    Args:
        df (pd.DataFrame): Dataset with GDP groups.

    Returns:
        dict: Dictionary of regression models per GDP group.
    """
    results = {}
    for group in df['GDP_Group'].unique():
        subset = df[df['GDP_Group'] == group]
        model = smf.ols('Malnutrition_Death_Rate ~ Health_Expenditure_GDP', data=subset).fit(cov_type='HC0')
        results[group] = model
    return results


def interpret_results(results: dict) -> None:
    """
    Prints a regression summary table and interprets economic meaning of results.

    Args:
        results (dict): Dictionary of fitted regression models.
    """
    summary = summary_col([results[g] for g in results],
                          stars=True,
                          model_names=list(results.keys()),
                          info_dict={'N': lambda x: f"{int(x.nobs)}",
                                     'R2': lambda x: f"{x.rsquared:.2f}"})
    print(summary)

    print("\nECONOMIC INTERPRETATION:")
    for group, model in results.items():
        print(f"\n{group} Group:")
        for var, coef in model.params.items():
            pval = model.pvalues[var]
            sig = "significant" if pval < 0.05 else "not significant"
            direction = "decrease" if coef < 0 else "increase"

            if var != "Intercept":
                print(f"  - A 1 unit increase in {var} is associated with a {abs(coef):.2f} unit {direction} "
                      f"in malnutrition death rate. This effect is {sig} (p = {pval:.4f}).")

        r2 = model.rsquared
        print(f"  - R² = {r2:.3f}: This means that {r2*100:.1f}% of the variation in malnutrition death rate "
              f"is explained by health expenditure in this GDP group.")

        # Economic intuition
        if model.pvalues["Health_Expenditure_GDP"] < 0.05:
            print("  ➤ Economically, this supports the idea that greater investment in health (as % of GDP) "
                  "is associated with fewer deaths from malnutrition in this income tier.")
        else:
            print("  ➤ Economically, this suggests that healthcare spending is not significantly linked to malnutrition "
                  "outcomes in this group — possibly due to inefficiencies or other stronger determinants.")


def breusch_pagan_tests(results: dict) -> None:
    """
    Performs Breusch-Pagan test for heteroskedasticity.

    Args:
        results (dict): Dictionary of regression models.
    """
    print("\nBreusch-Pagan Test Results:")
    for group, model in results.items():
        bp = het_breuschpagan(model.resid, model.model.exog)
        labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
        print(f"\n{group} Group:")
        for label, stat in zip(labels, bp):
            print(f"{label}: {stat:.4f}")


def main() -> None:
    """
    Main execution pipeline for the analysis.
    """
    df = load_and_merge_data(
        "data/death-rate-from-malnutrition-ghe.csv",
        "data/total-healthcare-expenditure-gdp.csv",
        "data/gdp-per-capita-worldbank.csv"
    )
    df = assign_gdp_groups(df)

    summarize_statistics(df)
    plot_distributions(df)
    plot_scatter(df)

    results = run_regressions(df)
    interpret_results(results)
    breusch_pagan_tests(results)


if __name__ == "__main__":
    main()
